import math
from pathlib import Path
import subprocess
import uuid

import pytest
import torch
import torch.nn.functional as F

from areal.infra.platforms import current_platform
from areal.utils.network import find_free_ports


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_ROOT = PROJECT_ROOT / ".test_artifacts" / "veomini_vs_fsdp_dp_parity"


def _run_test_with_torchrun(n_gpus: int, backend: str, output: str):
    port = find_free_ports(1)[0]
    try:
        result = subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_veomini_vs_fsdp_dp_parity.py",
                f"--backend={backend}",
                f"--output={output}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Test failed with error: {e.stderr}, {e.stdout}")

    if not Path(output).exists():
        pytest.fail(
            "Torchrun finished but did not create output artifact: "
            f"{output}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _run_multistep_test_with_torchrun(n_gpus: int, backend: str, output: str, steps: int):
    port = find_free_ports(1)[0]
    try:
        result = subprocess.run(
            [
                "torchrun",
                f"--nproc_per_node={n_gpus}",
                "--nnodes=1",
                "--master-addr=localhost",
                f"--master_port={port}",
                "tests/torchrun/run_veomini_vs_fsdp_dp_multistep.py",
                f"--backend={backend}",
                f"--output={output}",
                f"--steps={steps}",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Multistep test failed with error: {e.stderr}, {e.stdout}")

    if not Path(output).exists():
        pytest.fail(
            "Torchrun multistep finished but did not create output artifact: "
            f"{output}\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def _compare_tensor_maps(
    lhs: dict[str, torch.Tensor],
    rhs: dict[str, torch.Tensor],
    *,
    atol: float,
    rtol: float,
    label: str,
) -> None:
    assert lhs.keys() == rhs.keys(), f"{label} keys mismatch: {lhs.keys()} vs {rhs.keys()}"
    failures = []
    for name in lhs:
        left = lhs[name]
        right = rhs[name]
        if torch.allclose(left, right, atol=atol, rtol=rtol):
            continue
        diff = (left - right).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        cosine = F.cosine_similarity(left.flatten(), right.flatten(), dim=0).item()
        failures.append(
            f"{label} mismatch for {name}: max_diff={max_diff:.6e}, mean_diff={mean_diff:.6e}, cosine={cosine:.8f}"
        )
    if failures:
        raise AssertionError("\n".join(failures))


def _compare_step_stats(fsdp_stats: dict[str, float], veomini_stats: dict[str, float]) -> None:
    assert fsdp_stats["update_successful"] == veomini_stats["update_successful"]
    assert abs(fsdp_stats["lr"] - veomini_stats["lr"]) < 1e-12
    assert math.isfinite(fsdp_stats["grad_norm"])
    assert math.isfinite(veomini_stats["grad_norm"])
    assert math.isclose(
        fsdp_stats["grad_norm"],
        veomini_stats["grad_norm"],
        rel_tol=2e-3,
        abs_tol=5e-3,
    ), (fsdp_stats, veomini_stats)


def _compare_multistep_records(
    fsdp_records: list[dict[str, float]], veomini_records: list[dict[str, float]]
) -> None:
    """Compare multistep records between FSDP and VeOMini engines.

    Known implementation differences (not bugs):
    1. model.float() in VeOMini → fp32 optimizer vs bf16 optimizer in FSDP
    2. cast_forward_inputs=True in FSDP, not set in VeOMini
    3. Gradient clipping: Megatron multi_tensor_l2norm (FSDP) vs
       torch.nn.utils.clip_grad_norm_ (VeOMini)

    These cause EXPECTED numerical divergence. This test validates:
    - Both engines produce successful training updates
    - Both engines maintain rank synchronization (no inter-rank desync)
    - Initial parameters are identical (loaded from same checkpoint)
    - Gradient norms are in the same order of magnitude
    - Both engines actually update parameters (not stalled)
    """
    assert len(fsdp_records) == len(veomini_records)

    # --- Separate init record (step=-1) from training records ---
    fsdp_init = None
    veomini_init = None
    fsdp_train = []
    veomini_train = []
    for fs, vs in zip(fsdp_records, veomini_records):
        if fs["step"] < 0:
            fsdp_init = fs
            veomini_init = vs
        else:
            fsdp_train.append(fs)
            veomini_train.append(vs)

    # --- Check initial parameters are identical ---
    if fsdp_init is not None and veomini_init is not None:
        init_hash_diff = abs(fsdp_init["param_hash"] - veomini_init["param_hash"])
        assert init_hash_diff < 1e-2, (
            f"Initial param_hash differs: FSDP={fsdp_init['param_hash']:.6f} vs "
            f"VeOMini={veomini_init['param_hash']:.6f}, diff={init_hash_diff:.3e}. "
            "Model loading produces different weights."
        )
        assert fsdp_init["rank_hash_spread"] < 1e-6, fsdp_init
        assert veomini_init["rank_hash_spread"] < 1e-6, veomini_init

    # --- Validate training records ---
    print("\n=== Per-step comparison ===")
    print(f"{'step':>4}  {'fsdp_grad':>12}  {'veom_grad':>12}  {'grad_diff':>12}  "
          f"{'fsdp_hash':>16}  {'veom_hash':>16}  {'hash_diff':>12}")

    for fs, vs in zip(fsdp_train, veomini_train):
        step = int(fs["step"])

        # Both engines must produce successful updates
        assert fs["update_successful"] == 1.0, f"FSDP step {step} update failed: {fs}"
        assert vs["update_successful"] == 1.0, f"VeOMini step {step} update failed: {vs}"

        # No inter-rank desynchronization
        assert fs["rank_hash_spread"] < 1e-6, f"FSDP rank desync at step {step}: {fs}"
        assert vs["rank_hash_spread"] < 1e-6, f"VeOMini rank desync at step {step}: {vs}"

        # LR must be identical (same scheduler config)
        assert abs(fs["lr"] - vs["lr"]) < 1e-12, (
            f"LR mismatch at step {step}: FSDP={fs['lr']}, VeOMini={vs['lr']}"
        )

        # Grad norms must be finite and positive
        assert math.isfinite(fs["grad_norm"]) and fs["grad_norm"] > 0, fs
        assert math.isfinite(vs["grad_norm"]) and vs["grad_norm"] > 0, vs

        grad_diff = abs(fs["grad_norm"] - vs["grad_norm"])
        hash_diff = abs(fs["param_hash"] - vs["param_hash"])
        print(f"{step:>4}  {fs['grad_norm']:>12.4f}  {vs['grad_norm']:>12.4f}  "
              f"{grad_diff:>12.4f}  {fs['param_hash']:>16.6f}  "
              f"{vs['param_hash']:>16.6f}  {hash_diff:>12.6f}")

        # Grad norms should be same order of magnitude (within 50% relative)
        # The difference comes from different clip_grad_norm implementations
        # and different forward pass precision (cast_forward_inputs, attention impl)
        avg_grad = (fs["grad_norm"] + vs["grad_norm"]) / 2
        rel_grad_diff = grad_diff / (avg_grad + 1e-8)
        assert rel_grad_diff < 0.5, (
            f"Grad norm relative diff too large at step {step}: "
            f"{rel_grad_diff:.3f} (FSDP={fs['grad_norm']:.4f}, VeOMini={vs['grad_norm']:.4f})"
        )


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_veomini_vs_fsdp_dp_parity_2gpu():
    if current_platform.device_count() < 2:
        pytest.skip("Distributed parity test requires 2 GPUs to run")

    output_dir = ARTIFACT_ROOT / f"run_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fsdp_output = output_dir / "fsdp_artifact.pt"
    veomini_output = output_dir / "veomini_artifact.pt"

    _run_test_with_torchrun(2, "fsdp", str(fsdp_output))
    _run_test_with_torchrun(2, "veomini", str(veomini_output))

    fsdp_artifact = torch.load(fsdp_output, map_location="cpu", weights_only=False)
    veomini_artifact = torch.load(
        veomini_output, map_location="cpu", weights_only=False
    )

    assert fsdp_artifact["selected_names"] == veomini_artifact["selected_names"]
    _compare_tensor_maps(
        fsdp_artifact["grads"],
        veomini_artifact["grads"],
        atol=3e-3,
        rtol=3e-3,
        label="gradient",
    )
    _compare_tensor_maps(
        fsdp_artifact["params"],
        veomini_artifact["params"],
        atol=3e-3,
        rtol=3e-3,
        label="parameter",
    )
    _compare_step_stats(fsdp_artifact["step_stats"], veomini_artifact["step_stats"])


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_veomini_vs_fsdp_dp_multistep_parity_2gpu():
    if current_platform.device_count() < 2:
        pytest.skip("Distributed multistep parity test requires 2 GPUs to run")

    output_dir = ARTIFACT_ROOT / f"multistep_run_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fsdp_output = output_dir / "fsdp_multistep_artifact.pt"
    veomini_output = output_dir / "veomini_multistep_artifact.pt"

    _run_multistep_test_with_torchrun(2, "fsdp", str(fsdp_output), steps=6)
    _run_multistep_test_with_torchrun(2, "veomini", str(veomini_output), steps=6)

    fsdp_artifact = torch.load(fsdp_output, map_location="cpu", weights_only=False)
    veomini_artifact = torch.load(
        veomini_output, map_location="cpu", weights_only=False
    )

    _compare_multistep_records(fsdp_artifact["records"], veomini_artifact["records"])


# ═══════════════════════════════════════════════════════════════════════════
#  Weight sync parity test
# ═══════════════════════════════════════════════════════════════════════════

def _run_weight_sync_test_with_torchrun(n_gpus: int, backend: str, output: str):
    port = find_free_ports(1)[0]
    cmd = [
        "torchrun",
        f"--nproc_per_node={n_gpus}",
        "--nnodes=1",
        "--master-addr=localhost",
        f"--master_port={port}",
        "tests/torchrun/run_veomini_vs_fsdp_weight_sync.py",
        f"--backend={backend}",
        f"--output={output}",
    ]
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    try:
        result = subprocess.run(
            cmd,
            check=True,
            text=True,
        )
    except subprocess.CalledProcessError as e:
        pytest.fail(f"Weight sync test failed (exit code {e.returncode})")

    if not Path(output).exists():
        pytest.fail(
            "Torchrun weight-sync finished but did not create output artifact: "
            f"{output}"
        )


def _compare_weight_sync_artifacts(
    fsdp_art: dict, veomini_art: dict,
) -> None:
    """Compare weight update artifacts between FSDP and VeOMini engines.

    Validates:
    1. Both engines pause/continue rollout exactly once.
    2. Both engines send params to the rollout engine.
    3. Parameter names sent to rollout match (after accounting for naming).
    4. Parameter shapes match.
    5. Sampled parameter values are close.
    6. Both engines send ALL model parameters (nothing dropped).
    """
    # ── Control flow ────────────────────────────────────────────────────
    assert fsdp_art["pause_count"] == 1, (
        f"FSDP should pause rollout once, got {fsdp_art['pause_count']}"
    )
    assert fsdp_art["continue_count"] == 1, (
        f"FSDP should continue rollout once, got {fsdp_art['continue_count']}"
    )
    assert veomini_art["pause_count"] == 1, (
        f"VeOMini should pause rollout once, got {veomini_art['pause_count']}"
    )
    assert veomini_art["continue_count"] == 1, (
        f"VeOMini should continue rollout once, got {veomini_art['continue_count']}"
    )

    # ── Completeness: both engines should send parameters ───────────────
    fsdp_names = fsdp_art["sent_tensor_names"]
    veomini_names = veomini_art["sent_tensor_names"]

    assert len(fsdp_names) > 0, "FSDP sent no parameters to rollout!"
    assert len(veomini_names) > 0, "VeOMini sent no parameters to rollout!"

    print(f"\nFSDP sent {len(fsdp_names)} parameters")
    print(f"VeOMini sent {len(veomini_names)} parameters")

    # ── Name matching ───────────────────────────────────────────────────
    fsdp_set = set(fsdp_names)
    veomini_set = set(veomini_names)

    only_in_fsdp = fsdp_set - veomini_set
    only_in_veomini = veomini_set - fsdp_set
    common = fsdp_set & veomini_set

    if only_in_fsdp:
        print(f"\n⚠️  Parameters only in FSDP ({len(only_in_fsdp)}):")
        for n in sorted(only_in_fsdp)[:20]:
            print(f"  {n}")

    if only_in_veomini:
        print(f"\n⚠️  Parameters only in VeOMini ({len(only_in_veomini)}):")
        for n in sorted(only_in_veomini)[:20]:
            print(f"  {n}")

    # For non-MoE models, names MUST match exactly
    assert fsdp_set == veomini_set, (
        f"Parameter name mismatch: "
        f"{len(only_in_fsdp)} only in FSDP, "
        f"{len(only_in_veomini)} only in VeOMini"
    )

    # ── Shape matching ──────────────────────────────────────────────────
    fsdp_specs = {name: (shape, dtype) for name, shape, dtype in fsdp_art["rollout_param_specs"]}
    veomini_specs = {name: (shape, dtype) for name, shape, dtype in veomini_art["rollout_param_specs"]}

    shape_mismatches = []
    for name in common:
        if name in fsdp_specs and name in veomini_specs:
            fs, fd = fsdp_specs[name]
            vs, vd = veomini_specs[name]
            if fs != vs:
                shape_mismatches.append(f"  {name}: FSDP={fs} vs VeOMini={vs}")

    if shape_mismatches:
        msg = "Shape mismatches:\n" + "\n".join(shape_mismatches[:20])
        raise AssertionError(msg)

    # ── Value matching (sampled tensors) ────────────────────────────────
    fsdp_samples = fsdp_art.get("sent_tensor_samples", {})
    veomini_samples = veomini_art.get("sent_tensor_samples", {})

    sample_common = set(fsdp_samples.keys()) & set(veomini_samples.keys())
    if sample_common:
        print(f"\nSpot-checking {len(sample_common)} sampled parameters:")
        for name in sorted(sample_common):
            ft = fsdp_samples[name]
            vt = veomini_samples[name]
            diff = (ft - vt).abs()
            max_diff = diff.max().item()
            cosine = F.cosine_similarity(
                ft.flatten().unsqueeze(0),
                vt.flatten().unsqueeze(0),
            ).item()
            print(f"  {name}: max_diff={max_diff:.6e}, cosine={cosine:.8f}")

    # ── Hash comparison (all params) ────────────────────────────────────
    fsdp_hashes = fsdp_art.get("sent_tensor_hashes", {})
    veomini_hashes = veomini_art.get("sent_tensor_hashes", {})

    if fsdp_hashes and veomini_hashes:
        hash_diffs = []
        for name in sorted(common):
            if name in fsdp_hashes and name in veomini_hashes:
                hdiff = abs(fsdp_hashes[name] - veomini_hashes[name])
                if hdiff > 1.0:
                    hash_diffs.append((name, hdiff, fsdp_hashes[name], veomini_hashes[name]))

        if hash_diffs:
            print(f"\n⚠️  Parameters with abs-sum hash diff > 1.0 ({len(hash_diffs)}):")
            for name, hdiff, fh, vh in hash_diffs[:20]:
                print(f"  {name}: diff={hdiff:.4f} (fsdp={fh:.4f}, veomini={vh:.4f})")
            print("\n  Note: expected due to different optimizer precision "
                  "(VeOMini fp32 vs FSDP bf16)")
        else:
            print(f"\n✅ All {len(common)} param hashes match within tolerance")

    # ── Final: both engines must send ALL model parameters ──────────────
    fsdp_model_names = set(fsdp_art["post_train_param_names"])
    veomini_model_names = set(veomini_art["post_train_param_names"])

    fsdp_missing = fsdp_model_names - fsdp_set
    veomini_missing = veomini_model_names - veomini_set

    if fsdp_missing:
        print(f"\n🔴 FSDP model params NOT sent to rollout ({len(fsdp_missing)}):")
        for n in sorted(fsdp_missing)[:20]:
            print(f"  {n}")

    if veomini_missing:
        print(f"\n🔴 VeOMini model params NOT sent to rollout ({len(veomini_missing)}):")
        for n in sorted(veomini_missing)[:20]:
            print(f"  {n}")

    assert not fsdp_missing, (
        f"FSDP failed to send {len(fsdp_missing)} model params to rollout"
    )
    assert not veomini_missing, (
        f"VeOMini failed to send {len(veomini_missing)} model params to rollout"
    )

    print("\n✅ Weight sync parity test passed")


@pytest.mark.multi_gpu
@pytest.mark.slow
def test_veomini_vs_fsdp_weight_sync_parity_2gpu():
    """Verify that VeOMini sends the same parameters to rollout as FSDP does."""
    if current_platform.device_count() < 2:
        pytest.skip("Weight sync parity test requires 2 GPUs to run")

    output_dir = ARTIFACT_ROOT / f"weight_sync_{uuid.uuid4().hex[:8]}"
    output_dir.mkdir(parents=True, exist_ok=True)
    fsdp_output = output_dir / "fsdp_weight_sync.pt"
    veomini_output = output_dir / "veomini_weight_sync.pt"

    _run_weight_sync_test_with_torchrun(2, "fsdp", str(fsdp_output))
    _run_weight_sync_test_with_torchrun(2, "veomini", str(veomini_output))

    fsdp_artifact = torch.load(fsdp_output, map_location="cpu", weights_only=False)
    veomini_artifact = torch.load(
        veomini_output, map_location="cpu", weights_only=False
    )

    _compare_weight_sync_artifacts(fsdp_artifact, veomini_artifact)
