"""Capture-and-Replay Parity Test: FSDP vs VeOMini Engine.

Generates fixed realistic PPO-like training batches, then replays them through
both FSDP and VeOMini engines to pinpoint where training divergence occurs.

Three modes
-----------
  generate  Create N realistic variable-length PPO batches and save to disk.
            No GPU / torchrun required.
  train     Load saved batches, train with the specified backend, record
            per-step metrics (loss, grad norm, importance weights, param norms).
            Requires torchrun.
  compare   Load metrics from both backends and print a detailed comparison
            table highlighting the first significant divergence.

Usage
-----
  # 1) Generate batches (CPU only)
  python tests/torchrun/run_capture_and_replay.py generate \\
      --steps 10 --batch-size 4 --output-dir ./parity_debug

  # 2) Train FSDP
  torchrun --nproc_per_node=1 tests/torchrun/run_capture_and_replay.py train \\
      --backend fsdp --steps 10 --output-dir ./parity_debug

  # 3) Train VeOMini
  torchrun --nproc_per_node=1 tests/torchrun/run_capture_and_replay.py train \\
      --backend veomini --steps 10 --output-dir ./parity_debug

  # 4) Compare
  python tests/torchrun/run_capture_and_replay.py compare \\
      --output-dir ./parity_debug
"""

from __future__ import annotations

import argparse
import gc
import os
import sys
from pathlib import Path
from typing import Any

import torch

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PREFERRED_PARAM_SUFFIXES = [
    "embed_tokens.weight",
    "layers.0.self_attn.q_proj.weight",
    "layers.0.self_attn.k_proj.weight",
    "layers.0.self_attn.v_proj.weight",
    "layers.0.self_attn.o_proj.weight",
    "layers.0.mlp.gate_proj.weight",
    "layers.0.mlp.up_proj.weight",
    "layers.0.mlp.down_proj.weight",
    "norm.weight",
    "lm_head.weight",
]

# ===================================================================
# MODE: generate -- create realistic PPO batches (CPU only)
# ===================================================================


def generate_realistic_batch(
    step: int,
    batch_size: int = 4,
    min_prompt: int = 16,
    max_prompt: int = 64,
    min_resp: int = 16,
    max_resp: int = 128,
    vocab_size: int = 151936,
) -> dict[str, torch.Tensor]:
    """Create a single left-padded PPO training batch with variable-length
    sequences, realistic old_logprobs, and occasionally extreme advantages.

    All tensors are created on CPU so the batch can be saved to disk and
    loaded by any backend later.
    """
    g = torch.Generator().manual_seed(20260311 + step * 7)

    prompt_lens = torch.randint(
        min_prompt, max_prompt + 1, (batch_size,), generator=g
    )
    resp_lens = torch.randint(min_resp, max_resp + 1, (batch_size,), generator=g)
    seq_lens = prompt_lens + resp_lens
    max_len = int(seq_lens.max().item())

    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    loss_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
    old_logprobs = torch.zeros(batch_size, max_len, dtype=torch.float32)
    advantages = torch.zeros(batch_size, max_len, dtype=torch.float32)

    for i in range(batch_size):
        plen = int(prompt_lens[i].item())
        rlen = int(resp_lens[i].item())
        slen = plen + rlen
        start = max_len - slen

        # Random token IDs (skip id=0 which is the pad token)
        input_ids[i, start:] = torch.randint(
            100, vocab_size, (slen,), generator=g
        )
        attention_mask[i, start:] = True
        loss_mask[i, start + plen :] = True

        # Realistic old_logprobs: mostly in [-6, -0.5]
        old_lp = -torch.abs(torch.randn(slen, generator=g)) * 2.5 - 0.5
        old_logprobs[i, start:] = old_lp

        # Advantages: normal with occasional extremes
        if step % 4 == 0 and i == 0:
            # Extreme advantage batch (mimics high-reward outlier)
            adv = torch.randn(rlen, generator=g) * 5.0
        elif step % 4 == 1 and i == batch_size - 1:
            # Very short effective response (only 2 valid tokens)
            loss_mask[i, start + plen + 2 :] = False
            adv = torch.randn(rlen, generator=g) * 1.0
        else:
            adv = torch.randn(rlen, generator=g) * 1.5
        advantages[i, start + plen :] = adv

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "loss_mask": loss_mask,
        "old_logprobs": old_logprobs,
        "advantages": advantages,
    }


def cmd_generate(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for step in range(args.steps):
        batch = generate_realistic_batch(step, batch_size=args.batch_size)
        out_path = output_dir / f"batch_{step}.pt"
        torch.save(batch, out_path)

        seq_lens = batch["attention_mask"].sum(dim=1).tolist()
        valid_tokens = int(batch["loss_mask"].sum().item())
        print(
            f"  Batch {step:>3d}: seq_lens={seq_lens}, "
            f"valid_tokens={valid_tokens}, "
            f"adv range=[{batch['advantages'].min():.2f}, {batch['advantages'].max():.2f}]"
        )

    print(f"\n✅ {args.steps} batches saved to {output_dir}/")


# ===================================================================
# MODE: train -- replay saved batches through an engine
# ===================================================================


class InstrumentedLossFn:
    """PPO-style clipped loss that records importance weight statistics.

    Called once per micro-batch by ``_compute_logprobs_and_loss``.
    After ``train_batch`` finishes, call ``get_aggregated_stats()`` to
    retrieve weighted-average statistics across all micro-batches.
    """

    def __init__(self, eps_clip: float = 0.2, entropy_coeff: float = 0.01):
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self._mb_records: list[dict[str, float]] = []

    def reset(self) -> None:
        self._mb_records.clear()

    # ----- callable interface used by engine._compute_logprobs_and_loss -----
    def __call__(
        self,
        logprobs: torch.Tensor,
        entropy: torch.Tensor,
        input_data: dict[str, Any],
        **kwargs: Any,
    ) -> torch.Tensor:
        loss_mask = input_data["loss_mask"]
        adv = input_data.get("advantages", torch.zeros_like(logprobs))
        old_lp = input_data.get("old_logprobs", torch.zeros_like(logprobs))

        # Flatten 2-D remnants (shouldn't happen after pack, but be safe)
        if loss_mask.dim() == 2:
            loss_mask = loss_mask.reshape(-1)
            adv = adv.reshape(-1)
            old_lp = old_lp.reshape(-1)

        # Align lengths (micro-batch padding trim may differ by ±1)
        n = min(logprobs.size(0), loss_mask.size(0))
        logprobs_t = logprobs[:n]
        entropy_t = entropy[:n]
        mask_t = loss_mask[:n].bool()
        adv_t = adv[:n]
        old_lp_t = old_lp[:n]

        n_valid = int(mask_t.sum().item())
        if n_valid == 0:
            self._mb_records.append({"n_valid": 0})
            return (logprobs * 0).sum()

        v_lp = logprobs_t[mask_t]
        v_ent = entropy_t[mask_t]
        v_adv = adv_t[mask_t]
        v_old = old_lp_t[mask_t]

        # Importance weight  π_θ / π_old
        ratio = torch.exp(v_lp - v_old)

        # PPO clipped surrogate
        surr1 = ratio * v_adv
        surr2 = torch.clamp(ratio, 1.0 - self.eps_clip, 1.0 + self.eps_clip) * v_adv
        policy_loss = -torch.min(surr1, surr2).mean()
        entropy_loss = -self.entropy_coeff * v_ent.mean()
        loss = policy_loss + entropy_loss

        # Record per-micro-batch statistics
        self._mb_records.append(
            {
                "n_valid": n_valid,
                "iw_max": ratio.max().item(),
                "iw_min": ratio.min().item(),
                "iw_mean": ratio.mean().item(),
                "iw_std": ratio.std().item() if n_valid > 1 else 0.0,
                "logp_mean": v_lp.mean().item(),
                "logp_min": v_lp.min().item(),
                "logp_max": v_lp.max().item(),
                "old_lp_mean": v_old.mean().item(),
                "logp_diff_abs_max": (v_lp - v_old).abs().max().item(),
                "logp_diff_mean": (v_lp - v_old).mean().item(),
                "entropy_mean": v_ent.mean().item(),
                "adv_mean": v_adv.mean().item(),
                "adv_std": v_adv.std().item() if n_valid > 1 else 0.0,
                "policy_loss": policy_loss.item(),
                "entropy_loss": entropy_loss.item(),
                "total_loss": loss.item(),
                "ratio_gt_2": int((ratio > 2.0).sum().item()),
                "ratio_lt_half": int((ratio < 0.5).sum().item()),
            }
        )

        if torch.isnan(loss):
            return (logprobs * 0).sum()
        return loss

    # ----- aggregation -----
    def get_aggregated_stats(self) -> dict[str, float]:
        valid = [r for r in self._mb_records if r.get("n_valid", 0) > 0]
        if not valid:
            return {"n_valid_total": 0}

        total_tokens = sum(r["n_valid"] for r in valid)

        def _wavg(key: str) -> float:
            return sum(r[key] * r["n_valid"] for r in valid) / total_tokens

        return {
            "n_valid_total": total_tokens,
            "iw_max": max(r["iw_max"] for r in valid),
            "iw_min": min(r["iw_min"] for r in valid),
            "iw_mean": _wavg("iw_mean"),
            "iw_std": max(r["iw_std"] for r in valid),
            "logp_mean": _wavg("logp_mean"),
            "logp_min": min(r["logp_min"] for r in valid),
            "logp_max": max(r["logp_max"] for r in valid),
            "old_lp_mean": _wavg("old_lp_mean"),
            "logp_diff_abs_max": max(r["logp_diff_abs_max"] for r in valid),
            "logp_diff_mean": _wavg("logp_diff_mean"),
            "entropy_mean": _wavg("entropy_mean"),
            "policy_loss": sum(r["policy_loss"] for r in valid) / len(valid),
            "total_loss": sum(r["total_loss"] for r in valid) / len(valid),
            "ratio_gt_2": sum(r["ratio_gt_2"] for r in valid),
            "ratio_lt_half": sum(r["ratio_lt_half"] for r in valid),
        }


def _materialize(t: torch.Tensor) -> torch.Tensor:
    """Return a plain float32 tensor from a DTensor or regular tensor."""
    from torch.distributed.tensor import DTensor

    if isinstance(t, DTensor):
        t = t.full_tensor()
    return t.detach().float()


def _select_param_names(all_names: list[str]) -> list[str]:
    selected: list[str] = []
    for suffix in PREFERRED_PARAM_SUFFIXES:
        for name in all_names:
            if name.endswith(suffix) and name not in selected:
                selected.append(name)
                break
    return selected if selected else sorted(all_names)[:10]


def _compute_param_hash(engine: Any, selected: list[str]) -> float:
    """Sum of abs values of selected parameters (float64 for precision)."""
    h = torch.zeros((), dtype=torch.float64, device=engine.device)
    for name, param in engine._get_model_name_parameters():
        if name not in selected:
            continue
        h += _materialize(param).abs().sum().to(torch.float64)
    return float(h.item())


def _compute_param_norms(engine: Any, selected: list[str]) -> dict[str, float]:
    norms: dict[str, float] = {}
    for name, param in engine._get_model_name_parameters():
        if name not in selected:
            continue
        norms[name] = _materialize(param).norm().item()
    return norms


def cmd_train(args: argparse.Namespace) -> None:
    import torch.distributed as dist

    from areal.api.alloc_mode import ParallelStrategy
    from areal.api.io_struct import FinetuneSpec
    from areal.engine.fsdp_engine import FSDPEngine
    from areal.infra.platforms import current_platform
    from tests.torchrun.veomini_debug_common import make_debug_train_config
    from veomni.veomini_engine import VeOMiniEngine

    backend = args.backend
    output_dir = Path(args.output_dir)
    steps = args.steps

    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    current_platform.set_device(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")

    # Verify batches exist
    for s in range(steps):
        p = output_dir / f"batch_{s}.pt"
        if not p.exists():
            print(f"❌ Missing {p}. Run 'generate' mode first.", file=sys.stderr)
            sys.exit(1)

    # Create engine
    config_name = f"parity_{backend}"
    config = make_debug_train_config(config_name, lr=args.lr)
    if args.model_path:
        config.path = args.model_path

    ft_spec = FinetuneSpec(
        total_train_epochs=1,
        dataset_size=steps * args.batch_size,
        train_batch_size=args.batch_size,
    )
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    if backend == "fsdp":
        engine: FSDPEngine | VeOMiniEngine = FSDPEngine(config)
    elif backend == "veomini":
        engine = VeOMiniEngine(config)
    else:
        raise ValueError(f"Unknown backend: {backend}")

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)

    # Select parameters to track
    all_names = [n for n, _ in engine._get_model_name_parameters()]
    selected = _select_param_names(all_names)

    init_hash = _compute_param_hash(engine, selected)
    init_norms = _compute_param_norms(engine, selected)

    if rank == 0:
        print(f"\n[{backend}] Model loaded.  #params tracked: {len(selected)}")
        print(f"[{backend}] Init param_hash = {init_hash:.6f}")
        print(f"[{backend}] Selected params: {selected[:5]}...")
        print()

    metrics: list[dict[str, Any]] = []
    metrics.append(
        {
            "step": -1,
            "param_hash": init_hash,
            "param_norms": init_norms,
        }
    )

    # ---- Training loop ----
    loss_fn = InstrumentedLossFn(eps_clip=0.2, entropy_coeff=0.01)

    for step in range(steps):
        # Load batch
        batch = torch.load(
            output_dir / f"batch_{step}.pt",
            map_location=device,
            weights_only=True,
        )

        # Pre-step param norms
        pre_norms = _compute_param_norms(engine, selected)

        # Train
        loss_fn.reset()
        engine.train()
        step_result = engine.train_batch(
            input_=batch,
            loss_fn=loss_fn,
            loss_weight_fn=lambda x: x["cu_seqlens"][-1],
        )
        engine.lr_scheduler_step()

        # Post-step param norms
        post_norms = _compute_param_norms(engine, selected)
        param_hash = _compute_param_hash(engine, selected)

        # Weight update deltas
        weight_deltas = {
            k: abs(post_norms[k] - pre_norms[k])
            for k in selected
            if k in pre_norms and k in post_norms
        }

        # Aggregate loss-function stats
        loss_stats = loss_fn.get_aggregated_stats()

        record: dict[str, Any] = {
            "step": step,
            "param_hash": param_hash,
            "pre_param_norms": pre_norms,
            "post_param_norms": post_norms,
            "weight_deltas": weight_deltas,
        }
        record.update(step_result)  # update_successful, grad_norm, lr
        record.update(loss_stats)  # iw_max, logp_mean, total_loss, ...
        metrics.append(record)

        if rank == 0:
            iw_max = loss_stats.get("iw_max", float("nan"))
            iw_mean = loss_stats.get("iw_mean", float("nan"))
            logp_diff = loss_stats.get("logp_diff_abs_max", float("nan"))
            t_loss = loss_stats.get("total_loss", float("nan"))
            gn = step_result.get("grad_norm", float("nan"))
            ratio_gt2 = loss_stats.get("ratio_gt_2", 0)

            flag = " ⚠️" if iw_max > 2.0 or gn > 5.0 else ""
            print(
                f"[{backend}] step={step:>3d}  loss={t_loss:>9.5f}  "
                f"grad_norm={gn:>8.4f}  "
                f"iw_max={iw_max:>8.3f}  iw_mean={iw_mean:>7.4f}  "
                f"|Δlogp|_max={logp_diff:>7.3f}  "
                f"ratio>2={ratio_gt2:>4d}  "
                f"hash={param_hash:.4f}{flag}"
            )

    # Save
    if rank == 0:
        out_path = output_dir / f"{backend}_metrics.pt"
        torch.save(metrics, out_path)
        print(f"\n✅ Saved {backend} metrics ({len(metrics)} records) → {out_path}")

    if dist.is_initialized():
        dist.barrier()
    engine.destroy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ===================================================================
# MODE: compare -- load & diff metrics from both backends
# ===================================================================


def cmd_compare(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)

    fsdp_path = output_dir / "fsdp_metrics.pt"
    veomini_path = output_dir / "veomini_metrics.pt"

    for p in (fsdp_path, veomini_path):
        if not p.exists():
            print(f"❌ Missing {p}. Run 'train' mode for both backends first.", file=sys.stderr)
            sys.exit(1)

    fsdp_metrics: list[dict] = torch.load(fsdp_path, weights_only=False)
    veomini_metrics: list[dict] = torch.load(veomini_path, weights_only=False)

    # --- Step -1: initial state comparison ---
    f_init = fsdp_metrics[0]
    v_init = veomini_metrics[0]
    f_hash = f_init.get("param_hash", 0.0)
    v_hash = v_init.get("param_hash", 0.0)
    hash_diff = abs(f_hash - v_hash)

    print("=" * 110)
    print("FSDP vs VeOMini  --  Capture-and-Replay Parity Report".center(110))
    print("=" * 110)

    print(f"\nInitial param_hash:  FSDP={f_hash:.6f}   VeOMini={v_hash:.6f}   "
          f"diff={hash_diff:.2e}")
    if hash_diff > 1e-3:
        print("  ⚠️  Initial weights DIFFER — models loaded differently!")
    else:
        print("  ✅  Initial weights match.")

    # Compare initial per-layer norms
    f_inorms = f_init.get("param_norms", {})
    v_inorms = v_init.get("param_norms", {})
    common_params = sorted(set(f_inorms.keys()) & set(v_inorms.keys()))
    if common_params:
        init_norm_diffs = {k: abs(f_inorms[k] - v_inorms[k]) for k in common_params}
        worst_k = max(init_norm_diffs, key=init_norm_diffs.get)
        worst_v = init_norm_diffs[worst_k]
        print(f"  Max initial param-norm diff: {worst_v:.2e} at [{worst_k}]")

    # --- Training steps ---
    # Skip step=-1 entries
    f_steps = [r for r in fsdp_metrics if r.get("step", -1) >= 0]
    v_steps = [r for r in veomini_metrics if r.get("step", -1) >= 0]
    n_steps = min(len(f_steps), len(v_steps))

    print(f"\nComparing {n_steps} training steps:\n")

    header = (
        f"{'Step':>4} │ "
        f"{'Loss(F)':>10} {'Loss(V)':>10} {'Δ':>9} │ "
        f"{'GradN(F)':>9} {'GradN(V)':>9} {'Δ':>9} │ "
        f"{'IWmax(F)':>9} {'IWmax(V)':>9} │ "
        f"{'|ΔLP|(F)':>9} {'|ΔLP|(V)':>9} │ "
        f"{'Hash(F)':>10} {'Hash(V)':>10}"
    )
    print(header)
    print("─" * len(header))

    first_diverge_step = None

    for i in range(n_steps):
        f = f_steps[i]
        v = v_steps[i]

        loss_f = f.get("total_loss", float("nan"))
        loss_v = v.get("total_loss", float("nan"))
        gn_f = f.get("grad_norm", float("nan"))
        gn_v = v.get("grad_norm", float("nan"))
        iw_f = f.get("iw_max", float("nan"))
        iw_v = v.get("iw_max", float("nan"))
        ld_f = f.get("logp_diff_abs_max", float("nan"))
        ld_v = v.get("logp_diff_abs_max", float("nan"))
        h_f = f.get("param_hash", float("nan"))
        h_v = v.get("param_hash", float("nan"))

        loss_d = abs(loss_f - loss_v)
        gn_d = abs(gn_f - gn_v)

        flag = ""
        if loss_d > 0.01 or gn_d > 0.01:
            flag = " ⚠️"
            if first_diverge_step is None:
                first_diverge_step = i

        print(
            f"{i:>4} │ "
            f"{loss_f:>10.5f} {loss_v:>10.5f} {loss_d:>9.2e} │ "
            f"{gn_f:>9.4f} {gn_v:>9.4f} {gn_d:>9.2e} │ "
            f"{iw_f:>9.3f} {iw_v:>9.3f} │ "
            f"{ld_f:>9.3f} {ld_v:>9.3f} │ "
            f"{h_f:>10.4f} {h_v:>10.4f}{flag}"
        )

    # --- Weight update comparison ---
    print("\n" + "=" * 110)
    print("WEIGHT UPDATE DELTAS (per-layer)".center(110))
    print("=" * 110)

    for i in range(n_steps):
        f_wd = f_steps[i].get("weight_deltas", {})
        v_wd = v_steps[i].get("weight_deltas", {})
        common = set(f_wd.keys()) & set(v_wd.keys())
        if not common:
            continue

        diffs = {k: abs(f_wd[k] - v_wd[k]) for k in common}
        worst_key = max(diffs, key=diffs.get)
        worst_val = diffs[worst_key]

        if worst_val > 1e-6:
            print(f"\n  Step {i}: max weight-delta diff = {worst_val:.2e}")
            top3 = sorted(diffs.items(), key=lambda x: x[1], reverse=True)[:3]
            for k, d in top3:
                print(f"    {k}")
                print(f"      FSDP:    {f_wd[k]:.6e}")
                print(f"      VeOMini: {v_wd[k]:.6e}")
                print(f"      diff:    {d:.6e}")

    # --- Summary ---
    print("\n" + "=" * 110)
    print("SUMMARY".center(110))
    print("=" * 110)

    if first_diverge_step is not None:
        print(f"\n  ⚠️  First significant divergence at step {first_diverge_step}")
        f0 = f_steps[first_diverge_step]
        v0 = v_steps[first_diverge_step]
        print(f"      Loss:      FSDP={f0.get('total_loss', 'N/A'):.6f}  "
              f"VeOMini={v0.get('total_loss', 'N/A'):.6f}")
        print(f"      Grad norm: FSDP={f0.get('grad_norm', 'N/A'):.4f}  "
              f"VeOMini={v0.get('grad_norm', 'N/A'):.4f}")
        print(f"      IW max:    FSDP={f0.get('iw_max', 'N/A'):.4f}  "
              f"VeOMini={v0.get('iw_max', 'N/A'):.4f}")
    else:
        print("\n  ✅  No significant divergence detected in all steps.")

    # Final step comparison
    if n_steps > 0:
        ff = f_steps[-1]
        vv = v_steps[-1]
        print(f"\n  Final step ({n_steps - 1}):")
        print(f"    Loss diff:      {abs(ff.get('total_loss', 0) - vv.get('total_loss', 0)):.6e}")
        print(f"    Grad norm diff: {abs(ff.get('grad_norm', 0) - vv.get('grad_norm', 0)):.6e}")
        print(f"    Param hash diff:{abs(ff.get('param_hash', 0) - vv.get('param_hash', 0)):.6e}")


# ===================================================================
# MAIN
# ===================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Capture-and-Replay Parity Test: FSDP vs VeOMini",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="mode")

    # ---- generate ----
    gen = subparsers.add_parser("generate", help="Generate realistic PPO batches")
    gen.add_argument("--steps", type=int, default=10, help="Number of batches to create")
    gen.add_argument("--batch-size", type=int, default=4, help="Sequences per batch")
    gen.add_argument("--output-dir", type=str, default="./parity_debug")

    # ---- train ----
    trn = subparsers.add_parser("train", help="Train with specified backend")
    trn.add_argument("--backend", type=str, required=True, choices=["fsdp", "veomini"])
    trn.add_argument("--steps", type=int, default=10, help="Number of training steps")
    trn.add_argument("--batch-size", type=int, default=4, help="Must match generate")
    trn.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    trn.add_argument("--model-path", type=str, default="", help="Path to the reference model")
    trn.add_argument("--output-dir", type=str, default="./parity_debug")

    # ---- compare ----
    cmp = subparsers.add_parser("compare", help="Compare FSDP vs VeOMini metrics")
    cmp.add_argument("--output-dir", type=str, default="./parity_debug")

    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    if args.mode == "generate":
        cmd_generate(args)
    elif args.mode == "train":
        cmd_train(args)
    elif args.mode == "compare":
        cmd_compare(args)


if __name__ == "__main__":
    main()
