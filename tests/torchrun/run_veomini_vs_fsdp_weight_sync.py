"""Distributed weight-sync parity test: VeOMini vs FSDP.

Validates the complete weight update pipeline that sends trained parameters
to the rollout/inference engine.  Both engines:

  1. Load the same checkpoint.
  2. Run one training step (identical input, loss).
  3. Execute ``_update_weights_from_distributed`` with a mock rollout engine
     that captures every (name, tensor) pair broadcasted.

The test then compares:

  - Parameter **names** sent (completeness, ordering, naming convention).
  - Parameter **shapes** (layout must match what sglang expects).
  - Parameter **values** (post-training weights must be numerically close).

This catches the scenario where training works fine but rollout never
sees the updated weights (reward stagnation root-cause).

Usage:
    torchrun --nproc_per_node=2 tests/torchrun/run_veomini_vs_fsdp_weight_sync.py \
        --backend=fsdp --output=/tmp/fsdp_ws.pt
    torchrun --nproc_per_node=2 tests/torchrun/run_veomini_vs_fsdp_weight_sync.py \
        --backend=veomini --output=/tmp/veomini_ws.pt
"""

from __future__ import annotations

import argparse
import gc
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.api.alloc_mode import ParallelStrategy
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.platforms import current_platform
from tests.torchrun.veomini_debug_common import make_debug_train_config
from veomni.veomini_engine import VeOMiniEngine


# ── helpers ──────────────────────────────────────────────────────────────
def _make_config(experiment_name: str):
    return make_debug_train_config(experiment_name)


def _mock_input(
    step: int = 0,
    batch_size: int = 4,
    min_seqlen: int = 8,
    max_seqlen: int = 32,
    device: torch.device | str = "cuda",
) -> dict[str, Any]:
    pad_token_id = 0
    g = torch.Generator(device=device).manual_seed(20260310 + step)
    seqlens = torch.randint(
        min_seqlen, max_seqlen + 1, (batch_size,), dtype=torch.int,
        device=device, generator=g,
    )
    max_len = int(max(seqlens))
    input_ids = torch.randint(
        1000, 5000, (batch_size, max_len), dtype=torch.long,
        device=device, generator=g,
    )
    attention_mask = torch.zeros(
        (batch_size, max_len), dtype=torch.bool, device=device,
    )
    attention_mask[
        torch.arange(0, max_len, device=device).unsqueeze(0)
        < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attention_mask, pad_token_id)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def _mock_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    **kwargs,
) -> torch.Tensor:
    return torch.mean(logprobs) + 0.01 * torch.mean(entropy)


class _DummyFuture:
    """Immediately-resolved future for mock rollout engine."""
    def result(self):
        return None


class _CapturingRolloutEngine:
    """Mock rollout that captures every parameter broadcasted during weight update.

    Only rank-0 actually receives the ``update_weights_from_distributed`` calls
    from the train engine, so only rank-0's ``captured`` list is populated.
    """
    def __init__(self):
        self.captured: list[tuple[str, tuple, str]] = []   # (name, shape, dtype)
        self.paused = 0
        self.continued = 0

    def pause_generation(self):
        self.paused += 1

    def continue_generation(self):
        self.continued += 1

    def update_weights_from_distributed(self, meta, param_specs):
        for spec in param_specs:
            self.captured.append((spec.name, tuple(spec.shape), spec.dtype))
        return _DummyFuture()


def _materialize(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().float().cpu()


# ── main flow ────────────────────────────────────────────────────────────
def run_weight_sync_test(backend: str, output: str | None = None) -> bool:
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    current_platform.set_device(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=64, train_batch_size=4)
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    if backend == "fsdp":
        engine: FSDPEngine | VeOMiniEngine = FSDPEngine(
            _make_config("test_fsdp_weight_sync")
        )
    elif backend == "veomini":
        engine = VeOMiniEngine(_make_config("test_veomini_weight_sync"))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)

    if rank == 0:
        print(f"[{backend}] ✓ Engine initialized (world_size={world_size})", flush=True)

    # ── Step 1: One training step ────────────────────────────────────────
    input_ = _mock_input(step=0, device=device)
    engine.train()
    step_stats = engine.train_batch(
        input_=input_,
        loss_fn=_mock_loss_fn,
        loss_weight_fn=lambda x: x["cu_seqlens"][-1],
    )

    if rank == 0:
        print(f"[{backend}] ✓ Step 1 done: train_batch grad_norm={step_stats['grad_norm']:.4f}", flush=True)

    # ── Step 2: Collect post-training parameters (from engine's model) ───
    # NOTE: _get_full_tensor triggers DTensor all-gather — ALL ranks must call it.
    post_train_params: dict[str, torch.Tensor] = {}
    for name, param in engine._get_model_name_parameters():
        tensor = engine._get_full_tensor(param)
        if rank == 0:
            post_train_params[name] = _materialize(tensor)

    if rank == 0:
        print(f"[{backend}] ✓ Step 2 done: collected {len(post_train_params)} post-train params", flush=True)

    # ── Step 3: Execute weight update with mock rollout engine ───────────
    mock_rollout = _CapturingRolloutEngine()
    engine.rollout_engine = mock_rollout

    # Monkeypatch _update_bucket_weights_from_distributed to skip the real
    # NCCL broadcast (no rollout process in weight_update_group) but still
    # capture every (name, tensor) that would be sent.
    sent_tensors: dict[str, torch.Tensor] = {}

    def _capturing_update_bucket(meta, named_tensors):
        """Capture tensors instead of broadcasting via NCCL."""
        from areal.api.io_struct import ParamSpec

        if not named_tensors:
            return

        param_specs = [
            ParamSpec(
                name=name,
                shape=tuple(tensor.shape),
                dtype=str(tensor.dtype).split("torch.")[1],
            )
            for name, tensor in named_tensors
        ]

        # Call the mock rollout engine (same as real code)
        fut = mock_rollout.update_weights_from_distributed(
            meta, param_specs
        )

        # Capture the actual tensor data (instead of broadcasting)
        for name, tensor in named_tensors:
            sent_tensors[name] = tensor.detach().float().cpu()

        fut.result()
        named_tensors.clear()

    engine._update_bucket_weights_from_distributed = _capturing_update_bucket

    # Build a fake WeightUpdateMeta
    meta = SimpleNamespace(
        weight_chunked_mem_mb=2048,
        nccl_master_address="127.0.0.1",
        nccl_master_port=12345,
        nccl_group_name="test_group",
    )
    engine.weight_update_master_addr = "127.0.0.1"
    engine.weight_update_master_port = 12345
    engine.weight_update_group_name = "test_group"

    if rank == 0:
        print(f"[{backend}] → Step 3: calling _update_weights_from_distributed ...", flush=True)

    engine._update_weights_from_distributed(meta)

    if rank == 0:
        print(f"[{backend}] ✓ Step 3 done: sent {len(sent_tensors)} tensors, "
              f"rollout captured {len(mock_rollout.captured)} param_specs, "
              f"pause={mock_rollout.paused}, continue={mock_rollout.continued}", flush=True)

    # ── Step 4: Save results (rank 0 only) ──────────────────────────────
    if rank == 0 and output is not None:
        # Instead of saving all full tensors (too large for 1.7B model),
        # save lightweight summaries: hash, shape, dtype per param,
        # plus a few sampled tensors for spot-checking values.
        SAMPLE_SUFFIXES = [
            "embed_tokens.weight",
            "layers.0.self_attn.q_proj.weight",
            "layers.0.mlp.gate_proj.weight",
            "norm.weight",
            "lm_head.weight",
        ]

        def _pick_samples(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
            sampled = {}
            for name, t in tensors.items():
                if any(name.endswith(s) for s in SAMPLE_SUFFIXES):
                    sampled[name] = t
            return sampled

        def _compute_hashes(tensors: dict[str, torch.Tensor]) -> dict[str, float]:
            return {name: float(t.abs().sum().item()) for name, t in tensors.items()}

        artifact = {
            "backend": backend,
            "step_stats": step_stats,
            # Names sent to rollout (via ParamSpec)
            "rollout_param_specs": mock_rollout.captured,
            # Lightweight: names only
            "sent_tensor_names": sorted(sent_tensors.keys()),
            "post_train_param_names": sorted(post_train_params.keys()),
            # Hashes for all params (scalar per param — tiny)
            "sent_tensor_hashes": _compute_hashes(sent_tensors),
            "post_train_param_hashes": _compute_hashes(post_train_params),
            # A few sampled full tensors for spot-checking
            "sent_tensor_samples": _pick_samples(sent_tensors),
            "post_train_param_samples": _pick_samples(post_train_params),
            # Control flow
            "pause_count": mock_rollout.paused,
            "continue_count": mock_rollout.continued,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(artifact, output)
        print(f"[{backend}] ✓ Step 4 done: saved artifact to {output}", flush=True)

    print(f"[{backend}] rank={rank} → entering final barrier ...", flush=True)
    if dist.is_initialized():
        dist.barrier()

    print(f"[{backend}] rank={rank} → barrier passed, calling engine.destroy() ...", flush=True)
    # Cleanup
    engine.destroy()
    print(f"[{backend}] rank={rank} → engine destroyed", flush=True)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=["fsdp", "veomini"], required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    ok = run_weight_sync_test(backend=args.backend, output=args.output)
    if not ok:
        raise AssertionError("Weight sync test failed")


if __name__ == "__main__":
    main()
