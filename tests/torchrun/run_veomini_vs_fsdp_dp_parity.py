import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.tensor import DTensor

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.core import compute_total_loss_weight
from areal.engine.fsdp_engine import FSDPEngine, FSDPTrainContext
from areal.infra.platforms import current_platform
from tests.torchrun.veomini_debug_common import make_debug_train_config
from veomni.veomini_engine import VeOMiniEngine


def write_result(out: str, succ: bool):
    with open(out, "w") as f:
        f.write("Passed" if succ else "Failed")


def _save_artifact(out: str, payload: dict[str, Any]) -> None:
    torch.save(payload, out)


def mock_input(
    batch_size: int = 4,
    min_seqlen: int = 8,
    max_seqlen: int = 32,
    device: torch.device | str = current_platform.device_type,
) -> dict[str, Any]:
    pad_token_id = 0
    seqlens = torch.randint(
        min_seqlen, max_seqlen + 1, (batch_size,), dtype=torch.int, device=device
    )
    max_len = int(max(seqlens))
    input_ids = torch.randint(
        1000, 5000, (batch_size, max_len), dtype=torch.long, device=device
    )
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    attention_mask[
        torch.arange(0, max_len, device=device).unsqueeze(0) < seqlens.unsqueeze(1)
    ] = 1
    input_ids.masked_fill_(~attention_mask, pad_token_id)
    return {"input_ids": input_ids, "attention_mask": attention_mask}


def mock_loss_fn(
    logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs
) -> torch.Tensor:
    return torch.mean(logprobs) + 0.01 * torch.mean(entropy)


def make_config(experiment_name: str) -> TrainEngineConfig:
    return make_debug_train_config(experiment_name)


def make_parallel_strategy() -> ParallelStrategy:
    return ParallelStrategy(data_parallel_size=int(torch.distributed.get_world_size()))


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().float().cpu().clone()


def _select_parameter_names(all_names: list[str]) -> list[str]:
    names = sorted(all_names)
    selected: list[str] = []

    preferred_suffixes = [
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

    for suffix in preferred_suffixes:
        for name in names:
            if name.endswith(suffix) and name not in selected:
                selected.append(name)
                break

    fallback = [name for name in names if name not in selected]
    head = fallback[:6]
    mid_start = max(0, len(fallback) // 2 - 2)
    mid = fallback[mid_start : mid_start + 4]
    tail = fallback[-6:]

    for name in [*head, *mid, *tail]:
        if name not in selected:
            selected.append(name)
    return selected


def _collect_named_gradients(
    engine: FSDPEngine | VeOMiniEngine, selected_names: list[str] | None = None
) -> tuple[dict[str, torch.Tensor], list[str]]:
    named_parameters = list(engine._get_model_name_parameters())
    grad_names = [name for name, param in named_parameters if param.grad is not None]
    if selected_names is None:
        selected_names = _select_parameter_names(grad_names)

    grad_map: dict[str, torch.Tensor] = {}
    for name, param in named_parameters:
        if name not in selected_names or param.grad is None:
            continue
        grad_map[name] = _materialize_tensor(param.grad)
    return grad_map, selected_names


def _collect_named_parameters(
    engine: FSDPEngine | VeOMiniEngine, selected_names: list[str]
) -> dict[str, torch.Tensor]:
    param_map: dict[str, torch.Tensor] = {}
    for name, param in engine._get_model_name_parameters():
        if name not in selected_names:
            continue
        if hasattr(engine, "_get_full_tensor"):
            full_tensor = engine._get_full_tensor(param)
        else:
            full_tensor = param.data
        param_map[name] = _materialize_tensor(full_tensor)
    return param_map


def _run_backward_then_step(
    engine: FSDPEngine | VeOMiniEngine,
    input_: dict[str, Any],
    selected_names: list[str] | None = None,
) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor], dict[str, float], list[str]]:
    engine.train()
    engine._ensure_ready()
    engine.optimizer_zero_grad()

    mb_list = engine._prepare_mb_list(input_).to(engine.device)
    total_loss_weight = compute_total_loss_weight(
        mb_list, lambda x: x["cu_seqlens"][-1], engine.data_parallel_group
    )

    def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> torch.Tensor:
        ctx = FSDPTrainContext(**ctx_dict)
        return engine._compute_logprobs_and_loss(
            logits,
            ctx,
            mock_loss_fn,
            lambda x: x["cu_seqlens"][-1],
            total_loss_weight,
            loss_multiplier=engine.data_parallel_world_size,
        )

    engine.forward_backward_batch(mb_list, process_output, forward_only=False)
    grad_map, selected_names = _collect_named_gradients(engine, selected_names)
    step_stats = engine.optimizer_step()
    param_map = _collect_named_parameters(engine, selected_names)
    return grad_map, param_map, step_stats, selected_names


def _destroy_engine(engine: FSDPEngine | VeOMiniEngine) -> None:
    engine.destroy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_backend_test(backend: str, output: str | None = None) -> bool:
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    assert world_size >= 1

    current_platform.set_device(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=8, train_batch_size=4)
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    if backend == "fsdp":
        engine: FSDPEngine | VeOMiniEngine = FSDPEngine(make_config("test_fsdp_dp_parity"))
    elif backend == "veomini":
        engine = VeOMiniEngine(make_config("test_veomini_dp_parity"))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)

    try:
        torch.manual_seed(20260308)
        input_ = mock_input(device=device)
        grads, params, step_stats, selected_names = _run_backward_then_step(engine, input_)

        if rank == 0 and output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            _save_artifact(
                output,
                {
                    "backend": backend,
                    "selected_names": selected_names,
                    "grads": grads,
                    "params": params,
                    "step_stats": step_stats,
                },
            )
            print(f"Saved {backend} artifact to {output}", flush=True)
            if not Path(output).exists():
                raise RuntimeError(f"Artifact was not written successfully: {output}")
        if dist.is_initialized():
            dist.barrier()
    finally:
        _destroy_engine(engine)

    return True


def main():
    parser = argparse.ArgumentParser(description="Run VeOMini vs FSDP DP parity test")
    parser.add_argument("--backend", type=str, choices=["fsdp", "veomini"], required=True)
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()
    success = run_backend_test(backend=args.backend, output=args.output)
    if not success:
        raise AssertionError("VeOMini vs FSDP DP parity test failed")


if __name__ == "__main__":
    main()