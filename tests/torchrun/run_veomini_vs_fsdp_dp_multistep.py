import argparse
import gc
import os
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from areal.api.alloc_mode import ParallelStrategy
from areal.api.cli_args import MicroBatchSpec, OptimizerConfig, TrainEngineConfig
from areal.api.io_struct import FinetuneSpec
from areal.engine.fsdp_engine import FSDPEngine
from areal.infra.platforms import current_platform
from veomni.veomini_engine import VeOMiniEngine

MODEL_PATH = "/mnt/dolphinfs/ssd_pool/docker/user/hadoop-nlp-sh02/hadoop-aipnlp/FMG/pengxinyu05/huggingface.co/Qwen/Qwen3-1.7B-Base"


def make_config(experiment_name: str) -> TrainEngineConfig:
    return TrainEngineConfig(
        experiment_name=experiment_name,
        trial_name="test",
        path=MODEL_PATH,
        mb_spec=MicroBatchSpec(n_mbs=1),
        optimizer=OptimizerConfig(
            type="adam",
            lr=1e-5,
            weight_decay=0.01,
            beta1=0.9,
            beta2=0.999,
            eps=1e-8,
            lr_scheduler_type="constant",
            warmup_steps_proportion=0.0,
            gradient_clipping=1.0,
        ),
        disable_dropout=True,
    )


def mock_input(
    step: int,
    batch_size: int = 4,
    min_seqlen: int = 8,
    max_seqlen: int = 32,
    device: torch.device | str = current_platform.device_type,
) -> dict[str, Any]:
    pad_token_id = 0
    g = torch.Generator(device=device).manual_seed(20260310 + step)

    seqlens = torch.randint(
        min_seqlen, max_seqlen + 1, (batch_size,), dtype=torch.int, device=device, generator=g
    )
    max_len = int(max(seqlens))
    input_ids = torch.randint(
        1000, 5000, (batch_size, max_len), dtype=torch.long, device=device, generator=g
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


def _materialize_tensor(tensor: torch.Tensor) -> torch.Tensor:
    if isinstance(tensor, DTensor):
        tensor = tensor.full_tensor()
    return tensor.detach().float()


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

    if not selected:
        selected = names[:10]
    return selected


def _compute_selected_param_hash(
    engine: FSDPEngine | VeOMiniEngine,
    selected_names: list[str],
) -> torch.Tensor:
    h = torch.zeros((), dtype=torch.float64, device=engine.device)
    for name, param in engine._get_model_name_parameters():
        if name not in selected_names:
            continue
        if hasattr(engine, "_get_full_tensor"):
            tensor = engine._get_full_tensor(param)
        else:
            tensor = param.data
        tensor = _materialize_tensor(tensor)
        h += tensor.abs().sum().to(torch.float64)
    return h


def _rank_spread(value: torch.Tensor) -> float:
    v_min = value.clone()
    v_max = value.clone()
    dist.all_reduce(v_min, op=dist.ReduceOp.MIN)
    dist.all_reduce(v_max, op=dist.ReduceOp.MAX)
    return float((v_max - v_min).item())


def _destroy_engine(engine: FSDPEngine | VeOMiniEngine) -> None:
    engine.destroy()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_backend_multistep(backend: str, output: str | None = None, steps: int = 6) -> bool:
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    current_platform.set_device(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=64, train_batch_size=4)
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    if backend == "fsdp":
        engine: FSDPEngine | VeOMiniEngine = FSDPEngine(make_config("test_fsdp_dp_multistep"))
    elif backend == "veomini":
        engine = VeOMiniEngine(make_config("test_veomini_dp_multistep"))
    else:
        raise ValueError(f"Unknown backend: {backend}")

    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)

    records: list[dict[str, float]] = []

    try:
        all_names = [name for name, _ in engine._get_model_name_parameters()]
        selected_names = _select_parameter_names(all_names)

        # Record initial param_hash (before any training step)
        init_hash = _compute_selected_param_hash(engine, selected_names)
        init_spread = _rank_spread(init_hash)
        records.append(
            {
                "step": -1.0,
                "update_successful": 1.0,
                "grad_norm": 0.0,
                "lr": 0.0,
                "param_hash": float(init_hash.item()),
                "rank_hash_spread": init_spread,
            }
        )
        if rank == 0:
            print(f"[{backend}] init param_hash={init_hash.item():.6f}", flush=True)

        for step in range(steps):
            input_ = mock_input(step=step, device=device)
            engine.train()
            step_stats = engine.train_batch(
                input_=input_,
                loss_fn=mock_loss_fn,
                loss_weight_fn=lambda x: x["cu_seqlens"][-1],
            )
            param_hash = _compute_selected_param_hash(engine, selected_names)
            spread = _rank_spread(param_hash)

            rec = {
                "step": float(step),
                "update_successful": float(step_stats["update_successful"]),
                "grad_norm": float(step_stats["grad_norm"]),
                "lr": float(step_stats["lr"]),
                "param_hash": float(param_hash.item()),
                "rank_hash_spread": spread,
            }
            records.append(rec)

            if rank == 0:
                print(
                    f"[{backend}] step={step} grad_norm={rec['grad_norm']:.4f} "
                    f"param_hash={rec['param_hash']:.6f} spread={spread:.2e}",
                    flush=True,
                )

            engine.lr_scheduler_step()

        if rank == 0 and output is not None:
            Path(output).parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "backend": backend,
                    "selected_names": selected_names,
                    "records": records,
                },
                output,
            )
            print(f"Saved {backend} multistep artifact to {output}", flush=True)

        if dist.is_initialized():
            dist.barrier()
    finally:
        _destroy_engine(engine)

    return True


def main():
    parser = argparse.ArgumentParser(description="Run VeOMini vs FSDP DP multistep parity test")
    parser.add_argument("--backend", type=str, choices=["fsdp", "veomini"], required=True)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--steps", type=int, default=6)
    args = parser.parse_args()

    ok = run_backend_multistep(backend=args.backend, output=args.output, steps=args.steps)
    if not ok:
        raise AssertionError("VeOMini vs FSDP DP multistep test failed")


if __name__ == "__main__":
    main()
