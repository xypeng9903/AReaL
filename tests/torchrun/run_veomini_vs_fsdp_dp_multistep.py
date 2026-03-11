import argparse
import gc
import os
from pathlib import Path
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


def make_config(experiment_name: str):
    return make_debug_train_config(experiment_name)


def mock_input(
    step: int,
    batch_size: int = 4,
    min_prompt_len: int = 8,
    max_prompt_len: int = 16,
    min_resp_len: int = 8,
    max_resp_len: int = 16,
    device: torch.device | str = current_platform.device_type,
) -> dict[str, Any]:
    pad_token_id = 0
    g = torch.Generator(device=device).manual_seed(20260310 + step)

    prompt_lens = torch.randint(min_prompt_len, max_prompt_len + 1, (batch_size,), dtype=torch.int, device=device, generator=g)
    resp_lens = torch.randint(min_resp_len, max_resp_len + 1, (batch_size,), dtype=torch.int, device=device, generator=g)
    seqlens = prompt_lens + resp_lens
    max_len = int(max(seqlens))
    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long, device=device)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    loss_mask = torch.zeros((batch_size, max_len), dtype=torch.bool, device=device)
    
    # Real RL inputs also often have old_logprobs and advantages
    old_logprobs = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)
    advantages = torch.zeros((batch_size, max_len), dtype=torch.float32, device=device)

    for i in range(batch_size):
        plen = prompt_lens[i].item()
        rlen = resp_lens[i].item()
        slen = plen + rlen
        # Left-pad text (SGLang outputs often left-padded prompts and right-padded responses, but here we do left-padded to stress test)
        # Actually standard for autoregressive is left padding for inputs. Let's do left padding where prompt starts at (max_len - slen)
        start_idx = max_len - slen
        input_ids[i, start_idx:] = torch.randint(1000, 5000, (slen,), dtype=torch.long, device=device, generator=g)
        attention_mask[i, start_idx:] = True
        loss_mask[i, start_idx + plen:] = True
        old_logprobs[i, start_idx:] = -torch.rand((slen,), device=device, generator=g) * 2
        advantages[i, start_idx + plen:] = torch.randn((rlen,), device=device, generator=g)

    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "loss_mask": loss_mask,
        "old_logprobs": old_logprobs,
        "advantages": advantages
    }


def mock_loss_fn(
    logprobs: torch.Tensor, entropy: torch.Tensor, input_data: dict, **kwargs
) -> torch.Tensor:
    # Use loss_mask to compute actual loss
    loss_mask = input_data["loss_mask"]
    adv = input_data.get("advantages", torch.ones_like(logprobs))
    old_logprobs = input_data.get("old_logprobs", torch.zeros_like(logprobs))

    # Real models pack the tensors, applying attention_mask, 
    # making them 1D tensors of size num_valid_tokens
    
    # If loss_mask refers to the tokens, in AReaL models `logprobs` is either returned as sequence layout or packed.
    # Note: `engine.train_batch` automatically packs the input dictionary, including loss_mask if defined.
    # We should flat logprobs and mask.
    if loss_mask.dim() == 2:
        loss_mask = loss_mask.reshape(-1)
        adv = adv.reshape(-1)
        old_logprobs = old_logprobs.reshape(-1)
        
    # Some packing logics modify sizes or padding
    if loss_mask.size(0) > logprobs.size(0):
        loss_mask = loss_mask[:logprobs.size(0)]
        adv = adv[:logprobs.size(0)]
        old_logprobs = old_logprobs[:logprobs.size(0)]
    elif loss_mask.size(0) < logprobs.size(0):
        logprobs = logprobs[:loss_mask.size(0)]
        entropy = entropy[:loss_mask.size(0)]
    
    # Compute masked sum
    if loss_mask.shape == logprobs.shape:
        valid_logprobs = logprobs[loss_mask]
        valid_entropy = entropy[loss_mask]
        valid_adv = adv[loss_mask]
        valid_old_logprobs = old_logprobs[loss_mask]
    else:
        # Fallback if shape mismatch happens
        valid_logprobs = logprobs
        valid_entropy = entropy
        valid_adv = adv
        valid_old_logprobs = old_logprobs
        
    # Simple PPO-like loss
    ratio = torch.exp(valid_logprobs - valid_old_logprobs)
    surr1 = ratio * valid_adv
    surr2 = torch.clamp(ratio, 1.0 - 0.2, 1.0 + 0.2) * valid_adv
    loss = -torch.min(surr1, surr2).mean() - 0.01 * torch.mean(valid_entropy)

    # prevent NaN if mask is empty
    if torch.isnan(loss) or (loss_mask.shape == logprobs.shape and loss_mask.sum() == 0):
        return (logprobs * 0).sum()
    return loss


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
