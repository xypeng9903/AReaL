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
from veomni.veomini_engine import VeOMiniEngine
from tests.torchrun.run_veomini_vs_fsdp_dp_multistep import make_config, mock_input, mock_loss_fn, _materialize_tensor

def _get_detailed_state(engine, input_, device):
    """Run forward and backward without step, return detailed state."""
    engine.train()
    engine._ensure_ready()
    engine.optimizer_zero_grad()

    mb_list = engine._prepare_mb_list(input_).to(engine.device)
    from areal.engine.core import compute_total_loss_weight
    from areal.engine.fsdp_engine import FSDPTrainContext
    
    dp_group = getattr(engine, "data_parallel_group", getattr(engine, "_dp_group", None))
    total_loss_weight = compute_total_loss_weight(
        mb_list, lambda x: x["cu_seqlens"][-1], dp_group
    )
    
    losses = []
    
    def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> torch.Tensor:
        ctx = FSDPTrainContext(**ctx_dict)
        dp_size = getattr(engine, "data_parallel_world_size", getattr(dist, "get_world_size", lambda: 1)())
        loss = engine._compute_logprobs_and_loss(
            logits,
            ctx,
            mock_loss_fn,
            lambda x: x["cu_seqlens"][-1],
            total_loss_weight,
            loss_multiplier=dp_size,
        )
        losses.append(loss.detach().item())
        return loss

    engine.forward_backward_batch(mb_list, process_output, forward_only=False)
    
    named_parameters = list(engine._get_model_name_parameters())
    grad_norms = {}
    param_norms = {}
    
    for name, param in named_parameters:
        if param.grad is not None:
            g = _materialize_tensor(param.grad)
            grad_norms[name] = g.norm().item()
        else:
            grad_norms[name] = 0.0
            
        if hasattr(engine, "_get_full_tensor"):
            p_full = _materialize_tensor(engine._get_full_tensor(param))
        else:
            p_full = _materialize_tensor(param.data)
        param_norms[name] = p_full.norm().item()
        
    return param_norms, grad_norms, sum(losses)

def run_tracker(backend: str, output: str, steps: int = 3):
    rank = dist.get_rank() if dist.is_initialized() else int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))

    current_platform.set_device(local_rank)
    device = torch.device(f"{current_platform.device_type}:{local_rank}")

    ft_spec = FinetuneSpec(total_train_epochs=1, dataset_size=64, train_batch_size=4)
    parallel_strategy = ParallelStrategy(data_parallel_size=world_size)

    config_name = "test_fsdp_dp_multistep" if backend == "fsdp" else "test_veomini_dp_multistep"
    engine_cls = FSDPEngine if backend == "fsdp" else VeOMiniEngine
    engine = engine_cls(make_config(config_name))
    
    engine.create_process_group(parallel_strategy=parallel_strategy)
    engine.initialize(addr=None, ft_spec=ft_spec)

    history = []
    
    for step in range(steps):
        input_ = mock_input(step=step, batch_size=4, min_prompt_len=16, max_prompt_len=32, min_resp_len=16, max_resp_len=32, device=device)
        p_norms, g_norms, loss_val = _get_detailed_state(engine, input_, device)
        
        step_stats = engine.optimizer_step()
        engine.lr_scheduler_step()
        
        p_norms_after = {}
        for name, param in engine._get_model_name_parameters():
            if hasattr(engine, "_get_full_tensor"):
                p_full = _materialize_tensor(engine._get_full_tensor(param))
            else:
                p_full = _materialize_tensor(param.data)
            p_norms_after[name] = p_full.norm().item()
            
        history.append({
            "step": step,
            "loss": loss_val,
            "p_norms_before": p_norms,
            "g_norms": g_norms,
            "p_norms_after": p_norms_after,
            "step_stats": step_stats
        })
        if rank == 0:
            print(f"[{backend}] Step {step}: Loss={loss_val:.6f}, EngineGradNorm={step_stats.get('grad_norm', 0.0):.4f}")
            
    if rank == 0:
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        torch.save(history, output)
        print(f"Saved {backend} detailed history to {output}", flush=True)

    engine.destroy()
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    run_tracker(args.backend, args.output, args.steps)
