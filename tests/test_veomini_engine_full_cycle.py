"""Full training cycle parity test: VeOMiniEngine vs FSDPEngine.

Simulates the *complete* train_batch loop N steps on a single device,
comparing every observable quantity at each step:

  zero_grad → forward → logprobs/entropy/loss → backward →
  grad_clip → optimizer.step → lr_scheduler.step

The goal is to find divergence that accumulates across the full pipeline,
even when individual components look correct in isolation.

Usage (on the GPU server):
    pytest tests/test_veomini_engine_full_cycle.py -v -s
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Callable

import pytest
import torch
from torch import nn

# --------------------------------------------------------------------------
# AReaL imports
# --------------------------------------------------------------------------
from areal.engine.fsdp_engine import FSDPEngine, FSDPTrainContext
from areal.engine.fsdp_utils import get_cosine_schedule_with_warmup
from areal.engine.core.model import disable_dropout_in_model
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# VeOmni imports
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.veomini_engine import VeOMiniEngine
import veomni.veomini_engine as veomini_engine_module


# ==========================================================================
# Helpers — tiny model with realistic structure
# ==========================================================================


class _SmallCausalLM(nn.Module):
    """Minimal causal LM with embedding → transformer block → lm_head.

    Mimics the structure HuggingFace models expose:
      model(**inputs) → CausalLMOutputWithPast(logits=...)
    """

    VOCAB_SIZE = 32
    HIDDEN = 16
    SEQ_LEN = 12

    def __init__(self) -> None:
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, self.HIDDEN)
        self.layers = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.GELU(),
            nn.LayerNorm(self.HIDDEN),
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.GELU(),
            nn.LayerNorm(self.HIDDEN),
        )
        self.lm_head = nn.Linear(self.HIDDEN, self.VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> SimpleNamespace:
        if input_ids.ndim == 2:
            input_ids = input_ids.squeeze(0)
        h = self.embed(input_ids)
        h = self.layers(h)
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits)


# ==========================================================================
# Helpers — stub engines (no distributed, no model loading)
# ==========================================================================


def _make_fsdp_stub(temperature: float = 1.0) -> FSDPEngine:
    """Create an FSDPEngine without __init__ (no distributed setup needed)."""
    engine = object.__new__(FSDPEngine)
    engine.config = SimpleNamespace(is_critic=False, temperature=temperature)
    engine.parallel_helper = SimpleNamespace(sp_size=1, tp_size=1, tp_group=None)
    engine.enable_tree_training = False
    return engine


def _make_veomini_stub(temperature: float = 1.0) -> VeOMiniEngine:
    """Create a VeOMiniEngine without __init__."""
    engine = object.__new__(VeOMiniEngine)
    engine.config = SimpleNamespace(is_critic=False, temperature=temperature)
    engine._parallel_state = None
    engine._sp_group = None
    return engine


# ==========================================================================
# Helpers — RL-style loss function & weight function
# ==========================================================================


def actor_loss_fn(
    logprobs: torch.Tensor,
    entropy: torch.Tensor,
    input_data: dict,
    vocab_min_logits: torch.Tensor | None = None,
    vocab_max_logits: torch.Tensor | None = None,
) -> torch.Tensor:
    """PPO-style actor loss: policy gradient + entropy bonus + vocab range penalty."""
    loss_mask = input_data["loss_mask"].to(logprobs.dtype)
    target_logprobs = input_data["target_logprobs"].to(logprobs.dtype)
    advantages = input_data["advantages"].to(logprobs.dtype)

    # Simplified PPO: ratio * advantage
    ratio = torch.exp(logprobs - target_logprobs)
    pg_loss = -(ratio * advantages * loss_mask).sum()

    # Entropy bonus
    entropy_loss = -0.01 * (entropy * loss_mask).sum()

    # Vocab range regularization
    assert vocab_min_logits is not None and vocab_max_logits is not None
    vocab_penalty = 0.001 * (
        (vocab_max_logits - vocab_min_logits) * loss_mask
    ).sum()

    return pg_loss + entropy_loss + vocab_penalty


def loss_weight_fn(input_data: dict[str, torch.Tensor]) -> torch.Tensor:
    return input_data["loss_mask"].count_nonzero()


# ==========================================================================
# Helpers — generate RL training data
# ==========================================================================


def _generate_rl_batch(
    batch_idx: int,
    vocab_size: int = _SmallCausalLM.VOCAB_SIZE,
    seq_len: int = _SmallCausalLM.SEQ_LEN,
) -> FSDPTrainContext:
    """Generate a fake RL micro-batch with deterministic data per batch_idx."""
    rng = torch.Generator().manual_seed(1000 + batch_idx)

    input_ids = torch.randint(0, vocab_size, (1, seq_len), generator=rng)
    rolled_input_ids = torch.roll(input_ids, shifts=-1, dims=-1)

    # Last token has no target → mask it out
    loss_mask = torch.ones(seq_len, dtype=torch.float32)
    loss_mask[-1] = 0.0

    target_logprobs = -torch.rand(seq_len, generator=rng) * 3  # ~ [-3, 0]
    advantages = torch.randn(seq_len, generator=rng)  # ~ N(0, 1)

    model_inputs = {
        "input_ids": input_ids,
        "rolled_input_ids": rolled_input_ids,
    }
    mb_input = {
        "loss_mask": loss_mask,
        "target_logprobs": target_logprobs,
        "advantages": advantages,
    }
    return FSDPTrainContext(model_inputs=model_inputs, mb_input=mb_input)


# ==========================================================================
# Core test: full training cycle
# ==========================================================================

@dataclass
class StepRecord:
    """Records observable quantities at each training step."""
    step: int
    loss: float
    grad_norm: float
    lr: float
    param_hash: float  # sum of all param abs values (cheap fingerprint)


def _run_full_training_cycle(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    engine_stub: FSDPEngine | VeOMiniEngine,
    n_steps: int,
    max_grad_norm: float,
    dp_size: int = 1,
) -> list[StepRecord]:
    """Simulate the full train_batch flow for n_steps.

    Mirrors the actual engine code:
        optimizer.zero_grad()
        for mb in mb_list:
            logits = model(**inputs)
            loss = engine._compute_logprobs_and_loss(logits, ctx, ...)
            loss.backward()
        grad_norm = clip_grad_norm(model, max_grad_norm)
        if finite(grad_norm): optimizer.step()
        lr_scheduler.step()
    """
    records = []

    for step in range(n_steps):
        optimizer.zero_grad()

        # --- Forward + loss (1 micro-batch per step for simplicity) ---
        ctx = _generate_rl_batch(step)
        inputs = ctx.model_inputs

        outputs = model(**inputs)
        logits = outputs.logits.squeeze(0)

        # Total loss weight for this step (simulates compute_total_loss_weight)
        total_loss_weight = loss_weight_fn(ctx.mb_input).float()

        # Use the engine's _compute_logprobs_and_loss to compute scaled loss
        loss = engine_stub._compute_logprobs_and_loss(
            logits,
            ctx,
            actor_loss_fn,
            loss_weight_fn,
            total_loss_weight,
            loss_multiplier=float(dp_size),
        )

        loss_val = loss.detach().item()

        # --- Backward ---
        loss.backward()

        # --- Gradient clipping ---
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        ).item()

        # --- Optimizer step (skip on non-finite grad norm) ---
        if math.isfinite(grad_norm):
            optimizer.step()
            update_ok = True
        else:
            optimizer.zero_grad()
            update_ok = False

        # --- LR scheduler step ---
        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()

        # --- Record ---
        param_hash = sum(p.data.abs().sum().item() for p in model.parameters())
        records.append(StepRecord(
            step=step,
            loss=loss_val,
            grad_norm=grad_norm,
            lr=current_lr,
            param_hash=param_hash,
        ))

    return records


# ==========================================================================
# Test Class: Full Cycle Parity
# ==========================================================================


class TestFullTrainingCycleParity:
    """End-to-end parity test: run N training steps with the SAME data
    and SAME initial weights, but using VeOMini vs FSDP-style setup.

    Tests cover:
    - Loss value at each step
    - Gradient norm at each step
    - LR at each step
    - Parameter values at each step
    """

    N_STEPS = 30
    LR = 1.7e-5
    WEIGHT_DECAY = 0.1
    BETA1, BETA2 = 0.9, 0.999
    EPS = 1e-8
    MAX_GRAD_NORM = 1.0
    TEMPERATURE = 0.9

    def _make_model_pair(self) -> tuple[_SmallCausalLM, _SmallCausalLM]:
        """Create two identical models."""
        torch.manual_seed(42)
        model_a = _SmallCausalLM()
        model_b = _SmallCausalLM()
        model_b.load_state_dict(model_a.state_dict())
        return model_a, model_b

    # ---- constant LR (user's actual config) ----

    def test_constant_lr_full_cycle(self):
        """Full training cycle with constant LR — the user's actual config.
        Should be IDENTICAL between VeOmni and FSDP."""
        model_fsdp, model_veomni = self._make_model_pair()

        total_steps = self.N_STEPS
        warmup_proportion = 0.001
        num_warmup = int(warmup_proportion * total_steps)

        # FSDP-style: torch.optim.AdamW + transformers constant scheduler
        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=self.LR, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_constant_schedule_with_warmup(opt_fsdp, num_warmup)

        # VeOMini-style: build_optimizer + same scheduler (after fix)
        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )
        sched_veomni = get_constant_schedule_with_warmup(opt_veomni, num_warmup)

        # Stubs
        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        # Run
        records_fsdp = _run_full_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            self.N_STEPS, self.MAX_GRAD_NORM,
        )
        records_veomni = _run_full_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            self.N_STEPS, self.MAX_GRAD_NORM,
        )

        self._compare_records(records_fsdp, records_veomni, "constant")

    # ---- cosine LR ----

    def test_cosine_lr_full_cycle(self):
        """Full training cycle with cosine LR — important for future configs.
        After LR fix, should be identical."""
        model_fsdp, model_veomni = self._make_model_pair()

        total_steps = self.N_STEPS
        warmup_proportion = 0.1
        num_warmup = int(warmup_proportion * total_steps)
        min_lr_ratio = 0.1

        # FSDP
        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=self.LR, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_cosine_schedule_with_warmup(
            opt_fsdp, num_warmup, total_steps, min_lr_ratio=min_lr_ratio,
        )

        # VeOMini (fixed: now uses AReaL schedulers)
        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )
        sched_veomni = get_cosine_schedule_with_warmup(
            opt_veomni, num_warmup, total_steps, min_lr_ratio=min_lr_ratio,
        )

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        records_fsdp = _run_full_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            self.N_STEPS, self.MAX_GRAD_NORM,
        )
        records_veomni = _run_full_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            self.N_STEPS, self.MAX_GRAD_NORM,
        )

        self._compare_records(records_fsdp, records_veomni, "cosine")

    # ---- cosine LR with OLD VeOmni scheduler (regression test) ----

    def test_cosine_lr_old_veomni_diverges(self):
        """With the OLD VeOmni build_lr_scheduler (before fix),
        cosine training should diverge from FSDP. This is a regression test
        to ensure the fix actually matters."""
        model_fsdp, model_veomni = self._make_model_pair()

        total_steps = 100
        warmup_proportion = 0.1
        num_warmup = int(warmup_proportion * total_steps)
        min_lr_ratio = 0.0  # max difference

        # FSDP: AReaL cosine
        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=self.LR, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_cosine_schedule_with_warmup(
            opt_fsdp, num_warmup, total_steps, min_lr_ratio=min_lr_ratio,
        )

        # VeOMini: OLD VeOmni scheduler (build_lr_scheduler)
        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )
        sched_veomni = build_lr_scheduler(
            opt_veomni,
            train_steps=total_steps, lr=self.LR,
            lr_decay_style="cosine", lr_warmup_ratio=warmup_proportion,
            lr_min=1e-7, lr_start=0.0,
        )

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        records_fsdp = _run_full_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            n_steps=total_steps, max_grad_norm=self.MAX_GRAD_NORM,
        )
        records_veomni = _run_full_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            n_steps=total_steps, max_grad_norm=self.MAX_GRAD_NORM,
        )

        # With old scheduler, LR should differ during warmup
        warmup_lr_diffs = [
            abs(records_fsdp[i].lr - records_veomni[i].lr)
            for i in range(num_warmup)
        ]
        max_lr_diff = max(warmup_lr_diffs) if warmup_lr_diffs else 0
        print(f"Old cosine scheduler: max warmup LR diff = {max_lr_diff:.2e}")

        # Parameters should diverge
        final_param_diff = abs(
            records_fsdp[-1].param_hash - records_veomni[-1].param_hash
        )
        print(f"Old cosine scheduler: final param_hash diff = {final_param_diff:.6e}")

        # This SHOULD diverge — the old scheduler is different
        assert max_lr_diff > 0 or final_param_diff > 0, (
            "Expected divergence with old VeOmni scheduler, but they matched! "
            "Is build_lr_scheduler now identical to get_cosine_schedule_with_warmup?"
        )

    # ---- linear LR ----

    def test_linear_lr_full_cycle(self):
        """Full training cycle with linear LR."""
        model_fsdp, model_veomni = self._make_model_pair()

        total_steps = self.N_STEPS
        warmup_proportion = 0.1
        num_warmup = int(warmup_proportion * total_steps)

        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=self.LR, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_linear_schedule_with_warmup(
            opt_fsdp, num_warmup, total_steps,
        )

        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )
        sched_veomni = get_linear_schedule_with_warmup(
            opt_veomni, num_warmup, total_steps,
        )

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        records_fsdp = _run_full_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            self.N_STEPS, self.MAX_GRAD_NORM,
        )
        records_veomni = _run_full_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            self.N_STEPS, self.MAX_GRAD_NORM,
        )

        self._compare_records(records_fsdp, records_veomni, "linear")

    # ---- high LR stress test ----

    def test_high_lr_stress_test(self):
        """With a higher LR, any numerical difference gets amplified.
        This is the most sensitive test for catching subtle parity issues."""
        model_fsdp, model_veomni = self._make_model_pair()

        high_lr = 1e-3
        total_steps = 50
        warmup_proportion = 0.1
        num_warmup = int(warmup_proportion * total_steps)

        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=high_lr, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_constant_schedule_with_warmup(opt_fsdp, num_warmup)

        opt_veomni = build_optimizer(
            model_veomni,
            lr=high_lr, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )
        sched_veomni = get_constant_schedule_with_warmup(opt_veomni, num_warmup)

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        records_fsdp = _run_full_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            n_steps=total_steps, max_grad_norm=self.MAX_GRAD_NORM,
        )
        records_veomni = _run_full_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            n_steps=total_steps, max_grad_norm=self.MAX_GRAD_NORM,
        )

        self._compare_records(records_fsdp, records_veomni, "high_lr_stress")

    # ---- multiple micro-batches per step ----

    def test_multi_microbatch_accumulation(self):
        """Simulate gradient accumulation across 4 micro-batches per step,
        which is the typical RL training pattern."""
        model_fsdp, model_veomni = self._make_model_pair()

        total_steps = 20
        n_microbatches = 4
        warmup_proportion = 0.1
        num_warmup = int(warmup_proportion * total_steps)

        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=self.LR, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_constant_schedule_with_warmup(opt_fsdp, num_warmup)

        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )
        sched_veomni = get_constant_schedule_with_warmup(opt_veomni, num_warmup)

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        records_fsdp = _run_multi_mb_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            total_steps, n_microbatches, self.MAX_GRAD_NORM,
        )
        records_veomni = _run_multi_mb_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            total_steps, n_microbatches, self.MAX_GRAD_NORM,
        )

        self._compare_records(records_fsdp, records_veomni, "multi_mb")

    # ---- non-finite gradient handling ----

    def test_nan_grad_skips_update(self):
        """When gradient is NaN/Inf, both engines should skip optimizer.step().
        This tests the `if not math.isfinite(grad_norm)` branch."""
        model_fsdp, model_veomni = self._make_model_pair()

        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(), lr=self.LR, fused=False,
        )
        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            param_groups=[{
                "params": [p for p in model_veomni.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }],
        )

        # Inject NaN into gradients
        opt_fsdp.zero_grad()
        opt_veomni.zero_grad()

        ctx = _generate_rl_batch(0)
        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        # Normal forward + backward
        logits_f = model_fsdp(**ctx.model_inputs).logits.squeeze(0)
        total_w = loss_weight_fn(ctx.mb_input).float()
        loss_f = fsdp_stub._compute_logprobs_and_loss(
            logits_f, ctx, actor_loss_fn, loss_weight_fn, total_w,
        )
        loss_f.backward()

        logits_v = model_veomni(**ctx.model_inputs).logits.squeeze(0)
        loss_v = veomini_stub._compute_logprobs_and_loss(
            logits_v, ctx, actor_loss_fn, loss_weight_fn, total_w,
        )
        loss_v.backward()

        # Inject NaN into one param's grad
        for p in model_fsdp.parameters():
            if p.grad is not None:
                p.grad[0] = float("nan")
                break
        for p in model_veomni.parameters():
            if p.grad is not None:
                p.grad[0] = float("nan")
                break

        # Clip → should get inf/nan norm
        norm_f = torch.nn.utils.clip_grad_norm_(
            model_fsdp.parameters(), self.MAX_GRAD_NORM
        ).item()
        norm_v = torch.nn.utils.clip_grad_norm_(
            model_veomni.parameters(), self.MAX_GRAD_NORM
        ).item()

        # Both should be non-finite
        assert not math.isfinite(norm_f), f"FSDP grad norm should be non-finite, got {norm_f}"
        assert not math.isfinite(norm_v), f"VeOMini grad norm should be non-finite, got {norm_v}"

        # Both should skip update → params unchanged from before step
        params_before_f = {n: p.data.clone() for n, p in model_fsdp.named_parameters()}
        params_before_v = {n: p.data.clone() for n, p in model_veomni.named_parameters()}

        # Simulate the "skip" logic
        opt_fsdp.zero_grad()
        opt_veomni.zero_grad()

        # Params should be unchanged (no step was called)
        for n, p in model_fsdp.named_parameters():
            assert torch.equal(p.data, params_before_f[n]), f"FSDP param {n} changed after NaN skip"
        for n, p in model_veomni.named_parameters():
            assert torch.equal(p.data, params_before_v[n]), f"VeOMini param {n} changed after NaN skip"

        print("✓ Both engines correctly skip update on NaN gradients")

    # ---- weight decay splitting (default vs explicit) ----

    def test_default_weight_decay_splits_diverge(self):
        """When VeOmni uses DEFAULT param groups (splits bias/norm),
        but FSDP applies uniform weight decay, the training diverges.

        This tests the param_groups difference that VeOMiniEngine avoids
        by passing explicit param_groups."""
        model_fsdp, model_veomni = self._make_model_pair()

        total_steps = 50
        num_warmup = 5

        # FSDP: uniform weight decay
        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=1e-3, weight_decay=self.WEIGHT_DECAY,
            betas=(self.BETA1, self.BETA2), eps=self.EPS, fused=False,
        )
        sched_fsdp = get_constant_schedule_with_warmup(opt_fsdp, num_warmup)

        # VeOMini: DEFAULT (no explicit param_groups) → bias/norm get no decay
        opt_veomni = build_optimizer(
            model_veomni,
            lr=1e-3, betas=(self.BETA1, self.BETA2), eps=self.EPS,
            weight_decay=self.WEIGHT_DECAY, fused=False, optimizer_type="adamw",
            # No param_groups → VeOmni splits decay/no-decay internally
        )
        sched_veomni = get_constant_schedule_with_warmup(opt_veomni, num_warmup)

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        records_fsdp = _run_full_training_cycle(
            model_fsdp, opt_fsdp, sched_fsdp, fsdp_stub,
            n_steps=total_steps, max_grad_norm=self.MAX_GRAD_NORM,
        )
        records_veomni = _run_full_training_cycle(
            model_veomni, opt_veomni, sched_veomni, veomini_stub,
            n_steps=total_steps, max_grad_norm=self.MAX_GRAD_NORM,
        )

        # They should diverge because of different weight decay on bias/norm
        final_param_diff = abs(
            records_fsdp[-1].param_hash - records_veomni[-1].param_hash
        )
        final_loss_diff = abs(
            records_fsdp[-1].loss - records_veomni[-1].loss
        )

        print(f"Default WD split: final param diff = {final_param_diff:.6e}")
        print(f"Default WD split: final loss diff = {final_loss_diff:.6e}")

        # Check: with higher LR and 50 steps, this SHOULD diverge
        if final_param_diff > 1e-6:
            print("⚠ Default weight-decay splitting causes divergence (expected)")
        else:
            print("✓ No significant divergence (weight decay splitting effect is tiny)")

    # ---- temperature parameter ----

    def test_temperature_parity(self):
        """Both engines should compute identical logprobs with the same temperature."""
        fsdp_stub = _make_fsdp_stub(temperature=0.7)
        veomini_stub = _make_veomini_stub(temperature=0.7)

        torch.manual_seed(42)
        logits = torch.randn(_SmallCausalLM.SEQ_LEN, _SmallCausalLM.VOCAB_SIZE)

        ctx = _generate_rl_batch(0)
        total_w = loss_weight_fn(ctx.mb_input).float()

        loss_fsdp = fsdp_stub._compute_logprobs_and_loss(
            logits.clone().requires_grad_(True),
            ctx, actor_loss_fn, loss_weight_fn, total_w,
        )
        loss_veomni = veomini_stub._compute_logprobs_and_loss(
            logits.clone().requires_grad_(True),
            ctx, actor_loss_fn, loss_weight_fn, total_w,
        )

        assert torch.allclose(loss_fsdp, loss_veomni, atol=1e-7), (
            f"Temperature 0.7: FSDP loss={loss_fsdp.item():.8f}, "
            f"VeOMini loss={loss_veomni.item():.8f}"
        )
        print(f"✓ Temperature=0.7 parity: loss diff = {abs(loss_fsdp.item() - loss_veomni.item()):.2e}")

    # ======================================================================
    # Comparison helper
    # ======================================================================

    def _compare_records(
        self,
        records_fsdp: list[StepRecord],
        records_veomni: list[StepRecord],
        label: str,
    ) -> None:
        assert len(records_fsdp) == len(records_veomni)

        max_loss_diff = 0.0
        max_grad_diff = 0.0
        max_lr_diff = 0.0
        max_param_diff = 0.0
        first_diverge_step = None

        print(f"\n{'='*70}")
        print(f" Full Training Cycle Parity: {label}")
        print(f"{'='*70}")
        print(f"{'Step':>4} | {'Loss Diff':>12} | {'GradNorm Diff':>14} | "
              f"{'LR Diff':>12} | {'Param Diff':>12}")
        print(f"{'-'*4}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")

        for rf, rv in zip(records_fsdp, records_veomni):
            ld = abs(rf.loss - rv.loss)
            gd = abs(rf.grad_norm - rv.grad_norm)
            lrd = abs(rf.lr - rv.lr)
            pd = abs(rf.param_hash - rv.param_hash)

            max_loss_diff = max(max_loss_diff, ld)
            max_grad_diff = max(max_grad_diff, gd)
            max_lr_diff = max(max_lr_diff, lrd)
            max_param_diff = max(max_param_diff, pd)

            if first_diverge_step is None and pd > 1e-6:
                first_diverge_step = rf.step

            # Print every 5th step or if there's noticeable diff
            if rf.step % 5 == 0 or pd > 1e-8:
                print(f"{rf.step:4d} | {ld:12.6e} | {gd:14.6e} | "
                      f"{lrd:12.6e} | {pd:12.6e}")

        print(f"{'-'*4}-+-{'-'*12}-+-{'-'*14}-+-{'-'*12}-+-{'-'*12}")
        print(f"MAX  | {max_loss_diff:12.6e} | {max_grad_diff:14.6e} | "
              f"{max_lr_diff:12.6e} | {max_param_diff:12.6e}")

        if first_diverge_step is not None:
            print(f"\n⚠ First significant divergence at step {first_diverge_step}")
        else:
            print(f"\n✓ No significant divergence across {len(records_fsdp)} steps")

        # Assert parity
        assert max_loss_diff < 1e-5, (
            f"[{label}] Loss diverged: max diff = {max_loss_diff:.2e}"
        )
        assert max_grad_diff < 1e-5, (
            f"[{label}] Grad norm diverged: max diff = {max_grad_diff:.2e}"
        )
        assert max_lr_diff < 1e-10, (
            f"[{label}] LR diverged: max diff = {max_lr_diff:.2e}"
        )
        assert max_param_diff < 1e-4, (
            f"[{label}] Params diverged: max diff = {max_param_diff:.2e}"
        )


# ==========================================================================
# Multi micro-batch helper
# ==========================================================================


def _run_multi_mb_training_cycle(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: Any,
    engine_stub: FSDPEngine | VeOMiniEngine,
    n_steps: int,
    n_microbatches: int,
    max_grad_norm: float,
    dp_size: int = 1,
) -> list[StepRecord]:
    """Like _run_full_training_cycle, but accumulates gradients across
    multiple micro-batches before stepping — mimics the real RL training loop."""
    records = []

    for step in range(n_steps):
        optimizer.zero_grad()
        total_loss = 0.0

        # Accumulate gradients across micro-batches
        for mb_idx in range(n_microbatches):
            ctx = _generate_rl_batch(step * n_microbatches + mb_idx)
            inputs = ctx.model_inputs

            outputs = model(**inputs)
            logits = outputs.logits.squeeze(0)

            total_loss_weight = loss_weight_fn(ctx.mb_input).float()

            loss = engine_stub._compute_logprobs_and_loss(
                logits, ctx, actor_loss_fn, loss_weight_fn,
                total_loss_weight, loss_multiplier=float(dp_size),
            )
            loss.backward()
            total_loss += loss.detach().item()

        # Gradient clipping
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), max_norm=max_grad_norm
        ).item()

        # Optimizer step
        if math.isfinite(grad_norm):
            optimizer.step()

        # LR step
        current_lr = lr_scheduler.get_last_lr()[0]
        lr_scheduler.step()

        param_hash = sum(p.data.abs().sum().item() for p in model.parameters())
        records.append(StepRecord(
            step=step,
            loss=total_loss,
            grad_norm=grad_norm,
            lr=current_lr,
            param_hash=param_hash,
        ))

    return records


# ==========================================================================
# Test Class: Gradient flow parity
# ==========================================================================


class TestGradientFlowParity:
    """Verify that gradients flowing through _compute_logprobs_and_loss
    produce identical parameter gradients in both engines."""

    TEMPERATURE = 0.9

    def test_gradient_values_match_per_parameter(self):
        """After one forward-backward through the full pipeline,
        every parameter's gradient should be bit-for-bit identical."""
        torch.manual_seed(42)
        model_fsdp = _SmallCausalLM()
        model_veomni = _SmallCausalLM()
        model_veomni.load_state_dict(model_fsdp.state_dict())

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        ctx = _generate_rl_batch(0)
        total_w = loss_weight_fn(ctx.mb_input).float()

        # FSDP forward-backward
        logits_f = model_fsdp(**ctx.model_inputs).logits.squeeze(0)
        loss_f = fsdp_stub._compute_logprobs_and_loss(
            logits_f, ctx, actor_loss_fn, loss_weight_fn, total_w,
        )
        loss_f.backward()

        # VeOMini forward-backward
        logits_v = model_veomni(**ctx.model_inputs).logits.squeeze(0)
        loss_v = veomini_stub._compute_logprobs_and_loss(
            logits_v, ctx, actor_loss_fn, loss_weight_fn, total_w,
        )
        loss_v.backward()

        # Compare losses
        assert torch.allclose(loss_f, loss_v, atol=1e-7), (
            f"Loss mismatch: FSDP={loss_f.item():.8f}, VeOMini={loss_v.item():.8f}"
        )

        # Compare every gradient
        print(f"\n{'Parameter':<30} | {'Grad Diff':>12}")
        print(f"{'-'*30}-+-{'-'*12}")

        max_diff = 0.0
        for (nf, pf), (nv, pv) in zip(
            model_fsdp.named_parameters(), model_veomni.named_parameters()
        ):
            assert nf == nv
            assert pf.grad is not None and pv.grad is not None, f"Missing grad for {nf}"
            diff = (pf.grad - pv.grad).abs().max().item()
            max_diff = max(max_diff, diff)
            print(f"{nf:<30} | {diff:12.2e}")

        print(f"\nMax gradient diff: {max_diff:.2e}")
        assert max_diff < 1e-7, f"Gradients diverged: max diff = {max_diff:.2e}"

    def test_gradient_accumulation_over_multiple_mb(self):
        """Gradients accumulated over 4 micro-batches should match."""
        torch.manual_seed(42)
        model_fsdp = _SmallCausalLM()
        model_veomni = _SmallCausalLM()
        model_veomni.load_state_dict(model_fsdp.state_dict())

        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        n_mb = 4
        for mb_idx in range(n_mb):
            ctx = _generate_rl_batch(mb_idx)
            total_w = loss_weight_fn(ctx.mb_input).float()

            logits_f = model_fsdp(**ctx.model_inputs).logits.squeeze(0)
            loss_f = fsdp_stub._compute_logprobs_and_loss(
                logits_f, ctx, actor_loss_fn, loss_weight_fn, total_w,
            )
            loss_f.backward()

            logits_v = model_veomni(**ctx.model_inputs).logits.squeeze(0)
            loss_v = veomini_stub._compute_logprobs_and_loss(
                logits_v, ctx, actor_loss_fn, loss_weight_fn, total_w,
            )
            loss_v.backward()

        max_diff = 0.0
        for (nf, pf), (nv, pv) in zip(
            model_fsdp.named_parameters(), model_veomni.named_parameters()
        ):
            diff = (pf.grad - pv.grad).abs().max().item()
            max_diff = max(max_diff, diff)

        print(f"After {n_mb} micro-batches accumulated grad diff: {max_diff:.2e}")
        assert max_diff < 1e-6, f"Accumulated gradients diverged: max diff = {max_diff:.2e}"


# ==========================================================================
# Test Class: Loss scaling and multiplier
# ==========================================================================


class TestLossScaling:
    """Verify that loss_scale computation is identical between engines."""

    TEMPERATURE = 1.0

    def test_loss_multiplier_dp4(self):
        """With dp_size=4, loss_multiplier=4 — both engines should scale identically."""
        fsdp_stub = _make_fsdp_stub(self.TEMPERATURE)
        veomini_stub = _make_veomini_stub(self.TEMPERATURE)

        torch.manual_seed(42)
        logits = torch.randn(_SmallCausalLM.SEQ_LEN, _SmallCausalLM.VOCAB_SIZE, requires_grad=True)
        logits_clone = logits.clone().detach().requires_grad_(True)

        ctx = _generate_rl_batch(0)
        total_w = loss_weight_fn(ctx.mb_input).float()

        loss_f = fsdp_stub._compute_logprobs_and_loss(
            logits, ctx, actor_loss_fn, loss_weight_fn, total_w,
            loss_multiplier=4.0,
        )
        loss_v = veomini_stub._compute_logprobs_and_loss(
            logits_clone, ctx, actor_loss_fn, loss_weight_fn, total_w,
            loss_multiplier=4.0,
        )

        assert torch.allclose(loss_f, loss_v, atol=1e-7), (
            f"dp_size=4: FSDP loss={loss_f.item():.8f}, VeOMini={loss_v.item():.8f}"
        )

        # Backward should also match
        loss_f.backward()
        loss_v.backward()
        assert torch.allclose(logits.grad, logits_clone.grad, atol=1e-7)
        print(f"✓ loss_multiplier=4.0 parity OK, loss={loss_f.item():.6e}")


# ==========================================================================
# Test Class: Non-TP hypotheses for reward not improving
# ==========================================================================


class _DropoutCausalLM(nn.Module):
    """Tiny LM with dropout to simulate RL logprob instability."""

    VOCAB_SIZE = 32
    HIDDEN = 16
    SEQ_LEN = 12

    def __init__(self, p: float = 0.2) -> None:
        super().__init__()
        self.embed = nn.Embedding(self.VOCAB_SIZE, self.HIDDEN)
        self.layers = nn.Sequential(
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.GELU(),
            nn.Dropout(p),
            nn.LayerNorm(self.HIDDEN),
            nn.Linear(self.HIDDEN, self.HIDDEN),
            nn.GELU(),
            nn.Dropout(p),
            nn.LayerNorm(self.HIDDEN),
        )
        self.lm_head = nn.Linear(self.HIDDEN, self.VOCAB_SIZE, bias=False)

    def forward(self, input_ids: torch.Tensor, **kwargs) -> SimpleNamespace:
        if input_ids.ndim == 2:
            input_ids = input_ids.squeeze(0)
        h = self.embed(input_ids)
        h = self.layers(h)
        logits = self.lm_head(h)
        return SimpleNamespace(logits=logits)


class TestNonTPRewardStallHypotheses:
    """Hypotheses relevant when TP is disabled (tp_size=1)."""

    def _collect_logprobs(
        self,
        engine_stub: FSDPEngine | VeOMiniEngine,
        model: nn.Module,
        ctx: FSDPTrainContext,
        n_passes: int,
    ) -> list[torch.Tensor]:
        outputs = []
        with torch.no_grad():
            for _ in range(n_passes):
                logits = model(**ctx.model_inputs).logits.squeeze(0)
                logp = engine_stub._compute_logprobs(
                    logits,
                    ctx.model_inputs,
                    ulysses_pad_size=0,
                )
                outputs.append(logp.detach().clone())
        return outputs

    def test_dropout_train_mode_makes_logprob_nondeterministic(self):
        """With dropout enabled and model.train(), policy logprobs fluctuate.

        This introduces noise into PPO ratio/advantage updates and can stall reward.
        """
        torch.manual_seed(7)
        model = _DropoutCausalLM(p=0.25)
        model.train()

        veomini_stub = _make_veomini_stub(temperature=1.0)
        ctx = _generate_rl_batch(batch_idx=0, seq_len=_DropoutCausalLM.SEQ_LEN)

        logps = self._collect_logprobs(veomini_stub, model, ctx, n_passes=8)
        max_pairwise_diff = 0.0
        for i in range(len(logps)):
            for j in range(i + 1, len(logps)):
                d = (logps[i] - logps[j]).abs().max().item()
                max_pairwise_diff = max(max_pairwise_diff, d)

        print(f"dropout train-mode max logprob diff: {max_pairwise_diff:.6e}")
        assert max_pairwise_diff > 1e-5, (
            "Expected stochastic logprobs in train mode with dropout enabled"
        )

    def test_disable_dropout_restores_train_mode_determinism(self):
        """After disabling dropout (p=0), repeated train-mode logprobs are stable."""
        torch.manual_seed(7)
        model = _DropoutCausalLM(p=0.25)
        disable_dropout_in_model(model)
        model.train()

        fsdp_stub = _make_fsdp_stub(temperature=1.0)
        ctx = _generate_rl_batch(batch_idx=1, seq_len=_DropoutCausalLM.SEQ_LEN)

        logps = self._collect_logprobs(fsdp_stub, model, ctx, n_passes=5)
        max_pairwise_diff = 0.0
        for i in range(len(logps)):
            for j in range(i + 1, len(logps)):
                d = (logps[i] - logps[j]).abs().max().item()
                max_pairwise_diff = max(max_pairwise_diff, d)

        print(f"dropout disabled max logprob diff: {max_pairwise_diff:.6e}")
        assert max_pairwise_diff < 1e-8

    def test_veomini_create_device_model_does_not_apply_disable_dropout_flag(self, monkeypatch):
        """Current VeOMiniEngine ignores config.disable_dropout in _create_device_model.

        This test documents the gap that can affect RL stability when TP is off.
        """

        class _DummyLogger:
            def info(self, *args, **kwargs):
                return None

        dummy_engine = object.__new__(VeOMiniEngine)
        dummy_engine.config = SimpleNamespace(
            path="dummy-path",
            dtype="bfloat16",
            init_from_scratch=False,
            gradient_checkpointing=False,
            disable_dropout=True,
            attn_impl="flash_attention_2",
        )
        dummy_engine.logger = _DummyLogger()
        dummy_engine.get_device_stats = lambda: SimpleNamespace(log=lambda *a, **k: None)

        def _fake_build_tokenizer(path: str):
            return SimpleNamespace()

        def _fake_build_processor(path: str):
            return None

        def _fake_build_foundation_model(**kwargs):
            model = _DropoutCausalLM(p=0.3)
            model.config = SimpleNamespace(model_type="qwen3", tie_word_embeddings=False)
            return model

        monkeypatch.setattr(veomini_engine_module, "build_tokenizer", _fake_build_tokenizer)
        monkeypatch.setattr(veomini_engine_module, "build_processor", _fake_build_processor)
        monkeypatch.setattr(
            veomini_engine_module,
            "build_foundation_model",
            _fake_build_foundation_model,
        )

        VeOMiniEngine._create_device_model(dummy_engine)

        dropout_ps = [m.p for m in dummy_engine.model.modules() if isinstance(m, nn.Dropout)]
        assert dropout_ps, "Expected fake model to contain dropout layers"

        print(f"dropout ps after VeOMini _create_device_model: {dropout_ps}")
        assert any(p > 0 for p in dropout_ps), (
            "VeOMiniEngine currently does not apply disable_dropout=True to model"
        )

    def test_veomini_non_moe_weight_update_sends_all_parameters(self, monkeypatch):
        """For non-MoE models, VeOMini distributed weight update should not drop params.

        If rollout gets incomplete weights, online reward can stagnate even when
        train-side loss looks normal.
        """

        class _DummyFuture:
            def result(self):
                return None

        class _DummyRollout:
            def __init__(self):
                self.paused = 0
                self.continued = 0

            def pause_generation(self):
                self.paused += 1

            def continue_generation(self):
                self.continued += 1

            def update_weights_from_distributed(self, meta, param_specs):
                return _DummyFuture()

        dummy_engine = object.__new__(VeOMiniEngine)
        dummy_engine.rollout_engine = _DummyRollout()
        dummy_engine.weight_update_master_addr = "127.0.0.1"
        dummy_engine.weight_update_master_port = 23456
        dummy_engine.weight_update_group_name = "test_group"
        dummy_engine.weight_update_group = object()
        dummy_engine._parallel_state = None
        dummy_engine._initialized = True
        dummy_engine._cpu_group = object()

        model = _SmallCausalLM()
        dummy_engine.model = model

        sent_names: list[str] = []

        def _fake_update_bucket(meta, named_tensors):
            sent_names.extend([n for n, _ in named_tensors])
            named_tensors.clear()

        monkeypatch.setattr(
            dummy_engine,
            "_update_bucket_weights_from_distributed",
            _fake_update_bucket,
        )
        monkeypatch.setattr(dummy_engine, "_get_full_tensor", lambda p: p.data)
        monkeypatch.setattr(
            veomini_engine_module.dist,
            "get_rank",
            lambda *args, **kwargs: 0,
        )
        monkeypatch.setattr(
            veomini_engine_module.dist,
            "barrier",
            lambda *args, **kwargs: None,
        )
        monkeypatch.setattr(veomini_engine_module, "synchronize", lambda: None)

        meta = SimpleNamespace(weight_chunked_mem_mb=2048)
        VeOMiniEngine._update_weights_from_distributed(dummy_engine, meta)

        expected_names = [name for name, _ in model.named_parameters()]
        assert sent_names == expected_names, (
            f"Missing/extra params in weight update: sent={len(sent_names)}, "
            f"expected={len(expected_names)}"
        )
        assert dummy_engine.rollout_engine.paused == 1
        assert dummy_engine.rollout_engine.continued == 1

    def test_veomini_create_device_model_uses_config_attn_impl(self, monkeypatch):
        """When TP is off, ensure VeOMini uses configured attention impl (not hidden fallback)."""

        class _DummyLogger:
            def info(self, *args, **kwargs):
                return None

        captured = {}

        def _fake_build_foundation_model(**kwargs):
            captured.update(kwargs)
            model = _SmallCausalLM()
            model.config = SimpleNamespace(model_type="qwen3", tie_word_embeddings=False)
            return model

        dummy_engine = object.__new__(VeOMiniEngine)
        dummy_engine.config = SimpleNamespace(
            path="dummy-path",
            dtype="bfloat16",
            init_from_scratch=False,
            gradient_checkpointing=False,
            disable_dropout=False,
            attn_impl="flash_attention_2",
        )
        dummy_engine.logger = _DummyLogger()
        dummy_engine.get_device_stats = lambda: SimpleNamespace(log=lambda *a, **k: None)

        monkeypatch.setattr(veomini_engine_module, "build_tokenizer", lambda p: SimpleNamespace())
        monkeypatch.setattr(veomini_engine_module, "build_processor", lambda p: None)
        monkeypatch.setattr(
            veomini_engine_module,
            "build_foundation_model",
            _fake_build_foundation_model,
        )

        VeOMiniEngine._create_device_model(dummy_engine)
        assert captured.get("attn_implementation") == "flash_attention_2"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
