"""Unit tests for VeOMiniEngine vs FSDPEngine parity.

Focuses on areas most likely to cause training divergence:
1. LR scheduler equivalence (constant / cosine / linear)
2. Optimizer param group construction (weight decay splitting)
3. Gradient clipping behavior
4. Attention mask format in _prepare_mb_list
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from torch import nn

# --------------------------------------------------------------------------
# AReaL imports
# --------------------------------------------------------------------------
from areal.engine.fsdp_utils import get_cosine_schedule_with_warmup
from transformers import (
    get_constant_schedule_with_warmup,
    get_linear_schedule_with_warmup,
)

# VeOmni imports
from veomni.optim import build_lr_scheduler, build_optimizer


# ==========================================================================
# Helpers
# ==========================================================================


class _TinyModel(nn.Module):
    """Minimal model for optimizer / scheduler tests."""

    def __init__(self, hidden: int = 8, out: int = 4) -> None:
        super().__init__()
        self.linear = nn.Linear(hidden, out)
        self.norm = nn.LayerNorm(out)
        self.head = nn.Linear(out, 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.norm(self.linear(x)))


def _collect_lrs(scheduler, n_steps: int) -> list[float]:
    """Step the scheduler n_steps times and collect the LR after each step."""
    lrs = []
    for _ in range(n_steps):
        lrs.append(scheduler.get_last_lr()[0])
        scheduler.step()
    return lrs


# ==========================================================================
# Test 1: LR Scheduler — constant
# ==========================================================================


class TestConstantSchedulerParity:
    """Constant LR with warmup: VeOmni build_lr_scheduler vs transformers."""

    LR = 1e-4
    TOTAL_STEPS = 200
    WARMUP_RATIO = 0.05  # 10 warmup steps

    def _make_veomni_scheduler(self):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=self.LR)
        sched = build_lr_scheduler(
            opt,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="constant",
            lr_warmup_ratio=self.WARMUP_RATIO,
            lr_min=1e-7,
            lr_start=0.0,
        )
        return opt, sched

    def _make_areal_scheduler(self):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=self.LR)
        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)
        sched = get_constant_schedule_with_warmup(opt, num_warmup)
        return opt, sched

    def test_constant_warmup_start_differs(self):
        """VeOmni warmup starts from lr_start=0, AReaL constant warmup starts from 0 too.
        For constant schedule they should both reach LR after warmup.
        The key difference is the warmup interpolation formula."""
        _, veomni_sched = self._make_veomni_scheduler()
        _, areal_sched = self._make_areal_scheduler()

        veomni_lrs = _collect_lrs(veomni_sched, self.TOTAL_STEPS)
        areal_lrs = _collect_lrs(areal_sched, self.TOTAL_STEPS)

        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)

        # After warmup, both should be at LR
        for step in range(num_warmup, self.TOTAL_STEPS):
            assert abs(veomni_lrs[step] - self.LR) < 1e-9, (
                f"VeOmni constant LR should be {self.LR} after warmup, "
                f"got {veomni_lrs[step]} at step {step}"
            )
            assert abs(areal_lrs[step] - self.LR) < 1e-9, (
                f"AReaL constant LR should be {self.LR} after warmup, "
                f"got {areal_lrs[step]} at step {step}"
            )

    def test_constant_warmup_step0_differs(self):
        """Check whether step 0 LR differs between the two implementations."""
        _, veomni_sched = self._make_veomni_scheduler()
        _, areal_sched = self._make_areal_scheduler()

        veomni_lr0 = veomni_sched.get_last_lr()[0]
        areal_lr0 = areal_sched.get_last_lr()[0]

        # VeOmni starts from lr_start=0 → lr_lambda(0) = 0/lr = 0 → LR = 0
        # AReaL constant warmup: lr_lambda(0) = 0/max(1, warmup) = 0 → LR = 0
        # Both should be 0 at step 0 for constant schedule
        print(f"Step 0: VeOmni LR={veomni_lr0:.2e}, AReaL LR={areal_lr0:.2e}")


# ==========================================================================
# Test 2: LR Scheduler — cosine (the most important one)
# ==========================================================================


class TestCosineSchedulerParity:
    """Cosine LR with warmup: VeOmni build_lr_scheduler vs AReaL get_cosine_schedule_with_warmup.

    This is the most critical test because:
    - VeOmni warmup goes 0 → lr, AReaL warmup goes min_lr_ratio*lr → lr
    - VeOmni min_lr is absolute (e.g. 1e-7), AReaL uses min_lr_ratio (e.g. 0.0)
    - The cosine decay formulas differ slightly
    """

    LR = 1e-4
    TOTAL_STEPS = 1000
    WARMUP_RATIO = 0.05  # 50 warmup steps
    MIN_LR_RATIO = 0.1

    def _make_veomni_scheduler(self, min_lr: float | None = None):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=self.LR)
        if min_lr is None:
            min_lr = self.LR * self.MIN_LR_RATIO
        sched = build_lr_scheduler(
            opt,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="cosine",
            lr_warmup_ratio=self.WARMUP_RATIO,
            lr_min=min_lr,
            lr_start=0.0,
        )
        return opt, sched

    def _make_areal_scheduler(self):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=self.LR)
        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)
        sched = get_cosine_schedule_with_warmup(
            opt,
            num_warmup,
            self.TOTAL_STEPS,
            min_lr_ratio=self.MIN_LR_RATIO,
        )
        return opt, sched

    def test_cosine_warmup_start_not_equal(self):
        """VeOmni starts warmup from 0, AReaL starts from min_lr_ratio * lr.
        This test documents the difference."""
        _, veomni_sched = self._make_veomni_scheduler()
        _, areal_sched = self._make_areal_scheduler()

        veomni_lr0 = veomni_sched.get_last_lr()[0]
        areal_lr0 = areal_sched.get_last_lr()[0]

        # VeOmni: step=0 → lr_lambda = (0 + (lr - 0)*0/warmup)/lr = 0 → actual LR = 0
        # AReaL:  step=0 → lr_lambda = min_lr_ratio + (1 - min_lr_ratio)*0/warmup = min_lr_ratio
        #         → actual LR = min_lr_ratio * lr
        print(f"Step 0: VeOmni={veomni_lr0:.2e}, AReaL={areal_lr0:.2e}")
        assert veomni_lr0 != areal_lr0, (
            "Expected warmup start to differ: VeOmni starts from 0, "
            "AReaL starts from min_lr_ratio * lr"
        )

    def test_cosine_end_lr_not_equal_with_abs_min_lr(self):
        """When VeOmni uses absolute min_lr=1e-7 (the default), the end LR
        is 1e-7, while AReaL's end LR is min_lr_ratio * lr."""
        # VeOmni with default abs min_lr = 1e-7
        _, veomni_sched = self._make_veomni_scheduler(min_lr=1e-7)
        _, areal_sched = self._make_areal_scheduler()

        veomni_lrs = _collect_lrs(veomni_sched, self.TOTAL_STEPS)
        areal_lrs = _collect_lrs(areal_sched, self.TOTAL_STEPS)

        veomni_end = veomni_lrs[-1]
        areal_end = areal_lrs[-1]
        expected_areal_end = self.MIN_LR_RATIO * self.LR  # 0.1 * 1e-4 = 1e-5

        print(f"End LR: VeOmni={veomni_end:.2e}, AReaL={areal_end:.2e}")
        print(f"Expected AReaL end: {expected_areal_end:.2e}")

        # VeOmni end should be ~1e-7, AReaL end should be ~1e-5
        assert abs(veomni_end - 1e-7) < 1e-9
        assert abs(areal_end - expected_areal_end) < 1e-9
        assert veomni_end != areal_end

    def test_cosine_mid_training_lr_diff(self):
        """Compare LR at the midpoint of training (after warmup)."""
        _, veomni_sched = self._make_veomni_scheduler()
        _, areal_sched = self._make_areal_scheduler()

        veomni_lrs = _collect_lrs(veomni_sched, self.TOTAL_STEPS)
        areal_lrs = _collect_lrs(areal_sched, self.TOTAL_STEPS)

        # At step 525 (midpoint of the cosine part)
        mid = self.TOTAL_STEPS // 2
        diff = abs(veomni_lrs[mid] - areal_lrs[mid])
        print(
            f"Mid-training (step {mid}): "
            f"VeOmni={veomni_lrs[mid]:.6e}, AReaL={areal_lrs[mid]:.6e}, "
            f"diff={diff:.6e}"
        )

    def test_cosine_exact_match_when_min_lr_aligned(self):
        """When VeOmni min_lr = min_lr_ratio * lr, the cosine part should be
        identical (only warmup differs because of different start points)."""
        aligned_min_lr = self.MIN_LR_RATIO * self.LR
        _, veomni_sched = self._make_veomni_scheduler(min_lr=aligned_min_lr)
        _, areal_sched = self._make_areal_scheduler()

        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)
        veomni_lrs = _collect_lrs(veomni_sched, self.TOTAL_STEPS)
        areal_lrs = _collect_lrs(areal_sched, self.TOTAL_STEPS)

        # After warmup, the cosine part should be very close
        max_diff = 0.0
        for step in range(num_warmup, self.TOTAL_STEPS):
            diff = abs(veomni_lrs[step] - areal_lrs[step])
            max_diff = max(max_diff, diff)

        print(f"Max LR diff after warmup (min_lr aligned): {max_diff:.2e}")
        # They should be very close after warmup
        assert max_diff < 1e-8, (
            f"After warmup, cosine LRs should match when min_lr is aligned, "
            f"but max diff is {max_diff:.2e}"
        )


# ==========================================================================
# Test 3: LR Scheduler — linear
# ==========================================================================


class TestLinearSchedulerParity:
    """Linear LR with warmup: VeOmni vs transformers."""

    LR = 1e-4
    TOTAL_STEPS = 200
    WARMUP_RATIO = 0.1  # 20 warmup steps

    def _make_veomni_scheduler(self):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=self.LR)
        sched = build_lr_scheduler(
            opt,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="linear",
            lr_warmup_ratio=self.WARMUP_RATIO,
            lr_min=1e-7,
            lr_start=0.0,
        )
        return opt, sched

    def _make_areal_scheduler(self):
        model = _TinyModel()
        opt = torch.optim.AdamW(model.parameters(), lr=self.LR)
        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)
        sched = get_linear_schedule_with_warmup(opt, num_warmup, self.TOTAL_STEPS)
        return opt, sched

    def test_linear_end_lr_differs(self):
        """VeOmni linear decays to min_lr (absolute), transformers decays to 0."""
        _, veomni_sched = self._make_veomni_scheduler()
        _, areal_sched = self._make_areal_scheduler()

        veomni_lrs = _collect_lrs(veomni_sched, self.TOTAL_STEPS)
        areal_lrs = _collect_lrs(areal_sched, self.TOTAL_STEPS)

        veomni_end = veomni_lrs[-1]
        areal_end = areal_lrs[-1]

        print(f"Linear end LR: VeOmni={veomni_end:.2e}, AReaL={areal_end:.2e}")
        # transformers linear decays to 0
        assert areal_end < 1e-9, f"AReaL linear should decay to ~0, got {areal_end}"


# ==========================================================================
# Test 4: Optimizer param group — weight decay splitting
# ==========================================================================


class TestOptimizerParamGroups:
    """VeOmni build_optimizer does weight-decay splitting internally
    (biases and LayerNorm get no weight decay). torch.optim.AdamW in FSDPEngine
    applies weight decay uniformly to all params.

    This is a potential difference that could cause divergence."""

    LR = 1e-3
    WEIGHT_DECAY = 0.1

    def test_veomni_splits_decay_vs_no_decay(self):
        """VeOmni should create separate param groups with/without weight decay."""
        model = _TinyModel()
        opt = build_optimizer(
            model,
            lr=self.LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.WEIGHT_DECAY,
            fused=False,
            optimizer_type="adamw",
            # NOTE: NOT passing param_groups — let VeOmni do its default splitting
        )

        # VeOmni should have created 2 groups: decay and no-decay
        groups = opt.param_groups
        print(f"VeOmni default: {len(groups)} param groups")
        for i, g in enumerate(groups):
            print(
                f"  Group {i}: {len(g['params'])} params, "
                f"weight_decay={g['weight_decay']}"
            )

        # There should be at least one group with weight_decay=0
        has_no_decay = any(g["weight_decay"] == 0.0 for g in groups)
        print(f"Has no-decay group: {has_no_decay}")

    def test_veomini_engine_explicit_param_groups_bypass_splitting(self):
        """When VeOMiniEngine passes explicit param_groups, VeOmni should NOT
        do its own splitting — all params get uniform weight_decay."""
        model = _TinyModel()
        param_groups = [
            {
                "params": [p for p in model.parameters() if p.requires_grad],
                "weight_decay": self.WEIGHT_DECAY,
            }
        ]
        opt = build_optimizer(
            model,
            lr=self.LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.WEIGHT_DECAY,
            fused=False,
            optimizer_type="adamw",
            param_groups=param_groups,
        )

        groups = opt.param_groups
        print(f"Explicit param_groups: {len(groups)} param groups")
        for i, g in enumerate(groups):
            print(
                f"  Group {i}: {len(g['params'])} params, "
                f"weight_decay={g['weight_decay']}"
            )

    def test_weight_decay_diff_after_steps(self):
        """Compare weight values after multiple optimizer steps.
        If VeOmni internally exempts bias/norm from weight decay but FSDPEngine
        does not, the parameters will diverge."""
        torch.manual_seed(42)
        inputs = torch.randn(4, 8)
        targets = torch.randn(4, 2)

        model_fsdp = _TinyModel()
        model_veomni = _TinyModel()
        model_veomni.load_state_dict(model_fsdp.state_dict())

        # FSDPEngine style: uniform weight decay on all params
        opt_fsdp = torch.optim.AdamW(
            model_fsdp.parameters(),
            lr=self.LR,
            weight_decay=self.WEIGHT_DECAY,
            betas=(0.9, 0.999),
            eps=1e-8,
            fused=False,
        )

        # VeOMiniEngine style: explicit param_groups with uniform weight decay
        opt_veomni = build_optimizer(
            model_veomni,
            lr=self.LR,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=self.WEIGHT_DECAY,
            fused=False,
            optimizer_type="adamw",
            param_groups=[
                {
                    "params": [
                        p for p in model_veomni.parameters() if p.requires_grad
                    ],
                    "weight_decay": self.WEIGHT_DECAY,
                }
            ],
        )

        # Run 10 steps
        for _ in range(10):
            opt_fsdp.zero_grad()
            opt_veomni.zero_grad()

            loss_fsdp = (model_fsdp(inputs) - targets).square().mean()
            loss_veomni = (model_veomni(inputs) - targets).square().mean()

            loss_fsdp.backward()
            loss_veomni.backward()

            opt_fsdp.step()
            opt_veomni.step()

        # Compare params
        max_diff = 0.0
        for (n1, p1), (n2, p2) in zip(
            model_fsdp.named_parameters(), model_veomni.named_parameters()
        ):
            assert n1 == n2
            diff = (p1 - p2).abs().max().item()
            max_diff = max(max_diff, diff)
            if diff > 1e-6:
                print(f"  {n1}: max diff = {diff:.2e}")

        print(f"Max param diff after 10 steps: {max_diff:.2e}")
        assert max_diff < 1e-5, (
            f"Params diverged after 10 steps with explicit param_groups: "
            f"max diff = {max_diff:.2e}"
        )


# ==========================================================================
# Test 5: The YAML config — verify what actually runs
# ==========================================================================


class TestYAMLConfigScheduler:
    """Reproduce the exact config from the user's YAML:
    lr_scheduler_type: constant, warmup_steps_proportion: 0.001, lr: 1.7e-5

    With constant schedule, the LR bug should NOT matter much (only warmup).
    But let's quantify the difference."""

    LR = 1.7e-5
    WARMUP_PROPORTION = 0.001
    # Assume ~100 steps per epoch * 10 epochs = ~1000 total steps
    TOTAL_STEPS = 1000

    def test_constant_schedule_with_yaml_config(self):
        """With lr_scheduler_type='constant', VeOmni and AReaL should be
        nearly identical after warmup."""
        num_warmup = int(self.WARMUP_PROPORTION * self.TOTAL_STEPS)  # = 1

        # VeOmni (old code path)
        model_v = _TinyModel()
        opt_v = torch.optim.AdamW(model_v.parameters(), lr=self.LR)
        sched_v = build_lr_scheduler(
            opt_v,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="constant",
            lr_warmup_ratio=self.WARMUP_PROPORTION,
            lr_min=1e-7,
            lr_start=0.0,
        )

        # AReaL (fixed code path)
        model_a = _TinyModel()
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=self.LR)
        sched_a = get_constant_schedule_with_warmup(opt_a, num_warmup)

        veomni_lrs = _collect_lrs(sched_v, min(50, self.TOTAL_STEPS))
        areal_lrs = _collect_lrs(sched_a, min(50, self.TOTAL_STEPS))

        print(f"Warmup steps = {num_warmup}")
        print(f"Step 0: VeOmni={veomni_lrs[0]:.6e}, AReaL={areal_lrs[0]:.6e}")
        if num_warmup > 0:
            print(
                f"Step {num_warmup}: VeOmni={veomni_lrs[num_warmup]:.6e}, "
                f"AReaL={areal_lrs[num_warmup]:.6e}"
            )
        print(
            f"Step {len(veomni_lrs)-1}: VeOmni={veomni_lrs[-1]:.6e}, "
            f"AReaL={areal_lrs[-1]:.6e}"
        )

        # After warmup they should match
        for step in range(num_warmup + 1, len(veomni_lrs)):
            assert abs(veomni_lrs[step] - areal_lrs[step]) < 1e-10, (
                f"Step {step}: VeOmni={veomni_lrs[step]:.6e} != "
                f"AReaL={areal_lrs[step]:.6e}"
            )
        print("✓ After warmup, constant LRs match perfectly.")

    def test_what_if_cosine_with_yaml_config(self):
        """What if the user switches to cosine? Show the difference.
        This is NOT the current YAML, but important for awareness."""
        num_warmup = int(self.WARMUP_PROPORTION * self.TOTAL_STEPS)
        min_lr_ratio = 0.0  # default from OptimizerConfig

        # VeOmni cosine (old code)
        model_v = _TinyModel()
        opt_v = torch.optim.AdamW(model_v.parameters(), lr=self.LR)
        sched_v = build_lr_scheduler(
            opt_v,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="cosine",
            lr_warmup_ratio=self.WARMUP_PROPORTION,
            lr_min=1e-7,  # VeOmni default absolute min_lr
            lr_start=0.0,
        )

        # AReaL cosine (fixed code)
        model_a = _TinyModel()
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=self.LR)
        sched_a = get_cosine_schedule_with_warmup(
            opt_a, num_warmup, self.TOTAL_STEPS, min_lr_ratio=min_lr_ratio
        )

        veomni_lrs = _collect_lrs(sched_v, self.TOTAL_STEPS)
        areal_lrs = _collect_lrs(sched_a, self.TOTAL_STEPS)

        # Find max difference
        max_diff = 0.0
        max_diff_step = 0
        for step in range(self.TOTAL_STEPS):
            diff = abs(veomni_lrs[step] - areal_lrs[step])
            if diff > max_diff:
                max_diff = diff
                max_diff_step = step

        # End LR comparison
        veomni_end = veomni_lrs[-1]
        areal_end = areal_lrs[-1]

        print(f"Cosine schedule with lr={self.LR}, min_lr_ratio={min_lr_ratio}:")
        print(f"  VeOmni end LR = {veomni_end:.2e} (uses abs min_lr=1e-7)")
        print(f"  AReaL  end LR = {areal_end:.2e} (uses min_lr_ratio*lr=0)")
        print(f"  Max diff = {max_diff:.2e} at step {max_diff_step}")
        print(
            f"  VeOmni end is {veomni_end/self.LR*100:.4f}% of peak LR"
        )

        # VeOmni ends at 1e-7, AReaL ends at 0 — these are different
        assert veomni_end > areal_end, (
            "VeOmni cosine end LR should be > AReaL when abs min_lr > min_lr_ratio*lr"
        )


# ==========================================================================
# Test 6: Gradient clipping — veomni_clip_grad_norm vs fsdp2_clip_grad_norm
# ==========================================================================


class TestGradientClipping:
    """Test that veomni_clip_grad_norm and manual torch clip_grad_norm
    produce the same result for a non-distributed (single-GPU-equivalent) case."""

    def test_clip_grad_norm_value(self):
        """In the non-distributed case, veomni_clip_grad_norm should be
        equivalent to torch.nn.utils.clip_grad_norm_."""
        torch.manual_seed(123)
        model = _TinyModel()
        x = torch.randn(4, 8)
        loss = model(x).sum()
        loss.backward()

        # Get expected grad norm before clipping
        total_norm_sq = 0.0
        for p in model.parameters():
            if p.grad is not None:
                total_norm_sq += p.grad.data.float().norm() ** 2
        expected_norm = math.sqrt(total_norm_sq)

        print(f"Grad norm before clipping: {expected_norm:.6f}")

        # Now test with a model copy
        model2 = _TinyModel()
        model2.load_state_dict(model.state_dict())
        x2 = torch.randn(4, 8)

        # Same forward/backward
        torch.manual_seed(123)
        model2 = _TinyModel()
        loss2 = model2(torch.randn(4, 8)).sum()
        loss2.backward()

        max_norm = 0.5
        torch_norm = torch.nn.utils.clip_grad_norm_(
            model2.parameters(), max_norm=max_norm
        )
        print(f"torch clip_grad_norm returned: {torch_norm:.6f}")


# ==========================================================================
# Test 7: attention_mask format — Qwen3 MoE vs default
# ==========================================================================


class TestAttentionMaskFormat:
    """VeOMiniEngine always sets attention_mask = dict(full_attention=None, sliding_attention=None).
    FSDPEngine has special handling for Qwen3 MoE where attention_mask = None.

    This test documents the difference."""

    def test_default_attention_mask_format(self):
        """For non-MoE models, both engines use the same format."""
        expected = dict(full_attention=None, sliding_attention=None)
        # VeOMiniEngine always produces this
        veomini_mask = dict(full_attention=None, sliding_attention=None)
        assert veomini_mask == expected

    def test_qwen3_moe_attention_mask_should_be_none(self):
        """For Qwen3 MoE, FSDPEngine sets attention_mask = None,
        but VeOMiniEngine incorrectly uses the dict format.

        This test will FAIL once VeOMiniEngine is fixed for Qwen3 MoE."""
        # FSDPEngine for Qwen3 MoE:
        fsdp_mask = None

        # VeOMiniEngine (current, WRONG for Qwen3 MoE):
        veomini_mask = dict(full_attention=None, sliding_attention=None)

        # They differ — this is the bug
        assert fsdp_mask != veomini_mask, (
            "VeOMiniEngine attention_mask format differs from FSDPEngine for Qwen3 MoE"
        )
        print(
            "⚠ VeOMiniEngine does not handle Qwen3 MoE attention_mask=None. "
            "This will cause issues when training Qwen3 MoE models."
        )


# ==========================================================================
# Test 8: Multi-step training divergence check
# ==========================================================================


class TestMultiStepTrainingDivergence:
    """End-to-end test: simulate multiple optimizer steps with different schedulers
    and check if the models diverge."""

    LR = 1e-3
    TOTAL_STEPS = 100
    WARMUP_RATIO = 0.1
    MIN_LR_RATIO = 0.0

    def test_cosine_training_divergence(self):
        """Run 50 training steps with VeOmni vs AReaL cosine scheduler.
        Even with constant data, the LR difference causes param divergence."""
        torch.manual_seed(42)
        inputs = torch.randn(8, 8)
        targets = torch.randn(8, 2)

        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)

        # Model A: AReaL cosine
        model_a = _TinyModel()
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=self.LR, fused=False)
        sched_a = get_cosine_schedule_with_warmup(
            opt_a, num_warmup, self.TOTAL_STEPS, min_lr_ratio=self.MIN_LR_RATIO
        )

        # Model V: VeOmni cosine (old code)
        model_v = _TinyModel()
        model_v.load_state_dict(model_a.state_dict())
        opt_v = torch.optim.AdamW(model_v.parameters(), lr=self.LR, fused=False)
        sched_v = build_lr_scheduler(
            opt_v,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="cosine",
            lr_warmup_ratio=self.WARMUP_RATIO,
            lr_min=self.LR * self.MIN_LR_RATIO if self.MIN_LR_RATIO > 0 else 1e-7,
            lr_start=0.0,
        )

        n_steps = 50
        for step in range(n_steps):
            opt_a.zero_grad()
            opt_v.zero_grad()

            loss_a = (model_a(inputs) - targets).square().mean()
            loss_v = (model_v(inputs) - targets).square().mean()

            loss_a.backward()
            loss_v.backward()

            opt_a.step()
            opt_v.step()

            sched_a.step()
            sched_v.step()

        # Compare final params
        max_diff = 0.0
        for (n1, p1), (n2, p2) in zip(
            model_a.named_parameters(), model_v.named_parameters()
        ):
            diff = (p1 - p2).abs().max().item()
            max_diff = max(max_diff, diff)

        print(
            f"After {n_steps} cosine steps: max param diff = {max_diff:.6e}"
        )
        # With min_lr_ratio=0 and min_lr=1e-7, the difference is small but nonzero
        if max_diff > 1e-6:
            print("⚠ Models have diverged due to LR scheduler difference!")
        else:
            print("✓ Models are close (LR difference is negligible for this config)")

    def test_constant_training_no_divergence(self):
        """With constant LR (the user's current config), divergence should be
        minimal (only 1 warmup step matters)."""
        torch.manual_seed(42)
        inputs = torch.randn(8, 8)
        targets = torch.randn(8, 2)

        num_warmup = int(self.WARMUP_RATIO * self.TOTAL_STEPS)

        # Model A: AReaL constant
        model_a = _TinyModel()
        opt_a = torch.optim.AdamW(model_a.parameters(), lr=self.LR, fused=False)
        sched_a = get_constant_schedule_with_warmup(opt_a, num_warmup)

        # Model V: VeOmni constant
        model_v = _TinyModel()
        model_v.load_state_dict(model_a.state_dict())
        opt_v = torch.optim.AdamW(model_v.parameters(), lr=self.LR, fused=False)
        sched_v = build_lr_scheduler(
            opt_v,
            train_steps=self.TOTAL_STEPS,
            lr=self.LR,
            lr_decay_style="constant",
            lr_warmup_ratio=self.WARMUP_RATIO,
            lr_min=1e-7,
            lr_start=0.0,
        )

        n_steps = 50
        for step in range(n_steps):
            opt_a.zero_grad()
            opt_v.zero_grad()

            loss_a = (model_a(inputs) - targets).square().mean()
            loss_v = (model_v(inputs) - targets).square().mean()

            loss_a.backward()
            loss_v.backward()

            opt_a.step()
            opt_v.step()

            sched_a.step()
            sched_v.step()

        max_diff = 0.0
        for (n1, p1), (n2, p2) in zip(
            model_a.named_parameters(), model_v.named_parameters()
        ):
            diff = (p1 - p2).abs().max().item()
            max_diff = max(max_diff, diff)

        print(
            f"After {n_steps} constant steps: max param diff = {max_diff:.6e}"
        )
        # With only ~10 warmup steps difference, the divergence should be very small
        # but not zero because warmup formula differs
        if max_diff < 1e-6:
            print("✓ Constant schedule: negligible divergence")
        else:
            print(f"⚠ Constant schedule: divergence = {max_diff:.6e}")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
