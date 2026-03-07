"""VeOMini Engine: Bridges VeOmni's model/distributed infrastructure with AReaL's TrainEngine API.

This engine uses VeOmni's model building, parallelization (FSDP1/FSDP2), optimizer,
lr scheduler, and gradient clipping, while exposing AReaL's TrainEngine interface
for RL training (PPO actor, critic, etc.).
"""

from __future__ import annotations

import dataclasses
import gc
import math
import os
import re
import time
from collections.abc import Callable, Iterator
from concurrent.futures import Future
from datetime import datetime
from typing import TYPE_CHECKING, Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions,
    get_model_state_dict,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor
from torchdata.stateful_dataloader import StatefulDataLoader
from transformers import (
    AutoConfig,
    PretrainedConfig,
    PreTrainedTokenizerFast,
    ProcessorMixin,
)

from areal.api.alloc_mode import FSDPParallelStrategy, ParallelStrategy
from areal.api.cli_args import PerfTracerConfig, TrainEngineConfig
from areal.api.engine_api import InferenceEngine, TrainEngine
from areal.api.io_struct import (
    DeviceRuntimeInfo,
    FinetuneSpec,
    ParamSpec,
    SaveLoadMeta,
    WeightUpdateMeta,
)
from areal.api.workflow_api import WorkflowLike
from areal.engine.core import (
    aggregate_eval_losses,
    compute_total_loss_weight,
    reorder_and_pad_outputs,
)
from areal.engine.core.distributed import (
    init_custom_process_group,
    patch_dist_group_timeout,
)
from areal.engine.fsdp_engine import FSDPTrainContext
from areal.engine.fsdp_utils import (
    fsdp2_load_full_state_dict,
)
from areal.engine.fsdp_utils.checkpoint import DCPState
from areal.infra.dist_rollout import DistRolloutCoordinator
from areal.infra.platforms import current_platform
from areal.models.fsdp.ulysses import (
    set_ulysses_sequence_parallel_group,
    ulysses_pad,
    ulysses_pad_and_slice_inputs,
    ulysses_prepare_inputs,
)
from areal.utils import (
    logging,
    name_resolve,
    names,
    perf_tracer,
    stats_tracker,
)
from areal.utils.constants import DIST_GROUP_DEFAULT_TIMEOUT
from areal.utils.data import (
    MicroBatchItem,
    MicroBatchList,
    amend_position_ids,
    pack_tensor_dict,
    pad_mb_list,
    split_padded_tensor_dict_into_mb_list,
    unsqueeze_mb_list,
)
from areal.utils.functional import gather_logprobs, gather_logprobs_entropy
from areal.utils.network import find_free_ports, gethostip
from areal.utils.perf_tracer import trace_perf, trace_scope
from areal.utils.save_load import get_state_dict_from_repo_id_or_path

# VeOmni imports
from veomni.distributed.clip_grad_norm import veomni_clip_grad_norm
from veomni.distributed.parallel_state import (
    ParallelState,
    get_parallel_state,
    init_parallel_state,
)
from veomni.distributed.torch_parallelize import build_parallelize_model
from veomni.models import (
    build_foundation_model,
    build_tokenizer,
    save_model_weights,
)
from veomni.models.auto import build_processor
from veomni.optim import build_lr_scheduler, build_optimizer
from veomni.utils.device import get_device_type, get_dist_comm_backend, get_torch_device, synchronize
from veomni.utils.helper import empty_cache

if TYPE_CHECKING:
    from areal.api.cli_args import PPOActorConfig, PPOCriticConfig
    from areal.api.scheduler_api import Scheduler

# ---------------------------------------------------------------------------
# MoE weight format conversion patterns
#
# VeOmni merges per-expert 2D weights into 3D tensors:
#   model.layers.{i}.mlp.experts.gate_proj  shape [num_experts, inter, hidden]
#   model.layers.{i}.mlp.experts.up_proj    shape [num_experts, inter, hidden]
#   model.layers.{i}.mlp.experts.down_proj  shape [num_experts, hidden, inter]
#
# HuggingFace / sglang uses per-expert 2D weights:
#   model.layers.{i}.mlp.experts.{j}.gate_proj.weight  shape [inter, hidden]
#   model.layers.{i}.mlp.experts.{j}.up_proj.weight    shape [inter, hidden]
#   model.layers.{i}.mlp.experts.{j}.down_proj.weight  shape [hidden, inter]
# ---------------------------------------------------------------------------
_MERGED_EXPERT_RE = re.compile(
    r"^(.*\.mlp\.experts)\.(gate_proj|up_proj|down_proj)$"
)
_HF_EXPERT_RE = re.compile(
    r"^(.*\.mlp\.experts)\.(\d+)\.(gate_proj|up_proj|down_proj)\.weight$"
)


class VeOMiniEngine(TrainEngine):
    """TrainEngine implementation backed by VeOmni's model/parallel/optim infrastructure.

    This engine replaces FSDPEngine's model creation and parallelization with VeOmni's
    equivalents (``build_foundation_model``, ``build_parallelize_model``, ``build_optimizer``,
    ``build_lr_scheduler``, ``veomni_clip_grad_norm``), while keeping AReaL's RL-specific
    logic (micro-batch preparation, logprobs computation, weight update coordination, etc.).
    """

    def __init__(self, config: TrainEngineConfig):
        self.config = config
        self.optimizer_config = config.optimizer

        self.model: torch.nn.Module
        self.optimizer: torch.optim.Optimizer
        self.tokenizer: PreTrainedTokenizerFast
        self.processor: ProcessorMixin | None = None
        self.model_config: PretrainedConfig
        self._version: int = 0

        self._initialized = False
        self.own_global_group = False
        self._cpu_group: dist.ProcessGroup
        self.weight_update_group_initialized = False
        self.weight_update_group_name: str
        self.weight_update_master_addr: str
        self.weight_update_master_port: int

        self.model_config = AutoConfig.from_pretrained(
            pretrained_model_name_or_path=self.config.path,
            trust_remote_code=True,
        )

        self.rollout_engine: InferenceEngine | None = None
        self.rollout_coordinator: DistRolloutCoordinator | None = None

        self.world_size: int
        self.rank: int
        self.dp_head: int
        self.dp_rank: int

        self.is_offload: bool = False

        # VeOmni parallel state (set during create_process_group)
        self._parallel_state: ParallelState | None = None

    # ------------------------------------------------------------------
    # Process group & distributed initialization
    # ------------------------------------------------------------------

    def create_process_group(self, parallel_strategy: ParallelStrategy | None = None):
        patch_dist_group_timeout(DIST_GROUP_DEFAULT_TIMEOUT)

        backend = get_dist_comm_backend()
        if not dist.is_initialized():
            dist.init_process_group(
                backend=backend,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )
            self.own_global_group = True

        self._cpu_group = dist.new_group(
            timeout=DIST_GROUP_DEFAULT_TIMEOUT, backend="gloo"
        )

        if parallel_strategy is None:
            parallel_strategy = ParallelStrategy()

        self.logger = logging.getLogger(f"[VeOMiniEngine Rank {dist.get_rank()}]")

        # Derive ulysses / tp / ep sizes from the parallel strategy
        fsdp_strategy = self._make_parallel_strategy(parallel_strategy)
        sp_size = getattr(fsdp_strategy, "sp_size", 1)
        tp_size = getattr(fsdp_strategy, "tp_size", 1)
        ep_size = getattr(fsdp_strategy, "ep_size", 1)

        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # Use VeOmni's init_parallel_state to set up device mesh.
        # NOTE: In VeOmni, EP is orthogonal to DP — it creates a separate
        # ``ep_fsdp_device_mesh`` of shape ``(ep_size, world_size // ep_size)``
        # for expert modules, while the primary device mesh (for regular
        # params) uses DP × TP × SP dimensions as usual.
        dp_size = self.world_size // (sp_size * tp_size)
        init_parallel_state(
            dp_size=dp_size,
            dp_replicate_size=1,
            dp_shard_size=dp_size,
            tp_size=tp_size,
            ep_size=ep_size,
            pp_size=1,
            cp_size=1,
            ulysses_size=sp_size,
            dp_mode="fsdp2",
        )
        self._parallel_state = get_parallel_state()

        # Model parallel.
        mesh = self._parallel_state.device_mesh
        if sp_size > 1 and tp_size > 1:
            assert mesh is not None
            self._mp_group = mesh["ulysses", "tp"]._flatten(mesh_dim_name="ulysses_tp").get_group()
        elif sp_size > 1:
            assert mesh is not None
            self._mp_group = mesh["ulysses"].get_group()
        elif tp_size > 1:
            assert mesh is not None
            self._mp_group = mesh["tp"].get_group()
        else:
            self._mp_group = None
            for r in range(self.world_size):
                g = dist.new_group(
                    ranks=[r],
                    timeout=DIST_GROUP_DEFAULT_TIMEOUT,
                    backend=get_dist_comm_backend(),
                )
                if r == self.rank:
                    self._mp_group = g
            assert self._mp_group is not None

        # Data parallel.
        self._dp_group = self._parallel_state.dp_group
        self._sp_group = (
            self._parallel_state.sp_group if self._parallel_state.sp_enabled else None
        )
        mp_ranks = dist.get_process_group_ranks(self._mp_group)
        self.dp_head = mp_ranks[0]
        self.dp_rank = self._parallel_state.dp_rank

        # Logging.
        self.logger.info(
            f"VeOmni parallel state initialized: "
            f"dp_size={dp_size}, sp_size={sp_size}, tp_size={tp_size}, ep_size={ep_size}"
        )
        self.logger.info(
            f"Group layout: dp_ranks={dist.get_process_group_ranks(self._dp_group)}, "
            f"mp_ranks={mp_ranks}, rank={self.rank}"
        )
        if ep_size > 1:
            self.logger.info(
                f"Expert Parallelism enabled: ep_size={ep_size}, "
                f"ep_fsdp_size={self._parallel_state.ep_fsdp_size}, "
                f"ep_rank={self._parallel_state.ep_rank}"
            )
        self.logger.info(f"Data parallel head {self.dp_head} and rank {self.dp_rank}")

    def initialize(self, addr: str | None, ft_spec: FinetuneSpec, *args, **kwargs):
        assert addr is None, "VeOMiniEngine does not support remote initialization."
        assert ft_spec is not None, "VeOMiniEngine requires FinetuneSpec to initialize."

        self.weight_update_group_name = "update_weight_group"

        # Set device
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_type = get_device_type()
        device_str = f"{device_type}:{local_rank}"
        get_torch_device().set_device(device_str)
        self.device = torch.device(device_str)

        # Build model using VeOmni
        self._create_device_model()

        # Parallelize model using VeOmni
        self._parallelize_model()

        # Create optimizer
        self._create_optimizer(ft_spec)
        self._initialized = True

    # ------------------------------------------------------------------
    # TrainEngine property implementations
    # ------------------------------------------------------------------

    @property
    def data_parallel_group(self) -> dist.ProcessGroup:
        return self._dp_group

    @property
    def data_parallel_rank(self) -> int:
        return self.dp_rank

    @property
    def data_parallel_world_size(self) -> int:
        return self._parallel_state.dp_size if self._parallel_state else 1

    @property
    def context_and_model_parallel_group(self) -> dist.ProcessGroup:
        return self._mp_group

    @property
    def cpu_group(self) -> dist.ProcessGroup:
        assert self._initialized
        return self._cpu_group

    def destroy(self):
        self._initialized = False
        if hasattr(self, "optimizer"):
            del self.optimizer
        if hasattr(self, "model"):
            del self.model
        gc.collect()
        empty_cache()
        gc.collect()
        if dist.is_initialized() and self.own_global_group:
            dist.destroy_process_group()

    @property
    def initialized(self) -> bool:
        return self._initialized

    def current_data_parallel_head(self) -> int:
        return self.dp_head

    def is_data_parallel_head(self) -> bool:
        return self.rank == self.dp_head

    def train(self, mode: bool = True):
        assert self.model is not None
        self.model.train(mode=mode)
        return self

    # ------------------------------------------------------------------
    # Rollout / inference engine connection
    # ------------------------------------------------------------------

    def connect_engine(self, engine: InferenceEngine, meta: WeightUpdateMeta):
        if self.rollout_engine is not None and self.rollout_engine != engine:
            self.logger.warning(
                f"Connected rollout engine changed from {self.rollout_engine} to {engine}."
            )
        self.rollout_engine = engine
        self.rollout_coordinator = DistRolloutCoordinator(
            rollout_engine=engine, train_engine=self
        )

        if meta.type == "xccl" and not self.weight_update_group_initialized:
            self._init_weight_update_from_distributed(meta)
            self.weight_update_group_initialized = True

        synchronize()
        dist.barrier(group=self.cpu_group)

    def rollout_batch(
        self,
        data: list[dict[str, Any]],
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        group_size: int = 1,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.rollout_batch(
            data,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            group_size=group_size,
        )

    def prepare_batch(
        self,
        dataloader: StatefulDataLoader,
        workflow: WorkflowLike,
        workflow_kwargs: dict[str, Any] | None = None,
        should_accept_fn: Callable[[dict[str, Any]], bool] | str | None = None,
        group_size: int = 1,
        dynamic_bs: bool = False,
    ) -> dict[str, Any]:
        self._check_rollout_engine_connected()
        return self.rollout_coordinator.prepare_batch(
            dataloader,
            workflow=workflow,
            workflow_kwargs=workflow_kwargs,
            should_accept_fn=should_accept_fn,
            group_size=group_size,
            dynamic_bs=dynamic_bs,
        )

    # ------------------------------------------------------------------
    # Weight management
    # ------------------------------------------------------------------

    def update_weights(self, meta: WeightUpdateMeta):
        self._check_rollout_engine_connected()
        if meta.type == "xccl":
            assert self.weight_update_group_initialized
            self._update_weights_from_distributed(meta)
        elif meta.type == "disk":
            self._update_weights_from_disk(meta)
        else:
            raise ValueError(f"Unknown weight update type {meta.type}")

    def set_version(self, version: int):
        self._version = version

    def get_version(self) -> int:
        return self._version

    def save(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._save_model_to_hf(meta.path, meta.tokenizer, meta.processor)
        elif meta.weight_format == "dcp":
            self._save_to_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        if meta.with_optim and meta.weight_format == "hf":
            self._save_optimizer_state(meta.path)

    def load(self, meta: SaveLoadMeta):
        if meta.weight_format == "hf":
            self._load_model_from_hf(meta.path)
        elif meta.weight_format == "dcp":
            self._load_from_dcp(meta.path, meta.with_optim)
        else:
            raise ValueError(f"Unknown weight format {meta.weight_format}.")

        if meta.with_optim and meta.weight_format == "hf":
            self._load_optimizer_state(meta.path)

    # ------------------------------------------------------------------
    # Optimizer / LR scheduler
    # ------------------------------------------------------------------

    def optimizer_zero_grad(self):
        assert self.optimizer is not None
        self.optimizer.zero_grad()

    def optimizer_step(self):
        assert self.optimizer is not None
        assert self.optimizer_config is not None
        assert self.lr_scheduler is not None

        # Use VeOmni's gradient clipping
        grad_norm = veomni_clip_grad_norm(
            self.model,
            max_norm=self.optimizer_config.gradient_clipping,
        )

        if not math.isfinite(grad_norm):
            self.optimizer_zero_grad()
            update_successful = False
        else:
            with trace_scope("veomini_engine.step"):
                self.optimizer.step()
            update_successful = True

        current_lr = self.lr_scheduler.get_last_lr()[0]
        return dict(
            update_successful=float(update_successful),
            grad_norm=float(grad_norm) if grad_norm is not None else float("nan"),
            lr=current_lr,
        )

    def lr_scheduler_step(self):
        assert self.lr_scheduler is not None
        self.lr_scheduler.step()

    # ------------------------------------------------------------------
    # Forward / backward
    # ------------------------------------------------------------------

    def forward_backward_batch(
        self,
        mb_list: MicroBatchList,
        process_output_fn: Callable[
            [torch.Tensor, dict[str, Any]], torch.Tensor | None
        ],
        forward_only: bool = False,
    ) -> None:
        for mb_item in mb_list:
            inputs, ctx = self._prepare_mb_inputs(mb_item)

            with trace_scope("veomini_engine.forward"):
                outputs = self.model(**inputs)
            logits = outputs.logits.squeeze(0)

            ctx_dict = ctx.to_dict()
            loss = process_output_fn(logits, ctx_dict)

            if not forward_only and loss is not None:
                with trace_scope("veomini_engine.backward"):
                    loss.backward()

    def train_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> dict[str, float]:
        self._ensure_ready()
        self.optimizer_zero_grad()

        # Step 1: Prepare micro-batches
        mb_list = self._prepare_mb_list(input_).to(self.device)

        # Step 2: Compute total loss weight
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self._dp_group
        )

        # Step 3: Forward-backward
        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)
            return self._compute_logprobs_and_loss(
                logits,
                ctx,
                loss_fn,
                loss_weight_fn,
                total_loss_weight,
                loss_multiplier=self.data_parallel_world_size,
            )

        self.forward_backward_batch(mb_list, process_output, forward_only=False)

        # Step 4: Optimizer step
        return self.optimizer_step()

    @torch.no_grad()
    def eval_batch(
        self,
        input_: dict[str, Any],
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
    ) -> torch.Tensor | None:
        self._ensure_ready()

        mb_list = self._prepare_mb_list(input_).to(self.device)
        total_loss_weight = compute_total_loss_weight(
            mb_list, loss_weight_fn, self._dp_group
        )

        losses: list[torch.Tensor] = []

        def process_output(
            logits: torch.Tensor, ctx_dict: dict[str, Any]
        ) -> torch.Tensor:
            ctx = FSDPTrainContext(**ctx_dict)
            loss = self._compute_logprobs_and_loss(
                logits, ctx, loss_fn, loss_weight_fn, total_loss_weight,
            )
            losses.append(loss.detach())
            return loss

        self.forward_backward_batch(mb_list, process_output, forward_only=True)
        return aggregate_eval_losses(losses, self._dp_group)

    @torch.no_grad()
    def forward_batch(
        self,
        input_: dict[str, Any],
        output_seqlens: list[int] | None = None,
        aggregate_fn: Callable[[list[Any]], Any] = torch.cat,
    ) -> torch.Tensor:
        self._ensure_ready()

        cu_seqlens = pack_tensor_dict(input_)["cu_seqlens"]
        if output_seqlens is None:
            output_seqlens = (cu_seqlens[1:] - cu_seqlens[:-1]).cpu().numpy().tolist()
        assert output_seqlens is not None
        batch_size = len(output_seqlens)

        mb_list = self._prepare_mb_list(input_).to(self.device)
        outputs: list[torch.Tensor] = []

        def process_output(logits: torch.Tensor, ctx_dict: dict[str, Any]) -> None:
            ctx = FSDPTrainContext(**ctx_dict)
            result = self._compute_forward_result(logits, ctx)
            outputs.append(result)
            return None

        self.forward_backward_batch(mb_list, process_output, forward_only=True)
        return reorder_and_pad_outputs(outputs, output_seqlens, mb_list, aggregate_fn)

    def export_stats(self) -> dict[str, float]:
        return stats_tracker.export_all(reduce_group=self.data_parallel_group)

    # ------------------------------------------------------------------
    # Offload / onload
    # ------------------------------------------------------------------

    def offload(self) -> None:
        self.get_device_stats().log("before offload model")
        empty_cache()
        synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after offload model")
        self.is_offload = True

    def onload(self) -> None:
        synchronize()
        dist.barrier(group=self.cpu_group)
        self.get_device_stats().log("after onload model")
        self.is_offload = False

    def clear_batches(self, *args):
        """Placeholder method of single-controller API."""

    def get_device_stats(self) -> DeviceRuntimeInfo:
        return DeviceRuntimeInfo.get_current()

    def save_perf_tracer(self, step: int | None = None, force: bool = False) -> None:
        perf_tracer.save(step=step, force=force)

    def config_perf_tracer(
        self, config: PerfTracerConfig, rank: int, role: str
    ) -> None:
        if perf_tracer.is_configured():
            return
        perf_tracer.configure(config, rank=rank, role=role)

    # ==================================================================
    # Private helpers — model creation & parallelization (VeOmni-based)
    # ==================================================================

    def _make_parallel_strategy(
        self, parallel_strategy: ParallelStrategy
    ) -> FSDPParallelStrategy:
        return FSDPParallelStrategy(
            **dataclasses.asdict(parallel_strategy),
        )

    def _create_device_model(self):
        """Build the model using VeOmni's ``build_foundation_model``."""
        self.get_device_stats().log("before model creation/loading")

        dtype = self.config.dtype  # e.g. "bfloat16"
        init_from_scratch = self.config.init_from_scratch

        # Build tokenizer
        self.tokenizer = build_tokenizer(self.config.path)
        try:
            self.processor = build_processor(self.config.path)
        except Exception:
            self.processor = None

        tik = time.perf_counter()

        weights_path = None if init_from_scratch else self.config.path

        model = build_foundation_model(
            config_path=self.config.path,
            weights_path=weights_path,
            torch_dtype=dtype,
            attn_implementation=self.config.attn_impl or "veomni_flash_attention_2_with_sp",
            init_device="meta",  # FSDP2 requires meta init
        )

        self.model_config = model.config
        self.get_device_stats().log("after model creation/loading")

        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )

        self.logger.info(
            f"Model creation time: {time.perf_counter() - tik:.2f}s"
        )
        self.model = model

    def _parallelize_model(self):
        """Apply VeOmni's parallelization (FSDP2 + EP when enabled) to the model.

        When ``ep_size > 1``, VeOmni's ``build_parallelize_model`` will:
        1. Call ``model.get_parallel_plan()`` to get expert sharding specs.
        2. Slice expert tensors along the expert dimension across EP ranks.
        3. Apply FSDP2 to expert modules on the ``ep_fsdp`` mesh (dim-1 sharding).
        4. Apply FSDP2 to regular modules on the standard FSDP mesh (dim-0 sharding).
        """
        tik = time.perf_counter()

        basic_modules = list(
            set(getattr(self.model, "_no_split_modules", None) or [])
        )

        # Validate that the model provides a parallel plan when EP is enabled
        ep_enabled = self._parallel_state is not None and self._parallel_state.ep_enabled
        if ep_enabled and not hasattr(self.model, "get_parallel_plan"):
            raise ValueError(
                "Expert Parallelism (EP) requires the model to implement "
                "``get_parallel_plan()``. Please ensure the model class defines "
                "this method (see veomni/models/transformers/qwen3_moe/parallel_plan.py "
                "for an example)."
            )

        self.model = build_parallelize_model(
            self.model,
            init_device="meta",
            weights_path=None if self.config.init_from_scratch else self.config.path,
            enable_full_shard=True,
            enable_reshard_after_forward=True,
            enable_mixed_precision=True,
            enable_gradient_checkpointing=self.config.gradient_checkpointing,
            basic_modules=basic_modules,
            enable_reentrant=False,
            enable_forward_prefetch=True,
            broadcast_model_weights_from_rank0=False,
        )
        self.model.train()

        self.logger.info(
            f"Model parallelization took {time.perf_counter() - tik:.2f}s"
            + (f" (EP enabled, ep_size={self._parallel_state.ep_size})" if ep_enabled else "")
        )

    def _create_optimizer(self, ft_spec: FinetuneSpec) -> None:
        """Build optimizer and lr scheduler using VeOmni utilities."""
        if self.optimizer_config is None:
            return
        assert self.model is not None

        tik = time.perf_counter()

        lr = self.optimizer_config.lr
        weight_decay = self.optimizer_config.weight_decay

        optimizer_type = "adamw"
        if self.optimizer_config.type == "adam_bf16":
            optimizer_type = "anyprecision_adamw"
        elif self.optimizer_config.type == "sgd":
            self.logger.warning(
                "SGD is not natively supported by VeOmni's build_optimizer. "
                "Falling back to adamw."
            )

        self.optimizer = build_optimizer(
            self.model,
            lr=lr,
            weight_decay=weight_decay,
            fused=True,
            optimizer_type=optimizer_type,
        )

        total_train_steps = ft_spec.total_train_steps
        warmup_ratio = self.optimizer_config.warmup_steps_proportion

        lr_decay_style = "constant"
        if self.optimizer_config.lr_scheduler_type == "cosine":
            lr_decay_style = "cosine"
        elif self.optimizer_config.lr_scheduler_type == "linear":
            lr_decay_style = "linear"

        self.lr_scheduler = build_lr_scheduler(
            self.optimizer,
            train_steps=total_train_steps,
            lr=lr,
            lr_decay_style=lr_decay_style,
            lr_warmup_ratio=warmup_ratio,
            lr_min=getattr(self.optimizer_config, "min_lr", 1e-7)
            if hasattr(self.optimizer_config, "min_lr")
            else lr * getattr(self.optimizer_config, "min_lr_ratio", 0.0),
        )

        self.logger.info(f"Create optimizer time: {time.perf_counter() - tik:.2f}s")

    # ==================================================================
    # Private helpers — micro-batch preparation (aligned with FSDPEngine)
    # ==================================================================

    def _prepare_mb_list(self, input_: dict[str, Any]) -> MicroBatchList:
        assert "attention_mask" in input_ and "input_ids" in input_
        input_ = input_.copy()

        input_ = amend_position_ids(input_)

        mb_list = split_padded_tensor_dict_into_mb_list(input_, self.config.mb_spec)
        mb_list.mbs = [pack_tensor_dict(mb) for mb in mb_list.mbs]
        mb_list = pad_mb_list(
            mb_list,
            pad_value=0.0,
            pad_to_maximum=self.config.pad_to_maximum,
        )
        self.logger.info(
            f"Microbatch #tokens (rank {dist.get_rank()}): {mb_list.group_lens}, "
            f"padded to: {mb_list.padded_to_lengths}, padding lengths: {mb_list.padding_lengths}"
        )
        mb_list = unsqueeze_mb_list(mb_list)

        assert mb_list.padded_mbs is not None
        for i, mb in enumerate(mb_list.mbs):
            mb_list.mbs[i] = dict(**mb)
        for i, mb in enumerate(mb_list.padded_mbs):
            mb_list.padded_mbs[i] = dict(**mb)

        for mb, padded_mb in zip(mb_list.mbs, mb_list.padded_mbs):
            mb["max_length_q"] = mb["max_length_k"] = mb["max_seqlen"] = int(
                mb["max_seqlen"]
            )
            padded_mb["max_length_q"] = padded_mb["max_length_k"] = padded_mb[
                "max_seqlen"
            ] = int(padded_mb["max_seqlen"])
            mb["cu_seq_lens_q"] = mb["cu_seq_lens_k"] = mb["cu_seqlens"]
            padded_mb["cu_seq_lens_q"] = padded_mb["cu_seq_lens_k"] = padded_mb[
                "cu_seqlens"
            ]
            mb["use_cache"] = False
            padded_mb["use_cache"] = False
            mb["attention_mask"] = dict(full_attention=None, sliding_attention=None)
            padded_mb["attention_mask"] = dict(
                full_attention=None, sliding_attention=None
            )
        return mb_list

    def _prepare_mb_inputs(
        self, mb_item: MicroBatchItem
    ) -> tuple[dict[str, Any], FSDPTrainContext]:
        """Prepare micro-batch inputs with Ulysses sequence parallel handling."""
        sp_size = self._parallel_state.sp_size if self._parallel_state else 1

        if sp_size > 1:
            input_ids = mb_item.padded_mb["input_ids"]
            position_ids = mb_item.padded_mb.get("position_ids", None)

            (
                ulysses_input_ids,
                ulysses_position_ids,
                ulysses_pad_size,
            ) = ulysses_pad_and_slice_inputs(
                input_ids,
                position_ids,
                sp_size=sp_size,
            )
            if (
                ulysses_position_ids is not None
                and not ulysses_position_ids.is_contiguous()
            ):
                ulysses_position_ids = ulysses_position_ids.contiguous()

            inputs = ulysses_prepare_inputs(
                mb_item.padded_mb,
                ulysses_input_ids,
                ulysses_position_ids,
                sp_size,
            )
        else:
            inputs = mb_item.padded_mb
            ulysses_pad_size = 0

        ctx = FSDPTrainContext(
            model_inputs=inputs,
            mb_input=mb_item.orig_mb,
            pad_length=mb_item.padding_length,
            ulysses_pad_size=ulysses_pad_size,
        )
        return inputs, ctx

    # ==================================================================
    # Private helpers — logprobs / loss computation
    # ==================================================================

    def _sp_all_gather(self, tensor: torch.Tensor) -> torch.Tensor:
        import torch.distributed.nn.functional as dist_F

        gathered = dist_F.all_gather(tensor, group=self._sp_group)
        return torch.cat(gathered, dim=-1)

    def _get_vocab_min_max_logits(
        self,
        logits: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        vocab_min_logits = logits.detach().min(-1).values.float()
        vocab_max_logits = logits.detach().max(-1).values.float()
        sp_size = self._parallel_state.sp_size if self._parallel_state else 1
        if sp_size > 1:
            vocab_min_logits = self._sp_all_gather(vocab_min_logits)
            vocab_max_logits = self._sp_all_gather(vocab_max_logits)
            if ulysses_pad_size > 0:
                vocab_min_logits = vocab_min_logits[:-ulysses_pad_size]
                vocab_max_logits = vocab_max_logits[:-ulysses_pad_size]
        return vocab_min_logits, vocab_max_logits

    def _compute_logprobs_entropy(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs, entropy = gather_logprobs_entropy(
            logits,
            labels,
            temperature=self.config.temperature,
        )
        sp_size = self._parallel_state.sp_size if self._parallel_state else 1
        if sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            entropy = self._sp_all_gather(entropy)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
                entropy = entropy[:-ulysses_pad_size]
        return logprobs, entropy

    def _compute_logprobs(
        self,
        logits: torch.Tensor,
        inputs: dict[str, Any],
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        labels = inputs.get(
            "rolled_input_ids",
            torch.roll(inputs["input_ids"], shifts=-1, dims=-1),
        )
        if labels.ndim == 2 and labels.shape[0] == 1:
            labels = labels.squeeze(0)
        logprobs = gather_logprobs(
            logits,
            labels,
            temperature=self.config.temperature,
        )
        sp_size = self._parallel_state.sp_size if self._parallel_state else 1
        if sp_size > 1:
            logprobs = self._sp_all_gather(logprobs)
            if ulysses_pad_size > 0:
                logprobs = logprobs[:-ulysses_pad_size]
        return logprobs

    def _compute_values(
        self,
        values: torch.Tensor,
        ulysses_pad_size: int = 0,
    ) -> torch.Tensor:
        sp_size = self._parallel_state.sp_size if self._parallel_state else 1
        if sp_size > 1:
            values = self._sp_all_gather(values)
            if ulysses_pad_size > 0:
                values = values[:-ulysses_pad_size]
        return values

    def _compute_logprobs_and_loss(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
        loss_fn: Callable[..., torch.Tensor],
        loss_weight_fn: Callable[[dict[str, Any]], torch.Tensor],
        total_loss_weight: torch.Tensor,
        loss_multiplier: float = 1.0,
    ) -> torch.Tensor:
        """Compute logprobs/entropy and return scaled loss."""
        if not self.config.is_critic:
            logprobs, entropy = self._compute_logprobs_entropy(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
            vocab_min_logits, vocab_max_logits = self._get_vocab_min_max_logits(
                logits, ctx.ulysses_pad_size
            )
            if ctx.pad_length > 0:
                logprobs = logprobs[: -ctx.pad_length]
                entropy = entropy[: -ctx.pad_length]
                vocab_min_logits = vocab_min_logits[: -ctx.pad_length]
                vocab_max_logits = vocab_max_logits[: -ctx.pad_length]
            loss = loss_fn(
                logprobs,
                entropy,
                ctx.mb_input,
                vocab_min_logits=vocab_min_logits,
                vocab_max_logits=vocab_max_logits,
            )
        else:
            values = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
            if ctx.pad_length > 0:
                values = values[: -ctx.pad_length]
            loss = loss_fn(values, ctx.mb_input)

        loss_scale = loss_weight_fn(ctx.mb_input) / total_loss_weight * loss_multiplier
        return loss * loss_scale

    def _compute_forward_result(
        self,
        logits: torch.Tensor,
        ctx: FSDPTrainContext,
    ) -> torch.Tensor:
        """Compute forward output (logprobs or values)."""
        if not self.config.is_critic:
            result = self._compute_logprobs(
                logits, ctx.model_inputs, ctx.ulysses_pad_size
            )
        else:
            result = self._compute_values(logits.squeeze(-1), ctx.ulysses_pad_size)
        if ctx.pad_length > 0:
            result = result[: -ctx.pad_length]
        return result

    # ==================================================================
    # Private helpers — weight update coordination
    # ==================================================================

    def _check_rollout_engine_connected(self) -> None:
        if self.rollout_engine is None or self.rollout_coordinator is None:
            raise RuntimeError(
                "Rollout engine not connected. Call connect_engine()"
                " before using rollout/update_weight methods."
            )

    def _ensure_ready(self) -> None:
        if self.is_offload:
            self.onload()

        sp_size = self._parallel_state.sp_size if self._parallel_state else 1
        if sp_size > 1 and self._sp_group is not None:
            set_ulysses_sequence_parallel_group(self._sp_group)

    def _get_model_name_parameters(self) -> Iterator[tuple[str, nn.Parameter]]:
        yield from self.model.named_parameters()

    def _get_full_tensor(self, param: nn.Parameter) -> torch.Tensor:
        """Get full tensor from a parameter, handling DTensor and CPU offloaded tensors.

        NOTE: For EP models, this gathers within the FSDP mesh only.
        Expert params remain local (each EP rank holds its subset of experts).
        Use ``_get_full_tensor_ep_aware`` for gathering across EP as well.
        """
        tensor = param.data
        if isinstance(tensor, DTensor):
            if tensor.device.type != "cpu":
                return tensor.full_tensor()
            temp_dtensor = DTensor.from_local(
                tensor.to_local(),
                device_mesh=tensor.device_mesh,
                placements=tensor.placements,
            )
            return temp_dtensor.full_tensor()
        else:
            if tensor.device.type == "cpu":
                tensor = tensor.to(get_device_type())
            return tensor

    def _is_ep_expert_param(self, name: str) -> bool:
        """Check whether a parameter is an EP-sharded expert parameter.

        VeOmni's ``parallelize_model_fsdp2`` attaches ``_fqn2spec_info`` to
        the model when EP is enabled.  Each entry maps an FQN to a ``SpecInfo``
        with the original EP placement.
        """
        fqn2spec = getattr(self.model, "_fqn2spec_info", None)
        if fqn2spec is None:
            return False
        spec_info = fqn2spec.get(name)
        if spec_info is None:
            return False
        # SpecInfo.placement is Shard(0) for expert params, Replicate for others
        from torch.distributed.tensor import Shard as DTShard
        return isinstance(spec_info.placement, DTShard)

    def _gather_expert_tensor_across_ep(
        self, tensor: torch.Tensor, name: str
    ) -> torch.Tensor:
        """All-gather an expert tensor across the EP group.

        After FSDP2 ``full_tensor()``, each EP rank holds its local expert
        slice (e.g., 32 out of 128 experts for EP=4).  This method
        concatenates them along the expert dimension (dim 0) so that the
        result is the complete expert weight.

        Args:
            tensor: The local expert tensor (already FSDP-gathered).
            name: Parameter FQN (used to look up EP shard dim).

        Returns:
            The globally-gathered expert tensor on every EP rank.
        """
        assert self._parallel_state is not None and self._parallel_state.ep_enabled
        ep_group = self._parallel_state.ep_group
        ep_size = self._parallel_state.ep_size

        # Determine the shard dimension from the EP plan
        fqn2spec = getattr(self.model, "_fqn2spec_info", {})
        spec_info = fqn2spec.get(name)
        shard_dim = spec_info.placement.dim if spec_info else 0

        # All-gather across EP group
        gathered = [torch.empty_like(tensor) for _ in range(ep_size)]
        dist.all_gather(gathered, tensor.contiguous(), group=ep_group)
        return torch.cat(gathered, dim=shard_dim)

    # ------------------------------------------------------------------
    # MoE weight format conversion  (VeOmni ↔ HuggingFace)
    # ------------------------------------------------------------------

    def _split_merged_expert_to_hf(
        self, name: str, tensor: torch.Tensor
    ) -> list[tuple[str, torch.Tensor]]:
        """Convert a VeOmni merged 3D expert param to HF per-expert 2D params.

        VeOmni stores expert weights as merged 3D tensors::

            model.layers.{i}.mlp.experts.gate_proj   [num_experts, out, in]

        HF / sglang expects per-expert 2D tensors::

            model.layers.{i}.mlp.experts.{j}.gate_proj.weight   [out, in]

        For non-expert parameters the original ``(name, tensor)`` pair is
        returned unchanged.
        """
        match = _MERGED_EXPERT_RE.match(name)
        if match is None:
            return [(name, tensor)]

        prefix = match.group(1)   # e.g. "model.layers.0.mlp.experts"
        proj = match.group(2)     # e.g. "gate_proj"

        # Unbind along the expert dimension (dim 0) — returns views, no copy.
        expert_weights = torch.unbind(tensor, dim=0)
        return [
            (f"{prefix}.{expert_id}.{proj}.weight", w)
            for expert_id, w in enumerate(expert_weights)
        ]

    @staticmethod
    def _merge_hf_experts_to_veomni(
        state_dict: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Convert HF per-expert 2D weights to VeOmni merged 3D format.

        This is the reverse of ``_split_merged_expert_to_hf``, needed when
        loading from an HF-format checkpoint into a VeOmni model.

        Non-expert keys are passed through unchanged.
        """
        new_state_dict: dict[str, torch.Tensor] = {}
        expert_buffers: dict[str, list[tuple[int, torch.Tensor]]] = {}

        for name, tensor in state_dict.items():
            match = _HF_EXPERT_RE.match(name)
            if match:
                prefix = match.group(1)      # "model.layers.0.mlp.experts"
                expert_id = int(match.group(2))
                proj = match.group(3)        # "gate_proj"
                veomni_key = f"{prefix}.{proj}"
                if veomni_key not in expert_buffers:
                    expert_buffers[veomni_key] = []
                expert_buffers[veomni_key].append((expert_id, tensor))
            else:
                new_state_dict[name] = tensor

        # Stack collected per-expert tensors into merged 3D tensors.
        for veomni_key, experts in expert_buffers.items():
            experts.sort(key=lambda x: x[0])
            new_state_dict[veomni_key] = torch.stack([t for _, t in experts])

        return new_state_dict

    def _update_bucket_weights_from_distributed(
        self,
        meta: WeightUpdateMeta,
        named_tensors: list[tuple[str, nn.Parameter | torch.Tensor]],
    ):
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

        fut = self.rollout_engine.update_weights_from_distributed(meta, param_specs)

        handles = []
        for _, tensor in named_tensors:
            handles.append(
                dist.broadcast(
                    tensor, src=0, group=self.weight_update_group, async_op=True
                )
            )
        for handle in handles:
            handle.wait()

        fut.result()
        named_tensors.clear()

    def _init_weight_update_from_distributed(self, meta: WeightUpdateMeta):
        assert meta.type == "xccl"

        meta.nccl_master_address = self.weight_update_master_addr = gethostip()
        meta.nccl_master_port = self.weight_update_master_port = find_free_ports(1)[0]
        meta.nccl_group_name = self.weight_update_group_name

        os.environ["TORCHELASTIC_USE_AGENT_STORE"] = str(False)
        if dist.get_rank() == 0:
            assert meta.alloc_mode is not None

            fut = self.rollout_engine.init_weights_update_group(meta)

            self.logger.info(
                f"Initializing weight update group: type={meta.type} "
                f"init_method=tcp://{meta.nccl_master_address}:{meta.nccl_master_port} "
                f"group={meta.nccl_group_name}"
            )
            self.weight_update_group = init_custom_process_group(
                backend=get_dist_comm_backend(),
                world_size=meta.alloc_mode.gen.world_size + 1,
                init_method=f"tcp://{meta.nccl_master_address}:{meta.nccl_master_port}",
                rank=0,
                group_name=meta.nccl_group_name,
                timeout=DIST_GROUP_DEFAULT_TIMEOUT,
            )

            fut.result()

    @trace_perf("veomini_engine.update_weights_from_distributed", category="comm")
    def _update_weights_from_distributed(self, meta: WeightUpdateMeta):
        """Broadcast parameters (chunked) from rank 0.

        When EP is enabled, expert parameters are first gathered across the
        EP group so that rank 0 obtains the full expert weights before
        broadcasting them to the inference engine (e.g. sglang).

        VeOmni merged 3D expert tensors are split into HF per-expert 2D
        format before broadcasting so that sglang receives HF-compatible
        parameter names and shapes.
        """
        meta.nccl_master_address = self.weight_update_master_addr
        meta.nccl_master_port = self.weight_update_master_port
        meta.nccl_group_name = self.weight_update_group_name

        if dist.get_rank() == 0:
            self.rollout_engine.pause_generation()

        dist.barrier(group=self.cpu_group)

        ep_enabled = (
            self._parallel_state is not None
            and self._parallel_state.ep_enabled
        )

        weight_chunked_mem_size = meta.weight_chunked_mem_mb * 1024 * 1024
        main_rank = dist.get_rank() == 0

        buffer_size = 0
        named_tensors: list[tuple[str, torch.Tensor]] = []

        for name, param in self._get_model_name_parameters():
            # full_tensor() gathers within FSDP mesh only (not across EP)
            tensor = self._get_full_tensor(param)

            # For EP expert params, all-gather across EP group so that
            # every EP rank (including rank 0) has the full expert weights.
            if ep_enabled and self._is_ep_expert_param(name):
                tensor = self._gather_expert_tensor_across_ep(tensor, name)

            if not main_rank:
                continue

            # Convert VeOmni merged 3D expert params to HF per-expert 2D
            # format so that the inference engine (e.g. sglang) receives
            # HF-compatible parameter names and shapes.
            hf_pairs = self._split_merged_expert_to_hf(name, tensor)
            for hf_name, hf_tensor in hf_pairs:
                tensor_size = hf_tensor.numel() * hf_tensor.element_size()
                if tensor_size + buffer_size > weight_chunked_mem_size:
                    self._update_bucket_weights_from_distributed(meta, named_tensors)
                    buffer_size = 0

                named_tensors.append((hf_name, hf_tensor))
                buffer_size += tensor_size

        if named_tensors:
            self._update_bucket_weights_from_distributed(meta, named_tensors)

        dist.barrier(group=self.cpu_group)

        if dist.get_rank() == 0:
            self.rollout_engine.continue_generation()

        synchronize()
        dist.barrier(group=self.cpu_group)

    @trace_perf("veomini_engine.update_weights_from_disk", category="io")
    def _update_weights_from_disk(self, meta: WeightUpdateMeta):
        fut = Future()

        if dist.get_rank() == 0:
            fut = self.rollout_engine.update_weights_from_disk(meta)

        assert meta.path is not None
        self._save_model_to_hf(meta.path, self.tokenizer, self.processor)

        if dist.get_rank() == 0:
            update_name = names.update_weights_from_disk(
                self.config.experiment_name,
                self.config.trial_name,
                self.get_version(),
            )
            name_resolve.add(
                update_name, str(datetime.now().timestamp()), keepalive_ttl=120
            )
            fut.result()

        synchronize()
        dist.barrier(group=self.cpu_group)

    # ==================================================================
    # Private helpers — checkpoint save/load
    # ==================================================================

    def _save_model_to_hf(
        self,
        path: str,
        tokenizer: PreTrainedTokenizerFast | None,
        processor: ProcessorMixin | None,
    ):
        """Save model in HuggingFace format using VeOmni's save_model_weights.

        When EP is enabled, expert parameters are gathered across the EP group
        first so that the saved checkpoint contains all experts (not just the
        local shard).
        """
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)

        ep_enabled = (
            self._parallel_state is not None
            and self._parallel_state.ep_enabled
        )

        # Build state dict, gathering EP expert params if needed.
        # VeOmni's save_model_weights calls .full_tensor() internally for
        # DTensors, which only gathers within FSDP — not across EP.  For
        # expert params we need to pre-gather so the full expert weights
        # are available on rank 0.
        if ep_enabled:
            veomni_state_dict: dict[str, torch.Tensor | nn.Parameter] = {}
            for name, param in self.model.named_parameters():
                if self._is_ep_expert_param(name):
                    full_fsdp = self._get_full_tensor(param)
                    full_ep = self._gather_expert_tensor_across_ep(full_fsdp, name)
                    veomni_state_dict[name] = full_ep
                else:
                    veomni_state_dict[name] = param
        else:
            veomni_state_dict = dict(self.model.named_parameters())

        # Convert VeOmni merged 3D expert params to HF per-expert 2D format
        # so the saved checkpoint is loadable by HF / sglang.
        state_dict: dict[str, torch.Tensor | nn.Parameter] = {}
        for name, tensor in veomni_state_dict.items():
            for hf_name, hf_tensor in self._split_merged_expert_to_hf(name, tensor):
                state_dict[hf_name] = hf_tensor

        model_assets = []
        if self.model_config is not None:
            model_assets.append(self.model_config)
        if tokenizer is not None:
            model_assets.append(tokenizer)
        if processor is not None:
            model_assets.append(processor)

        save_model_weights(
            output_dir=path,
            state_dict=state_dict,
            global_rank=dist.get_rank(),
            save_dtype="bfloat16",
            model_assets=model_assets if model_assets else None,
        )
        dist.barrier(group=self.cpu_group)

    def _load_model_from_hf(self, path: str):
        """Load model from HuggingFace format.

        If the checkpoint contains HF per-expert keys (e.g. saved by
        ``_save_model_to_hf``), they are automatically merged into VeOmni's
        3D format before loading into the model.
        """
        if dist.get_rank() == 0:
            full_state = get_state_dict_from_repo_id_or_path(path)
            # Detect HF per-expert keys and convert to VeOmni merged format
            # if the checkpoint was saved in HF format.
            if any(_HF_EXPERT_RE.match(k) for k in full_state):
                full_state = self._merge_hf_experts_to_veomni(full_state)
        else:
            full_state = {}

        fsdp2_load_full_state_dict(
            self.model,
            full_state,
            cpu_offload=None,
            tie_word_embeddings=self.model_config.tie_word_embeddings,
        )

    def _save_to_dcp(self, path: str, with_optim: bool):
        """Save model in DCP format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        os.makedirs(path, exist_ok=True)
        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.save(state_dict, checkpoint_id=path)

    def _load_from_dcp(self, path: str, with_optim: bool):
        """Load model from DCP format."""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        dcp_state = DCPState(self.model, self.optimizer if with_optim else None)
        state_dict = {"dcp": dcp_state}
        dcp.load(state_dict=state_dict, checkpoint_id=path)

    def _save_optimizer_state(self, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        state_dict = self.optimizer.state_dict()
        torch.save(state_dict, shard_path)
        dist.barrier(group=self.cpu_group)

    def _load_optimizer_state(self, path: str):
        assert self.optimizer is not None
        assert dist.is_initialized()
        rank = dist.get_rank()
        shard_path = os.path.join(
            path, f"optim_world_size_{self.world_size}_rank_{rank}.pt"
        )
        optimizer_state_dict = torch.load(shard_path, weights_only=False)
        self.optimizer.load_state_dict(optimizer_state_dict)
        dist.barrier(group=self.cpu_group)


# =============================================================================
# Algorithm-specific VeOMini Engines
# =============================================================================


class VeOMiniPPOActor(VeOMiniEngine):
    """PPO Actor implementation using VeOMini backend."""

    def __init__(self, config: PPOActorConfig):
        from areal.trainer.ppo.actor import PPOActor

        super().__init__(config)
        self.actor = PPOActor(config, self)

    @torch.no_grad()
    def compute_logp(self, *args, **kwargs) -> torch.Tensor | None:
        return self.actor.compute_logp(*args, **kwargs)

    @torch.no_grad()
    def compute_advantages(self, *args, **kwargs) -> dict[str, Any]:
        return self.actor.compute_advantages(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.actor.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOActorConfig, scheduler: Scheduler):
        from areal.trainer.ppo.actor import PPOActorController

        return PPOActorController(train_engine=cls, config=config, scheduler=scheduler)


class VeOMiniPPOCritic(VeOMiniEngine):
    """PPO Critic implementation using VeOMini backend."""

    def __init__(self, config: PPOCriticConfig):
        from areal.trainer.ppo.critic import PPOCritic

        super().__init__(config)
        self.critic = PPOCritic(config, self)

    @torch.no_grad()
    def compute_values(self, *args, **kwargs) -> torch.Tensor:
        return self.critic.compute_values(*args, **kwargs)

    def ppo_update(self, *args, **kwargs) -> None:
        self.critic.ppo_update(*args, **kwargs)

    @classmethod
    def as_controller(cls, config: PPOCriticConfig, scheduler: Scheduler):
        from areal.trainer.ppo.critic import PPOCriticController

        return PPOCriticController(train_engine=cls, config=config, scheduler=scheduler)


class VeOMiniLMEngine(VeOMiniEngine):
    """Language model engine for SFT using VeOMini backend."""

    def __init__(self, config: TrainEngineConfig):
        from areal.trainer.sft.lm_engine import LMEngine

        super().__init__(config)
        self.lm_engine = LMEngine(self)

    def train_lm(self, data):
        return self.lm_engine.train_lm(data)

    def evaluate_lm(self, data):
        return self.lm_engine.evaluate_lm(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        from areal.trainer.sft.lm_engine import LMController

        return LMController(train_engine=cls, config=config, scheduler=scheduler)


class VeOMiniRWEngine(VeOMiniEngine):
    """Reward model engine using VeOMini backend."""

    def __init__(self, config: TrainEngineConfig):
        from copy import deepcopy

        from areal.trainer.rw.rw_engine import RWEngine

        super().__init__(config)
        self.rw_engine = RWEngine(self)
        if self.config.mb_spec.granularity != 2:
            rw_logger = logging.getLogger("RW engine")
            rw_logger.warning("mb_spec.granularity must be 2 for reward modeling")
            self.config = deepcopy(self.config)
            self.config.mb_spec.granularity = 2

    def train_rw(self, data):
        return self.rw_engine.train_rw(data)

    def evaluate_rw(self, data):
        return self.rw_engine.evaluate_rw(data)

    @classmethod
    def as_controller(cls, config: TrainEngineConfig, scheduler: Scheduler):
        from areal.trainer.rw.rw_engine import RWController

        return RWController(train_engine=cls, config=config, scheduler=scheduler)
