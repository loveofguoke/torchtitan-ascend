# SPDX-License-Identifier: Apache-2.0
"""HyperParallel MegaMoE execution with TorchTitan-owned expert parameters."""

from __future__ import annotations

import logging
import weakref
from dataclasses import dataclass
from typing import Literal, TYPE_CHECKING

import spmd_types as spmd
import torch
from torchtitan.config import derive, override
from torchtitan.distributed import ParallelismContext
from torchtitan.models.base.module import Module
from torchtitan.models.common.activation import SwiGLU
from torchtitan.models.common.linear import GroupedLinear
from torchtitan.models.common.moe import RoutedExperts

from torchtitan_ascend.device import require_device

if TYPE_CHECKING:
    from hyper_parallel.core.multicore import MegaMoeExperts

logger = logging.getLogger(__name__)


class _ExecutionGroup:
    """Collect compatible layers during parallelization, bind before first use."""

    _groups: weakref.WeakValueDictionary = weakref.WeakValueDictionary()

    def __init__(self, context: ParallelismContext) -> None:
        self.context = context
        self.members: weakref.WeakSet = weakref.WeakSet()
        self.prepared = False

    @classmethod
    def register(
        cls, module: MegaMoeRoutedExperts, context: ParallelismContext
    ) -> _ExecutionGroup:
        key = (
            id(context),
            module.w13.group_size,
            module.w13.in_features,
            module.w2.in_features,
            module.top_k,
            module.dispatch_mode,
        )
        group = cls._groups.get(key)
        if group is None:
            group = cls(context)
            cls._groups[key] = group
        if group.prepared:
            raise RuntimeError("Parallelize all MegaMoE layers before first forward")
        group.members.add(module)
        return group

    def prepare(self) -> None:
        if self.prepared:
            return
        executors = [module._executor for module in self.members]
        if any(executor is None for executor in executors):
            raise RuntimeError("Configure all MegaMoE layers before sharing resources")
        from hyper_parallel.core.multicore import MegaMoeExperts

        MegaMoeExperts.share_execution_resources(executors)
        self.prepared = True
        logger.info(
            "Hyper MegaMoE shared execution resources across %s layers", len(executors)
        )


class MegaMoeRoutedExperts(Module):
    """Fuse dispatch, SwiGLU experts and combine without changing state ownership.

    W13 retains ``[E, 2, F, D]`` and W2 retains ``[E, D, F]``. MegaMoE's
    external-parameter EOI interface consumes zero-copy views of these weights.
    Router scores already include the model's normalization and scaling.
    """

    @dataclass(kw_only=True, slots=True)
    class Config(Module.Config):
        w13: GroupedLinear.Config
        w2: GroupedLinear.Config
        top_k: int
        dispatch_mode: Literal["push", "pull"] = "push"

        def __post_init__(self) -> None:
            if self.w13.group_size != self.w2.group_size:
                raise ValueError("MegaMoE projections must have the same expert count")
            if self.w13.num_linears != 2 or self.w2.num_linears != 1:
                raise ValueError(
                    "MegaMoE requires stacked gate/up and one down projection"
                )
            if (
                self.w13.in_features != self.w2.out_features
                or self.w13.out_features != self.w2.in_features
            ):
                raise ValueError("MegaMoE projection dimensions do not match")
            if not 0 < self.top_k <= self.w13.group_size:
                raise ValueError("MegaMoE top_k must be within the expert count")
            if self.dispatch_mode not in ("push", "pull"):
                raise ValueError("MegaMoE dispatch_mode must be push or pull")

    def __init__(self, config: Config) -> None:
        super().__init__()
        require_device("Ascend910B", feature="Hyper MegaMoE TorchTitan adapter")
        self.w13 = config.w13.build()
        self.w2 = config.w2.build()
        self.top_k = config.top_k
        self.dispatch_mode = config.dispatch_mode
        self._executor: MegaMoeExperts | None = None
        self._execution_group: _ExecutionGroup | None = None

    def _parallelize(self, parallelism_context: ParallelismContext) -> None:
        """Bind an explicit EP group after TorchTitan establishes its meshes."""
        if (
            parallelism_context.tp != 1
            or parallelism_context.cp != 1
            or parallelism_context.pp != 1
            or parallelism_context.ep <= 1
        ):
            raise ValueError("MegaMoE adapter requires TP=CP=PP=1 and EP>1")
        super()._parallelize(parallelism_context)
        # HyperParallel/native libraries remain optional for ordinary recipes.
        from hyper_parallel.core.multicore import MegaMoeExperts

        group = parallelism_context.get_mesh("ep").get_group()
        self._executor = MegaMoeExperts(
            hidden_size=self.w13.in_features,
            intermediate_size=self.w2.in_features,
            num_experts=self.w13.group_size,
            top_k=self.top_k,
            ep_group=group,
            create_parameters=False,
            expert_weight_layout="EOI",
            dispatch_mode=self.dispatch_mode,
        )
        self._execution_group = _ExecutionGroup.register(self, parallelism_context)
        logger.info(
            "Hyper MegaMoE bound: EP=%s, experts=%s, top_k=%s, layout=EOI, mode=%s",
            group.size(),
            self.w13.group_size,
            self.top_k,
            self.dispatch_mode,
        )

    def close(self) -> None:
        """Release execution resources after backward; keep owned parameters intact."""
        executor = self._executor
        if executor is not None:
            executor.close()
            self._executor = None
        if self._execution_group is not None:
            self._execution_group.members.discard(self)
            self._execution_group = None

    def forward(
        self,
        x_TD: torch.Tensor,
        topk_scores_TK: torch.Tensor,
        topk_expert_ids_TK: torch.Tensor,
        num_local_tokens_per_expert_E: torch.Tensor,
        *,
        padding_mask_T: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Consume the unchanged router's IDs and differentiable scores.

        Like the stock RoutedExperts, compute padding rows too; the model loss
        masks them. Let MegaMoE derive its exact dispatch histogram from IDs.
        No EP gradient division is added: TorchTitan normalizes by global tokens.
        """
        del num_local_tokens_per_expert_E, padding_mask_T
        executor = self._executor
        if executor is None:
            raise RuntimeError("Parallelize the MegaMoE adapter before forward")
        if self._execution_group is not None:
            self._execution_group.prepare()
        # Native kernels are an opaque local compute region to spmd_types. The
        # enclosing routed-expert boundary still checks input/output placements.
        with spmd.no_typecheck():
            output = executor(
                x_TD.bfloat16().contiguous(),
                topk_ids=topk_expert_ids_TK.contiguous(),
                topk_weights=topk_scores_TK.float().contiguous(),
                expert_weights=(
                    self.w13.weight.bfloat16().flatten(1, 2).contiguous(),
                    self.w2.weight.bfloat16().contiguous(),
                ),
            ).to(x_TD.dtype)
        if spmd.is_type_checking():
            spmd.assert_type_like(output, x_TD)
        return output


@override(
    target=RoutedExperts.Config,
    exact=True,
    description="Hyper MegaMoE dispatch, expert compute and combine with native weights",
)
def hyper_megamoe(
    cfg: RoutedExperts.Config, *, dispatch_mode: Literal["push", "pull"] = "push"
) -> MegaMoeRoutedExperts.Config:
    """Keep weight configs/sharding; reject unsupported expert math explicitly."""
    if type(cfg.activation_fn) is not SwiGLU.Config:
        raise ValueError("Hyper MegaMoE override supports plain SwiGLU experts only")
    if cfg.output_postprocess is not None:
        raise ValueError(
            "Hyper MegaMoE override does not support expert postprocessing"
        )
    if (
        type(cfg.w13) is not GroupedLinear.Config
        or type(cfg.w2) is not GroupedLinear.Config
    ):
        raise ValueError("Hyper MegaMoE requires unmodified GroupedLinear projections")
    return derive(
        cfg,
        MegaMoeRoutedExperts.Config,
        top_k=cfg.token_dispatcher.top_k,
        dispatch_mode=dispatch_mode,
    )
