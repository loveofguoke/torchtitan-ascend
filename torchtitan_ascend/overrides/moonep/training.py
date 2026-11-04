# SPDX-License-Identifier: Apache-2.0
"""Bind MoonEP initialization and teardown to the existing training lifecycle."""

from dataclasses import dataclass, field, replace
from typing import Any, cast
from unittest.mock import patch

from torch.distributed.tensor import Shard
from torchtitan.components.checkpointer.utils import canonical_fqn
from torchtitan.components.optim import DistMuon
from torchtitan.config import derive
from torchtitan.distributed import fsdp
from torchtitan.distributed.flex_shard import ComputeLayout
from torchtitan.models.common.moe import RoutedExperts
from torchtitan.models.kimi_k3.model import KimiK3Model
from torchtitan.trainer import Trainer
from torchtitan.training_engine import TrainingEngine

from .experts import MoonEPRoutedExperts, MoonEPTokenDispatcher
from .runtime import close_runtime, current_runtime
from .settings import MoonEPConfig


def _exclude_experts(fully_shard):
    def wrapped(target, **kwargs):
        saved_expert_weights = {}
        try:
            for model_root in target if isinstance(target, list | tuple) else [target]:
                for module in model_root.modules():
                    if isinstance(module, MoonEPRoutedExperts):
                        for expert_linear in (module.w13, module.w2):
                            saved_expert_weights.setdefault(
                                expert_linear, expert_linear.weight
                            )
                            expert_linear.register_parameter("weight", None)
            return fully_shard(target, **kwargs)
        finally:
            for expert_linear, expert_weight in saved_expert_weights.items():
                expert_linear.register_parameter("weight", expert_weight)

    return wrapped


def _expert_optimizer_layout(optimizer_config, expert_weight_fqns):
    expert_compute_layout = ComputeLayout(shardings_by_mesh_axis={"ep": Shard(0)})
    return replace(
        optimizer_config,
        optimizers=[
            (
                replace(
                    optimizer,
                    compute_sharding_by_fqn={
                        **optimizer.compute_sharding_by_fqn,
                        **dict.fromkeys(expert_weight_fqns, expert_compute_layout),
                    },
                )
                if isinstance(optimizer, DistMuon.Config)
                else optimizer
            )
            for optimizer in optimizer_config.optimizers
        ],
    )


def _topology_tokens(parallelism, num_tokens_per_dp_rank):
    """Fold dense token partitions into EP, retaining one owner per expert shard."""
    if parallelism.pipeline_parallel_degree != 1:
        raise ValueError("Ascend MoonEP currently requires PP=1")
    if (
        parallelism.expert_parallel_degree <= 1
        or parallelism.data_parallel_replicate_degree != 1
    ):
        raise ValueError("Ascend MoonEP requires EP > 1 and DP replicate = 1")
    if min(parallelism.tensor_parallel_degree, parallelism.context_parallel_degree) < 1:
        raise ValueError("MoonEP TP and CP degrees must be positive")
    num_token_partitions = (
        parallelism.tensor_parallel_degree * parallelism.context_parallel_degree
    )
    if (
        parallelism.data_parallel_shard_degree * num_token_partitions
        != parallelism.expert_parallel_degree
    ):
        raise ValueError(
            "Resident MoonEP experts require DP shard * TP * CP = EP (no expert data parallelism)"
        )
    return (num_tokens_per_dp_rank + num_token_partitions - 1) // num_token_partitions


class MoonEPTrainingEngine(TrainingEngine):
    """Bind resident homes before optimizers; keep dense FSDP and ordinary checkpointing."""

    model_parts: list[Any]
    config: Any

    def _initialize_model(self, **kwargs):
        # Limit the collection change to this engine's model construction.
        with patch.object(fsdp, "fully_shard", _exclude_experts(fsdp.fully_shard)):
            return super()._initialize_model(**kwargs)

    def _initialize_optim(self):
        expert_weight_fqns = set()
        for model_part in self.model_parts:
            for module_fqn, module in model_part.named_modules():
                if isinstance(module, MoonEPRoutedExperts):
                    runtime = module.token_dispatcher.runtime
                    assert runtime is not None
                    module.weight_projections, home_parameters = runtime.bind(
                        module.w13.weight,
                        module.w2.weight,
                        module.token_dispatcher.ep_device_mesh,
                    )
                    for projection_name, parameter in zip(
                        ("w13", "w2"), home_parameters, strict=True
                    ):
                        getattr(module, projection_name).register_parameter(
                            "weight", parameter
                        )
                        expert_weight_fqns.add(
                            canonical_fqn(f"{module_fqn}.{projection_name}.weight")
                        )
        self.config.optim.optimizer = _expert_optimizer_layout(
            self.config.optim.optimizer, expert_weight_fqns
        )
        return super()._initialize_optim()

    def _initialize_forward_backward(self):
        current_runtime().start()
        return super()._initialize_forward_backward()

    def close(self):
        # Async checkpoint staging can still be reading resident parameters.
        super().close()
        close_runtime()


class MoonEPTrainer(Trainer):
    """Trainer with an explicit MoonEP resource owner."""

    engine_cls = MoonEPTrainingEngine

    @dataclass(kw_only=True, slots=True)
    class Config(Trainer.Config):
        moonep: MoonEPConfig = field(default_factory=MoonEPConfig)

        def __post_init__(self):
            config = cast(Trainer.Config, self)
            parallelism, training = config.parallelism, config.training
            num_max_tokens_per_rank = _topology_tokens(
                parallelism, training.num_tokens_per_microbatch_per_dp_rank
            )
            if parallelism.context_parallel_degree > 1 and isinstance(
                config.model, KimiK3Model.Config
            ):
                raise ValueError("Kimi K3 does not support context parallelism")
            if parallelism.tensor_parallel_degree > 1 and any(
                isinstance(optimizer, DistMuon.Config)
                for optimizer in config.optim.optimizer.optimizers
            ):
                raise ValueError("MoonEP TP with DistMuon requires TP=1")
            if (
                training.dtype != "bfloat16"
                or not training.disable_cuda_graphs
                or training.enable_cpu_offload
            ):
                raise ValueError(
                    "Ascend MoonEP requires BF16, disabled CUDA graphs and no CPU offload"
                )
            if config.create_seed_checkpoint or config.dist_moe_runtime is not None:
                raise ValueError(
                    "MoonEP cannot use seed-checkpoint creation or another EP runtime"
                )
            Trainer.Config.__post_init__(self)
            communication_options = self.moonep.options()
            for _, routed_experts, *_ in config.model.traverse(RoutedExperts.Config):
                routed_experts.token_dispatcher = derive(
                    routed_experts.token_dispatcher,
                    MoonEPTokenDispatcher.Config,
                    hidden_dim=routed_experts.w13.in_features,
                    num_max_tokens_per_rank=self.moonep.tokens_per_rank
                    or num_max_tokens_per_rank,
                    **communication_options,
                )
