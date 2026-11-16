# SPDX-License-Identifier: Apache-2.0
"""MoonEP topology validation before and after automatic DP resolution."""

from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

import pytest

from torchtitan_ascend import setup

setup()

from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.training_engine import TrainingEngine

from torchtitan_ascend.overrides.moonep.training import (
    _topology_tokens,
    MoonEPTrainingEngine,
)


@pytest.mark.parametrize("dp_shard", [-1, 2])
def test_auto_and_explicit_dp_have_the_same_token_capacity(dp_shard):
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=dp_shard,
        context_parallel_degree=2,
        expert_parallel_degree=4,
    )
    assert _topology_tokens(parallelism, 2049) == 1025


def test_explicit_expert_data_parallelism_is_rejected():
    parallelism = ParallelismConfig(
        data_parallel_shard_degree=4, expert_parallel_degree=2
    )
    with pytest.raises(ValueError, match="no expert data parallelism"):
        _topology_tokens(parallelism, 2048)


@pytest.mark.parametrize("resolved_dp_shard", [2, 4])
def test_runtime_checks_the_resolved_dp_shard(resolved_dp_shard):
    engine = cast(MoonEPTrainingEngine, object.__new__(MoonEPTrainingEngine))
    engine.config = SimpleNamespace(
        parallelism=ParallelismConfig(expert_parallel_degree=2),
        training=SimpleNamespace(num_tokens_per_microbatch_per_dp_rank=2048),
    )
    engine.parallelism_context = SimpleNamespace(
        dp_shard=resolved_dp_shard, tp=1, cp=1, ep=2
    )
    with patch.object(TrainingEngine, "_initialize_model") as initialize:
        if resolved_dp_shard == 2:
            engine._initialize_model(hf_assets_path="unused")
            initialize.assert_called_once_with(hf_assets_path="unused")
        else:
            with pytest.raises(ValueError, match="no expert data parallelism"):
                engine._initialize_model(hf_assets_path="unused")
            initialize.assert_not_called()
    assert engine.config.parallelism.data_parallel_shard_degree == -1


@pytest.mark.parametrize(
    "schedule",
    [
        "1F1B",
        "Interleaved1F1B",
        "GPipe",
        "DualPipeV",
        "ZBVZeroBubble",
        "InterleavedZeroBubble",
    ],
)
@pytest.mark.parametrize("tp,cp", [(1, 1), (1, 2), (1, 4), (2, 1), (2, 2)])
@pytest.mark.parametrize("replicate", [1, 2])
@pytest.mark.parametrize("enable_sp", [False, True])
def test_parallel_recipe_composition(tp, cp, replicate, enable_sp, schedule):
    from torchtitan.components.optim import AdamW, OptimizersContainer
    from torchtitan.config import apply_overrides
    from torchtitan.models.common.attention.kda import KDA

    from torchtitan_ascend.overrides.moonep.experts import MoonEPRoutedExperts
    from torchtitan_ascend_recipes.models.kimi_k3 import (
        apply_kimi_cp,
        kimi_k3_7b_moonep,
    )

    config = kimi_k3_7b_moonep(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=8,
            expert_parallel_degree=8,
            data_parallel_replicate_degree=replicate,
            enable_sequence_parallel=enable_sp,
            pipeline_parallel_degree=2,
            pipeline_parallel_schedule=schedule,
            num_pp_microbatches=8,
        )
    )
    if tp > 1:
        config.optim.optimizer = OptimizersContainer.Config(
            optimizers=[AdamW.Config(pattern=".*", lr=1e-5)]
        )
    config.parallelism.data_parallel_shard_degree = 8 // (tp * cp)
    config.parallelism.tensor_parallel_degree = tp
    config.parallelism.context_parallel_degree = cp
    if cp > 1:
        config = apply_kimi_cp(config)
    apply_overrides(config.override, config)
    config.__post_init__()
    experts = list(config.model.traverse(MoonEPRoutedExperts.Config))
    assert len(experts) == 12
    assert all(
        expert.token_dispatcher.num_max_tokens_per_rank == 2048 // (tp * cp)
        for _, expert, *_ in experts
    )
    assert config.parallelism.data_parallel_replicate_degree == replicate
    assert config.parallelism.pipeline_parallel_degree == 2
    assert config.parallelism.pipeline_parallel_schedule == schedule
    assert config.parallelism.enable_sequence_parallel is enable_sp
    if tp > 1:
        assert isinstance(config.optim.optimizer.optimizers[0], AdamW.Config)
    if cp > 1:
        assert all(
            type(layer.attention.inner_attention).__qualname__
            == "AscendUlyssesInnerKDA.Config"
            for layer in config.model.layers
            if isinstance(layer.attention, KDA.Config)
        )


def test_tp_distmuon_is_rejected():
    from torchtitan_ascend_recipes.models.kimi_k3 import kimi_k3_7b_moonep

    config = kimi_k3_7b_moonep(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=4,
            expert_parallel_degree=4,
        )
    )
    config.parallelism.data_parallel_shard_degree = 2
    config.parallelism.tensor_parallel_degree = 2
    with pytest.raises(ValueError, match="TP with DistMuon"):
        config.__post_init__()


@pytest.mark.parametrize("replicate", [1, 2])
def test_resident_replica_reduction_precedes_optimizer(replicate):
    import torch

    engine = cast(MoonEPTrainingEngine, object.__new__(MoonEPTrainingEngine))
    replica_mesh = SimpleNamespace(get_group=lambda: "replica-group")
    engine.parallelism_context = SimpleNamespace(
        get_optional_mesh=lambda name: replica_mesh if replicate > 1 else None
    )
    gradient = torch.tensor([0.25, 0.5], dtype=torch.bfloat16)
    parameter = SimpleNamespace(grad=SimpleNamespace(to_local=lambda: gradient))
    engine._expert_parameters = [parameter, SimpleNamespace(grad=None)]
    storage = gradient.data_ptr()
    trace = []

    def reduce_gradient(value, *, group):
        assert group == "replica-group" and value.dtype == torch.float32
        value.add_(torch.tensor([0.5, 1.0]))
        trace.append("reduce")

    def update():
        expected = [0.75, 1.5] if replicate > 1 else [0.25, 0.5]
        torch.testing.assert_close(gradient, gradient.new_tensor(expected))
        assert gradient.data_ptr() == storage
        trace.append("clip-and-update")
        return gradient.norm()

    with (
        patch(
            "torchtitan_ascend.overrides.moonep.training.dist.all_reduce",
            reduce_gradient,
        ),
        patch.object(TrainingEngine, "optim_step", side_effect=update),
    ):
        engine.optim_step()
    assert trace == (
        ["reduce", "clip-and-update"] if replicate > 1 else ["clip-and-update"]
    )


@pytest.mark.parametrize("virtual_stages_per_rank", [2, 4, 6])
def test_virtual_pipeline_splits_keep_experts_in_every_stage(virtual_stages_per_rank):
    from torchtitan.distributed.pipeline_parallel import (
        _generate_llm_fqn_per_model_part,
    )
    from torchtitan.models.common.moe import RoutedExperts

    from torchtitan_ascend_recipes.models.kimi_k3 import kimi_k3_debugmodel_npu

    model = kimi_k3_debugmodel_npu().model
    num_stages = 2 * virtual_stages_per_rank
    parts = _generate_llm_fqn_per_model_part(num_stages, len(model.layers), 0, 0)
    assert len(parts) == num_stages
    expert_layers = {
        f"layers.{index}"
        for index, layer in enumerate(model.layers)
        if list(layer.traverse(RoutedExperts.Config))
    }
    assert all(expert_layers.intersection(part) for part in parts)


@pytest.mark.parametrize("split_backward", [False, True])
@pytest.mark.parametrize("activation_remat", [None, object()])
@pytest.mark.parametrize("fail", [False, True])
def test_split_pipeline_restores_backward_settings(
    split_backward, activation_remat, fail
):
    import torch._functorch.config as functorch_config
    from torch.distributed.pipelining import stage as pipeline_stage
    from torch_remat import _api as remat_api

    engine = cast(MoonEPTrainingEngine, object.__new__(MoonEPTrainingEngine))
    engine._split_backward = split_backward
    engine.config = SimpleNamespace(activation_remat=activation_remat)
    load_input = remat_api._load_saved_input
    weight_backward = pipeline_stage.stage_backward_weight
    patch_remat = split_backward and activation_remat is not None

    def execute(_engine, **kwargs):
        assert functorch_config.donated_buffer is not split_backward
        assert (remat_api._load_saved_input is not load_input) == patch_remat
        assert (
            pipeline_stage.stage_backward_weight is not weight_backward
        ) == patch_remat
        if fail:
            raise RuntimeError("backward failed")
        return "completed"

    with (
        functorch_config.patch(donated_buffer=True),
        patch.object(TrainingEngine, "forward_backward", execute),
    ):
        if fail:
            with pytest.raises(RuntimeError, match="backward failed"):
                engine.forward_backward()
        else:
            assert engine.forward_backward() == "completed"
        assert functorch_config.donated_buffer is True
        assert remat_api._load_saved_input is load_input
        assert pipeline_stage.stage_backward_weight is weight_backward


@pytest.mark.parametrize(
    "schedule,kimi,cache",
    [
        ("ZBVZeroBubble", True, False),
        ("DualPipeV", True, False),
        ("ZBVZeroBubble", False, True),
        ("GPipe", True, True),
        ("InterleavedZeroBubble", True, True),
    ],
)
def test_v_schedule_uses_native_kimi_whole_stack_transport(schedule, kimi, cache):
    from torchtitan.models.kimi_k3.model import KimiK3Model
    from torchtitan.models.llama3.model import Llama3Model

    engine = cast(MoonEPTrainingEngine, object.__new__(MoonEPTrainingEngine))
    config_cls = KimiK3Model.Config if kimi else Llama3Model.Config
    engine.model_config = object.__new__(config_cls)
    engine.parallelism_context = SimpleNamespace(pp_enabled=True)
    engine.config = SimpleNamespace(
        parallelism=ParallelismConfig(pipeline_parallel_schedule=schedule)
    )
    regions = ["attention"]
    with patch.object(
        TrainingEngine,
        "_parallelize_compile_kwargs",
        return_value={"local_compile_regions": regions},
    ):
        options = engine._parallelize_compile_kwargs()
    assert options.get("attn_res_cache", True) is cache
    assert options["local_compile_regions"] is regions
