# SPDX-License-Identifier: Apache-2.0
"""Check NPU PP initialization and track upstream K3 FSDP finalization."""

import logging
import sys
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def runtime(monkeypatch):
    """Provide only the external initialization boundaries."""
    modules = {}
    for name in (
        "torch",
        "torch.distributed",
        "torchtitan",
        "torchtitan.distributed",
        "torchtitan.distributed.utils",
    ):
        module = ModuleType(name)
        module.__path__ = []
        modules[name] = module
        monkeypatch.setitem(sys.modules, name, module)
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(modules[parent], child, module)

    state = SimpleNamespace(initialized=False, original_calls=[])
    dist = modules["torch.distributed"]
    utils = modules["torchtitan.distributed.utils"]
    dist.is_initialized = lambda: state.initialized
    utils.device_type = "npu"
    utils.dist_config = SimpleNamespace(pipeline_per_edge_p2p=False)
    utils.logger = logging.getLogger(__name__)

    def original_init(*args, pipeline_parallel_degree=1, **kwargs):
        state.original_calls.append((args, pipeline_parallel_degree, kwargs))
        if state.initialized:
            return "already-initialized"
        # Mirror the upstream flag assignment that the patch must override.
        utils.dist_config.pipeline_per_edge_p2p = pipeline_parallel_degree > 1
        return "upstream"

    utils.init_distributed = original_init

    state.utils = utils
    state.comm = SimpleNamespace(backend="hccl")
    # Install this module's patch only; setup() would also apply the torch-level
    # patches, which need a real torch.
    from torchtitan_ascend.patches import pipeline_parallel

    pipeline_parallel.apply()
    return state


def test_npu_pp_keeps_shared_process_group(runtime):
    result = runtime.utils.init_distributed(runtime.comm, pipeline_parallel_degree=2)

    assert result == "upstream"
    assert runtime.original_calls == [((runtime.comm, False, "", None), 2, {})]
    assert runtime.utils.dist_config.pipeline_per_edge_p2p is False


def test_npu_pp_forwards_initialization_arguments(runtime):
    runtime.utils.init_distributed(
        runtime.comm, True, "/tmp/dump", [0, 2], pipeline_parallel_degree=2
    )

    assert runtime.original_calls[0][0] == (
        runtime.comm,
        True,
        "/tmp/dump",
        [0, 2],
    )


def test_npu_pp_logs_shared_process_group(runtime, caplog):
    with caplog.at_level(logging.WARNING):
        runtime.utils.init_distributed(runtime.comm, pipeline_parallel_degree=2)

    assert "shared process group" in caplog.text


def test_non_npu_initialization_retains_upstream_behavior(runtime):
    runtime.utils.device_type = "cuda"
    result = runtime.utils.init_distributed(runtime.comm, pipeline_parallel_degree=2)

    assert result == "upstream"
    assert len(runtime.original_calls) == 1
    assert runtime.utils.dist_config.pipeline_per_edge_p2p is True


def test_existing_process_group_is_not_reconfigured(runtime):
    runtime.initialized = True
    runtime.utils.dist_config.pipeline_per_edge_p2p = True
    result = runtime.utils.init_distributed(runtime.comm, pipeline_parallel_degree=2)

    assert result == "already-initialized"
    assert runtime.utils.dist_config.pipeline_per_edge_p2p is True


@pytest.mark.parametrize("backend", ["fake", "real_pp_fake_spmd"])
def test_fake_topology_receives_actual_pp_degree(runtime, backend):
    runtime.comm.backend = backend
    runtime.utils.init_distributed(runtime.comm, pipeline_parallel_degree=4)

    assert runtime.original_calls[0][1] == 4
    assert runtime.utils.dist_config.pipeline_per_edge_p2p is False


@pytest.mark.parametrize(
    "has_backward,is_fsdp,expected",
    [
        (True, True, True),
        (False, True, False),
        (True, False, None),
    ],
)
def test_k3_manual_finalization_precedes_forward(has_backward, is_fsdp, expected):
    """Verify the K3 stage enables deferred FSDP finalization before forward."""
    stage_module = pytest.importorskip(
        "torchtitan.models.kimi_k3.pipeline_parallel.stage"
    )
    from torch.distributed.fsdp import FSDPModule

    class FSDPProbe(FSDPModule):
        manual_finalization = False

        def set_manual_backward_finalization(self, enabled):
            self.manual_finalization = enabled

    class ForwardReachedError(Exception):
        pass

    forwarded = []

    def forward(*args, **kwargs):
        forwarded.append((args, kwargs))
        raise ForwardReachedError

    # Stop at the model forward boundary: this tests the real upstream method
    # without creating process groups, tensors, or a fully sharded model.
    submod = object.__new__(FSDPProbe) if is_fsdp else object()
    stage = SimpleNamespace(
        store=lambda: None,
        is_first=True,
        submod=submod,
        has_backward=has_backward,
        forward_maybe_with_nosync=forward,
    )
    with pytest.raises(ForwardReachedError):
        stage_module.AttnResPipelineStage.forward_one_chunk(
            stage, 3, ("input",), {"key": "value"}
        )

    assert forwarded == [(("input",), {"key": "value"})]
    assert getattr(submod, "manual_finalization", None) is expected
