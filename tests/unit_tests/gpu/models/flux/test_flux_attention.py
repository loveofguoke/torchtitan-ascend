# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-3-Clause file in the root directory of this source tree.

import unittest
from typing import TYPE_CHECKING

import pytest


# Load the NPU backend before spmd_types so spawned imports cannot re-enter it.
import torch  # usort: skip
import torch_npu  # noqa: F401  # usort: skip
import spmd_types as spmd  # usort: skip
import torch.nn.functional as F
from spmd_types.checker import typecheck
from torch.nn.attention import sdpa_kernel, SDPBackend

from torchtitan_ascend import setup

# Native DTensor workers import this module without pytest conftest.
setup()

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

from torchtitan.distributed.parallelism_context import ParallelismContext  # noqa: E402
from torchtitan.models.common.attention import VarlenInnerAttention  # noqa: E402
from torchtitan.models.flux.layers import _varlen_attention  # noqa: E402
from torchtitan.models.flux.model import _get_attention_metadata  # noqa: E402
from torchtitan.models.flux.sharding import (  # noqa: E402
    flux_inner_attention_sharding_config,
)


def _reference_attention(q, k, v):
    with sdpa_kernel(SDPBackend.MATH):
        return F.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=False
        ).transpose(1, 2)


if TYPE_CHECKING:
    _DTensorTestBase = unittest.TestCase
else:
    _DTensorTestBase = DTensorTestBase


@pytest.mark.multi_gpu
@unittest.skipUnless(torch.npu.device_count() >= 2, "requires two Ascend NPUs")
class TestFluxVarlenContextParallel(_DTensorTestBase):

    if TYPE_CHECKING:

        def assertEqual(  # noqa: N802 - PyTorch unittest API
            self,
            first: object,
            second: object,
            msg: str | None = None,
            *,
            atol: float | None = None,
            rtol: float | None = None,
        ) -> None:
            ...

    @property
    def device_type(self) -> str:
        return "npu"

    @property
    def world_size(self) -> int:
        return 2

    @with_comms
    def test_gathered_kv_preserves_batch_boundaries(self):
        parallelism_context = ParallelismContext(
            dp_replicate=1,
            dp_shard=1,
            cp=2,
            tp=1,
            pp=1,
            ep=1,
            world_size=2,
            enable_sequence_parallel=False,
        )
        config = VarlenInnerAttention.Config(
            window_size=(-1, -1), sharding_config=flux_inner_attention_sharding_config()
        )
        attention = config.build()
        with parallelism_context.activate_spmd():
            attention._parallelize(parallelism_context)

        torch.manual_seed(42)
        global_inputs = [
            torch.randn(2, 32, 4, 64, device=self.device_type, dtype=torch.bfloat16)
            for _ in range(3)
        ]
        local_inputs = [
            x.chunk(self.world_size, dim=1)[self.rank].clone().requires_grad_()
            for x in global_inputs
        ]
        reference_inputs = [x.float().requires_grad_() for x in global_inputs]
        expected = _reference_attention(*reference_inputs)
        expected[0].square().sum().backward()

        with parallelism_context.activate_spmd(), typecheck(local=False):
            for x in local_inputs:
                spmd.assert_type(x, {"dp": spmd.S(0), "cp": spmd.S(1)})
            metadata = _get_attention_metadata(
                config, 2, 16, device=local_inputs[0].device
            )
            actual = _varlen_attention(
                attention, *local_inputs, attention_metadata=metadata
            )
            actual[0].float().square().sum().backward()

        self.assertEqual(
            actual.float(),
            expected.chunk(self.world_size, dim=1)[self.rank],
            atol=0.01,
            rtol=0.01,
        )
        for actual_input, expected_input in zip(local_inputs, reference_inputs):
            assert actual_input.grad is not None
            assert expected_input.grad is not None
            self.assertEqual(
                actual_input.grad.float(),
                expected_input.grad.chunk(self.world_size, dim=1)[self.rank],
                atol=0.03,
                rtol=0.03,
            )
            self.assertEqual(torch.count_nonzero(actual_input.grad[1]).item(), 0)
