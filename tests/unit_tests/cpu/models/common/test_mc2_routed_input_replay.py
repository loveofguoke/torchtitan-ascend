# SPDX-License-Identifier: Apache-2.0

import importlib.util
import sys
from types import ModuleType

import torch


def test_mc2_replay_rebuilds_expert_major_rows(monkeypatch):
    spec = importlib.util.find_spec("torchtitan_ascend.overrides.common.moe.mc2_gmm")
    assert spec is not None and spec.loader is not None
    mc2 = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, "torch_npu", ModuleType("torch_npu"))
    spec.loader.exec_module(mc2)

    tokens_TD = torch.arange(8, dtype=torch.float32).view(4, 2)
    inverse_indices_N = torch.tensor([3, 0, 5, 1, 4, 2, 7, 6], dtype=torch.int32)
    counts_Pe = torch.tensor([[2, 1, 0], [1, 0, 3]])
    group = object()
    recv_from_peer = torch.arange(100, 108, dtype=torch.bfloat16).view(4, 2)
    all_to_all_calls = []

    def exchange(output, inputs, output_split_sizes, input_split_sizes, *, group):
        all_to_all_calls.append(group)
        assert output_split_sizes == [3, 4]
        assert input_split_sizes == [3, 5]
        expected_send = tokens_TD[[0, 1, 2, 0, 2, 1, 3, 3]].bfloat16()
        torch.testing.assert_close(inputs, expected_send, rtol=0, atol=0)
        output.copy_(torch.cat((inputs[:3], recv_from_peer)))

    monkeypatch.setattr(mc2.dist, "all_to_all_single", exchange)
    rows_RD = mc2._replay_routed_input(
        tokens_TD, inverse_indices_N, 2, counts_Pe, [3, 5], [3, 4], group
    )
    assert all_to_all_calls == [group]
    expected_rows = torch.cat(
        (
            tokens_TD[:2].bfloat16(),
            recv_from_peer[:1],
            tokens_TD[2:3].bfloat16(),
            recv_from_peer[1:],
        )
    )
    torch.testing.assert_close(rows_RD, expected_rows, rtol=0, atol=0)
