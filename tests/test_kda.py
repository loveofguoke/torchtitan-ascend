# SPDX-License-Identifier: Apache-2.0
"""Check the public KDA wrapper's autograd calling convention on CPU."""

import torch

from torchtitan_ascend.overrides.kda import npu_chunk_kda, NpuChunkKDAFunction


def test_kda_autograd_flags_are_positional(monkeypatch):
    calls = []

    def forward(ctx, *args):
        calls.append(args)
        return args[0].clone(), None

    monkeypatch.setattr(NpuChunkKDAFunction, "forward", forward)
    q = torch.randn(1, 4, 2, 128)
    out, state = npu_chunk_kda(
        q,
        q,
        q,
        q,
        q[..., 0],
        use_qk_l2norm_in_kernel=True,
        use_beta_sigmoid_in_kernel=True,
    )
    assert torch.equal(out, q)
    assert state is None
    assert len(calls[0]) == 19
    assert calls[0][-2:] == (True, True)
