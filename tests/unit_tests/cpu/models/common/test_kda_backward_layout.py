# SPDX-License-Identifier: Apache-2.0
"""Check packed TND KDA inputs, saved caches and model gradients."""

import sys
from types import SimpleNamespace

import pytest
import torch

from torchtitan_ascend.overrides.common.attention.kda import chunk_kda


@pytest.mark.parametrize("normalize", [False, True], ids=["raw-qk", "normalized-qk"])
@pytest.mark.parametrize("sigmoid", [False, True], ids=["raw-beta", "sigmoid-beta"])
def test_kda_backward_layout(monkeypatch, normalize, sigmoid):
    # Different head and token counts expose accidental transposes. The slices
    # also exercise the adapter's contiguous packed input requirement.
    shape = (1, 7, 3, 128)
    q, k, v, g = [
        torch.randn(*shape[:-1], 256)[..., ::2].requires_grad_() for _ in range(4)
    ]
    beta = torch.randn(shape[:-1], requires_grad=True)
    grad_out = torch.randn_like(v)
    A_log = torch.randn(3, requires_grad=True)
    dt_bias = torch.randn(3 * 128, requires_grad=True)

    def head_major(x):
        return x.squeeze(0).transpose(0, 1)

    def forward(
        q_in,
        k_in,
        v_in,
        g_in,
        beta_in,
        *,
        layout,
        q_hat_out,
        k_hat_out,
        q_rstd_out,
        k_rstd_out,
        beta_eff_out,
        **kwargs
    ):
        assert layout == "TND"
        for packed, model in zip(
            (q_in, k_in, v_in, g_in, beta_in), (q, k, v, g, beta), strict=True
        ):
            assert packed.is_contiguous()
            torch.testing.assert_close(packed, model.squeeze(0))
        if normalize:
            # FLA exports its normalized Q/K and reciprocal norms head-major
            # for both input layouts. Use distinct values to catch swapped caches.
            q_hat_out.copy_(head_major(q) + 1)
            k_hat_out.copy_(head_major(k) + 2)
            q_rstd_out.copy_(head_major(beta))
            k_rstd_out.copy_(head_major(beta) + 3)
        else:
            assert q_hat_out is k_hat_out is q_rstd_out is k_rstd_out is None
        if sigmoid:
            beta_eff_out.copy_(head_major(beta.sigmoid()))
        else:
            assert beta_eff_out is None
        return (v.squeeze(0).contiguous(),) + (None,) * 11

    def backward(
        q_in,
        k_in,
        v_in,
        effective_beta,
        *args,
        raw_g,
        q_rstd,
        k_rstd,
        A_log,
        dt_bias,
        layout,
        **kwargs
    ):
        assert layout == "TND"
        expected_beta = beta.sigmoid() if sigmoid else beta
        torch.testing.assert_close(
            q_in, head_major(q) + 1 if normalize else q.squeeze(0)
        )
        torch.testing.assert_close(
            k_in, head_major(k) + 2 if normalize else k.squeeze(0)
        )
        torch.testing.assert_close(v_in, v.squeeze(0))
        torch.testing.assert_close(effective_beta, expected_beta.squeeze(0))
        torch.testing.assert_close(raw_g, g.squeeze(0))
        torch.testing.assert_close(args[-2], grad_out.squeeze(0))  # d_o precedes scale
        assert dt_bias.shape == (3, 128)
        if normalize:
            torch.testing.assert_close(q_rstd, head_major(beta))
            torch.testing.assert_close(k_rstd, head_major(beta) + 3)
        else:
            assert q_rstd is k_rstd is None
        return (
            q.squeeze(0),
            k.squeeze(0),
            v_in,
            effective_beta,
            raw_g,
            None,
            A_log,
            dt_bias,
        )

    ops = SimpleNamespace(npu_chunk_kda_fwd=forward, npu_chunk_kda_bwd=backward)
    monkeypatch.setitem(sys.modules, "fla_npu.ops", SimpleNamespace(ascendc=ops))
    out, _ = chunk_kda(
        q,
        k,
        v,
        g,
        beta,
        cu_seqlens=[0, 2, 7],
        use_gate_in_kernel=True,
        use_qk_l2norm_in_kernel=normalize,
        use_beta_sigmoid_in_kernel=sigmoid,
        A_log=A_log,
        dt_bias=dt_bias,
    )
    torch.testing.assert_close(out, v)
    gradients = torch.autograd.grad(out, (q, k, v, g, beta, A_log, dt_bias), grad_out)

    beta_eff = beta.sigmoid()
    beta_gradient = beta_eff * beta_eff * (1 - beta_eff) if sigmoid else beta
    for actual, expected in zip(
        gradients, (q, k, v, g, beta_gradient, A_log, dt_bias), strict=True
    ):
        torch.testing.assert_close(actual, expected)
