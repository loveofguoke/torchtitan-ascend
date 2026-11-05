# SPDX-License-Identifier: Apache-2.0
"""Opt-in SiTU-GLU using the public CANN ops-nn forward and backward."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from torchtitan.config import derive, override
from torchtitan.models.common.activation import SiTUGLU

__all__ = [
    "AscendCSiTUGLU",
    "situ_glu",
    "ascendc_situ_glu",
]


def _operators():
    import cann_ops_nn

    # Import for operator registration; the public entry points live on the package.
    import cann_ops_nn.ops  # noqa: F401

    return cann_ops_nn.situ_glu, cann_ops_nn.situ_glu_grad


class _SiTUGLU(torch.autograd.Function):
    @staticmethod
    def forward(ctx, packed, beta, linear_beta):
        forward, _ = _operators()
        ctx.save_for_backward(packed)
        ctx.beta, ctx.linear_beta = beta, linear_beta
        return forward(
            packed, dim=-1, beta=beta, linear_beta=linear_beta, activate_left=True
        )

    @staticmethod
    def backward(ctx, dy):
        _, backward = _operators()
        (packed,) = ctx.saved_tensors
        dx = backward(
            dy.contiguous(),
            packed,
            dim=-1,
            beta=ctx.beta,
            linear_beta=ctx.linear_beta,
            activate_left=True,
        )
        return dx, None, None


def situ_glu(gate_up, beta, linear_beta):
    if gate_up.ndim < 2 or gate_up.shape[-2] != 2:
        raise ValueError("SiTU-GLU expects [..., 2, F] gate/up layout")
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("SiTU beta must be finite and positive")
    if linear_beta is not None and (not math.isfinite(linear_beta) or linear_beta <= 0):
        raise ValueError("SiTU linear_beta must be None or finite and positive")
    return _SiTUGLU.apply(
        gate_up.contiguous().flatten(-2),
        beta,
        0.0 if linear_beta is None else linear_beta,
    )


class AscendCSiTUGLU(SiTUGLU):
    beta: float
    linear_beta: float | None

    @dataclass(kw_only=True, slots=True)
    class Config(SiTUGLU.Config):
        pass

    def __call__(self, gate_up, *, offsets=None):
        # Like the upstream eager implementation, compute padded rows as well.
        del offsets
        return situ_glu(gate_up, self.beta, self.linear_beta)


@override(target=SiTUGLU.Config, exact=True, description="Ascend C SiTU-GLU")
def ascendc_situ_glu(config):
    return derive(config, AscendCSiTUGLU.Config)
