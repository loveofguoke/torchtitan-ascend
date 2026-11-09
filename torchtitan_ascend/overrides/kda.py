# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# Copyright (c) 2026 Tianjin University
# SPDX-License-Identifier: BSD-3-Clause
"""Ascend implementations of Kimi Delta Attention components.

The model uses [1, T, H, D], with a singleton batch and packed tokens T.
AscendC uses NTD, where N is the head count H, internally.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module

import spmd_types as spmd
import torch
from torchtitan.config import derive, override
from torchtitan.models.base.module import Module
from torchtitan.models.common.attention.kda import InnerKDA, KDAKernel
from torchtitan.models.common.attention.linear_attention import LinearAttentionMetadata

from torchtitan_ascend.device import require_device
from torchtitan_ascend.models.common.linear_attention import causal_conv1d

__all__ = [
    "AscendInnerKDA",
    "AscendKDAKernel",
    "NpuChunkKDAFunction",
    "npu_kda",
]


def _l2norm(x):
    """Normalize the last dimension using FLA's FP32 semantics and epsilon."""
    x_f = x.float()
    return (x_f * torch.rsqrt(x_f.pow(2).sum(-1, keepdim=True) + 1e-6)).type_as(x)


def _to_ntd(x):
    """Convert [1, T, H, ...] to packed [H, T, ...]."""
    return x.squeeze(0).transpose(0, 1).contiguous()


def _from_ntd(x):
    """Restore packed [H, T, ...] to the model's [1, T, H, ...]."""
    return x.transpose(0, 1).unsqueeze(0).contiguous()


class NpuChunkKDAFunction(torch.autograd.Function):
    """Fused KDA forward/backward with Q/K L2Norm and beta sigmoid.

    Backward supports head dimension 128 and consumes packed NTD intermediates.
    """

    @staticmethod
    def forward(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        scale,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel,
    ):
        """Run fused KDA through the public AscendC interface for varlen inputs."""
        from fla_npu.ops import ascendc

        npu_chunk_kda_fwd = ascendc.npu_chunk_kda_fwd

        # aclnn rejects bf16 gate params with 161002; the ascendc kernels
        # consume A_log/dt_bias in fp32.
        A_log_in_dt, dt_bias_in_dt = A_log.dtype, dt_bias.dtype
        A_log = A_log.float()
        dt_bias = dt_bias.float()

        ctx.num_heads = q.shape[2]
        ctx.g_in_dt, ctx.beta_in_dt = g.dtype, beta.dtype
        q, k, v, g, beta = map(_to_ntd, (q, k, v, g, beta))

        # The forward kernel exports normalized Q/K and their inverse norms.
        q_n = torch.empty_like(q) if use_qk_l2norm_in_kernel else q
        k_n = torch.empty_like(k) if use_qk_l2norm_in_kernel else k
        q_rstd = (
            torch.empty(q.shape[:-1], dtype=torch.float32, device=q.device)
            if use_qk_l2norm_in_kernel
            else None
        )
        k_rstd = (
            torch.empty(k.shape[:-1], dtype=torch.float32, device=k.device)
            if use_qk_l2norm_in_kernel
            else None
        )
        beta_f = (
            torch.empty_like(beta, dtype=torch.float32)
            if use_beta_sigmoid_in_kernel
            else beta
        )

        (o, final_state, gk, Aqk, Akk, w, _, qg, kg, v_new, h, _,) = npu_chunk_kda_fwd(
            q,
            k,
            v,
            g,
            beta,
            layout="NTD",
            initial_state=initial_state,
            output_final_state=output_final_state,
            cu_seqlens=cu_seqlens,
            chunk_indices=chunk_indices,
            safe_gate=safe_gate,
            lower_bound=lower_bound,
            use_gate_in_kernel=use_gate_in_kernel,
            A_log=A_log,
            dt_bias=dt_bias,
            disable_recompute=True,
            scale=float(scale),
            chunk_size=chunk_size,
            use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
            use_exp2=True,
            q_hat_out=q_n if use_qk_l2norm_in_kernel else None,
            k_hat_out=k_n if use_qk_l2norm_in_kernel else None,
            q_rstd_out=q_rstd,
            k_rstd_out=k_rstd,
            beta_eff_out=beta_f if use_beta_sigmoid_in_kernel else None,
        )

        ctx.save_for_backward(
            q_n,
            k_n,
            v,
            gk,
            g,
            beta_f,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            qg,
            kg,
            v_new,
            h,
            q_rstd,
            k_rstd,
        )
        # Plain host data cannot go through save_for_backward.
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.scale = scale
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.A_log_in_dt = A_log_in_dt
        ctx.dt_bias_in_dt = dt_bias_in_dt
        ctx.dt_bias_shape = dt_bias.shape
        return o.unsqueeze(0).type_as(q), final_state

    @staticmethod
    def backward(ctx, do, dht):
        (
            q_n,
            k_n,
            v,
            gk,
            g,
            beta_f,
            A_log,
            dt_bias,
            Aqk,
            Akk,
            w,
            qg,
            kg,
            v_new,
            h,
            q_rstd,
            k_rstd,
        ) = ctx.saved_tensors
        cu_seqlens = ctx.cu_seqlens

        head_dim = q_n.shape[-1]
        assert head_dim == 128, "NPU KDA backward requires head_dim == 128"
        from fla_npu.ops import ascendc

        dq, dk, dv, db, dg, dh0, dA, dbias = ascendc.npu_chunk_kda_bwd(
            q_n,
            k_n,
            v,
            beta_f,
            gk,
            Aqk,
            Akk,
            w,
            qg,
            kg,
            v_new,
            h,
            _to_ntd(do),
            float(ctx.scale),
            raw_g=g if ctx.use_gate_in_kernel else None,
            A_log=A_log,
            dt_bias=dt_bias.reshape(ctx.num_heads, head_dim)
            if dt_bias is not None
            else None,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            cu_seqlens=cu_seqlens,
            chunk_indices=ctx.chunk_indices,
            disable_recompute=True,
            use_exp2=True,
            state_v_first=False,
            implementation="optimized",
            q_rstd=q_rstd,
            k_rstd=k_rstd,
        )

        # Apply the sigmoid derivative using the forward beta result.
        if ctx.use_beta_sigmoid_in_kernel:
            db = db.float() * beta_f * (1 - beta_f)

        dq, dk, dv, dg, db = map(_from_ntd, (dq, dk, dv, dg, db))

        return (
            dq.to(q_n),
            dk.to(k_n),
            dv.to(v),
            dg.to(ctx.g_in_dt),
            db.to(ctx.beta_in_dt),
            None,  # scale
            None,  # initial_state
            None,  # output_final_state
            None,  # state_v_first
            None,  # cu_seqlens
            None,  # chunk_indices
            None,  # chunk_size
            None,  # safe_gate
            None,  # lower_bound
            None,  # use_gate_in_kernel
            dA.to(ctx.A_log_in_dt) if dA is not None else None,  # A_log
            dbias.reshape(ctx.dt_bias_shape).to(ctx.dt_bias_in_dt)
            if dbias is not None
            else None,  # dt_bias
            None,  # use_qk_l2norm_in_kernel
            None,  # use_beta_sigmoid_in_kernel
        )


# KDA runs on rank-local heads inside InnerKDA's local SPMD region.
# Register its autograd function for debug.spmd_typechecking.
spmd.register_local_autograd_function(NpuChunkKDAFunction)


def npu_chunk_kda(
    q,
    k,
    v,
    g,
    beta,
    scale=None,
    initial_state=None,
    output_final_state=False,
    use_qk_l2norm_in_kernel=False,
    use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False,
    allow_neg_eigval=False,
    safe_gate=False,
    lower_bound=None,
    chunk_size=64,
    disable_recompute=False,
    return_intermediate_states=False,
    state_v_first=False,
    cu_seqlens=None,
    cu_seqlens_cpu=None,
    cp_context=None,
    **kwargs,
):
    """Run AscendC KDA on packed [1, T, H, 128] inputs.

    Without cu_seqlens, all tokens belong to one packed sequence.
    """
    if q.ndim != 4 or q.shape[0] != 1:
        raise ValueError("NPU KDA expects packed inputs with shape [1, T, H, D].")
    if q.shape[-1] != 128:
        raise ValueError(
            f"NPU KDA backend requires head_dim == 128, got {q.shape[-1]}."
        )
    if initial_state is not None or output_final_state:
        raise ValueError("KDA training does not support initial or final states.")
    A_log = kwargs.pop("A_log", None)
    dt_bias = kwargs.pop("dt_bias", None)
    if disable_recompute is not False:
        raise ValueError("NPU KDA backend always disables recompute.")
    if return_intermediate_states:
        raise ValueError("NPU KDA backend does not support intermediate states.")
    # fla_npu's wrapper truth-tests ``cu_seqlens`` and converts it on the
    # host, so hand it a plain list (the model passes a device tensor).
    if cu_seqlens_cpu is not None:
        cu_seqlens = cu_seqlens_cpu
    chunk_indices = None
    if cu_seqlens is None:
        cu_seqlens = [0, q.shape[1]]
    elif torch.is_tensor(cu_seqlens):
        cu_seqlens = [int(x) for x in cu_seqlens.cpu().tolist()]
    else:
        cu_seqlens = [int(x) for x in cu_seqlens]
    if len(cu_seqlens) < 2:
        raise ValueError(
            "cu_seqlens must contain at least the start and end boundaries."
        )
    # Remove empty documents while preserving nonempty sequence boundaries.
    cu_seqlens = [
        cu_seqlens[0],
        *[
            end
            for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True)
            if end != start
        ],
    ]
    # The backend derives the canonical sequence-major chunk table;
    # let it select the ordering by passing None.
    o, final_state = NpuChunkKDAFunction.apply(
        q,
        k,
        v,
        g,
        beta,
        scale if scale is not None else q.shape[-1] ** -0.5,
        initial_state,
        output_final_state,
        state_v_first,
        cu_seqlens,
        chunk_indices,
        chunk_size,
        safe_gate,
        lower_bound,
        use_gate_in_kernel,
        A_log,
        dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    return o, final_state


class AscendKDAKernel(KDAKernel):
    """``KDAKernel`` using fla_npu AscendC kernels with head_dim=128."""

    @dataclass(kw_only=True, slots=True)
    class Config(KDAKernel.Config):
        pass

    def __init__(self, config: Config):
        require_device("Ascend950", feature="fla-npu KDA")
        import_module("fla_npu")
        Module.__init__(self)
        self.lower_bound = config.lower_bound

    def forward(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        v_1THV: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run KDA with the gate, Q/K normalization, and beta sigmoid fused."""
        output_1THV, _ = npu_chunk_kda(
            q_1THK,
            k_1THK,
            v_1THV,
            raw_gate_1THK,
            raw_beta_1TH,
            A_log=A_log_H,
            dt_bias=dt_bias_HK.reshape(-1),
            use_qk_l2norm_in_kernel=True,
            use_gate_in_kernel=True,
            use_beta_sigmoid_in_kernel=True,
            safe_gate=self.lower_bound is not None,
            lower_bound=self.lower_bound,
            cu_seqlens=cu_seqlens,
        )
        return output_1THV

    def prepare_inputs(
        self,
        q_1THK: torch.Tensor,
        k_1THK: torch.Tensor,
        raw_gate_1THK: torch.Tensor,
        raw_beta_1TH: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Eager preprocessing shared by local and CP KDA cores.

        ``lower_bound * sigmoid(exp(A_log) * (raw_gate + dt_bias))`` in FP32 is
        the bounded gate transform; the Q/K L2 norm is the standard per-head
        normalization. The local kernel path folds both in itself; this
        composite stays for the context-parallel core.
        """
        gate_1THK = self.lower_bound * torch.sigmoid(
            torch.exp(A_log_H.float()).unsqueeze(-1).unsqueeze(0)
            * (raw_gate_1THK.float() + dt_bias_HK.float())
        )
        return (
            _l2norm(q_1THK),
            _l2norm(k_1THK),
            gate_1THK,
            raw_beta_1TH.float().sigmoid(),
        )


class AscendInnerKDA(InnerKDA):
    """``InnerKDA`` with the Ascend short convolution."""

    head_dim: int
    kernel: torch.nn.Module

    @dataclass(kw_only=True, slots=True)
    class Config(InnerKDA.Config):
        pass

    def __init__(self, config: Config):
        require_device("Ascend950", feature="fla-npu KDA convolution")
        Module.__init__(self)
        self.head_dim = config.head_dim
        self.kernel = config.kernel.build()

    def forward(
        self,
        query_TC: torch.Tensor,
        key_TC: torch.Tensor,
        value_TC: torch.Tensor,
        raw_gate_THK: torch.Tensor,
        raw_beta_TH: torch.Tensor,
        conv_q_weight_C1W: torch.Tensor,
        conv_k_weight_C1W: torch.Tensor,
        conv_v_weight_C1W: torch.Tensor,
        A_log_H: torch.Tensor,
        dt_bias_HK: torch.Tensor,
        *,
        attention_metadata: LinearAttentionMetadata | None,
    ) -> torch.Tensor:
        if (
            attention_metadata is not None
            and type(attention_metadata) is not LinearAttentionMetadata
        ):
            raise NotImplementedError(
                "fla-npu KDA does not support context-parallel attention metadata"
            )
        return super().forward(
            query_TC,
            key_TC,
            value_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            A_log_H,
            dt_bias_HK,
            attention_metadata=attention_metadata,
        )

    def short_convolution(
        self, qkv_1TC, conv_weight_C1W, initial_state=None, *, cu_seqlens, routing
    ):
        if initial_state is not None or routing is not None:
            raise NotImplementedError(
                "fla-npu KDA convolution does not support cached state "
                "or context parallelism"
            )
        return causal_conv1d(qkv_1TC.squeeze(0), conv_weight_C1W, cu_seqlens).unsqueeze(
            0
        )


@override(
    target=InnerKDA.Config,
    exact=True,
    description="KDA on Ascend: fla-npu causal convolution + AscendC chunk KDA",
)
def npu_kda(cfg: InnerKDA.Config) -> AscendInnerKDA.Config:
    """Swap the whole KDA subtree in one override.

    ``InnerKDA.Config`` owns ``kernel: KDAKernel.Config``, and torchtitan rejects
    an override that claims a node whose ancestor another override already claims
    (nested overrides are order-dependent). So the nested kernel config is derived
    here rather than by a second ``@override``.
    """
    new = derive(cfg, AscendInnerKDA.Config)
    new.kernel = derive(cfg.kernel, AscendKDAKernel.Config)
    return new
