"""Ascend implementations of Kimi Delta Attention components."""

from __future__ import annotations

from dataclasses import dataclass

import spmd_types as spmd
import torch
import torch.nn.functional as F
from torchtitan.config import derive, override
from torchtitan.models.kimi_k3.kda import InnerKDA, KDAKernel
from torchtitan.protocols.module import Module

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


def _l2norm_bwd(x, dx):
    """Backpropagate through ``_l2norm`` with FP32 accumulation.

    Uses the derivative of ``x / sqrt(sum(x**2) + 1e-6)``.
    """
    x_f = x.float()
    norm = torch.sqrt(x_f.pow(2).sum(-1, keepdim=True) + 1e-6)
    q = x_f / norm
    return ((dx.float() - q * (dx.float() * q).sum(-1, keepdim=True)) / norm).to(dx.dtype)


def _to_bnsd(x):
    """(1, T, H, D) -> (1, H, T, D)."""
    return x.transpose(1, 2).contiguous()


def _to_bsnd(x):
    """(1, H, T, D) -> (1, T, H, D)."""
    return x.transpose(1, 2).contiguous()


class NpuChunkKDAFunction(torch.autograd.Function):
    """Autograd wrapper over fla_npu's fused ascendc chunk_kda fwd/bwd.

    Forward: ``npu_chunk_kda_fwd`` -- 12 outputs aligned with FLA's
    ``chunk_kda_fwd``: (o, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h,
    initial_state), with ``disable_recompute=True`` (saved tensors are real
    outputs, no recompute).

    Backward: the fused ``npu_chunk_kda_bwd`` for K=V=128 (the fused path's
    only supported head_dim). Both directions use fla_npu's public AscendC
    interfaces, which select the Stable-ABI launcher by default. Backend
    selection and any compatibility fallback remain owned by fla_npu.

    Pre/post processing mirrors fla's ``_chunk_kda``: l2-norm on q/k
    (``use_qk_l2norm_in_kernel``), sigmoid on beta
    (``use_beta_sigmoid_in_kernel``), and their backward chain rules. The
    *original* q/k/beta are what the backward chain rules need, so they are
    what gets saved. fla_npu's kernels require g and beta in fp32 -- bf16
    inputs are rejected by aclnn with 161002 -- so the wrapper upcasts at the
    forward boundary and returns gradients in the caller's dtype.
    """

    @staticmethod
    def forward(
        ctx,
        q, k, v, g, beta,
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

        q_n = _l2norm(q) if use_qk_l2norm_in_kernel else q
        k_n = _l2norm(k) if use_qk_l2norm_in_kernel else k

        beta_raw = beta
        beta_f = beta_raw.float().sigmoid() if use_beta_sigmoid_in_kernel else beta_raw.float()

        g_f = g.float()

        (o, final_state, gk, Aqk, Akk, w, u, qg, kg, v_new, h, initial_state_out) = (
            npu_chunk_kda_fwd(
                q_n, k_n, v, g_f, beta_f,
                layout="BSND",
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
            )
        )

        ctx.save_for_backward(
            q, k, v, gk, g_f, beta_raw, beta_f,
            A_log, dt_bias, Aqk, Akk, w, u, qg, kg, v_new, h,
        )
        # Plain host data cannot go through save_for_backward.
        ctx.cu_seqlens = cu_seqlens
        ctx.chunk_indices = chunk_indices
        ctx.chunk_size = chunk_size
        ctx.safe_gate = safe_gate
        ctx.lower_bound = lower_bound
        ctx.scale = scale
        ctx.use_gate_in_kernel = use_gate_in_kernel
        ctx.use_qk_l2norm_in_kernel = use_qk_l2norm_in_kernel
        ctx.use_beta_sigmoid_in_kernel = use_beta_sigmoid_in_kernel
        ctx.A_log_in_dt = A_log_in_dt
        ctx.dt_bias_in_dt = dt_bias_in_dt
        return o.type_as(q), final_state

    @staticmethod
    def backward(ctx, do, dht):
        (q, k, v, gk, g_f, beta_raw, beta_f,
         A_log, dt_bias, Aqk, Akk, w, u, qg, kg, v_new, h) = ctx.saved_tensors
        cu_seqlens, chunk_indices = ctx.cu_seqlens, ctx.chunk_indices

        head_dim = q.shape[-1]
        assert head_dim == 128, "NPU KDA backward requires head_dim == 128"
        from fla_npu.ops import ascendc

        # Fused ascendc backward: K=V=128 only. Consumes BNSD tensors;
        # dt_bias is [H, K] here. Varlen calls take [H, T, D] (batchless)
        # inputs and return batchless gradients; dense calls take
        # [B, H, T, D].
        _is_varlen = cu_seqlens is not None

        # The backward must consume the *normalized* q/k the forward fed the
        # kernel -- the saved Aqk/Akk/... intermediates are built from them.
        # The raw q/k saved for backward are used only for the l2-norm chain
        # rule further down.
        q_n = _l2norm(q) if ctx.use_qk_l2norm_in_kernel else q
        k_n = _l2norm(k) if ctx.use_qk_l2norm_in_kernel else k

        def _bwd_in(x_BHTD):
            return x_BHTD[0] if _is_varlen else x_BHTD

        def _bwd_out(x_HTD):
            return x_HTD.unsqueeze(0) if _is_varlen else x_HTD

        dq, dk, dv, db, dg, dh0, dA, dbias = ascendc.npu_chunk_kda_bwd(
            _bwd_in(_to_bnsd(q_n)), _bwd_in(_to_bnsd(k_n)), _bwd_in(_to_bnsd(v)),
            _bwd_in(beta_f.transpose(1, 2).contiguous()),
            _bwd_in(gk), _bwd_in(Aqk), _bwd_in(Akk), _bwd_in(w),
            _bwd_in(qg), _bwd_in(kg), _bwd_in(v_new),
            # h from the fwd is (B, NT, H, K, V); the varlen backward takes
            # (NT, H, K, V), the dense backward (B, NT, H, K, V).
            _bwd_in(h),
            _bwd_in(_to_bnsd(do)), float(ctx.scale),
            raw_g=_bwd_in(_to_bnsd(g_f)) if ctx.use_gate_in_kernel else None,
            A_log=A_log,
            dt_bias=dt_bias.reshape(q.shape[2], head_dim) if dt_bias is not None else None,
            chunk_size=ctx.chunk_size,
            safe_gate=ctx.safe_gate,
            lower_bound=ctx.lower_bound,
            use_gate_in_kernel=ctx.use_gate_in_kernel,
            cu_seqlens=cu_seqlens,
        )

        dq = _to_bsnd(_bwd_out(dq))
        dk = _to_bsnd(_bwd_out(dk))
        dv = _to_bsnd(_bwd_out(dv))
        dg = _to_bsnd(_bwd_out(dg))
        db = _to_bsnd(_bwd_out(db))  # ascendc bwd returns BNSD; beta_raw is BSND
        if ctx.use_qk_l2norm_in_kernel:
            dq = _l2norm_bwd(q, dq)
            dk = _l2norm_bwd(k, dk)
        if ctx.use_beta_sigmoid_in_kernel:
            db = db * beta_raw.float().sigmoid() * (1 - beta_raw.float().sigmoid())

        return (
            dq.to(q), dk.to(k), dv.to(v), dg.to(g_f), db.to(beta_raw),
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
            dbias.reshape(-1).to(ctx.dt_bias_in_dt)
            if dbias is not None
            else None,  # dt_bias
            None,  # use_qk_l2norm_in_kernel
            None,  # use_beta_sigmoid_in_kernel
        )


# KDA runs on rank-local heads inside InnerKDA's local SPMD region.
# Register its autograd function for debug.spmd_typechecking.
spmd.register_local_autograd_function(NpuChunkKDAFunction)


def npu_chunk_kda(
    q, k, v, g, beta, scale=None, initial_state=None, output_final_state=False,
    use_qk_l2norm_in_kernel=False, use_gate_in_kernel=False,
    use_beta_sigmoid_in_kernel=False, allow_neg_eigval=False,
    safe_gate=False, lower_bound=None, chunk_size=64, disable_recompute=False,
    return_intermediate_states=False, state_v_first=False,
    cu_seqlens=None, cu_seqlens_cpu=None, cp_context=None, **kwargs,
):
    """Run fla_npu's fused AscendC KDA with head_dim=128."""
    if q.shape[-1] != 128:
        raise ValueError(
            f"NPU KDA backend requires head_dim == 128, got {q.shape[-1]}."
        )
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
    if cu_seqlens is not None:
        if torch.is_tensor(cu_seqlens):
            cu_seqlens = [int(x) for x in cu_seqlens.cpu().tolist()]
        else:
            cu_seqlens = [int(x) for x in cu_seqlens]
        # The backend derives the canonical sequence-major chunk table;
        # let it select the ordering by passing None.
    o, final_state = NpuChunkKDAFunction.apply(
        q, k, v, g, beta, scale if scale is not None else q.shape[-1] ** -0.5,
        initial_state, output_final_state, state_v_first,
        cu_seqlens, chunk_indices, chunk_size, safe_gate, lower_bound,
        use_gate_in_kernel, A_log, dt_bias,
        use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
        use_beta_sigmoid_in_kernel=use_beta_sigmoid_in_kernel,
    )
    return o, final_state



class _CausalConv1d(torch.autograd.Function):
    """Bind the fla-npu packed convolution to its explicit backward operator."""

    @staticmethod
    def forward(ctx, x, weight, offsets):
        from fla_npu.ops import ascendc

        output = ascendc.causal_conv1d_fn(
            x, weight, None, query_start_loc_cpu=offsets, activation=None
        )
        ctx.save_for_backward(x, weight)
        ctx.offsets = offsets
        return output

    @staticmethod
    def backward(ctx, grad_output):
        from fla_npu.ops import ascendc

        x, weight = ctx.saved_tensors
        dx, dw, *_ = ascendc.causal_conv1d_bwd(
            x,
            None,
            weight,
            grad_output.contiguous(),
            query_start_loc=ctx.offsets,
            activation=0,
            input_layout="TND",
        )
        return dx, dw, None


def ascend_causal_conv1d(
    x_BTC: torch.Tensor,
    weight_CW: torch.Tensor,
    *,
    activation: str | None = None,
    cu_seqlens: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run fla-npu depthwise causal convolution with packed-sequence boundaries.

    Inputs use [B, T, C] and weights use [C, W]. The AscendC operators consume
    packed [T, C] inputs and [W, C] weights; SiLU remains a PyTorch operation.
    """
    if x_BTC.dim() != 3:
        raise ValueError(f"expected [B, T, C], got {tuple(x_BTC.shape)}")
    if activation not in (None, "silu"):
        raise NotImplementedError(f"activation {activation!r} is not implemented on NPU")
    batch, tokens, channels = x_BTC.shape
    if cu_seqlens is not None:
        if batch != 1:
            raise ValueError("packed cu_seqlens requires batch size 1")
        offsets = cu_seqlens.tolist()
    else:
        offsets = [index * tokens for index in range(batch + 1)]
    weight_WC = weight_CW.T.to(x_BTC.dtype).contiguous()
    output = _CausalConv1d.apply(
        x_BTC.reshape(batch * tokens, channels).contiguous(), weight_WC, offsets
    ).reshape(batch, tokens, channels)
    return F.silu(output) if activation == "silu" else output


class AscendKDAKernel(KDAKernel):
    """``KDAKernel`` using fla_npu AscendC kernels with head_dim=128."""

    @dataclass(kw_only=True, slots=True)
    class Config(KDAKernel.Config):
        pass

    def __init__(self, config: Config):
        """Initialize without loading KDAKernel's unrelated attn_gym dependency."""
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

    @dataclass(kw_only=True, slots=True)
    class Config(InnerKDA.Config):
        pass

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
        cu_seqlens: torch.Tensor | None,
    ) -> torch.Tensor:
        mixed_qkv_BTC = torch.cat((query_TC, key_TC, value_TC), dim=-1).unsqueeze(0)
        conv_weight_C1W = torch.cat(
            (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W), dim=0
        )
        conv_output_BTC = ascend_causal_conv1d(
            mixed_qkv_BTC,
            conv_weight_C1W[:, 0],
            activation="silu",
            cu_seqlens=cu_seqlens,
        )
        q_BTC, k_BTC, v_BTC = conv_output_BTC.chunk(3, dim=-1)
        q_BTNK, k_BTNK, v_BTNV = (
            tensor.unflatten(-1, (-1, self.head_dim)) for tensor in (q_BTC, k_BTC, v_BTC)
        )
        output_BTNV = self.kernel(
            q_BTNK,
            k_BTNK,
            v_BTNV,
            raw_gate_THK.unsqueeze(0),
            raw_beta_TH.unsqueeze(0),
            A_log_H,
            dt_bias_HK,
            cu_seqlens=cu_seqlens,
        )
        return output_BTNV.squeeze(0)


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
