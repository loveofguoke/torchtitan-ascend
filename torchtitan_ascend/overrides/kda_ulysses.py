# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""Ulysses KDA: exchange tokens for heads before the AscendC local kernels."""

from dataclasses import dataclass

import spmd_types as spmd
from torchtitan.config import derive, override
from torchtitan.distributed.parallelism_context import MeshAxisName
from torchtitan.distributed.spmd_types import spmd_mesh_group
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.attention.kda import InnerKDA

from torchtitan_ascend.overrides.kda import AscendInnerKDA, AscendKDAKernel


class AscendUlyssesInnerKDA(ContextParallelInnerKDA):
    """Run full packed sequences on disjoint heads within each CP group.

    The sequence/head exchange includes gate and beta projections. Convolution
    and gate parameters follow the same head partition. Their partial gradients
    are reduced over CP by the outer FSDP wrapper. Contiguous sharding preserves
    the global packed-document offsets, including cross-rank documents.
    """

    head_dim: int
    kernel: AscendKDAKernel
    run_stages = InnerKDA.run_stages

    @dataclass(kw_only=True, slots=True)
    class Config(ContextParallelInnerKDA.Config):
        pass

    def __init__(self, config: Config):
        AscendInnerKDA.__init__(self, config)

    @staticmethod
    def prepare_cp_metadata(attention_metadata, *, permutation):
        if permutation is not None:
            raise ValueError(
                "Ascend Ulysses KDA requires contiguous CP sharding without a load balancer."
            )
        return attention_metadata

    def short_convolution(self, *args, **kwargs):
        return AscendInnerKDA.short_convolution(self, *args, **kwargs)

    def kda_core(self, *args, **kwargs):
        return InnerKDA.kda_core(self, *args, **kwargs)

    def forward(
        self,
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
        *,
        attention_metadata,
    ):
        group = spmd_mesh_group(MeshAxisName.CP)
        if group is None:
            raise RuntimeError("Ascend Ulysses KDA requires an active CP group.")
        heads = raw_gate_THK.shape[1]
        if heads % group.size():
            raise ValueError(
                f"CP degree {group.size()} must divide the local KDA head count {heads}."
            )

        def tokens_to_heads(tensor):
            return spmd.redistribute(tensor, group, src=spmd.S(0), dst=spmd.S(1))

        query_TC, key_TC, value_TC = (
            tokens_to_heads(t.unflatten(-1, (heads, self.head_dim))).flatten(-2)
            for t in (query_TC, key_TC, value_TC)
        )
        raw_gate_THK = tokens_to_heads(raw_gate_THK)
        raw_beta_TH = tokens_to_heads(raw_beta_TH)

        def shard_heads(tensor):
            return spmd.redistribute(tensor, group, src=spmd.R, dst=spmd.S(0))

        conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W = (
            shard_heads(t.unflatten(0, (heads, self.head_dim))).flatten(0, 1)
            for t in (conv_q_weight_C1W, conv_k_weight_C1W, conv_v_weight_C1W)
        )
        output_THV = self.run_stages(
            query_TC,
            key_TC,
            value_TC,
            raw_gate_THK,
            raw_beta_TH,
            conv_q_weight_C1W,
            conv_k_weight_C1W,
            conv_v_weight_C1W,
            shard_heads(A_log_H),
            shard_heads(dt_bias_HK),
            cu_seqlens=attention_metadata.varlen.cu_seq_q
            if attention_metadata and attention_metadata.varlen
            else None,
            routing=None,
        )
        return spmd.redistribute(output_THV, group, src=spmd.S(1), dst=spmd.S(0))


@override(
    target=ContextParallelInnerKDA.Config,
    exact=True,
    description="KDA CP on Ascend: Ulysses exchange with AscendC kernels",
)
def npu_cp_kda(cfg: ContextParallelInnerKDA.Config) -> AscendUlyssesInnerKDA.Config:
    new = derive(cfg, AscendUlyssesInnerKDA.Config)
    new.kernel = derive(cfg.kernel, AscendKDAKernel.Config)
    return new
