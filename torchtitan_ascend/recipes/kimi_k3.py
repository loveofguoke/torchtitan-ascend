# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""Kimi K3 on Ascend using upstream recipes and Ascend KDA kernels."""

from typing import Literal

from torchtitan.config import apply_overrides, derive, OverrideConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.config.transform import apply_transforms, ContextParallelTransform
from torchtitan.models.common.attention.cp_kda import ContextParallelInnerKDA
from torchtitan.models.common.attention.cp_mla import (
    AllGatherKVMLAFlexAttention,
    UlyssesMLAAttention,
)
from torchtitan.models.common.attention.kda import InnerKDA, KDA
from torchtitan.models.common.attention.mla import MLAAttention
from torchtitan.models.kimi_k3.flavors import _kimi_k3_config, _vision_encoder_config
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    kimi_k3_debugmodel,
)

from torchtitan_ascend.config import override_target
from torchtitan_ascend.overrides.moonep.experts import npu_moonep
from torchtitan_ascend.overrides.moonep.training import MoonEPTrainer

_GATED_RMSNORM_REGION = "gated_rmsnorm"


def npu_changes(config: Trainer.Config) -> None:
    """Apply Ascend KDA kernels and runtime settings."""
    from torchtitan_ascend.overrides.kda import npu_kda

    config.model.local_compile_regions = [
        region
        for region in config.model.local_compile_regions
        if region != _GATED_RMSNORM_REGION
    ]
    config.comm.train_timeout_seconds = max(config.comm.train_timeout_seconds, 600)

    config.override.imports = [
        *config.override.imports,
        override_target(npu_kda),
    ]


def npu_fusion(config: Trainer.Config) -> None:
    """Apply Ascend fused operators."""
    from torchtitan_ascend.overrides.attn_res import triton_attn_res
    from torchtitan_ascend.overrides.fusion_attention import ascendc_fusion_attention
    from torchtitan_ascend.overrides.rmsnorm import ascendc_rmsnorm
    from torchtitan_ascend.overrides.rotary_mul import enable_ascendc_rotary_mul
    from torchtitan_ascend.overrides.situ_glu import ascendc_situ_glu

    # Apply child activations before a routed-expert override can claim them.
    activation_target = override_target(ascendc_situ_glu)
    apply_overrides(OverrideConfig(imports=[activation_target]), config)
    config.override.imports = [
        *config.override.imports,
        override_target(ascendc_fusion_attention),
        override_target(ascendc_rmsnorm),
        override_target(triton_attn_res),
    ]
    enable_ascendc_rotary_mul()


def apply_kimi_cp(
    config: Trainer.Config,
    *,
    kda_cp_backend: Literal["ulysses"] = "ulysses",
    mla_cp_backend: Literal["ulysses", "allgather_kv"] = "ulysses",
) -> Trainer.Config:
    """Return a transformed copy using Ulysses KDA and the selected MLA backend."""
    from torchtitan_ascend.overrides.kda_ulysses import npu_cp_kda

    if kda_cp_backend != "ulysses":
        raise ValueError(f"Unknown Ascend KDA CP backend: {kda_cp_backend}")
    if mla_cp_backend not in {"ulysses", "allgather_kv"}:
        raise ValueError(f"Unknown MLA CP backend: {mla_cp_backend}")
    parallel = config.parallelism
    if parallel.context_parallel_degree < 2:
        raise ValueError("Ascend KDA CP requires context_parallel_degree >= 2")
    if parallel.context_parallel_load_balancer is not None:
        raise ValueError(
            "Ascend KDA CP requires contiguous partitions without a load balancer"
        )
    for layer in config.model.layers:
        if isinstance(layer.attention, KDA.Config):
            heads = layer.attention.n_heads
            partitions = (
                parallel.tensor_parallel_degree * parallel.context_parallel_degree
            )
            if heads % partitions:
                raise ValueError(
                    f"KDA heads ({heads}) must be divisible by TP * CP ({partitions}) for Ulysses"
                )

    mla_cp = (
        UlyssesMLAAttention
        if mla_cp_backend == "ulysses"
        else AllGatherKVMLAFlexAttention
    )
    transformed = apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    MLAAttention: mla_cp,
                    InnerKDA: ContextParallelInnerKDA,
                }
            )
        ],
    )
    transformed.override.imports.append(override_target(npu_cp_kda))
    return transformed


def kimi_k3_debugmodel_npu(
    *, parallelism: ParallelismConfig | None = None, attn_backend: str = "varlen"
) -> Trainer.Config:
    """Build the NPU debug recipe with DistMuon and multimodal test data."""
    config = kimi_k3_debugmodel(parallelism=parallelism, attn_backend=attn_backend)
    npu_changes(config)
    return config


def kimi_k3_7b_npu(
    *, parallelism: ParallelismConfig | None = None, attn_backend: str = "varlen"
) -> Trainer.Config:
    """Build the 13-layer, 32-expert NPU recipe with 2 MiB-aligned expert shards."""
    # Validate the requested topology after replacing the eight-expert debug model.
    config = kimi_k3_debugmodel(attn_backend=attn_backend)
    if parallelism is not None:
        config.parallelism = parallelism
    dim = 4608
    config.model = _kimi_k3_config(
        max_context_length=config.training.max_context_length,
        dim=dim,
        vocab_size=2048,
        num_layers=13,
        full_attention_layers={3, 7, 11, 12},
        attn_res_block_size=4,
        num_heads=64,
        q_lora_rank=896,
        kv_lora_rank=288,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=19456,
        latent_dim=2048,
        expert_hidden_dim=1792,
        num_experts=32,
        top_k=2,
        num_shared_experts=2,
        vision_encoder=_vision_encoder_config(
            text_dim=dim,
            dim=512,
            qkv_dim=768,
            hidden_dim=2048,
            num_layers=2,
            num_heads=6,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        ),
        attn_backend=attn_backend,
        parallelism=config.parallelism,
    )
    config.optim.optimizer = _dist_muon_optimizer(
        config.model,
        muon_lr=1e-5,
        adamw_lr=1e-5,
        parallelism=config.parallelism,
    )
    config.__post_init__()
    npu_changes(config)
    return config


def kimi_k3_7b_moonep(
    *, parallelism: ParallelismConfig | None = None, attn_backend: str = "varlen"
) -> MoonEPTrainer.Config:
    """Build the same 7B model and optimizer with resident Ascend MoonEP experts."""
    if parallelism is None:
        parallelism = ParallelismConfig(
            expert_parallel_degree=2, data_parallel_shard_degree=2
        )
    config = kimi_k3_7b_npu(parallelism=parallelism, attn_backend=attn_backend)
    config.override.imports.append(override_target(npu_moonep))
    return derive(config, MoonEPTrainer.Config)
