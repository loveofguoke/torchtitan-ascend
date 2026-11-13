# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""Kimi K3.1 training recipes on Ascend."""

from torch import finfo, float32

from torchtitan.components.data.multimodal.mm_datasets import (
    MM_DATASETS,
    MMSamplePackingConfig,
)
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import apply_overrides, derive, OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.models.kimi_k3.flavors import _kimi_k3
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.kimi_k3 import (
    _dist_muon_optimizer,
    _kimi_k3_multimodal_dataloader,
    _KimiK3TrainerConfig,
)

from torchtitan_ascend.config import override_target
from torchtitan_ascend.models.kimi_k3_1 import build_model_config
from torchtitan_ascend.models.kimi_k3_1.flavors import _make_dsa_config
from torchtitan_ascend.models.kimi_k3_1.model import (
    KimiK31Model,
    KimiK31TransformerBlock,
)
from torchtitan_ascend.recipes.kimi_k3 import npu_changes


def npu_fusion_kimik31(config: Trainer.Config) -> None:
    """Apply the Kimi K3.1 attention, normalization, MLP, and vision fusions."""
    from torchtitan_ascend.overrides.attn_res import triton_attn_res
    from torchtitan_ascend.overrides.dsa import ascendc_lightning_indexer
    from torchtitan_ascend.overrides.grouped_linear import ascendc_grouped_linear
    from torchtitan_ascend.overrides.lig_kl import ascendc_lig_kl
    from torchtitan_ascend.overrides.packed_vision_attention import packed_vision_fa
    from torchtitan_ascend.overrides.rmsnorm import (
        ascendc_gated_rmsnorm,
        ascendc_rmsnorm,
    )
    from torchtitan_ascend.overrides.rotary_mul import enable_ascendc_rotary_mul
    from torchtitan_ascend.overrides.situ_glu import ascendc_situ_glu

    # Apply child overrides before a routed-expert override claims their parent.
    apply_overrides(
        OverrideConfig(
            imports=[
                override_target(ascendc_situ_glu),
                override_target(ascendc_grouped_linear),
            ]
        ),
        config,
    )
    config.override.imports = [
        *config.override.imports,
        override_target(ascendc_rmsnorm),
        override_target(ascendc_gated_rmsnorm),
        override_target(triton_attn_res),
        override_target(ascendc_lightning_indexer),
        override_target(ascendc_lig_kl),
    ]
    config.model.vision_encoder = packed_vision_fa(config.model.vision_encoder)
    enable_ascendc_rotary_mul()


def kimi_k3_1_pro_npu(
    *,
    hf_assets_path: str = "./tests/assets/tokenizer",
    seq_len: int = 2048,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build a BF16 training recipe for the released K3.1 Pro architecture."""
    if parallelism is None:
        parallelism = ParallelismConfig(enable_sequence_parallel=False)
    model = build_model_config(
        "Kimi-K3.1-Pro", parallelism=parallelism, seq_len=seq_len, multimodal=True
    )
    vocab_size = decoder_vocab_size(model)
    config = Trainer.Config(
        model=model,
        hf_assets_path=hf_assets_path,
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m"]),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-5)],
            ),
        ),
        training=TrainingConfig(
            max_context_length=seq_len,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        activation_remat=None,
    )
    npu_changes(config)
    return config


def kimi_k3_1_reduced_npu(
    *,
    seq_len: int = 16384,
    parallelism: ParallelismConfig | None = None,
    attn_backend: str = "varlen",
) -> Trainer.Config:
    """Build an eight-layer, full-width multimodal recipe with 64 experts."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    base = _kimi_k3(attn_backend, seq_len=seq_len, parallelism=parallelism)
    vision = base.vision_encoder
    for norm in (vision.block.norm1, vision.block.norm2, vision.final_norm):
        norm.eps = finfo(float32).eps
    base.layers = base.layers[:8]
    num_experts = 64
    for layer in base.layers[1:]:
        moe = layer.feed_forward
        moe.num_experts = num_experts
        moe.router.num_experts = num_experts
        moe.router.gate.out_features = num_experts
        moe.routed_experts.w13.group_size = num_experts
        moe.routed_experts.w2.group_size = num_experts
        moe.routed_experts.token_dispatcher.num_experts = num_experts
    layers = []
    for layer in base.layers:
        if isinstance(layer.attention, KimiMLAAttention.Config):
            attention = _make_dsa_config(
                layer.attention,
                num_index_heads=32,
                index_head_dim=128,
                index_topk=2048,
            )
            layer = derive(layer, KimiK31TransformerBlock.Config, attention=attention)
        layers.append(layer)
    model = derive(base, KimiK31Model.Config, layers=layers)
    dataloader = _kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m-test"])
    dataloader.dataset = MMSamplePackingConfig(
        dataset=dataloader.dataset,
        num_packing_bins=1,
    )
    vocab_size = decoder_vocab_size(model)
    config = _KimiK3TrainerConfig(
        model=model,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=vocab_size,
            ),
        ),
        hf_assets_path="./tests/assets/tokenizer",
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        dataloader=dataloader,
        optim=Optim.Config(
            optimizer=_dist_muon_optimizer(
                model,
                muon_lr=1e-5,
                adamw_lr=1e-5,
                parallelism=parallelism,
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            max_context_length=model.max_context_length,
            steps=10,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        metrics=MetricsProcessor.Config(log_freq=1),
    )
    npu_changes(config)
    return config
