# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""Qwen3.8 text training with fla-npu GDN."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import OverrideConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_8 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_ascend.config import override_target
from torchtitan_ascend.overrides.common.attention.gdn import npu_gated_delta_net


def _text_config(flavor: str, *, hf_assets_path: str, dataset) -> Trainer.Config:
    parallelism = ParallelismConfig()
    model = build_model_config(
        flavor, seq_len=256, attn_backend="varlen", parallelism=parallelism
    )
    # Upstream executes dummy vision inputs even for text batches if this is present.
    model.vision_encoder = None
    model.local_compile_regions = []
    return Trainer.Config(
        model=model,
        parallelism=parallelism,
        override=OverrideConfig(imports=[override_target(npu_gated_delta_net)]),
        hf_assets_path=hf_assets_path,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model)
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=dataset),
            collator=TextCollator.Config(),
            shuffle=False,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)],
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=512,
            max_context_length=model.max_context_length,
            steps=10,
        ),
        checkpointer=None,
        activation_remat=None,
        metrics=MetricsProcessor.Config(log_freq=1),
    )


def qwen38_debugmodel_npu() -> Trainer.Config:
    """Ten-step text training with the upstream dense debug decoder."""
    return _text_config(
        "debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        dataset=DATASETS["c4_test"],
    )


def qwen38_debugmodel_moe_npu() -> Trainer.Config:
    """Ten-step text training with the upstream MoE debug decoder."""
    return _text_config(
        "debugmodel_moe",
        hf_assets_path="./tests/assets/tokenizer",
        dataset=DATASETS["c4_test"],
    )
