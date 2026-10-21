# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""Qwen3.5 text training with fla-npu GDN."""

from torchtitan.components.data import (
    ConcatThenSplitPackingConfig,
    GrainDataLoader,
    SingleDatasetConfig,
)
from torchtitan.components.data.collators import TextCollator
from torchtitan.components.data.sources import HuggingFaceRandomAccessSource
from torchtitan.components.data.text_datasets import DATASETS, TextProcessor
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.config import OverrideConfig, TrainingConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.qwen3_5 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_ascend.changes import override_target
from torchtitan_ascend.modules.gdn import npu_gated_delta_net


def _text_config(flavor: str, *, hf_assets_path: str, dataset) -> Trainer.Config:
    model = build_model_config(flavor, seq_len=256, attn_backend="varlen")
    # Upstream executes dummy vision inputs even for text batches if this is present.
    model.vision_encoder = None
    model.local_compile_regions = []
    return Trainer.Config(
        model=model,
        override=OverrideConfig(imports=[override_target(npu_gated_delta_net)]),
        hf_assets_path=hf_assets_path,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=decoder_vocab_size(model)),
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
        activation_checkpoint=None,
        metrics=MetricsProcessor.Config(log_freq=1),
    )


def qwen35_debugmodel_npu() -> Trainer.Config:
    """Ten-step text training with the upstream debug decoder and test assets."""
    return _text_config(
        "debugmodel",
        hf_assets_path="./tests/assets/tokenizer",
        dataset=DATASETS["c4_test"],
    )


def qwen35_0_8b_npu(
    *,
    hf_assets_path: str = "./assets/hf/Qwen3.5-0.8B",
    data_path: str = "./assets/c4/en/c4-train.00000-of-01024.json.gz",
) -> Trainer.Config:
    """Ten-step text training with the upstream 0.8B decoder and local assets."""
    return _text_config(
        "0.8B",
        hf_assets_path=hf_assets_path,
        dataset=SingleDatasetConfig(
            source=HuggingFaceRandomAccessSource.Config(
                path="json",
                split="train",
                load_dataset_kwargs={"data_files": data_path},
            ),
            processor=TextProcessor.Config(),
            post_filters=(lambda sample: sample is not None,),
        ),
    )
