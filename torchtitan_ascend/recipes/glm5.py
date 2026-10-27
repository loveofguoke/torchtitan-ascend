# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""GLM-5.2 and GLM-5.3 training recipes for Ascend NPU."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, LRSchedulersContainer, Optim, OptimizersContainer
from torchtitan.config import TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.hf_datasets.text_datasets import DATASETS
from torchtitan.models.common.config_utils import decoder_vocab_size, DEFAULT_DEBUG_MODEL_SEQ_LEN
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_ascend.config.override_utils import add_override
from torchtitan_ascend.models.glm5 import build_model_config, MTPLoss


def npu_changes(config: Trainer.Config) -> None:
    """Apply the Ascend-specific differences to a GLM-5 recipe."""
    from torchtitan_ascend.overrides.glm5 import replace_trunc_normal_initializers
    from torchtitan_ascend.overrides.rope import npu_complex_rope

    add_override(config, npu_complex_rope)
    replace_trunc_normal_initializers(config.model)
    config.model.local_compile_regions = []
    config.activation_checkpoint = None


def _glm5_debugmodel_npu(
    seq_len: int = DEFAULT_DEBUG_MODEL_SEQ_LEN,
    *,
    parallelism: ParallelismConfig | None = None,
    flavor: str = "glm5_2_debugmodel",
    mtp_scale: float | None = None,
) -> Trainer.Config:
    """Build a deterministic GLM-5 recipe and apply Ascend overrides."""
    if parallelism is None:
        parallelism = ParallelismConfig()
    model_config = build_model_config(
        flavor,
        parallelism=parallelism,
        seq_len=seq_len,
    )
    num_mtp_layers = len(model_config.mtp_layers or ())
    if mtp_scale is not None and num_mtp_layers == 0:
        raise ValueError("mtp_scale may only be set for a GLM MTP flavor")
    if mtp_scale is not None and mtp_scale < 0:
        raise ValueError("mtp_scale must be non-negative")
    loss_config = (
        MTPLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
            mtp_scale=0.3 if mtp_scale is None else mtp_scale,
        )
        if model_config.mtp_layers
        else CrossEntropyLoss.Config(
            global_vocab_size=decoder_vocab_size(model_config),
        )
    )
    config = Trainer.Config(
        loss=ChunkedLossWrapper.Config(loss_fn=loss_config),
        hf_assets_path="./tests/assets/tokenizer",
        metrics=MetricsProcessor.Config(log_freq=1),
        model=model_config,
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)]),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=2 * seq_len,
            max_context_length=seq_len,
            steps=10,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_checkpoint=None,
    )
    npu_changes(config)
    return config


def glm5_2_debugmodel_npu(
    seq_len: int = 128,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build the GLM-5.2 debug model for Ascend NPU."""
    return _glm5_debugmodel_npu(seq_len=seq_len, parallelism=parallelism)


def glm5_2_debugmodel_mtp_npu(
    seq_len: int = 128,
    *,
    parallelism: ParallelismConfig | None = None,
    mtp_scale: float = 0.3,
) -> Trainer.Config:
    """Build the GLM-5.2 debug model with one NextN/MTP block."""
    return _glm5_debugmodel_npu(
        seq_len=seq_len,
        parallelism=parallelism,
        flavor="glm5_2_debugmodel_mtp",
        mtp_scale=mtp_scale,
    )


def glm5_2_scaled_debugmodel_npu(
    seq_len: int = 4096,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build the scaled GLM-5.2 validation model for Ascend NPU."""
    return _glm5_debugmodel_npu(
        seq_len=seq_len,
        parallelism=parallelism,
        flavor="glm5_2_scaled_debugmodel",
    )


def glm5_3_debugmodel_npu(
    seq_len: int = 128,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build the GLM-5.3 debug model for Ascend NPU."""
    return _glm5_debugmodel_npu(
        seq_len=seq_len,
        parallelism=parallelism,
        flavor="glm5_3_debugmodel",
    )


def glm5_3_debugmodel_mtp_npu(
    seq_len: int = 128,
    *,
    parallelism: ParallelismConfig | None = None,
    mtp_scale: float = 0.3,
) -> Trainer.Config:
    """Build the GLM-5.3 debug model with one NextN/MTP block."""
    return _glm5_debugmodel_npu(
        seq_len=seq_len,
        parallelism=parallelism,
        flavor="glm5_3_debugmodel_mtp",
        mtp_scale=mtp_scale,
    )


def glm5_3_scaled_debugmodel_npu(
    seq_len: int = 4096,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build the scaled GLM-5.3 validation model for Ascend NPU."""
    return _glm5_debugmodel_npu(
        seq_len=seq_len,
        parallelism=parallelism,
        flavor="glm5_3_scaled_debugmodel",
    )


def glm5_debugmodel_npu(
    seq_len: int = 128,
    *,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build the GLM-5.2 debug model for backward compatibility."""
    return glm5_2_debugmodel_npu(seq_len, parallelism=parallelism)
