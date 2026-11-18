# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Add the V4 NPU debug training recipe;
# 2. Enable the RoPE override and disable local compilation regions and CUDA graphs;
# 3. Configure packed C4 data, AdamW and FSDP/TP/EP training.

"""DeepSeek V4 training recipes with Ascend operator overrides."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import DebugConfig, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.deepseek_v4 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_ascend.config.override_utils import add_override


def npu_changes(config: Trainer.Config) -> None:
    """Activate Ascend operator overrides in place."""
    from torchtitan_ascend.overrides.common.rope import npu_real_cache_rope

    add_override(config, npu_real_cache_rope)
    # triton-ascend 3.6.0 cannot compile the upstream local compile regions, so
    # the verified path is eager execution, which also excludes CUDA graphs.
    config.model.local_compile_regions = []


def deepseek_v4_debugmodel_npu(
    *,
    tp: int = 1,
    ep: int = 1,
    pp: int = 1,
) -> Trainer.Config:
    """Validate the DeepSeek V4 debug model with ten steps on C4 test data."""
    parallelism = ParallelismConfig(
        tensor_parallel_degree=tp,
        expert_parallel_degree=ep,
        pipeline_parallel_degree=pp,
        enable_sequence_parallel=(tp > 1),
    )
    # Upstream debugmodel defaults to 2048, not the flavor's 16384 cap; without
    # this the token budget per DP rank inflates 8x and the DSA indexer OOMs.
    model = build_model_config(
        "debugmodel", seq_len=DEFAULT_DEBUG_MODEL_SEQ_LEN, parallelism=parallelism
    )
    config = Trainer.Config(
        model=model,
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=64,
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
            disable_cuda_graphs=True,
            num_tokens_per_microbatch_per_dp_rank=8 * model.max_context_length,
            max_context_length=model.max_context_length,
            steps=10,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
        metrics=MetricsProcessor.Config(log_freq=1),
        debug=DebugConfig(seed=42, deterministic=True),
    )
    npu_changes(config)
    return config
