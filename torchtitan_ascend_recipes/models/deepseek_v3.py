# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV3 implementations from the DeepSeek AI team.
# https://github.com/deepseek-ai/DeepSeek-V3
#
# Modification points:
# 1. Add the V3 NPU debug training recipe;
# 2. Enable RoPE, RMSNorm and SwiGLU overrides;
# 3. Configure packed C4 data, AdamW and FSDP/TP/EP training.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-3-Clause file in the root directory of this source tree.

"""DeepSeek V3 training recipes with Ascend operator overrides."""

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
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.deepseek_v3 import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_ascend.config.override_utils import add_override


def npu_changes(config: Trainer.Config) -> None:
    """Activate Ascend operator overrides in place."""
    from torchtitan_ascend.overrides.common.activation.swiglu import npu_fused_swiglu
    from torchtitan_ascend.overrides.common.norm import npu_rmsnorm
    from torchtitan_ascend.overrides.common.rope import npu_real_cache_rope

    for override_fn in (npu_real_cache_rope, npu_rmsnorm, npu_fused_swiglu):
        add_override(config, override_fn)


def deepseek_v3_debugmodel_npu(
    *,
    tp: int = 1,
    ep: int = 1,
) -> Trainer.Config:
    """Validate the DeepSeek V3 debug model with ten steps on C4 test data.

    NPU-specific adaptations:
    - varlen attention backend (flex crashes on NPU)
    - CUDA graphs disabled (varlen_attn bridge rejects fixed-shape padding)
    - max_num_documents=None (variable-shape metadata)
    """
    parallelism = ParallelismConfig(
        tensor_parallel_degree=tp,
        expert_parallel_degree=ep,
        enable_sequence_parallel=(tp > 1),
    )
    model = build_model_config(
        "debugmodel",
        parallelism=parallelism,
        attn_backend="varlen",
    )
    config = Trainer.Config(
        model=model,
        hf_assets_path="./tests/assets/tokenizer",
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=None,
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
            num_tokens_per_microbatch_per_dp_rank=8 * model.max_context_length,
            max_context_length=model.max_context_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        metrics=MetricsProcessor.Config(log_freq=1),
        debug=DebugConfig(seed=42, deterministic=True),
    )
    npu_changes(config)
    return config
