# SPDX-License-Identifier: Apache-2.0
# Copyright (c) Meta Platforms, Inc. and affiliates. All rights reserved.
#
# This code is based on the DeepSeekV4 implementations from the DeepSeek AI team.
# https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro
#
# Modification points:
# 1. Add the V4.1 NPU debug training recipe;
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

"""DeepSeek V4.1 training recipes with Ascend operator overrides."""

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
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer

from torchtitan_ascend.config.override_utils import add_override
from torchtitan_ascend.models.deepseek_v4_1 import build_model_config


def npu_changes(config: Trainer.Config) -> None:
    """Activate Ascend kernels without claiming optional routed-expert subtrees."""
    from torchtitan_ascend.overrides.rms_norm import npu_rms_norm
    from torchtitan_ascend.overrides.rope import real_cache_rope
    from torchtitan_ascend.overrides.swiglu import npu_shared_expert_swiglu

    for override_fn in (real_cache_rope, npu_rms_norm, npu_shared_expert_swiglu):
        add_override(config, override_fn)


def deepseek_v4_1_debugmodel_npu(
    *,
    tp: int = 1,
    ep: int = 1,
) -> Trainer.Config:
    """Validate the DeepSeek V4.1 debug model with ten steps on C4 test data."""
    parallelism = ParallelismConfig(
        tensor_parallel_degree=tp,
        expert_parallel_degree=ep,
        enable_sequence_parallel=(tp > 1),
    )
    model = build_model_config("debugmodel", parallelism=parallelism)
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
            num_tokens_per_microbatch_per_dp_rank=128,
            max_context_length=model.max_context_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=None,
        metrics=MetricsProcessor.Config(log_freq=1),
        debug=DebugConfig(seed=42, deterministic=True),
    )
    npu_changes(config)
    return config
