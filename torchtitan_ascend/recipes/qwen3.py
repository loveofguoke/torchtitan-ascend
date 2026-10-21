# SPDX-License-Identifier: Apache-2.0
from torchtitan.models.qwen3 import build_model_config
from torchtitan_recipes.tests.models.qwen3 import qwen3_debugmodel


def qwen3_npu():
    config = qwen3_debugmodel(seq_len=256)
    config.model = build_model_config("debugmodel", seq_len=256, attn_backend="varlen")
    config.model.local_compile_regions = []
    config.activation_checkpoint = None
    config.training.num_tokens_per_microbatch_per_dp_rank = 512
    return config
