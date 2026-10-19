# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""GLM-5 on Ascend using the upstream debug model recipe."""

from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.glm5 import glm5_debugmodel

from torchtitan_ascend.changes import add_override


def npu_changes(config: Trainer.Config) -> None:
    """Apply the Ascend-specific differences to an upstream GLM-5 recipe."""
    from torchtitan_ascend.modules.glm5 import replace_trunc_normal_initializers
    from torchtitan_ascend.ops.rope import npu_complex_rope

    add_override(config, npu_complex_rope)
    replace_trunc_normal_initializers(config.model)
    config.model.local_compile_regions = []
    config.activation_remat = None


def glm5_debugmodel_npu() -> Trainer.Config:
    """Build the upstream GLM-5 debug model with the Ascend test settings."""
    config = glm5_debugmodel(seq_len=128)
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    npu_changes(config)
    return config
