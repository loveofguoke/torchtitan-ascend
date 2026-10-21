# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""GLM-5.2 and GLM-5.3 on Ascend using upstream debug model recipes."""

from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.glm5 import (
    glm5_2_debugmodel,
    glm5_3_debugmodel,
)

from torchtitan_ascend.changes import add_override


def npu_changes(config: Trainer.Config) -> None:
    """Apply the Ascend-specific differences to an upstream GLM-5 recipe."""
    from torchtitan_ascend.modules.glm5 import replace_trunc_normal_initializers
    from torchtitan_ascend.ops.rope import npu_complex_rope

    add_override(config, npu_complex_rope)
    replace_trunc_normal_initializers(config.model)
    config.model.local_compile_regions = []
    config.activation_remat = None


def _debugmodel_npu(config: Trainer.Config) -> Trainer.Config:
    config.training.num_tokens_per_microbatch_per_dp_rank = 256
    npu_changes(config)
    return config


def glm5_2_debugmodel_npu() -> Trainer.Config:
    """Build the upstream GLM-5.2 debug model with Ascend test settings."""
    return _debugmodel_npu(glm5_2_debugmodel(seq_len=128))


def glm5_3_debugmodel_npu() -> Trainer.Config:
    """Build the upstream GLM-5.3 debug model with Ascend test settings."""
    return _debugmodel_npu(glm5_3_debugmodel(seq_len=128))


def glm5_debugmodel_npu() -> Trainer.Config:
    """Build the GLM-5.2 debug model for backward compatibility."""
    return glm5_2_debugmodel_npu()
