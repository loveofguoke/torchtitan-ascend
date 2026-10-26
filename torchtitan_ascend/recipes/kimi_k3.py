# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan training recipes. See LICENSE-BSD-3-Clause.

"""Kimi K3 on Ascend: the upstream DistMuon debugmodel recipe with the Ascend
KDA kernels.
"""

from torchtitan.trainer import Trainer

from torchtitan_ascend.config import override_target
from torchtitan_recipes.tests.models.kimi_k3 import kimi_k3_debugmodel

_GATED_RMSNORM_REGION = "gated_rmsnorm"


def npu_changes(config: Trainer.Config) -> None:
    """Apply the Ascend-specific differences to an upstream Kimi K3 recipe.

    Swaps the InnerKDA subtree to the fla_npu kernels and drops the
    gated_rmsnorm local-compile region.
    """
    from torchtitan_ascend.overrides.kda import npu_kda

    config.override.imports = [*config.override.imports, override_target(npu_kda)]
    config.model.local_compile_regions = [
        region for region in config.model.local_compile_regions if region != _GATED_RMSNORM_REGION
    ]
    config.comm.train_timeout_seconds = max(config.comm.train_timeout_seconds, 600)


def kimi_k3_debugmodel_npu() -> Trainer.Config:
    """Validate the Kimi K3 debug model with DistMuon and multimodal test data.

    Reuses the upstream recipe and its DistMuon layouts verbatim; only the
    NPU-required differences are applied on top.
    """
    config = kimi_k3_debugmodel()
    npu_changes(config)
    return config
