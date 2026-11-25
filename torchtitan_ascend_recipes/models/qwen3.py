# SPDX-License-Identifier: Apache-2.0
from torchtitan_recipes.tests.models.qwen3 import qwen3_debugmodel


def qwen3_debugmodel_npu():
    config = qwen3_debugmodel()
    config.model.local_compile_regions = []
    return config
