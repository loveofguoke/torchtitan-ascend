# SPDX-License-Identifier: Apache-2.0
"""Initialize Ascend before collecting copied upstream tests."""

# Register the NPU backend before loading its operator libraries.
import torch_npu  # noqa: F401, I001  # usort: skip
import fla_npu  # noqa: F401  # usort: skip

from torchtitan_ascend import setup

setup()
