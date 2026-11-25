# SPDX-License-Identifier: Apache-2.0
"""Destroy the DeepEP EPBuffer pool before process-group teardown."""

import sys

from torchtitan_ascend.patches import patch


@patch(
    target="torchtitan.training_engine:TrainingEngine.close", kind="wrap", deferred=True
)
def _close_with_deepep_cleanup(original):
    def close(self):
        runtime = sys.modules.get("torchtitan_ascend.overrides.deepep.runtime")
        if runtime is not None:
            runtime.destroy_all_buffers()
        return original(self)

    return close
