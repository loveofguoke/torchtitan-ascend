# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from TorchTitan distributed initialization at 45e33cd.
# See LICENSE-BSD-3-Clause.

"""Keep NPU pipeline communication on the shared HCCL process group."""

from __future__ import annotations

from functools import wraps
from typing import TYPE_CHECKING

from torchtitan_ascend.patches import patch

if TYPE_CHECKING:
    from torchtitan.config import CommConfig
    from torchtitan.distributed.parallelism_context import DistributedTopology


@patch(target="torchtitan.distributed.utils:init_distributed", kind="wrap")
def apply(original):
    """Wrap upstream initialization and keep NPU PP P2P on the shared group."""
    import torch
    from torchtitan.distributed import utils

    @wraps(original)
    def init_distributed(
        comm_config: CommConfig,
        enable_cpu_backend: bool = False,
        base_folder: str = "",
        ranks: list[int] | None = None,
        *,
        pipeline_parallel_degree: int = 1,
    ) -> DistributedTopology:
        # An externally initialized process group makes upstream return before
        # it touches the per-edge flag; leave that flag untouched as well.
        if utils.device_type != "npu" or torch.distributed.is_initialized():
            return original(
                comm_config,
                enable_cpu_backend,
                base_folder,
                ranks,
                pipeline_parallel_degree=pipeline_parallel_degree,
            )

        if pipeline_parallel_degree > 1:
            utils.logger.warning("NPU PP: using the shared process group for P2P.")

        # Upstream enables one P2P communicator per directed rank edge when PP
        # is enabled. PipelineStage reads this flag while building the P2P path,
        # after initialization; HCCL keeps P2P on the shared process group
        try:
            return original(
                comm_config,
                enable_cpu_backend,
                base_folder,
                ranks,
                pipeline_parallel_degree=pipeline_parallel_degree,
            )
        finally:
            utils.dist_config.pipeline_per_edge_p2p = False

    return init_distributed
