# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 multi-token prediction (NextN) training components."""

from dataclasses import dataclass

from torchtitan.models.deepseek_v3.mtp import MTPLoss, MTPTransformerBlock, roll_mtp_sequence


class Glm5MTPTransformerBlock(MTPTransformerBlock):
    """GLM checkpoint config using the upstream MTP forward implementation."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPTransformerBlock.Config):
        pass


__all__ = [
    "Glm5MTPTransformerBlock",
    "MTPLoss",
    "roll_mtp_sequence",
]
