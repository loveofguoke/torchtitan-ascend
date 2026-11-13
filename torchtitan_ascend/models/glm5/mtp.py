# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""GLM-5 multi-token prediction (NextN) training components."""

from dataclasses import dataclass

import torch
import torch_remat as remat
from torchtitan.models.common.moe import MoE

from torchtitan.models.deepseek_v3.mtp import (
    MTPLoss,
    MTPTransformerBlock,
    roll_mtp_sequence,
)


class Glm5MTPTransformerBlock(MTPTransformerBlock):
    """Reuse NextN construction and propagate the indexer's loss denominator."""

    @dataclass(kw_only=True, slots=True)
    class Config(MTPTransformerBlock.Config):
        pass

    def forward(
        self,
        mtp_input_embed,
        prev_embed,
        mtp_input_valid_mask,
        attention_metadata,
        positions=None,
        *,
        padding_mask=None,
        aux_loss_denominator=None,
    ):
        # Upstream MTP passes the denominator to MoE but not to attention.
        # GLM's trainable indexer needs the denominator for this MTP objective.
        mtp_padding = ~mtp_input_valid_mask
        if padding_mask is not None:
            mtp_padding = mtp_padding | padding_mask
        valid = self._maybe_shard_mtp_valid_mask_across_tp(mtp_input_valid_mask)
        prev_embed = prev_embed * valid.unsqueeze(-1).to(prev_embed.dtype)
        h = self.eh_proj(
            torch.cat([self.enorm(mtp_input_embed), self.hnorm(prev_embed)], dim=-1)
        )
        remat.recompute_needs_tensor(h)
        attention_out = self.attention(
            self.attention_norm(h),
            attention_metadata,
            positions,
            padding_mask=mtp_padding,
            aux_loss_denominator=aux_loss_denominator,
        )
        remat.recompute_needs_tensor(attention_out)
        h = h + attention_out
        if isinstance(self.feed_forward, MoE):
            ffn_out = self.feed_forward(
                self.ffn_norm(h),
                padding_mask_T=mtp_padding,
                aux_loss_denominator=aux_loss_denominator,
            )
        else:
            ffn_out = self.feed_forward(self.ffn_norm(h))
        remat.recompute_needs_tensor(ffn_out)
        return self.mtp_norm(h + ffn_out)


__all__ = [
    "Glm5MTPTransformerBlock",
    "MTPLoss",
    "roll_mtp_sequence",
]
