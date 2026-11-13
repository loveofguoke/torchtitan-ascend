# SPDX-License-Identifier: Apache-2.0
"""Dense packed TND flash attention for the Kimi K3.1 vision encoder."""

from dataclasses import dataclass, replace
from itertools import accumulate

import spmd_types as spmd
import torch
from torchtitan.config import derive
from torchtitan.models.common.attention import FlexInnerAttention
from torchtitan.models.kimi_k2_7 import vision_encoder
from torchtitan.models.kimi_k2_7.vision_encoder import _tpool_patch_merger
from torchtitan.models.kimi_k3.vision_encoder import KimiK3VisionEncoder


class AscendPackedVisionAttention(FlexInnerAttention):
    @dataclass(kw_only=True, slots=True)
    class Config(FlexInnerAttention.Config):
        pass

    def forward(self, q_THK, k_THK, v_THV, *, attention_metadata, **kwargs):
        import torch_npu

        if q_THK.device.type != "npu" or q_THK.dtype not in (
            torch.bfloat16,
            torch.float16,
        ):
            raise ValueError("Packed vision FA requires BF16 or FP16 Ascend tensors.")
        if not attention_metadata or attention_metadata[-1] != q_THK.size(0):
            raise ValueError("Packed vision FA requires the complete item boundaries.")
        with spmd.no_typecheck():
            return torch_npu.npu_fusion_attention(
                q_THK.contiguous(),
                k_THK.contiguous(),
                v_THV.contiguous(),
                q_THK.size(1),
                "TND",
                scale=q_THK.size(-1) ** -0.5,
                actual_seq_qlen=attention_metadata,
                actual_seq_kvlen=attention_metadata,
                sparse_mode=0,
            )[0]


class AscendPackedKimiVisionEncoder(KimiK3VisionEncoder):
    merge_kernel_size: tuple[int, int]
    patch_embed: torch.nn.Module
    layers: torch.nn.ModuleDict
    final_norm: torch.nn.Module
    projector: torch.nn.Module

    @dataclass(kw_only=True, slots=True)
    class Config(KimiK3VisionEncoder.Config):
        pass

    def forward(
        self, pixel_values: torch.Tensor, *, grid_thw: torch.Tensor
    ) -> torch.Tensor:
        # The upstream encoder already reads grids on the host for position
        # embeddings. Keep the resulting cumulative ends on the host for FA.
        grids = grid_thw.tolist()
        kh, kw = self.merge_kernel_size
        lengths = []
        for t, h, w in grids:
            if min(t, h, w) <= 0 or h % kh or w % kw:
                raise ValueError(f"Invalid packed visual grid {(t, h, w)}.")
            lengths.append(t * h * w)
        ends = tuple(accumulate(lengths))
        if not ends or pixel_values.size(0) != ends[-1]:
            raise ValueError("The packed visual grids must describe every input patch.")

        learned_pos, rope_cache = super().compute_position_embeddings(grids)
        x = self.patch_embed(pixel_values) + learned_pos
        for block in self.layers.values():
            x = block(
                x,
                rope_cache=rope_cache,
                rope_apply=vision_encoder.ComplexRoPE.apply_rotary_emb,
                attention_mask=ends,
            )
        x = self.final_norm(x)
        return self.projector(_tpool_patch_merger(x, grids, self.merge_kernel_size))


def packed_vision_fa(
    config: KimiK3VisionEncoder.Config,
) -> AscendPackedKimiVisionEncoder.Config:
    attention = config.block.attn
    return derive(
        config,
        AscendPackedKimiVisionEncoder.Config,
        block=replace(
            config.block,
            attn=replace(
                attention,
                inner_attention=derive(
                    attention.inner_attention, AscendPackedVisionAttention.Config
                ),
            ),
        ),
    )
