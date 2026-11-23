# SPDX-License-Identifier: Apache-2.0
"""Use a flattened embedding lookup for attn_gym's packed KV gather on NPU.

The reference ``_packed_gather_attn`` gathers slots with 4-D advanced indexing
``kv[b_idx, h_idx, indices]``. On NPU the forward dispatches to the AI_CPU
``Index`` kernel (~0.8s per call at kv [1, 1, 6144, 512] and indices
[1, 4096, 640]) and the backward to the sort-based ``IndexPutWithSortV2``.
Expressing the same lookup over the flattened ``[B*H*S, D]`` table with
``F.embedding`` returns identical rows while routing the backward through
``embedding_dense_backward`` (index_add). Non-NPU devices keep upstream.
"""

import torch
from torch.nn import functional as F

from torchtitan_ascend.patches import patch


@patch(
    target="attn_gym.sparse.gather_attn.impl.reference:_packed_gather_attn",
    kind="wrap",
)
def apply(original):
    def _packed_gather_attn(
        query,
        local_kv,
        sparse_kv,
        kv_indices,
        attention_sink,
        cu_seqlens,
        cu_seqlens_k,
        sliding_window_size,
        *,
        scale,
    ):
        if query.device.type != "npu":
            return original(
                query,
                local_kv,
                sparse_kv,
                kv_indices,
                attention_sink,
                cu_seqlens,
                cu_seqlens_k,
                sliding_window_size,
                scale=scale,
            )

        batch, heads, tokens, dim = query.shape
        positions = torch.arange(tokens, device=query.device)
        window = min(sliding_window_size, tokens)
        local_indices = (
            positions[:, None] - window + 1 + torch.arange(window, device=query.device)
        )

        # torch.ge instead of `>=` so type checkers resolve a Tensor result.
        local_valid = torch.ge(local_indices, 0).unsqueeze(0)
        documents = torch.searchsorted(cu_seqlens[1:], positions, right=True)
        starts = cu_seqlens[documents]
        local_valid = local_valid & (local_indices >= starts[:, None])
        local_indices = local_indices.clamp_min(0).expand(batch, -1, -1)

        sparse_starts = cu_seqlens_k[documents][None, :, None]
        sparse_ends = cu_seqlens_k[
            (documents + 1).clamp(max=cu_seqlens_k.shape[0] - 1)
        ][None, :, None]
        sparse_valid = (kv_indices >= 0) & (kv_indices < sparse_ends - sparse_starts)
        sparse_indices = (
            torch.where(sparse_valid, kv_indices, 0) + sparse_starts + tokens
        )
        sparse_indices = torch.where(sparse_valid, sparse_indices, 0)

        indices = torch.cat((local_indices, sparse_indices), dim=-1).long()
        valid = torch.cat((local_valid, sparse_valid), dim=-1)[:, None, :, :, None]
        kv = torch.cat((local_kv, sparse_kv), dim=2)
        kv_heads = kv.shape[1]

        # Same rows as kv[b_idx, h_idx, indices], flattened to one table:
        # row (b*H + h)*S + idx of kv.reshape(B*H*S, D).
        kv_table = kv.reshape(batch * kv_heads * kv.shape[2], dim)
        row_offsets = torch.arange(batch * kv_heads, device=query.device) * kv.shape[2]
        flat_indices = indices[:, None] + row_offsets.view(batch, kv_heads, 1, 1)
        slots = F.embedding(flat_indices, kv_table)

        # Multiplication by a zero probability does not neutralize NaN values or gradients.
        slots = torch.where(valid, slots, 0)
        accumulation_dtype = torch.promote_types(query.dtype, torch.float32)
        slots_acc = slots.to(accumulation_dtype)

        # Share the gathered slots across query heads, including for the two matmuls.
        grouped_query = query.reshape(batch, kv_heads, heads // kv_heads, tokens, dim)
        grouped_query = grouped_query.permute(0, 1, 3, 2, 4).to(accumulation_dtype)
        logits = torch.matmul(grouped_query, slots_acc.transpose(-2, -1)) * scale
        logits = logits.permute(0, 1, 3, 2, 4).reshape(
            batch, heads, tokens, indices.shape[-1]
        )
        logits = logits.masked_fill(~valid.squeeze(-1), float("-inf"))

        # Inlined _softmax_with_sink: append the sink column so it joins the
        # softmax denominator, then strip it from the returned probabilities.
        sink = attention_sink.to(logits.dtype)[None, :, None, None]
        sinked_logits = torch.cat((logits, sink.expand(*logits.shape[:-1], 1)), dim=-1)
        empty = torch.isneginf(sinked_logits).all(dim=-1, keepdim=True)
        safe_logits = torch.where(empty, 0, sinked_logits)
        probabilities = torch.where(empty, 0, torch.softmax(safe_logits, dim=-1))[
            ..., :-1
        ]
        lse = torch.where(
            empty.squeeze(-1), float("-inf"), torch.logsumexp(safe_logits, dim=-1)
        )
        probabilities = probabilities.to(query.dtype)

        grouped_probabilities = probabilities.reshape(
            batch, kv_heads, heads // kv_heads, tokens, indices.shape[-1]
        ).permute(0, 1, 3, 2, 4)
        output = torch.matmul(grouped_probabilities.to(accumulation_dtype), slots_acc)
        output = output.permute(0, 1, 3, 2, 4).reshape(batch, heads, tokens, dim)
        return output.to(query.dtype), lse

    return _packed_gather_attn
