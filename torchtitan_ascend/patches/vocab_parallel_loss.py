# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""NPU-compatible vocab-parallel cross entropy."""

import torch

from torchtitan_ascend.patches.registry import patch


def _local_vocab_labels(
    labels: torch.Tensor,
    *,
    vocab_start: int,
    local_vocab_size: int,
    ignore_index: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map global labels to one vocab shard without boolean assignment."""
    safe_labels = torch.where(labels != ignore_index, labels, 0)
    out_of_range = (safe_labels < vocab_start) | (safe_labels >= vocab_start + local_vocab_size)
    local_labels = torch.where(
        out_of_range,
        torch.zeros_like(safe_labels),
        safe_labels - vocab_start,
    )
    return local_labels, out_of_range


@patch(
    target="torchtitan.components.loss:_LossParallelCrossEntropy.forward",
    kind="replace",
)
def _replace_forward(original):
    import torch.distributed as dist
    import torch.distributed._functional_collectives as funcol
    from torchtitan.components.loss import IGNORE_INDEX

    def forward(
        ctx,
        logits: torch.Tensor,
        labels: torch.Tensor,
        tp_group: dist.ProcessGroup,
        global_vocab_size: int,
        reduction: str = "sum",
    ) -> torch.Tensor:
        logits_dtype = logits.dtype
        logits = logits.float()
        tp_world_size = dist.get_world_size(tp_group)
        tp_rank = dist.get_rank(tp_group)
        chunk_size = (global_vocab_size + tp_world_size - 1) // tp_world_size
        vocab_start = min(global_vocab_size, chunk_size * tp_rank)
        vocab_end = min(global_vocab_size, vocab_start + chunk_size)
        local_vocab_size = max(0, vocab_end - vocab_start)
        if logits.shape[-1] != local_vocab_size:
            raise ValueError(
                "_LossParallelCrossEntropy expected local vocab size "
                f"{local_vocab_size} for global vocab size {global_vocab_size}, "
                f"got {logits.shape[-1]}."
            )
        if local_vocab_size == 0:
            raise ValueError("_LossParallelCrossEntropy does not support empty vocab shards.")

        local_max = torch.amax(logits, dim=-1, keepdim=True)
        local_max = funcol.all_reduce(local_max, reduceOp=dist.ReduceOp.MAX.name, group=tp_group)
        shifted = logits - local_max
        shifted_sumexp = torch.sum(torch.exp(shifted), dim=-1, keepdim=True)
        shifted_sumexp = funcol.all_reduce(shifted_sumexp, reduceOp=dist.ReduceOp.SUM.name, group=tp_group)
        log_probs = shifted - torch.log(shifted_sumexp)

        local_labels, out_of_range = _local_vocab_labels(
            labels,
            vocab_start=vocab_start,
            local_vocab_size=local_vocab_size,
            ignore_index=IGNORE_INDEX,
        )
        local_result = torch.gather(log_probs, -1, local_labels.unsqueeze(-1))
        local_result = torch.where(
            out_of_range.unsqueeze(-1),
            torch.zeros_like(local_result),
            local_result,
        )
        local_result = funcol.all_reduce(local_result, reduceOp=dist.ReduceOp.SUM.name, group=tp_group)

        result = -local_result.squeeze(-1)
        result = torch.where(labels != IGNORE_INDEX, result, 0)
        ctx.save_for_backward(log_probs, labels)
        ctx.logits_dtype = logits_dtype
        ctx.vocab_start = vocab_start
        ctx.local_vocab_size = local_vocab_size
        ctx.reduction = reduction
        return result if reduction == "none" else result.sum()

    return forward


@patch(
    target="torchtitan.components.loss:_LossParallelCrossEntropy.backward",
    kind="replace",
)
def _replace_backward(original):
    from torchtitan.components.loss import IGNORE_INDEX

    def backward(
        ctx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, None, None, None, None]:
        log_probs, labels = ctx.saved_tensors
        local_labels, out_of_range = _local_vocab_labels(
            labels,
            vocab_start=ctx.vocab_start,
            local_vocab_size=ctx.local_vocab_size,
            ignore_index=IGNORE_INDEX,
        )
        grad_update = out_of_range.to(log_probs.dtype) - 1.0
        grad_input = torch.zeros_like(log_probs).scatter_(
            -1,
            local_labels.unsqueeze(-1),
            grad_update.unsqueeze(-1),
        )
        if ctx.reduction == "none":
            grad_output = grad_output.unsqueeze(-1)
        grad_output = torch.where((labels != IGNORE_INDEX).unsqueeze(-1), grad_output, 0)
        grad_logits = (grad_input + torch.exp(log_probs)) * grad_output
        return grad_logits.to(ctx.logits_dtype), None, None, None, None

    return backward
