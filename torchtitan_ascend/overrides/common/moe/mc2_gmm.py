# SPDX-License-Identifier: Apache-2.0
from typing import NamedTuple

import torch
import torch.distributed as dist
import torch_npu


class RoutedInputReplay(NamedTuple):
    local_tokens: torch.Tensor
    inverse_indices: torch.Tensor
    top_k: int


def _rows_by_expert(tokens_RD, counts_Pe):
    """Change source-rank-major token blocks to expert-major blocks."""
    rank_counts = counts_Pe.flatten()
    rank_starts = (rank_counts.cumsum(0) - rank_counts).view_as(counts_Pe)
    expert_counts = counts_Pe.T.reshape(-1)
    expert_starts = expert_counts.cumsum(0) - expert_counts
    row_delta = rank_starts.T.reshape(-1) - expert_starts
    gather_R = torch.repeat_interleave(
        row_delta, expert_counts, output_size=tokens_RD.size(0)
    )
    gather_R.add_(torch.arange(tokens_RD.size(0), device=tokens_RD.device))
    return tokens_RD.index_select(0, gather_R)


def _replay_routed_input(
    tokens_TD, inverse_indices_N, top_k, counts_Pe, send_splits, recv_splits, group
):
    # Invert the forward's actual local permutation to reproduce its row order.
    rows_N = torch.arange(inverse_indices_N.numel(), device=tokens_TD.device)
    gather_N = torch.empty_like(rows_N).scatter_(
        0, inverse_indices_N.long(), rows_N // top_k
    )
    send_ND = tokens_TD.index_select(0, gather_N).bfloat16()
    recv_RD = send_ND.new_empty((sum(recv_splits), send_ND.size(1)))
    dist.all_to_all_single(recv_RD, send_ND, recv_splits, send_splits, group=group)
    del send_ND
    return _rows_by_expert(recv_RD, counts_Pe)


class AllToAllGroupedMatmul(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
        replay_routed_input,
    ):
        rank = torch.distributed.get_rank(group)
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        ep_world_size = torch.distributed.get_world_size(group)
        counts_Pe = recv_counts.reshape(ep_world_size, -1)
        group_list_tensor = counts_Pe.sum(dim=0)

        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()

        output, shared_output, permute_output = torch_npu.npu_alltoallv_gmm(
            inputs,
            weights,
            hcomm,
            ep_world_size,
            send_counts,
            recv_counts,
            mm_x=shared_inputs,
            mm_weight=shared_weight,
            permute_out_flag=replay_routed_input is None,
        )

        tokens_TD = inverse_indices_N = replay_counts = None
        if replay_routed_input is not None:
            tokens_TD, inverse_indices_N, ctx.top_k = replay_routed_input
            replay_counts = counts_Pe
            num_experts = counts_Pe.size(1)
            ctx.ep_group = group
            ctx.send_splits = [
                sum(send_counts[start : start + num_experts])
                for start in range(0, len(send_counts), num_experts)
            ]
            ctx.recv_splits = [
                sum(recv_counts[start : start + num_experts])
                for start in range(0, len(recv_counts), num_experts)
            ]
        ctx.save_for_backward(
            weights,
            shared_inputs,
            shared_weight,
            permute_output,
            tokens_TD,
            inverse_indices_N,
            replay_counts,
        )
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group_list_tensor = group_list_tensor
        return output, shared_output

    @staticmethod
    def backward(ctx, *grad_output):
        output_grad, shared_output_grad = grad_output
        (
            weights,
            shared_inputs,
            shared_weight,
            permute_output,
            tokens_TD,
            inverse_indices_N,
            counts_Pe,
        ) = ctx.saved_tensors
        hcomm = ctx.hcomm
        ep_world_size = ctx.ep_world_size
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        group_list_tensor = ctx.group_list_tensor

        inputs_grad, shared_inputs_grad = torch_npu.npu_gmm_alltoallv(
            output_grad,
            weights,
            hcomm,
            ep_world_size,
            recv_counts,
            send_counts,
            mm_x=shared_output_grad,
            mm_weight=shared_weight,
            trans_gmm_weight=True,
        )

        if tokens_TD is not None:
            permute_output = _replay_routed_input(
                tokens_TD,
                inverse_indices_N,
                ctx.top_k,
                counts_Pe,
                ctx.send_splits,
                ctx.recv_splits,
                ctx.ep_group,
            )

        weights_grad = torch_npu.npu_grouped_matmul(
            [permute_output.T],
            [output_grad],
            bias=None,
            group_list=group_list_tensor,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        shared_weight_grad = (
            None
            if shared_inputs is None
            else torch.matmul(shared_inputs.T, shared_output_grad)
        )
        return (
            inputs_grad,
            weights_grad,
            None,
            None,
            None,
            shared_inputs_grad,
            shared_weight_grad,
            None,
        )


class GroupedMatmulAllToAll(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
    ):
        rank = torch.distributed.get_rank(group)
        global_rank = torch.distributed.get_global_rank(group, rank)
        hcomm = group._get_backend(torch.device("npu")).get_hccl_comm_name(global_rank)
        ep_world_size = torch.distributed.get_world_size(group)
        group_list_tensor = send_counts.reshape(ep_world_size, -1).sum(dim=0)

        send_counts = send_counts.tolist()
        recv_counts = recv_counts.tolist()

        output, shared_output = torch_npu.npu_gmm_alltoallv(
            inputs,
            weights,
            hcomm,
            ep_world_size,
            send_counts,
            recv_counts,
            mm_x=shared_inputs,
            mm_weight=shared_weight,
        )

        ctx.save_for_backward(inputs, weights, shared_inputs, shared_weight)
        ctx.hcomm = hcomm
        ctx.ep_world_size = ep_world_size
        ctx.send_counts = send_counts
        ctx.recv_counts = recv_counts
        ctx.group_list_tensor = group_list_tensor
        return output, shared_output

    @staticmethod
    def backward(ctx, *grad_output):
        output_grad, shared_output_grad = grad_output
        inputs, weights, shared_inputs, shared_weight = ctx.saved_tensors
        hcomm = ctx.hcomm
        ep_world_size = ctx.ep_world_size
        send_counts = ctx.send_counts
        recv_counts = ctx.recv_counts
        group_list_tensor = ctx.group_list_tensor

        inputs_grad, shared_inputs_grad, permute_grad = torch_npu.npu_alltoallv_gmm(
            output_grad,
            weights,
            hcomm,
            ep_world_size,
            recv_counts,
            send_counts,
            mm_x=shared_output_grad,
            mm_weight=shared_weight,
            permute_out_flag=True,
            trans_gmm_weight=True,
        )

        weights_grad = torch_npu.npu_grouped_matmul(
            [inputs.T],
            [permute_grad],
            bias=None,
            group_list=group_list_tensor,
            split_item=3,
            group_type=2,
            group_list_type=1,
        )[0]

        shared_weight_grad = (
            None
            if shared_inputs is None
            else torch.matmul(shared_inputs.T, shared_output_grad)
        )
        return (
            inputs_grad,
            weights_grad,
            None,
            None,
            None,
            shared_inputs_grad,
            shared_weight_grad,
        )


# send_counts: [num_global_experts], recv_counts: [num_global_experts]
def all2all_grouped_matmul(
    inputs,
    weights,
    group,
    send_counts,
    recv_counts,
    shared_inputs=None,
    shared_weight=None,
    *,
    replay_routed_input: RoutedInputReplay | None = None,
):
    output = AllToAllGroupedMatmul.apply(
        inputs,
        weights,
        group,
        send_counts,
        recv_counts,
        shared_inputs,
        shared_weight,
        replay_routed_input,
    )
    if shared_inputs is not None:
        return output[0], output[1]  # experts output and shared experts outputs
    return output[0]


def grouped_matmul_all2all(
    inputs,
    weights,
    group,
    send_counts,
    recv_counts,
    shared_inputs=None,
    shared_weight=None,
):
    output = GroupedMatmulAllToAll.apply(
        inputs, weights, group, send_counts, recv_counts, shared_inputs, shared_weight
    )
    if shared_inputs is not None:
        return output[0], output[1]  # experts output and shared experts outputs
    return output[0]
