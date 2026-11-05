# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# Adapted from pytorch/torchtitan PR #4751. See LICENSE-BSD-3-Clause.
"""Ordered MoonEP operations retained across activation checkpoint replay."""

from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch_remat as remat
from torch import Tensor

from .gmm import rows_mm, weight_grad
from .runtime import current_runtime


@dataclass
class _Plan:
    comm_plan: object
    num_valid_rows: Tensor
    activation: Callable[..., Tensor] | None = None
    weight_projections: tuple = ()


_plans: dict[int, _Plan] = {}
_next_plan_id = 0


def _clear_tail(rows, num_valid_rows):
    # masked_fill also removes NaN/Inf in undefined storage (multiplying by 0 does not).
    return rows.masked_fill(
        torch.arange(rows.shape[0], device=rows.device)[:, None] >= num_valid_rows, 0
    )


def _validate_plan(
    full_group_end_offsets, num_dispatched_rows, ep_rank, num_local_experts, num_experts
):
    # Debug-only device-to-host read: never pay this synchronization by default.
    full_group_end_offsets = full_group_end_offsets.detach().to(
        device="cpu", dtype=torch.int64
    )
    if (
        full_group_end_offsets.ndim != 1
        or full_group_end_offsets.numel() != num_experts + num_local_experts
    ):
        raise ValueError("MoonEP plan group ends have an invalid shape")
    group_token_counts = full_group_end_offsets.diff(
        prepend=full_group_end_offsets.new_zeros(1)
    )
    if (
        bool((group_token_counts < 0).any())
        or int(full_group_end_offsets[-1]) > num_dispatched_rows
    ):
        raise ValueError(
            "MoonEP plan group ends must be monotonic and within dispatched capacity"
        )
    home_group_start = ep_rank * num_local_experts
    if bool(group_token_counts[:home_group_start].any()) or bool(
        group_token_counts[home_group_start + num_local_experts : num_experts].any()
    ):
        raise ValueError(
            "MoonEP plan locality violation: nonlocal home groups contain tokens"
        )


@torch.library.custom_op("ascend_moonep::dispatch", mutates_args=())
def _dispatch(
    source_input: Tensor,
    topk_scores: Tensor,
    topk_expert_ids: Tensor,
    tokens_per_expert: Tensor,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    global _next_plan_id
    if _plans:
        raise RuntimeError(
            "Combine the outstanding MoonEP dispatch before starting another"
        )
    runtime = current_runtime()
    dispatched_input, route_weights, full_group_end_offsets, comm_plan = runtime.call(
        "dispatch",
        source_input,
        topk_scores,
        topk_expert_ids,
        tokens_per_expert,
        zero_copy=False,
    )
    if runtime.communication_options.get("validate_routing"):
        _validate_plan(
            full_group_end_offsets,
            dispatched_input.shape[0],
            runtime.ep_rank,
            runtime.num_local_experts,
            runtime.num_experts,
        )
    # Ascend returns global [E+B] ends, but GEMM only reads local home rows and slots.
    home_group_start = runtime.ep_rank * runtime.num_local_experts
    group_end_offsets = torch.cat(
        (
            full_group_end_offsets[
                home_group_start : home_group_start + runtime.num_local_experts
            ],
            full_group_end_offsets[runtime.num_experts :],
        )
    )
    group_end_offsets[-1] = dispatched_input.shape[0]
    _next_plan_id += 1
    _plans[_next_plan_id] = _Plan(comm_plan, full_group_end_offsets[-1])
    dispatched_input = _clear_tail(dispatched_input, full_group_end_offsets[-1])
    route_weights = _clear_tail(
        route_weights[:, None], full_group_end_offsets[-1]
    ).flatten()
    return (
        dispatched_input,
        route_weights,
        group_end_offsets,
        torch.tensor([_next_plan_id], device="cpu"),
    )


def _dispatch_setup(ctx, inputs, output):
    ctx.plan_state = _plans.get(int(output[3]))
    ctx.mark_non_differentiable(output[2], output[3])
    ctx.set_materialize_grads(False)


def _dispatch_backward(ctx, grad_dispatched_input, grad_route_weights, *_):
    grad_source_input, grad_topk_scores, _ = current_runtime().call(
        "combine",
        plan=ctx.plan_state.comm_plan,
        hidden_nvsh=grad_dispatched_input.contiguous(),
        route_weights_nvs=None
        if grad_route_weights is None
        else grad_route_weights.float().contiguous(),
    )
    return grad_source_input, grad_topk_scores, None, None


@torch.library.custom_op("ascend_moonep::experts", mutates_args=())
def _experts(
    dispatched_input: Tensor,
    route_weights: Tensor,
    w13_weight: Tensor,
    w2_weight: Tensor,
    group_end_offsets: Tensor,
    plan_id: Tensor,
    source_input: Tensor | None,
) -> tuple[Tensor, Tensor, Tensor]:
    plan_state = _plans[int(plan_id)]
    current_runtime().trace("experts.forward.begin")
    assert plan_state.activation is not None
    prefetched_weights = current_runtime().prefetch(
        plan_state.comm_plan, plan_state.weight_projections
    )
    w13_projection = next(prefetched_weights)
    gate_up_features = _clear_tail(
        rows_mm(dispatched_input, w13_projection, group_end_offsets),
        plan_state.num_valid_rows,
    )
    activated_features = plan_state.activation(
        gate_up_features.unflatten(-1, (2, -1)), offsets=group_end_offsets
    )
    expert_output = _clear_tail(
        rows_mm(activated_features, next(prefetched_weights), group_end_offsets),
        plan_state.num_valid_rows,
    )
    current_runtime().trace("experts.forward.end")
    return (
        (expert_output.float() * route_weights[:, None]).to(dispatched_input.dtype),
        gate_up_features,
        expert_output,
    )


def _experts_setup(ctx, inputs, output):
    (
        dispatched_input,
        route_weights,
        w13_weight,
        w2_weight,
        group_end_offsets,
        plan_id,
        source_input,
    ) = inputs
    ctx.plan_state = _plans.get(int(plan_id))
    ctx.mark_non_differentiable(output[1], output[2])
    ctx.set_materialize_grads(False)
    ctx.recompute_dispatch = source_input is not None
    ctx.save_for_backward(
        source_input if source_input is not None else dispatched_input,
        route_weights,
        output[1],
        output[2],
        w13_weight,
        w2_weight,
        group_end_offsets,
    )


def _experts_backward(ctx, grad_weighted_output, *_):
    runtime = current_runtime()
    runtime.trace("experts.backward.begin")
    (
        saved_input,
        route_weights,
        gate_up_features,
        expert_output,
        w13_weight,
        w2_weight,
        group_end_offsets,
    ) = ctx.saved_tensors
    dispatched_input = saved_input
    plan_state = ctx.plan_state
    if ctx.recompute_dispatch:
        dispatched_input = _clear_tail(
            runtime.call(
                "dispatch",
                saved_input.contiguous(),
                plan=plan_state.comm_plan,
                zero_copy=False,
            )[0],
            plan_state.num_valid_rows,
        )
    prefetched_weights = runtime.prefetch(
        plan_state.comm_plan, tuple(reversed(plan_state.weight_projections))
    )
    w2_projection = next(prefetched_weights)
    grad_weighted_output = _clear_tail(
        grad_weighted_output.contiguous(), plan_state.num_valid_rows
    )
    gate_up_features = gate_up_features.detach().requires_grad_()
    with torch.enable_grad():
        activated_features = plan_state.activation(
            gate_up_features.unflatten(-1, (2, -1)), offsets=group_end_offsets
        )
    grad_route_weights = (grad_weighted_output.float() * expert_output.float()).sum(-1)
    # Match the forward BF16 cast: scale the output gradient before the down GEMM.
    grad_expert_output = (grad_weighted_output.float() * route_weights[:, None]).to(
        dispatched_input.dtype
    )
    grad_activated_features = _clear_tail(
        rows_mm(grad_expert_output, w2_projection, group_end_offsets, transpose=False),
        plan_state.num_valid_rows,
    )
    (grad_gate_up_features,) = torch.autograd.grad(
        activated_features, gate_up_features, grad_activated_features
    )
    grouped_grad_w13 = weight_grad(
        grad_gate_up_features, dispatched_input, group_end_offsets
    )
    grouped_grad_w2 = weight_grad(
        grad_expert_output, activated_features.detach(), group_end_offsets
    )
    grad_dispatched_input = _clear_tail(
        rows_mm(
            grad_gate_up_features,
            next(prefetched_weights),
            group_end_offsets,
            transpose=False,
        ),
        plan_state.num_valid_rows,
    )
    grad_w13_weight, grad_w2_weight = runtime.reduce(
        plan_state.comm_plan,
        (grouped_grad_w13, grouped_grad_w2),
        (w13_weight.dtype, w2_weight.dtype),
    )
    runtime.trace("experts.backward.end")
    return (
        grad_dispatched_input,
        grad_route_weights,
        grad_w13_weight.reshape(w13_weight.shape),
        grad_w2_weight,
        None,
        None,
        None,
    )


@torch.library.custom_op("ascend_moonep::combine", mutates_args=())
def _combine(
    weighted_expert_output: Tensor, plan_id: Tensor, will_backward: bool
) -> Tensor:
    plan_key = int(plan_id)
    plan_state = _plans[plan_key] if will_backward else _plans.pop(plan_key)
    return current_runtime().call(
        "combine", plan=plan_state.comm_plan, hidden_nvsh=weighted_expert_output
    )[0]


def _combine_setup(ctx, inputs, output):
    ctx.plan_state = _plans.pop(int(inputs[1]), None)


def _combine_backward(ctx, grad_combined_output):
    grad_weighted_output = current_runtime().call(
        "dispatch",
        grad_combined_output.contiguous(),
        plan=ctx.plan_state.comm_plan,
        zero_copy=False,
    )[0]
    return _clear_tail(grad_weighted_output, ctx.plan_state.num_valid_rows), None, None


for _op, _backward, _setup in (
    (_dispatch, _dispatch_backward, _dispatch_setup),
    (_experts, _experts_backward, _experts_setup),
    (_combine, _combine_backward, _combine_setup),
):
    _op.register_autograd(_backward, setup_context=_setup)
    _op.register_effect(torch.library.EffectType.ORDERED)


def dispatch_tokens(
    source_input, topk_scores, topk_expert_ids, tokens_per_expert, region
):
    """Dispatch through a retained region with owned, nonaliasing outputs."""
    outputs = remat.region(_dispatch, region, recompute=False)(
        source_input.contiguous(),
        topk_scores.float().contiguous(),
        topk_expert_ids.int().contiguous(),
        tokens_per_expert.int().contiguous(),
    )
    remat.recompute_needs_tensor(*outputs)
    return outputs


def routed_experts(
    activation,
    weight_projections,
    dispatched_input,
    route_weights,
    w13_weight,
    w2_weight,
    group_end_offsets,
    plan_id,
    region,
    *,
    source_input=None,
):
    """Evaluate and differentiate experts using shared pools, retaining gate/up activations."""
    plan_state = _plans.get(int(plan_id))
    if plan_state is not None:
        plan_state.activation = activation
        plan_state.weight_projections = weight_projections
    weighted_expert_output, _, _ = remat.region(_experts, region, recompute=False)(
        dispatched_input,
        route_weights,
        w13_weight,
        w2_weight,
        group_end_offsets,
        plan_id,
        source_input,
    )
    remat.recompute_needs_tensor(weighted_expert_output)
    return weighted_expert_output


def combine_tokens(weighted_expert_output, plan_id, region):
    """Sum routed outputs, retaining communication across checkpoint replay."""
    will_backward = torch.is_grad_enabled() and weighted_expert_output.requires_grad
    combined_output = remat.region(_combine, region, recompute=False)(
        weighted_expert_output.contiguous(), plan_id, will_backward
    )
    remat.recompute_needs_tensor(combined_output)
    return combined_output


_expert_calls: dict[int, tuple] = {}


@torch.library.custom_op("ascend_moonep::recomputed_experts", mutates_args=())
def _recomputed_experts(
    source_input: Tensor,
    topk_scores: Tensor,
    topk_expert_ids: Tensor,
    tokens_per_expert: Tensor,
    w13_weight: Tensor,
    w2_weight: Tensor,
    call_id: int,
    will_backward: bool,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    dispatched_input, route_weights, group_end_offsets, plan_id = _dispatch(
        source_input, topk_scores, topk_expert_ids, tokens_per_expert
    )
    plan_state = _plans[int(plan_id)]
    plan_state.activation, plan_state.weight_projections = _expert_calls[call_id]
    weighted_expert_output, gate_up_features, expert_output = _experts(
        dispatched_input,
        route_weights,
        w13_weight,
        w2_weight,
        group_end_offsets,
        plan_id,
        source_input,
    )
    combined_output = _combine(weighted_expert_output, plan_id, will_backward)
    return (
        combined_output,
        route_weights,
        gate_up_features,
        expert_output,
        group_end_offsets,
        plan_id,
    )


def _recomputed_setup(ctx, inputs, output):
    source_input, _, _, _, w13_weight, w2_weight, _, _ = inputs
    ctx.plan_state = _plans.pop(int(output[5]), None)
    ctx.recompute_dispatch = True
    ctx.mark_non_differentiable(*output[1:])
    ctx.set_materialize_grads(False)
    ctx.save_for_backward(
        source_input, output[1], output[2], output[3], w13_weight, w2_weight, output[4]
    )


def _recomputed_backward(ctx, grad_combined_output, *_):
    grad_weighted_output = _combine_backward(ctx, grad_combined_output)[0]
    (
        grad_dispatched_input,
        grad_route_weights,
        grad_w13_weight,
        grad_w2_weight,
        *_,
    ) = _experts_backward(ctx, grad_weighted_output)
    grad_source_input, grad_topk_scores, *_ = _dispatch_backward(
        ctx, grad_dispatched_input, grad_route_weights
    )
    return (
        grad_source_input,
        grad_topk_scores,
        None,
        None,
        grad_w13_weight,
        grad_w2_weight,
        None,
        None,
    )


_recomputed_experts.register_autograd(
    _recomputed_backward, setup_context=_recomputed_setup
)
_recomputed_experts.register_effect(torch.library.EffectType.ORDERED)


def recomputed_experts(
    activation,
    weight_projections,
    source_input,
    topk_scores,
    topk_expert_ids,
    tokens_per_expert,
    w13_weight,
    w2_weight,
    region,
):
    """Retain one ordered MoE region; only redispatch the saved source in backward."""
    call_id = id(weight_projections)
    _expert_calls[call_id] = (activation, weight_projections)
    will_backward = torch.is_grad_enabled() and any(
        tensor.requires_grad
        for tensor in (source_input, topk_scores, w13_weight, w2_weight)
    )
    try:
        combined_output, *_ = remat.region(
            _recomputed_experts, region, recompute=False
        )(
            source_input.contiguous(),
            topk_scores.float().contiguous(),
            topk_expert_ids.int().contiguous(),
            tokens_per_expert.int().contiguous(),
            w13_weight,
            w2_weight,
            call_id,
            will_backward,
        )
    finally:
        _expert_calls.pop(call_id)
    remat.recompute_needs_tensor(combined_output)
    return combined_output
