"""Ascend token permutation and unpermutation with fused NPU operators."""

import torch
import torch_npu


def permute(
    tokens: torch.Tensor,
    indices: torch.Tensor,
    num_out_tokens: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Group token copies by expert using the fused NPU operator.

    Args:
        tokens: Input token features of shape [T, D].
        indices: Selected expert IDs of shape [T, K].
        num_out_tokens: Slice the expert-sorted rows; None or 0 keeps all copies.

    Returns:
        permuted_tokens: Expert-sorted token features of shape [N, D], with
            N=T*K by default or truncated by num_out_tokens.
        sorted_indices: Full inverse mapping of shape [T*K], used by unpermute.
            For the second EP permutation, K=1 and IDs refer to local experts.
    """
    return torch_npu.npu_moe_token_permute(tokens, indices, num_out_tokens=num_out_tokens)


def unpermute(
    permuted_tokens: torch.Tensor,
    sorted_indices: torch.Tensor,
    probs: torch.Tensor | None = None,
    restore_shape: tuple[int, ...] | list[int] | None = None,
) -> torch.Tensor:
    """Restore token order using the fused NPU operator.

    Args:
        permuted_tokens: Expert-sorted token features of shape [N, D].
        sorted_indices: Inverse mapping [N] returned by the paired permute.
        probs: Routing weights [T, K]; None restores order with K=1.
        restore_shape: Optional restored shape passed to the NPU operator.

    Returns:
        unpermuted_tokens: Tokens restored to original order.
    """
    token_dtype = permuted_tokens.dtype
    if probs is not None:
        permuted_tokens = permuted_tokens.to(probs.dtype)
    output = torch_npu.npu_moe_token_unpermute(
        permuted_tokens, sorted_indices, probs, restore_shape=restore_shape
    )
    # Router weights can be FP32; preserve the token dtype at the MoE boundary.
    return output.to(token_dtype)
