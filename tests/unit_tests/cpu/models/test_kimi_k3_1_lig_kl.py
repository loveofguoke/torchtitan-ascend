# SPDX-License-Identifier: Apache-2.0
"""Check fused indexer KL against an independent dense CPU objective."""

from dataclasses import replace

import pytest
import torch
import torch.nn.functional as F
from torchtitan.models.common.attention import VarlenAttentionMetadata

from torchtitan_ascend.models.kimi_k3_1.dsa import IndexerFeatures
from torchtitan_ascend.overrides.kimi_k3_1.lig_kl import npu_lig_kl
from torchtitan_ascend.recipes.kimi_k3_1 import kimi_k3_1_multimodal_debugmodel_npu


def _debug_lig_config(num_heads: int):
    attention = (
        kimi_k3_1_multimodal_debugmodel_npu(seq_len=1152)
        .model.layers[3]
        .attention.inner_attention
    )
    # SFA/LIG require a 512-wide latent; retain the debug model's 64-wide RoPE tail.
    return replace(
        attention,
        n_heads=num_heads,
        kv_lora_rank=512,
        wkv_b=replace(
            attention.wkv_b,
            in_features=512,
            out_features=num_heads
            * (attention.qk_nope_head_dim + attention.v_head_dim),
        ),
        aux_loss=replace(attention.aux_loss, num_heads=num_heads),
    )


def test_lig_kl_requires_unweighted_teacher():
    attention = _debug_lig_config(4)
    with pytest.raises(ValueError, match="unweighted"):
        npu_lig_kl(
            replace(attention, aux_loss=replace(attention.aux_loss, mass_weighted=True))
        )
    fused = npu_lig_kl(attention)
    assert fused.aux_loss.coeff == attention.aux_loss.coeff == 0.01


@pytest.mark.npu
@pytest.mark.parametrize(
    "heads,tokens,empty",
    [(65, 128, "none"), (96, 128, "partial"), (96, 128, "all"), (4, 1152, "partial")],
)
def test_lig_kl_loss_gradients_scaling_and_teacher_detach(heads, tokens, empty):
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires one visible Ascend NPU")
    torch.manual_seed(42)
    top_k, index_heads = 1024, 8
    config = _debug_lig_config(heads)
    fused = npu_lig_kl(config).build().to("npu")
    q = torch.randn(tokens, heads, 576).bfloat16()
    k = torch.randn(tokens, 576).bfloat16()
    qi = (torch.randn(tokens, index_heads, 128) * 0.1).bfloat16().requires_grad_()
    ki = (torch.randn(tokens, 128) * 0.1).bfloat16().requires_grad_()
    weights = (torch.randn(tokens, index_heads) * 0.1).requires_grad_()
    positions = torch.arange(tokens)
    documents = torch.ge(positions, 32)
    valid = torch.le(positions[None, :], positions[:, None]) & torch.eq(
        documents[None, :], documents[:, None]
    )
    padding = torch.ge(positions, tokens - 16 if empty == "partial" else tokens)
    if empty == "all":
        padding.fill_(True)
    valid &= ~padding[:, None] & ~padding[None, :]
    index_logits = torch.einsum(
        "thd,sd->ths",  # codespell:ignore ths
        qi.float(),
        ki.float(),
    )
    scores = (index_logits.relu() * weights[:, :, None]).sum(1)
    selected, indices = (
        scores.detach().masked_fill(~valid, -torch.inf).topk(min(top_k, tokens), dim=-1)
    )
    indices = indices.masked_fill(~selected.isfinite(), -1)
    indices = F.pad(indices, (0, top_k - indices.size(1)), value=-1)
    selected_valid = torch.ge(indices, 0)
    logits = (
        torch.einsum("thd,sd->ths", q.float(), k.float())  # codespell:ignore ths
        * config.softmax_scale
    )  # codespell:ignore ths
    logits = logits.gather(2, indices.clamp_min(0)[:, None].expand(-1, heads, -1))
    logits = logits.masked_fill(~selected_valid[:, None], -torch.inf)
    lse = logits.logsumexp(-1)
    safe_lse = lse.masked_fill(~lse.isfinite(), 0)
    teacher = (logits - safe_lse[:, :, None]).exp().mean(1)
    teacher = teacher / teacher.sum(-1, keepdim=True).clamp_min(
        torch.finfo(torch.float32).tiny
    )
    student = scores.gather(1, indices.clamp_min(0)).masked_fill(
        ~selected_valid, -torch.inf
    )
    student = student.masked_fill(~selected_valid.any(-1)[:, None], 0).log_softmax(-1)
    student = student.masked_fill(~selected_valid, 0)
    raw_loss = (torch.special.xlogy(teacher, teacher) - teacher * student).sum()
    denominator = tokens * 3
    reference = torch.autograd.grad(raw_loss * 0.01 / denominator, (qi, ki, weights))

    def leaf(value):
        return value.detach().npu().requires_grad_()

    actual_features = IndexerFeatures(*(leaf(value) for value in (qi, ki, weights)))
    teacher_q, teacher_k, teacher_lse = (leaf(value) for value in (q, k, lse))
    carrier = torch.randn(tokens, 1, device="npu", requires_grad=True)
    offsets = torch.tensor([0, 32, tokens], dtype=torch.int32, device="npu")
    actual = fused._apply_aux_loss(
        teacher_q,
        teacher_k,
        indices.int().npu(),
        None,
        teacher_lse,
        carrier=carrier,
        denominator=torch.tensor(denominator, device="npu"),
        padding_mask_T=padding.npu(),
        attention_metadata=VarlenAttentionMetadata(
            offsets, offsets, tokens - 32, tokens - 32
        ),
        indexer_features=actual_features,
    )
    torch.testing.assert_close(actual, carrier, rtol=0, atol=0)
    torch.testing.assert_close(
        fused.aux_loss.instance_acc.cpu(),
        raw_loss.detach() / denominator,
        rtol=1e-4,
        atol=1e-6,
    )
    actual.sum().backward()
    assert all(value.grad is None for value in (teacher_q, teacher_k, teacher_lse))
    torch.testing.assert_close(carrier.grad, torch.ones_like(carrier))
    for value, expected in zip(actual_features, reference):
        actual_gradient = value.grad.cpu().float()
        expected = expected.float()
        assert actual_gradient.isfinite().all()
        if empty == "all":
            assert actual_gradient.count_nonzero() == 0
        else:
            assert (actual_gradient - expected).norm() <= 0.01 * expected.norm()
            assert (
                actual_gradient - expected
            ).abs().max() <= 0.01 * expected.abs().max() + 1e-10
