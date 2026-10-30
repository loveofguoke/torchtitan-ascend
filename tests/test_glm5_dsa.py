# Copyright (c) 2026, Huawei Technologies Co., Ltd. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
"""DSA document isolation, native attention parity, and indexer KL gradients."""

from dataclasses import replace

import pytest
import torch

from torchtitan_ascend.models.glm5.dsa import DsaMetadata


@pytest.mark.parametrize(
    "order,start,end",
    [
        ((), 0, None),
        ((), 2, 6),
        ((0, 4, 2, 6, 1, 5, 3, 7), 0, 4),
    ],
)
def test_packing_preserves_document_causality_and_padding(order, start, end):
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3])
    padding = torch.tensor([False, False, True, False, False, True, False, False])
    metadata = replace(DsaMetadata.from_positions(positions, padding), order=order, query_start=start, query_end=end)
    packing = metadata.packing
    order = metadata.order or tuple(range(8))
    local_order = order[start:end]
    actual = torch.zeros(len(local_order), 8, dtype=torch.bool)
    qs = ks = 0
    for qe, ke in zip(packing.query_lengths, packing.key_lengths, strict=True):
        qcount, kcount = qe - qs, ke - ks
        for i, qrow in enumerate(packing.queries[qs:qe]):
            for key in packing.keys[ks : ks + kcount - qcount + i + 1]:
                actual[qrow, order[key]] = True
        qs, ks = qe, ke
    document_ids = torch.eq(positions, 0).cumsum(0)
    original = torch.tensor(local_order)
    keys = torch.arange(8)
    expected = (
        (original[:, None] >= keys[None, :])
        & (document_ids[original, None] == document_ids[None, :])
        & ~padding[original, None]
        & ~padding[None, :]
    )
    torch.testing.assert_close(actual, expected)


def test_all_padding_has_no_kernel_queries():
    metadata = DsaMetadata.from_positions(torch.arange(4), torch.ones(4, dtype=torch.bool))
    assert metadata.packing.queries == ()
    assert metadata.packing.keys == ()
    assert metadata.packing.query_count == 4


@pytest.fixture
def npu():
    torch_npu = pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("requires an Ascend NPU")
    torch.npu.set_device(0)
    return torch_npu


def _rand(shape, *, scale=0.5):
    return (torch.randn(shape) * scale).to(torch.bfloat16)


@pytest.mark.npu
def test_fused_indexer_topk_matches_packed_document_scores(npu):
    torch.manual_seed(89)
    tokens, index_heads, head_dim, topk = 32, 8, 128, 8
    qi, ki = _rand((tokens, index_heads, head_dim)), _rand((tokens, 1, head_dim))
    weights = torch.randn(tokens, index_heads) * (index_heads * head_dim) ** -0.5
    lengths = torch.tensor([16, 32], dtype=torch.int32).npu()
    indices, _ = npu.npu_lightning_indexer(
        qi.npu(),
        ki.npu(),
        weights.npu(),
        actual_seq_lengths_query=lengths,
        actual_seq_lengths_key=lengths,
        layout_query="TND",
        layout_key="TND",
        sparse_count=topk,
    )
    indices = indices.cpu().squeeze(1)
    for start in (0, 16):
        q = qi[start : start + 16].float()
        k = ki[start : start + 16, 0].float()
        scores = torch.einsum(
            "tn,tns->ts",
            weights[start : start + 16],
            torch.einsum("tnd,sd->tns", q, k).relu(),
        )
        for row in range(16):
            selected = indices[start + row]
            selected = selected[selected >= 0].long()
            assert selected.numel() == min(topk, row + 1)
            assert (selected <= row).all()
            actual = scores[row, selected].sort(descending=True).values
            expected = scores[row, : row + 1].topk(min(topk, row + 1)).values
            torch.testing.assert_close(actual, expected, rtol=0.01, atol=0.001)


@pytest.mark.npu
def test_sparse_attention_forward_backward_matches_reference(npu):
    torch.manual_seed(61)
    tokens, heads, latent_dim, rope_dim = 16, 32, 512, 64
    inputs = [
        _rand((tokens, heads, latent_dim)),
        _rand((tokens, 1, latent_dim)),
        _rand((tokens, heads, rope_dim)),
        _rand((tokens, 1, rope_dim)),
    ]
    q, k, qr, kr = [x.npu().detach().requires_grad_() for x in inputs]
    indices = torch.full((tokens, 1, 1024), -1, dtype=torch.int32)
    for row in range(tokens):
        indices[row, 0, : row + 1] = torch.arange(row + 1)
    lengths = torch.tensor([tokens], dtype=torch.int32).npu()
    scale = 256**-0.5
    out, _, _ = npu.npu_sparse_flash_attention(
        q,
        k,
        k,
        indices.npu(),
        scale,
        query_rope=qr,
        key_rope=kr,
        actual_seq_lengths_query=lengths,
        actual_seq_lengths_kv=lengths,
        layout_query="TND",
        layout_kv="TND",
        attention_mode=2,
        return_softmax_lse=True,
    )
    refs = [x.float().requires_grad_() for x in inputs]
    rq, rk, rqr, rkr = refs
    scores = (torch.einsum("tnc,sc->nts", rq, rk[:, 0]) + torch.einsum("tnr,sr->nts", rqr, rkr[:, 0])) * scale
    scores = scores.masked_fill(torch.ones(tokens, tokens, dtype=torch.bool).triu(1), -torch.inf)
    expected = torch.einsum("nts,sc->tnc", scores.softmax(-1), rk[:, 0])
    grad = torch.randn_like(expected)
    out.backward(grad.to(device="npu", dtype=torch.bfloat16))
    expected.backward(grad.to(torch.bfloat16).float())
    torch.testing.assert_close(out.float().cpu(), expected, rtol=0.015, atol=0.005)
    for actual, reference in zip((q, k, qr, kr), refs, strict=True):
        torch.testing.assert_close(actual.grad.float().cpu(), reference.grad, rtol=0.04, atol=0.025)


@pytest.mark.npu
@pytest.mark.parametrize("dense", [True, False])
def test_fused_kl_matches_loss_and_scaled_indexer_gradients(npu, dense):
    from torchtitan_ascend.overrides.glm5_dsa import _IndexerKL

    torch.manual_seed(73)
    tokens, heads, index_heads, head_dim, rope_dim = 16, 32, 8, 128 if dense else 512, 64
    qi_cpu, ki_cpu = _rand((tokens, index_heads, 128)), _rand((tokens, 1, 128))
    weights_cpu = torch.randn(tokens, index_heads) * 0.03
    qi, ki, weights = [x.npu().detach().requires_grad_() for x in (qi_cpu, ki_cpu, weights_cpu)]
    qcpu = _rand((tokens, heads, head_dim))
    kcpu = _rand((tokens, heads if dense else 1, head_dim))
    qrcpu, krcpu = _rand((tokens, heads, rope_dim)), _rand((tokens, heads if dense else 1, rope_dim))
    q, k, qr, kr = [x.npu() for x in (qcpu, kcpu, qrcpu, krcpu)]
    indices = torch.full((tokens, 1, 1024), -1, dtype=torch.int32)
    for row in range(tokens):
        indices[row, 0, : row + 1] = torch.arange(row + 1)
    scale = 256**-0.5
    if dense:
        qfull, kfull = torch.cat((q, qr), -1), torch.cat((k, kr), -1)
        mask = torch.ones(2048, 2048, dtype=torch.bool).triu(1).npu()
        _, maximum, total, *_ = npu.npu_fusion_attention(
            qfull,
            kfull,
            q,
            heads,
            "TND",
            atten_mask=mask,
            scale=scale,
            actual_seq_qlen=[tokens],
            actual_seq_kvlen=[tokens],
            sparse_mode=3,
            softmax_layout="TND",
        )
        maximum, total = maximum.transpose(0, 1)[..., :1].contiguous(), total.transpose(0, 1)[..., :1].contiguous()
        scores = (
            torch.einsum("tnd,snd->nts", qcpu.float(), kcpu.float())
            + torch.einsum("tnr,snr->nts", qrcpu.float(), krcpu.float())
        ) * scale
    else:
        lengths = torch.tensor([tokens], dtype=torch.int32).npu()
        _, maximum, total = npu.npu_sparse_flash_attention(
            q,
            k,
            k,
            indices.npu(),
            scale,
            query_rope=qr,
            key_rope=kr,
            actual_seq_lengths_query=lengths,
            actual_seq_lengths_kv=lengths,
            layout_query="TND",
            layout_kv="TND",
            attention_mode=2,
            return_softmax_lse=True,
        )
        scores = (
            torch.einsum("tnd,sd->nts", qcpu.float(), kcpu[:, 0].float())
            + torch.einsum("tnr,sr->nts", qrcpu.float(), krcpu[:, 0].float())
        ) * scale
    actual = _IndexerKL.apply(
        qi,
        ki,
        weights,
        q,
        k,
        qr,
        kr,
        maximum,
        total,
        indices.npu(),
        (tokens,),
        (tokens,),
        scale,
        dense,
    )
    rqi, rki, rw = [x.float().requires_grad_() for x in (qi_cpu, ki_cpu, weights_cpu)]
    logits = torch.einsum("tn,tns->ts", rw, torch.einsum("tnd,sd->tns", rqi, rki[:, 0]).relu())
    valid = torch.ones(tokens, tokens, dtype=torch.bool).tril()
    target = scores.masked_fill(~valid, -torch.inf).softmax(-1).mean(0).detach()
    log_probs = logits.masked_fill(~valid, -torch.inf).log_softmax(-1)
    expected = (target * (target.clamp_min(1e-30).log() - log_probs.masked_fill(~valid, 0))).sum()
    (actual * 0.37).backward()
    (expected * 0.37).backward()
    torch.testing.assert_close(actual.cpu(), expected, rtol=0.03, atol=0.002)
    for fused, reference in zip((qi, ki, weights), (rqi, rki, rw), strict=True):
        torch.testing.assert_close(fused.grad.float().cpu(), reference.grad, rtol=0.06, atol=0.002)


@pytest.mark.npu
@pytest.mark.parametrize("stage", ["sparse", "warmup"])
def test_all_padding_keeps_attention_and_indexer_backward_dependencies(npu, stage):
    from torchtitan.config.override import apply_overrides
    from torchtitan_ascend import setup
    from torchtitan_ascend.recipes.glm5 import glm5_2_debugmodel_npu

    setup()
    config = glm5_2_debugmodel_npu(seq_len=8, indexer_stage=stage)
    apply_overrides(config.override, config)
    attention = config.model.layers[0].attention.build().to(device="npu", dtype=torch.bfloat16)
    attention.init_states()
    if stage == "warmup":
        attention.requires_grad_(False)
        attention.indexer.requires_grad_(True)
    inputs = torch.randn(8, 256, device="npu", dtype=torch.bfloat16, requires_grad=stage != "warmup")
    positions = torch.arange(8, device="npu")
    metadata = DsaMetadata.from_positions(positions, torch.ones(8, device="npu", dtype=torch.bool))
    output = attention(inputs, metadata, positions)
    output.float().sum().backward()
    torch.testing.assert_close(output, torch.zeros_like(output))
    for parameter in attention.indexer.parameters():
        assert parameter.grad is not None
        torch.testing.assert_close(parameter.grad, torch.zeros_like(parameter.grad))
    if stage == "sparse":
        for name in ("wkv_a", "wq_b"):
            gradient = getattr(attention, name).weight.grad
            assert gradient is not None
            torch.testing.assert_close(gradient, torch.zeros_like(gradient))


@pytest.mark.npu
@pytest.mark.parametrize("stage", ["sparse", "warmup"])
def test_scaled_recipe_fused_attention_and_indexer_backward(npu, stage):
    from torchtitan.config.override import apply_overrides
    from torchtitan.models.common.aux_loss import AuxLoss
    from torchtitan_ascend import setup
    from torchtitan_ascend.recipes.glm5 import glm5_2_scaled_debugmodel_npu

    setup()
    config = glm5_2_scaled_debugmodel_npu(seq_len=8, indexer_stage=stage)
    apply_overrides(config.override, config)
    attention_config = config.model.layers[0].attention
    attention = attention_config.build().to(device="npu", dtype=torch.bfloat16)
    attention.init_states()
    if stage == "warmup":
        attention.requires_grad_(False)
        attention.indexer.requires_grad_(True)
    inputs = torch.randn(8, attention_config.dim, device="npu", dtype=torch.bfloat16)
    positions = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], device="npu")
    metadata = DsaMetadata.from_positions(positions)
    AuxLoss.set_step_denominator(torch.tensor(8.0, device="npu"))
    output = attention(inputs, metadata, positions)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    gradient = attention.indexer.wq_b.weight.grad
    assert gradient is not None and torch.isfinite(gradient).all()
    assert gradient.abs().max() > 0
    if stage == "warmup":
        assert attention.wkv_a.weight.grad is None
    else:
        assert torch.isfinite(attention.wkv_a.weight.grad).all()


@pytest.mark.npu
@pytest.mark.parametrize("nope_dim", [128, 192])
@pytest.mark.parametrize("keys", [8192, 8193])
def test_warmup_full_key_width_boundary(npu, nope_dim, keys):
    from torchtitan.config.override import apply_overrides
    from torchtitan.models.common.aux_loss import AuxLoss
    from torchtitan_ascend import setup
    from torchtitan_ascend.recipes.glm5 import glm5_2_debugmodel_npu

    torch.manual_seed(61)
    setup()
    queries, heads, index_heads, latent_dim, rope_dim = 32, 32, 8, 512, 64
    config = glm5_2_debugmodel_npu(seq_len=queries, indexer_stage="warmup")
    attention_config = config.model.layers[0].attention
    attention_config.qk_nope_head_dim = nope_dim
    attention_config.wq_b.out_features = heads * (nope_dim + rope_dim)
    attention_config.wkv_b.out_features = heads * (nope_dim + attention_config.v_head_dim)
    apply_overrides(config.override, config)
    attention = config.model.layers[0].attention.build().to(device="npu", dtype=torch.bfloat16)
    attention.init_states()
    attention.requires_grad_(False)

    q_nope = _rand((queries, heads, nope_dim), scale=0.1).npu()
    wk = _rand((heads, nope_dim, latent_dim), scale=0.02).npu()
    query = torch.einsum("tnp,npc->tnc", q_nope, wk)
    key = _rand((keys, 1, latent_dim), scale=0.1).npu()
    qr, kr = _rand((queries, heads, rope_dim), scale=0.1).npu(), _rand((keys, 1, rope_dim), scale=0.1).npu()
    features = (
        _rand((queries, index_heads, 128), scale=0.1).npu().requires_grad_(),
        _rand((keys, 1, 128), scale=0.1).npu().requires_grad_(),
        (torch.randn(queries, index_heads, device="npu") * 0.1 * (index_heads * 128) ** -0.5).requires_grad_(),
    )
    # The final 32 queries see a long KV prefix, as on a CP rank. This tests
    # the production attention/KL boundary without a full long-context model.
    metadata = DsaMetadata(
        documents=(tuple(range(keys)),), token_count=keys, query_start=keys - queries, query_end=keys
    )
    AuxLoss.set_step_denominator(torch.tensor(float(queries), device="npu"))
    if nope_dim == 192 and keys > 8192:
        with pytest.raises(ValueError, match="Sparse KL supports at most 8192 selected keys per query"):
            attention._fused_attention(query, key, qr, kr, q_nope, wk, metadata, features, None)
        return

    output, _ = attention._fused_attention(query, key, qr, kr, q_nope, wk, metadata, features, None)
    output.float().square().mean().backward()
    assert torch.isfinite(output).all()
    for feature in features:
        assert feature.grad is not None and torch.isfinite(feature.grad).all()
        assert feature.grad.abs().max() > 0

    rq, rk, rqr, rkr = [tensor.float().cpu() for tensor in (query, key, qr, kr)]
    scores = torch.einsum("tnc,sc->nts", rq, rk[:, 0]) + torch.einsum("tnr,sr->nts", rqr, rkr[:, 0])
    scores = scores * attention.softmax_scale
    valid = torch.arange(keys)[None, :] <= torch.arange(keys - queries, keys)[:, None]
    probabilities = scores.masked_fill(~valid, -torch.inf).softmax(-1)
    expected = torch.einsum("nts,sc->tnc", probabilities, rk[:, 0])
    torch.testing.assert_close(output.float().cpu(), expected, rtol=0.015, atol=0.005)
