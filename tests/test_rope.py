import sys
from types import SimpleNamespace

import torch

from torchtitan.models.common.rope import ComplexRoPE
from torchtitan_ascend.overrides.rope import AscendComplexRoPE, npu_apply_complex_rope, npu_reshape_for_broadcast


def test_complex_cache_uses_token_first_positions():
    real = torch.arange(32, dtype=torch.float32).view(8, 4)
    cache = torch.complex(real, real + 100)
    positions = torch.tensor([0, 1, 2, 0, 1])

    actual = npu_reshape_for_broadcast(cache, (5, 3, 8), positions)
    expected = cache[positions].view(5, 1, 4)

    assert actual.shape == (5, 1, 4)
    torch.testing.assert_close(actual, expected)


def test_cache_without_positions_uses_token_dimension():
    cache = torch.arange(32, dtype=torch.float32).view(8, 4)

    actual = npu_reshape_for_broadcast(cache, (5, 3, 8))

    assert actual.shape == (5, 1, 4)
    torch.testing.assert_close(actual, cache[:5].view(5, 1, 4))


def test_fused_complex_rope_matches_torchtitan_math(monkeypatch):
    def fake_rotary_mul(x, cos, sin, *, rotary_mode):
        assert rotary_mode == "interleave"
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated_pairs = torch.stack(
            (-x_pairs[..., 1], x_pairs[..., 0]),
            dim=-1,
        )
        return x * cos + rotated_pairs.flatten(-2) * sin

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rotary_mul=fake_rotary_mul),
    )
    query = torch.randn(5, 3, 8, dtype=torch.bfloat16)
    key = torch.randn(5, 1, 8, dtype=torch.bfloat16)
    real = torch.randn(5, 1, 4)
    imag = torch.randn(5, 1, 4)
    norm = torch.sqrt(real.square() + imag.square())
    cache = torch.complex(real / norm, imag / norm)

    actual_query, actual_key = npu_apply_complex_rope(query, key, cache)
    expected_query = (
        torch.view_as_real(torch.view_as_complex(query.float().reshape(5, 3, 4, 2)) * cache).flatten(-2).to(query.dtype)
    )
    expected_key = (
        torch.view_as_real(torch.view_as_complex(key.float().reshape(5, 1, 4, 2)) * cache).flatten(-2).to(key.dtype)
    )

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)


def test_ascend_complex_rope_matches_upstream_on_cpu():
    config = ComplexRoPE.Config(dim=8, max_context_length=16)
    upstream = ComplexRoPE(config)
    ascend = AscendComplexRoPE(AscendComplexRoPE.Config(dim=8, max_context_length=16))
    query = torch.randn(6, 2, 8)
    key = torch.randn(6, 2, 8)
    positions = torch.tensor([0, 3, 2, 5, 4, 1])

    expected_query, expected_key = upstream(query, key, positions)
    actual_query, actual_key = ascend(query, key, positions)  # pyrefly: ignore [not-callable]

    torch.testing.assert_close(actual_query, expected_query)
    torch.testing.assert_close(actual_key, expected_key)
    assert torch.is_complex(ascend.cache)
    assert ascend.cache.shape == (16, 4)
