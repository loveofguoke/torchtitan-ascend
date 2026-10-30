import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
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


@pytest.fixture
def npu_rotary_mul(monkeypatch):
    def fake_rotary_mul(x, cos, sin, *, rotary_mode):
        assert rotary_mode == "interleave"
        x_pairs = x.reshape(*x.shape[:-1], -1, 2)
        rotated_pairs = torch.stack(
            (-x_pairs[..., 1], x_pairs[..., 0]),
            dim=-1,
        )
        return x * cos + rotated_pairs.flatten(-2) * sin

    rotary_mul = Mock(side_effect=fake_rotary_mul)
    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(npu_rotary_mul=rotary_mul),
    )
    return rotary_mul


def test_fused_complex_rope_matches_torchtitan_math(npu_rotary_mul):
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
    assert npu_rotary_mul.call_count == 2


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("with_key", [False, True])
def test_ascend_complex_rope_uses_fused_kernel(npu_rotary_mul, inverse, with_key):
    config = ComplexRoPE.Config(dim=8, max_context_length=16)
    upstream = ComplexRoPE(config)
    ascend = AscendComplexRoPE(AscendComplexRoPE.Config(dim=8, max_context_length=16))
    query = torch.randn(6, 2, 8)
    key = torch.randn(6, 1, 8) if with_key else None
    positions = torch.tensor([0, 3, 2, 5, 4, 1])

    expected = upstream(query, key, positions, inverse=inverse)
    actual = ascend(query, key, positions, inverse=inverse)  # pyrefly: ignore [not-callable]

    torch.testing.assert_close(actual, expected)
    assert npu_rotary_mul.call_count == (2 if with_key else 1)
    assert torch.is_complex(ascend.cache)
    assert ascend.cache.shape == (16, 4)
