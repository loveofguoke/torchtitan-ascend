# SPDX-License-Identifier: Apache-2.0
"""CPU reference tests for DeepEP custom-op gradients and buffer selection."""

from types import SimpleNamespace

import torch

from torchtitan_ascend.overrides.deepep import AscendDeepEPTokenDispatcher, ops, runtime


def test_dispatch_backward_preserves_fp32_router_gradients():
    """Token dtype must not truncate gradients of the FP32 routing weights."""
    expected_TK = torch.tensor([[1.001, 1.01]], dtype=torch.float32)

    def combine(grad_recv_x, *, handle, topk_weights):
        del handle
        return torch.ones_like(grad_recv_x), topk_weights, None

    ctx = SimpleNamespace(
        entry=SimpleNamespace(buffer=SimpleNamespace(combine=combine)),
        saved_handle=object(),
        input_dtype=torch.bfloat16,
    )
    gradients = ops._dispatch_backward(
        ctx, torch.ones(1, 1, dtype=torch.bfloat16), expected_TK, None, None
    )

    assert gradients[0].dtype == torch.bfloat16
    assert gradients[2].dtype == torch.float32
    torch.testing.assert_close(gradients[2], expected_TK, rtol=0, atol=0)


def test_existing_buffer_is_reactivated_before_another_op(monkeypatch):
    """Alternating configurations must not retain another entry as active."""
    dispatcher = object.__new__(AscendDeepEPTokenDispatcher)
    dispatcher._buffer_entry = SimpleNamespace(name="own")
    monkeypatch.setattr(runtime, "_ACTIVE_BUFFER", SimpleNamespace(name="other"))

    dispatcher._ensure_buffer()

    assert runtime._ACTIVE_BUFFER is dispatcher._buffer_entry


def test_destroy_all_buffers_is_idempotent(monkeypatch):
    """Cleanup must release each buffer once and clear the active pointer."""
    destroyed = []
    entry = SimpleNamespace(
        buffer=SimpleNamespace(destroy=lambda: destroyed.append(True)),
        handles={1: object()},
    )
    monkeypatch.setattr(runtime, "_BUFFERS", {"test": entry})
    monkeypatch.setattr(runtime, "_ACTIVE_BUFFER", entry)

    runtime.destroy_all_buffers()
    runtime.destroy_all_buffers()

    assert destroyed == [True]
    assert entry.handles == {}
    assert runtime._ACTIVE_BUFFER is None
