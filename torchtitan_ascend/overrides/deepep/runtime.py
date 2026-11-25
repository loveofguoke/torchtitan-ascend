# SPDX-License-Identifier: Apache-2.0
"""EPBuffer pool, handle cache and cleanup for the DeepEP backend."""

from __future__ import annotations

import weakref
from dataclasses import dataclass, field
from functools import partial
from typing import TYPE_CHECKING

import torch
from torch.distributed import ProcessGroup

if TYPE_CHECKING:
    from deep_ep import EPBuffer, EPHandle


@dataclass
class _BufferEntry:
    buffer: EPBuffer
    handles: dict[int, EPHandle] = field(default_factory=dict)


_BUFFERS: dict[tuple, _BufferEntry] = {}
_HANDLE_COUNTER: int = 0

# The entry the custom ops act on; dispatchers reactivate their own entry
# before every op so alternating buffers cannot leak across configurations.
_ACTIVE_BUFFER: _BufferEntry | None = None


def _next_handle_id() -> torch.Tensor:
    """A fresh CPU int64 handle-id, avoiding any device-host sync."""
    global _HANDLE_COUNTER
    _HANDLE_COUNTER += 1
    return torch.tensor([_HANDLE_COUNTER], dtype=torch.int64, device="cpu")


def get_buffer(
    group: ProcessGroup,
    *,
    hidden: int,
    num_max_tokens_per_rank: int,
    num_topk: int,
    deterministic: bool = False,
) -> _BufferEntry:
    """Get or create the EPBuffer entry for this configuration.

    All ranks must pass identical arguments; destruction is owned by
    ``destroy_all_buffers``, not the C++ destructor.
    """
    # Deferred: deep_ep is optional for import-time (CPU-only) consumers.
    from deep_ep import EPBuffer

    key = (group, hidden, num_max_tokens_per_rank, num_topk, deterministic)
    entry = _BUFFERS.get(key)
    if entry is not None:
        set_active_buffer(entry)
        return entry

    num_bytes = EPBuffer.get_buffer_size_hint(
        group, num_max_tokens_per_rank, hidden, num_topk=num_topk
    )
    buffer = EPBuffer(
        group,
        num_bytes=num_bytes,
        num_max_tokens_per_rank=num_max_tokens_per_rank,
        hidden=hidden,
        num_topk=num_topk,
        deterministic=deterministic,
        explicitly_destroy=True,
    )
    entry = _BufferEntry(buffer=buffer)
    _BUFFERS[key] = entry
    set_active_buffer(entry)
    return entry


def set_active_buffer(entry: _BufferEntry) -> None:
    """Point the custom ops at this buffer entry."""
    global _ACTIVE_BUFFER
    _ACTIVE_BUFFER = entry


def register_handle(entry: _BufferEntry, handle: EPHandle) -> torch.Tensor:
    """Cache an EPHandle and return its int handle-id tensor.

    The finalizer covers a dispatch whose combine never runs, e.g. when
    activation checkpointing saves the combine region.
    """
    handle_id = _next_handle_id()
    key = handle_id.item()
    entry.handles[key] = handle
    weakref.finalize(handle_id, partial(entry.handles.pop, key, None))
    return handle_id


def take_handle(
    entry: _BufferEntry, handle_id: torch.Tensor, *, consume: bool
) -> EPHandle:
    """Fetch the EPHandle for a handle-id; ``consume`` transfers ownership out."""
    key = handle_id.item()
    handle = entry.handles.pop(key, None) if consume else entry.handles.get(key)
    assert handle is not None, f"EPHandle not found for handle_id={key}"
    return handle


def destroy_all_buffers() -> None:
    """Destroy every EPBuffer and clear the caches; safe to call repeatedly.

    Runs from the TrainingEngine.close patch before process-group teardown;
    a no-op when the DeepEP backend was never enabled.
    """
    for entry in _BUFFERS.values():
        entry.handles.clear()
        entry.buffer.destroy()
    _BUFFERS.clear()
    global _ACTIVE_BUFFER
    _ACTIVE_BUFFER = None
