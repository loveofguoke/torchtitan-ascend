"""Process-local hardware identity and requirements for Ascend implementations."""

from functools import cache

import torch


@cache
def get_device_name() -> str:
    """Cache the worker's NPU model on first use, after its device is selected."""
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("An available Ascend NPU is required.")
    return npu.get_device_name(npu.current_device())


def require_device(*supported: str, feature: str) -> None:
    """Require an NPU whose name starts with one of the supported model prefixes."""
    name = get_device_name()
    if not name.startswith(supported):
        raise RuntimeError(f"{feature} requires {' or '.join(supported)}; got {name}.")
