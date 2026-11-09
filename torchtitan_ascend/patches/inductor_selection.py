# SPDX-License-Identifier: Apache-2.0
"""Preserve PyTorch's deterministic selection when NPU installs its selector."""

import inspect
from functools import wraps

from torchtitan_ascend.patches import patch


def _restore_deterministic_selection():
    from torch._inductor import config
    from torch._inductor.select_algorithm import AlgorithmSelectorCache, TritonTemplateCaller

    original = AlgorithmSelectorCache.__call__
    # Newer NPU backends can restore this branch themselves. Do not replace
    # their policy, the native PyTorch selector, or our own installed wrapper.
    if not original.__module__.startswith("torch_npu.") or "pick_deterministic_choice" in original.__code__.co_names:
        return
    compile_only_selection = "_nobench_select_first_compilable" in inspect.getsource(original)

    @wraps(original)
    def select(self, name, choices, input_nodes, layout, *args, **kwargs):
        if config.deterministic and layout.device.type == "npu":
            usable = [choice for choice in choices if choice is not None]
            if usable and compile_only_selection and all(isinstance(candidate, TritonTemplateCaller) for candidate in usable):
                # Also validate a sole deferred-epilogue template: selecting an
                # uncompiled tile can defer a buffer overflow to kernel launch.
                absent = object()
                previous = [getattr(candidate, "_nobench_select_first_compilable", absent) for candidate in usable]
                try:
                    for candidate in usable:
                        candidate._nobench_select_first_compilable = True
                    return original(self, name, usable, input_nodes, layout, *args, **kwargs)
                finally:
                    for candidate, value in zip(usable, previous, strict=True):
                        if value is absent:
                            del candidate._nobench_select_first_compilable
                        else:
                            candidate._nobench_select_first_compilable = value
            if len(usable) == 1 and isinstance(usable[0], TritonTemplateCaller):
                return usable[0].output_node(), usable[0]
            if len(usable) > 1:
                choice = self.pick_deterministic_choice(usable)
                return choice.output_node(), choice
        return original(self, name, choices, input_nodes, layout, *args, **kwargs)

    AlgorithmSelectorCache.__call__ = select


@patch(target="torch_npu._inductor.select_algorithm:patch_algorithm_selector", deferred=True)
def apply(original):
    _restore_deterministic_selection()

    @wraps(original)
    def install(*args, **kwargs):
        result = original(*args, **kwargs)
        _restore_deterministic_selection()
        return result

    return install
