# SPDX-License-Identifier: Apache-2.0
"""DeepEP-Ascend overrides; importing the package initializes no hardware."""

import dataclasses

from torchtitan.models.common.moe import RoutedExperts

from torchtitan_ascend.overrides.deepep.experts import (
    AscendDeepEPTokenDispatcher,
    npu_deepep,
)


def expert_hidden_dim(model_config) -> int:
    """The input width of the first RoutedExperts.Config in a model config.

    The dispatcher override only replaces its own node, so the hidden dim is
    threaded from ``w13.in_features`` in the recipe.
    """
    seen: set[int] = set()
    stack = [model_config]
    while stack:
        node = stack.pop()
        if node is None or id(node) in seen:
            continue
        seen.add(id(node))
        if isinstance(node, RoutedExperts.Config):
            return node.w13.in_features
        if dataclasses.is_dataclass(node) and not isinstance(node, type):
            for f in dataclasses.fields(node):
                stack.append(getattr(node, f.name))
        elif isinstance(node, (list, tuple)):
            stack.extend(node)
        elif isinstance(node, dict):
            stack.extend(node.values())
    raise ValueError("No RoutedExperts.Config found in the model config")


__all__ = ["AscendDeepEPTokenDispatcher", "expert_hidden_dim", "npu_deepep"]
