# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# This source code is licensed under the BSD-style license found in the
# LICENSE-BSD-3-Clause file in the root directory of this source tree.

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from torchtitan.config import Configurable
from torchtitan.trainer import Trainer


DEFAULT_ENTRYPOINT = "torchtitan.train"


@dataclass
class IntegrationTestDefinition:
    """A named integration test backed by one or more complete config recipes."""

    test_descr: str = "default"
    test_name: str = "default"
    ngpu: int = 4
    disabled: bool = False
    skip_rocm_test: bool = False
    timeout: int | None = None
    golden_numerics_path: str | None = None
    """Run through loss_compare.py using this mode-specific golden path."""
    loss_compare_seed_config: Callable[[], Trainer.Config] | None = None
    """Model-equivalent config for loss_compare.py's single-GPU seed run.

    Use this when the test config applies a parallel transform. The seed run
    disables parallelism but does not undo the transform.
    """
    use_real_pg: bool = False
    """Whether the test requires communication semantics from a real PG."""
    configs: Sequence[Callable[[], Configurable.Config]] = ()
    """One complete configuration per run, selected by module and function."""
    entrypoints: Sequence[str] = ()
    """One launcher module per config; empty selects ``torchtitan.train``."""

    def __post_init__(self):
        if not self.configs:
            raise ValueError(f"{self.test_name} must define at least one config")
        if self.entrypoints and len(self.entrypoints) != len(self.configs):
            raise ValueError(
                f"{self.test_name} defines {len(self.entrypoints)} entrypoints "
                f"for {len(self.configs)} configs"
            )

    def entrypoint(self, index: int) -> str:
        """Return the module launched for ``configs[index]``."""
        return self.entrypoints[index] if self.entrypoints else DEFAULT_ENTRYPOINT

    def __repr__(self):
        return self.test_descr
