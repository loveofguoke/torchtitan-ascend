# SPDX-License-Identifier: Apache-2.0

"""Kimi K3.1 model and configuration builder."""

from .flavors import build_model_config
from .model import KimiK31Model

__all__ = ["KimiK31Model", "build_model_config"]
