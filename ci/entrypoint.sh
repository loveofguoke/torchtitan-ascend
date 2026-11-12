#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ "$CI_DEVICE" == a2 ]]; then
    # shellcheck source=ci/npu-env.sh
    source "$(dirname "${BASH_SOURCE[0]}")/npu-env.sh"
    # An explicit check also catches startup-hook failures, which Python can log and ignore.
    python -c 'import torch, torch_npu; from torchtitan_ascend import setup; setup(); assert torch.npu.is_available(), "No visible Ascend NPU"'
fi
cd /opt/torchtitan
exec "$@"
