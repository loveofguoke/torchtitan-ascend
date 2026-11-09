#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

if [[ "$CI_DEVICE" == a2 ]]; then
    set +u
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    # Preserve the CANN base image's compiler and NNAL initialization.
    # shellcheck source=/dev/null
    source /usr/local/Ascend/cann-9.2.0/share/info/ascendnpu-ir/bin/set_env.sh
    # shellcheck source=/dev/null
    source /usr/local/Ascend/nnal/atb/set_env.sh --cxx_abi="${ATB_CXX_ABI:-1}"
    set -u
    # Build-only stubs must not shadow the mounted driver's runtime libraries.
    export LD_LIBRARY_PATH
    LD_LIBRARY_PATH=$(printf '%s' "${LD_LIBRARY_PATH:-}" | tr ':' '\n' | sed '/devlib/d' | paste -sd: -)
    export TORCH_DEVICE_BACKEND_AUTOLOAD=1
    export PYTHONPATH="/opt/npu-bootstrap:${PYTHONPATH:-}"
    # An explicit check also catches startup-hook failures, which Python can log and ignore.
    python -c 'import torch, torch_npu; from torchtitan_ascend import setup; setup(); assert torch.npu.is_available(), "No visible Ascend NPU"'
fi
cd /opt/torchtitan
exec "$@"
