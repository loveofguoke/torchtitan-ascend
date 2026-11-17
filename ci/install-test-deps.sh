#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

target="${1:?Usage: install-test-deps.sh cpu|a2}"
case "$target" in
    cpu|a2) ;;
    *) echo "Unknown target: $target" >&2; exit 2 ;;
esac

# TorchTitan v0.4.0 uses quantization APIs from TorchAO nightly.
torchao_version='0.19.0.dev20261116+cpu'
torchao_wheel="torchao-${torchao_version}-py3-none-any.whl"
torchao_url="https://download.pytorch.org/whl/nightly/cpu/${torchao_wheel/+/%2B}"
torchao_sha256='1e3a183d2ed1a56106e8ea8583a38ad5425ed961941584c143f9ca3250581f7d'
dist_moe_version='0.1.0'
dist_moe_revision='894d8b87edce5e858deaefd39872f5c5d0676fc1'

requirements=()
add_if_needed() {
    local package="$1" version="$2" requirement="$3"
    if ! python - "$package" "$version" <<'PY'
from importlib.metadata import PackageNotFoundError, version
import sys

try:
    installed = version(sys.argv[1])
except PackageNotFoundError:
    installed = None
sys.exit(installed != sys.argv[2])
PY
    then
        requirements+=("$requirement")
    fi
}

add_if_needed torchao "$torchao_version" "torchao @ $torchao_url#sha256=$torchao_sha256"
if [[ "$target" == cpu ]]; then
    add_if_needed dist-moe "$dist_moe_version" \
        "dist-moe @ git+https://github.com/meta-pytorch/dist_moe.git@$dist_moe_revision"
fi

if [[ -z "${requirements[*]:-}" ]]; then
    echo "[test-deps] $target: required versions already installed"
    exit 0
fi

# CPU tests use the Dist-MoE Python API without its CUDA runtime dependencies.
uv pip install --python "$(command -v python)" --no-deps "${requirements[@]}"
