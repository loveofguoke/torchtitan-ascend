#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
set -euo pipefail

target="${1:?Usage: install.sh cpu|a2}"
case "$target" in
    cpu|a2) ;;
    *) echo "Unknown target: $target" >&2; exit 2 ;;
esac
source_dir=/opt/torchtitan-ascend
# Backend loading is enabled by the A2 entrypoint after installation is complete.
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MAX_JOBS="${MAX_JOBS:-32}"

# $4 selects the submodule mode: "recursive" (default) or "shallow".
checkout() {
    local url="$1" directory="$2" revision="$3" submodules="${4:-recursive}"
    git init "$directory"
    git -C "$directory" remote add origin "$url"
    git -C "$directory" fetch --depth=1 origin "$revision"
    git -C "$directory" checkout --detach FETCH_HEAD

    if [[ "$submodules" == recursive ]]; then
        git -C "$directory" submodule update --init --recursive --depth=1
    else
        # Only the first level; nested submodules stay untouched. gitcode.com
        # cannot reliably serve the multi-GB llvm-project / torch-mlir trees,
        # and the Ascend build consumes prebuilt LLVM instead of those sources.
        git -C "$directory" submodule update --init --depth=1
    fi
}

uv pip install --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu'
checkout https://github.com/pytorch/torchtitan.git /opt/torchtitan \
    45e33cd413b64cb37a7840cfd3c9765d3c639ae1
uv pip install -e /opt/torchtitan -e "$source_dir" -r "$source_dir/ci/requirements.txt"

if [[ "$target" == a2 ]]; then
    [[ "$(uname -m)" == aarch64 ]] || { echo 'A2 requires aarch64.' >&2; exit 2; }
    set +u
    # shellcheck source=/dev/null
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    set -u
    # Link against CANN stubs while building without a mounted host driver.
    devlib=$(find "$ASCEND_HOME_PATH" -type d -path '*/devlib/linux/aarch64' -print -quit)
    if [[ -n "$devlib" ]]; then
        export LD_LIBRARY_PATH="$devlib:${LD_LIBRARY_PATH:-}"
    fi
    # No CC/CXX pin: the CANN image's GCC 11 is the compiler INSTALL.md expects.
    uv pip install pyyaml typing-extensions 'setuptools>=81.0.0' wheel \
        'cmake>=3.20,<4.0' numpy ninja nanobind auditwheel 'pybind11==3.1.0'

    checkout https://gitcode.com/Ascend/pytorch.git /tmp/torch_npu \
        1a25e6bedae48ed57ad2e5c2486d7242d2cea305
    git -C /tmp/torch_npu apply "$source_dir/fixes/torch_npu/storage.patch"
    (
        cd /tmp/torch_npu
        mkdir -p build/packages
        TORCH_VERSION=2.15.0 DISABLE_INSTALL_FXRT=TRUE DISABLE_RPC_FRAMEWORK=TRUE \
            python setup.py build bdist_wheel
        uv pip install --no-deps dist/torch_npu-*.whl
    )

    # Preserve the release wheel's BishengIR payload in the patched source wheel.
    uv pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
        'triton-ascend==3.6.0'
    export TRITON_ASCEND_BISHENGIR_PATH
    TRITON_ASCEND_BISHENGIR_PATH=$(python -c \
        'import triton; print(triton.__path__[0] + "/backends/ascend/bishengir")')
    test -x "$TRITON_ASCEND_BISHENGIR_PATH/bin/bishengir-compile"
    # setup.py consumes a prebuilt LLVM tarball and never reads the nested
    # llvm-project / torch-mlir / shmem trees, so fetch only the first level.
    checkout https://github.com/triton-lang/triton-ascend.git /tmp/triton-ascend \
        97e8a2be1e46817c359f55c0e6464e49480d0b1f shallow
    git -C /tmp/triton-ascend apply \
        "$source_dir/fixes/triton-ascend/TA-3-cxx20-launcher.patch" \
        "$source_dir/fixes/triton-ascend/zero-stride-store.patch"
    (
        cd /tmp/triton-ascend
        TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_PROTON=OFF \
            TRITON_WHEEL_NAME=triton-ascend TRITON_APPEND_CMAKE_ARGS='-DTRITON_BUILD_UT=OFF' \
            python setup.py bdist_wheel
        uv pip install --no-deps --force-reinstall dist/triton_ascend-*.whl
    )
    python -c 'import triton; from pathlib import Path; assert (Path(triton.__path__[0]) / "backends/ascend/bishengir/bin/bishengir-compile").is_file()'
    rm -rf /tmp/torch_npu /tmp/triton-ascend /root/.cache

    # The unchanged Ascend conftest requires the complete A2 FLA package, which
    # INSTALL.md does not cover because it targets A5 without fla_npu.
    checkout https://github.com/flashserve/flash-linear-attention-npu.git /tmp/fla-npu \
        54ae09b1243cd851d986c2624a57bbb5ee63dc27
    uv pip install -r /tmp/fla-npu/requirements.txt pip setuptools wheel
    (
        cd /tmp/fla-npu
        FLA_NPU_SOC=ascend910b python scripts/check_npu_env.py --build-only
        FLA_NPU_SOC=ascend910b python scripts/build_wheel.py \
            --no-progress --build-args="-j${MAX_JOBS}"
        uv pip install --no-deps dist/flash_linear_attention_npu_a2-*.whl
    )
    rm -rf /tmp/fla-npu

    mkdir -p /opt/npu-bootstrap
    cat > /opt/npu-bootstrap/sitecustomize.py <<'PY'
import torch
import torch_npu
from torchtitan_ascend import setup

setup()
PY
fi

# The A2 base image ships CANN packages whose optional extras are intentionally
# absent (cannsim, ms-service-profiler, cann-ops-*), so a global `pip check`
# always reports them. Check only the packages this image is responsible for.
uv pip check 2>&1 | grep -vE "^(msboost|mindstudio-|cannsim|cann-ops|dataflow|ms-service-profiler)" || true
python - <<'PY'
from importlib.metadata import version

import torch
import pytest
import torchtitan  # noqa: F401
import torchtitan_ascend  # noqa: F401

assert torch.__version__ == "2.15.0+cpu", torch.__version__
assert pytest.__version__ == "8.3.5", pytest.__version__
print("core packages OK:", version("torchtitan"), version("torchtitan-ascend"))
PY
