# 安装指南

## 环境要求

- 硬件与系统：A5（Ascend 950）、Linux aarch64。
- 软件：Python 3.12、CANN 9.2.0 及配套驱动和固件。
- 构建工具：Git、支持 C++20 的编译器、Clang/LLD 和 zlib 开发库。

| 组件 | 版本 |
|---|---|
| PyTorch | `2.15.0+cpu` |
| Torchvision | `0.29.2+cpu` |
| torch_npu | `1a25e6bedae48ed57ad2e5c2486d7242d2cea305` |
| Triton-Ascend | `3.6.0` |
| TorchTitan | `0.4.0`，提交 `45e33cd413b64cb37a7840cfd3c9765d3c639ae1` |

A2 / A3 的兼容性状态见 [Issue #7](https://gitcode.com/Ascend/torchtitan-ascend/issues/7)，各模型的支持范围见[模型文档](README.md#模型文档)。

## 准备环境

在同一工作目录中依次执行以下命令，源码分别放在 `torchtitan-ascend/`、`torch_npu/` 和 `torchtitan/`：

```bash
git clone --branch refactor/npu-override https://gitcode.com/Ascend/torchtitan-ascend.git
python3.12 -m venv torchtitan-ascend/.venv
source torchtitan-ascend/.venv/bin/activate
python -m pip install uv
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export UV_OVERRIDE="$(pwd)/torchtitan-ascend/overrides.txt"
export UV_CONSTRAINT="$(pwd)/torchtitan-ascend/constraints.txt"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
```

## 安装依赖

### PyTorch

```bash
uv pip install --index-url https://download.pytorch.org/whl/cpu \
    'torch==2.15.0+cpu' 'torchvision==0.29.2+cpu'
```

### torch_npu

```bash
git clone https://gitcode.com/Ascend/pytorch.git torch_npu
git -C torch_npu checkout 1a25e6bedae48ed57ad2e5c2486d7242d2cea305
git -C torch_npu submodule update --init --recursive
git -C torch_npu apply ../torchtitan-ascend/fixes/torch_npu/storage.patch

cd torch_npu
uv pip install pyyaml typing-extensions 'setuptools>=81.0.0' wheel cmake numpy ninja nanobind auditwheel
mkdir -p build/packages
TORCH_VERSION=2.15.0 MAX_JOBS=16 DISABLE_INSTALL_FXRT=TRUE DISABLE_RPC_FRAMEWORK=TRUE \
    python setup.py build bdist_wheel
uv pip install --no-deps dist/torch_npu-*.whl
cd ..
```

### Triton-Ascend

```bash
uv pip install --reinstall-package triton-ascend \
    --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    'triton-ascend==3.6.0'
export TRITON_ASCEND_BISHENGIR_PATH="$(python -c \
    'import triton; print(triton.__path__[0] + "/backends/ascend/bishengir")')"
test -x "$TRITON_ASCEND_BISHENGIR_PATH/bin/bishengir-compile"

uv pip install 'cmake>=3.20,<4.0' 'ninja>=1.11.1' wheel setuptools 'pybind11==3.1.0'
git clone --branch v3.6.0 --single-branch \
    https://github.com/triton-lang/triton-ascend.git
git -C triton-ascend checkout --detach 97e8a2be1e46817c359f55c0e6464e49480d0b1f
git -C triton-ascend apply \
    ../torchtitan-ascend/fixes/triton-ascend/TA-3-cxx20-launcher.patch \
    ../torchtitan-ascend/fixes/triton-ascend/zero-stride-store.patch

cd triton-ascend
MAX_JOBS=16 TRITON_BUILD_WITH_CLANG_LLD=true TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME=triton-ascend TRITON_APPEND_CMAKE_ARGS='-DTRITON_BUILD_UT=OFF' \
    python setup.py bdist_wheel
uv pip install --no-deps --force-reinstall dist/triton_ascend-*.whl
cd ..
```

### TorchTitan

```bash
git clone https://github.com/pytorch/torchtitan.git
git -C torchtitan checkout 45e33cd413b64cb37a7840cfd3c9765d3c639ae1
```

## 安装本项目

```bash
uv pip install -e torchtitan -e torchtitan-ascend
uv pip check
```

## 运行验证

在上述工作目录中加载环境并启动单卡 Qwen3 debug 配方：

```bash
source torchtitan-ascend/.venv/bin/activate
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
export PET_LOCAL_ADDR=127.0.0.1
unset PYTORCH_NPU_ALLOC_CONF

cd torchtitan
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

该配方使用内置 tokenizer 和测试数据，运行 10 步。其他模型的依赖、数据和启动方式见[模型文档](README.md#模型文档)。
