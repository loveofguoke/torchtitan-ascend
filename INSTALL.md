# 安装指南

## 环境要求

Linux aarch64、Ascend NPU、CANN 9.2.0-beta.2、Python 3.12，以及支持 C++20 的编译器。

| 组件 | 固定版本 |
|---|---|
| PyTorch | `2.15.0.dev20261005+cpu` |
| Torchvision | `0.30.0.dev20261005+cpu` |
| torch_npu | `2c8682df9271e545e7f50f8c45e2395563a05483` |
| Triton-Ascend | `3.6.0` |
| TorchTitan | `edde21fb648d446fecb5b78ef42d8c432274b935` |

以下命令在同一 shell 中依次执行。安装所需的源码修复存放于 [`fixes/`](fixes/README.md)，随对应依赖的安装步骤应用。

## 准备环境

在本仓库根目录创建 Python 环境，并设置依赖源码目录：

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install uv
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export ASCEND_REPO="$(pwd)"
export TORCH_NPU_SRC=/absolute/path/to/torch_npu
export TORCHTITAN_SRC=/absolute/path/to/torchtitan
export UV_OVERRIDE="$ASCEND_REPO/overrides.txt"
export UV_CONSTRAINT="$ASCEND_REPO/constraints.txt"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0

uv pip install --index-url https://download.pytorch.org/whl/nightly/cpu \
    'torch==2.15.0.dev20261005+cpu' 'torchvision==0.30.0.dev20261005+cpu'
```

## 构建 torch_npu

```bash
git clone https://gitcode.com/Ascend/pytorch.git "$TORCH_NPU_SRC"
git -C "$TORCH_NPU_SRC" checkout 2c8682df9271e545e7f50f8c45e2395563a05483
git -C "$TORCH_NPU_SRC" submodule update --init --recursive
git -C "$TORCH_NPU_SRC" apply "$ASCEND_REPO/fixes/torch_npu/storage.patch"

cd "$TORCH_NPU_SRC"
uv pip install pyyaml typing-extensions setuptools wheel cmake numpy ninja nanobind auditwheel
mkdir -p build/packages
MAX_JOBS=16 DISABLE_INSTALL_FXRT=TRUE DISABLE_RPC_FRAMEWORK=TRUE \
    python setup.py build bdist_wheel
uv pip install --no-deps dist/torch_npu-*.whl
```

## 安装 Triton-Ascend

```bash
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    'triton-ascend==3.6.0' 'pybind11==3.1.0'

TRITON_BACKEND="$(python -c 'import triton.backends.ascend; print(triton.backends.ascend.__path__[0])')"
git -C "$TRITON_BACKEND" apply -p4 \
    "$ASCEND_REPO/fixes/triton-ascend/TA-3-cxx20-launcher.patch"
```

## 安装 TorchTitan 与扩展

```bash
git clone https://github.com/pytorch/torchtitan.git "$TORCHTITAN_SRC"
git -C "$TORCHTITAN_SRC" checkout edde21fb648d446fecb5b78ef42d8c432274b935
git -C "$TORCHTITAN_SRC" apply \
    "$ASCEND_REPO/fixes/torchtitan/optional-attn-gym-backends.patch" \
    "$ASCEND_REPO/fixes/torchtitan/entrypoint.patch"
uv pip install -e "$TORCHTITAN_SRC" -e "$ASCEND_REPO"
uv pip check
```

## 验证安装

在 TorchTitan 源码目录运行 Qwen3 debug 配方，使用上游自带的 tokenizer 和 C4 测试数据：

```bash
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

训练应完成 10 步并正常退出。可追加 `--output-dir <路径>` 指定输出目录。

新建 shell 后，重新激活该环境、加载 CANN，并设置「准备环境」中的路径及 uv 环境变量；训练时设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=1`。模型专用依赖与启动配置见[模型文档](README.md#模型文档)。
