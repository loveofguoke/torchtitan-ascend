# 安装指南

## 环境要求

Linux aarch64、Ascend NPU、CANN 9.2.0-beta.2、Python 3.12，以及支持 C++20 的编译器。

| 组件 | 固定版本 |
|---|---|
| PyTorch | `2.15.0.dev20261005+cpu` |
| Torchvision | `0.30.0.dev20261005+cpu` |
| TorchTitan | `edde21fb648d446fecb5b78ef42d8c432274b935` |
| Triton-Ascend | `3.6.0` |

## 准备环境

激活 Python 3.12 环境，安装表中版本的 PyTorch、Torchvision，并加载匹配的 CANN 环境。

将 `ASCEND_REPO` 设为本仓库源码目录，其余路径设为待创建的依赖源码目录。以下配置与安装命令在同一 shell 中执行：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh

export TORCH_NPU_SRC=/absolute/path/to/torch_npu
export TORCHTITAN_SRC=/absolute/path/to/torchtitan
export ASCEND_REPO=/absolute/path/to/torchtitan-ascend
```

## 构建 torch_npu

```bash
git clone https://gitcode.com/Ascend/pytorch.git "$TORCH_NPU_SRC"
git -C "$TORCH_NPU_SRC" checkout 2c8682df9271e545e7f50f8c45e2395563a05483
git -C "$TORCH_NPU_SRC" submodule update --init --recursive
git -C "$TORCH_NPU_SRC" apply \
  "$ASCEND_REPO/patches/torch_npu/flash-attention.patch" \
  "$ASCEND_REPO/patches/torch_npu/inductor-compat.patch" \
  "$ASCEND_REPO/patches/torch_npu/storage.patch"
git -C "$TORCH_NPU_SRC/third_party/op-plugin" apply \
  "$ASCEND_REPO/patches/op-plugin/NPU-6-zero-uint64.patch"

cd "$TORCH_NPU_SRC"
mkdir -p build/packages
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=16 \
DISABLE_INSTALL_TORCHAIR=TRUE DISABLE_INSTALL_FXRT=TRUE DISABLE_RPC_FRAMEWORK=TRUE \
    python setup.py build bdist_wheel
python -m pip install --no-deps dist/torch_npu-*.whl
```

## 安装 TorchTitan 与扩展

```bash
git clone https://github.com/pytorch/torchtitan.git "$TORCHTITAN_SRC"
git -C "$TORCHTITAN_SRC" checkout edde21fb648d446fecb5b78ef42d8c432274b935
git -C "$TORCHTITAN_SRC" apply \
  "$ASCEND_REPO/patches/torchtitan/TT-21-attn-gym-optional-backends.patch"
python -m pip install --no-deps -e "$TORCHTITAN_SRC" -e "$ASCEND_REPO"
```

构建环境中的 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 用于避免提前加载 NPU 后端；训练 shell 使用 `TORCH_DEVICE_BACKEND_AUTOLOAD=1`。

## 安装 Triton-Ascend

```bash
python -m pip uninstall -y triton triton-ascend
python -m pip install --extra-index-url https://mirrors.huaweicloud.com/ascend/repos/pypi \
    'triton-ascend==3.6.0' 'pybind11==3.1.0'

TRITON_BACKEND="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -c 'import triton.backends.ascend; print(triton.backends.ascend.__path__[0])')"
git -C "$TRITON_BACKEND" apply -p4 \
    "$ASCEND_REPO/patches/triton-ascend/TA-3-cxx20-launcher.patch"
```

## 安装 fla 和 fla_npu

```bash
python -m pip install --no-deps 'fla-core==0.5.2' 'flash-linear-attention==0.5.2'
```

fla_npu 的安装有两条路径：

1. 在 [fla_npu release](https://github.com/flashserve/flash-linear-attention-npu/releases) 下载 SOC 与架构对应的编译 wheel 包。
2. 参考以下过程，针对特定 commit 进行本地编译安装：

```bash
export FLA_NPU_SRC=/absolute/path/to/flash-linear-attention-npu
git clone https://github.com/flashserve/flash-linear-attention-npu.git "$FLA_NPU_SRC"
git -C "$FLA_NPU_SRC" checkout <commit-id>
cd "$FLA_NPU_SRC"
source /usr/local/Ascend/ascend-toolkit/set_env.sh
python -m pip install -r requirements.txt

export FLA_NPU_SOC=ascend_version  # A5→ascend950；A3→ascend910_93，A2→ascend910b
python scripts/build_wheel.py
python -m pip install dist/flash_linear_attention_npu-*.whl
```

## 验证安装

在 TorchTitan 源码目录执行 Qwen3 debug 配方，使用上游自带的 tokenizer 和 C4 测试数据：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

训练应完成 10 步并正常退出。在 `./run_train.sh` 后追加 `--output-dir <路径>` 可指定输出目录。

其他配方见 [模型文档](README.md#模型文档)。
