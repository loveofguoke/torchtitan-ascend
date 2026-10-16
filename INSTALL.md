# 安装

验证基线：Linux aarch64、Ascend 910B2、CANN 9.2.0-beta.2、Python 3.12。

| 组件 | 固定版本 |
|---|---|
| PyTorch（CPU wheel，设备后端由 torch_npu 提供） | `2.16.0.dev20261008+cpu` |
| torchvision | `0.30.0.dev20261008+cpu`，源码 `4e19bd2c7e8172f94bdbb093695cf315a7639322` |
| TorchTitan | `6f22255756ca247f343f2f517671c61f0a407e9b` |
| torch_npu | `95c7e465426ab894ffdb43ae878ee111e090fb17` + 下述两项修复 |
| op-plugin | 上述 torch_npu 的子模块 `0abe20992cf389c4e9615f96f7198205a155c769` |
| Triton（上游 Python 导入依赖） | `3.6.0`，官方 wheel |

此配方不需要 Triton-Ascend、BishengIR、自定义算子包或 Docker。使用独立 Python 环境，避免与带有其他源码补丁的开发环境混用。

## Python 依赖

先安装匹配的 PyTorch、torchvision。若固定 nightly 的 torchvision wheel 已不可下载，可从表中源码版本构建 CPU wheel（`FORCE_CUDA=0`）。然后安装上游依赖：

```bash
export ASCEND_REPO=/absolute/path/to/torchtitan-ascend
export TORCHTITAN_SRC=/absolute/path/to/torchtitan

git clone https://github.com/pytorch/torchtitan.git "$TORCHTITAN_SRC"
git -C "$TORCHTITAN_SRC" checkout 6f22255756ca247f343f2f517671c61f0a407e9b
python -m pip install -r "$TORCHTITAN_SRC/requirements.txt" 'triton==3.6.0'
python -m pip install --no-deps -e "$TORCHTITAN_SRC"
python -m pip install --no-deps -e "$ASCEND_REPO"
python -m pip install pytest
```

上游 `attn-gym` 的导入链还需要 `nvidia-cutlass-dsl`；它是 Python 导入依赖，本配方不执行 CUDA kernel。若上游依赖安装未包含它，运行 `python -m pip install nvidia-cutlass-dsl==4.8.0`。

## 最小 torch_npu 构建

以下命令用于新建的源码目录；不要重复应用补丁或覆盖已有开发源码。需要 C++20 编译器、CMake、Ninja 和 CANN 开发环境。

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
export TORCH_NPU_SRC=/absolute/path/to/torch_npu-pr1

git clone https://gitcode.com/Ascend/pytorch.git "$TORCH_NPU_SRC"
git -C "$TORCH_NPU_SRC" checkout 95c7e465426ab894ffdb43ae878ee111e090fb17
git -C "$TORCH_NPU_SRC" submodule update --init --recursive

git -C "$TORCH_NPU_SRC" apply \
  "$ASCEND_REPO/patches/torch_npu/NPU-1-flash-attention-privateuse1.patch"
git -C "$TORCH_NPU_SRC/third_party/op-plugin" apply \
  "$ASCEND_REPO/patches/op-plugin/NPU-6-zero-uint64.patch"

# 构建依赖文件中的旧 torch pin 与本基线不同；保留已安装的固定 torch。
sed '/^torch==/d; /^--extra-index-url/d' \
  "$TORCH_NPU_SRC/requirements-build.txt" > "$TORCH_NPU_SRC/requirements-build-pr1.txt"
python -m pip install -r "$TORCH_NPU_SRC/requirements-build-pr1.txt" 'numpy==1.26.4' 'cmake==4.4.4'
cd "$TORCH_NPU_SRC"
mkdir -p build/packages
TORCH_DEVICE_BACKEND_AUTOLOAD=0 MAX_JOBS=16 \
DISABLE_INSTALL_TORCHAIR=TRUE DISABLE_INSTALL_FXRT=TRUE DISABLE_RPC_FRAMEWORK=TRUE \
python setup.py bdist_wheel
python -m pip install --no-deps dist/torch_npu-*.whl
```

训练时使用默认的设备后端自动加载，不要保留构建时的 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。配方通过上游 `model.local_compile_regions=[]` 选择 eager，无需设置全局 `TORCH_COMPILE_DISABLE`。
