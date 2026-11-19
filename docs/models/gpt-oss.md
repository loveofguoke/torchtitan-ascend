# GPT-OSS

## 1. 模型概述

GPT-OSS 使用带 per-head attention sinks 的注意力，奇偶层交替使用滑动窗口（窗口 128）与全注意力，前馈采用 MoE 和 ClampedSwiGLU 激活，并用 YaRN RoPE 缩放支持长上下文。本仓复用 TorchTitan 的模型和训练配方，为 NPU 运行提供 Ascend 适配。

### 1.1 模型规格与训练配置

| 模型规格 / flavor | 类型 | 上下文长度 / 输入尺寸 | MODULE | CONFIG | 默认数据 |
|---|---|---|---|---|---|
| `debugmodel` | MoE，8 专家、top-4、4 层 | 默认 2048 tokens | `torchtitan_ascend_recipes.models.gpt_oss` | `gpt_oss_debugmodel_npu` | torchtitan 内置 C4 测试数据 |

debugmodel 为验收用迷你模型，具体参数见 [CONFIG 设置](../../torchtitan_ascend_recipes/models/gpt_oss.py)。

### 1.2 特性支持

| 特性 | 说明 |
|---|---|
| BF16 训练 | 支持基于 BF16 低精度数据类型进行模型训练 |
| FSDP2 | 默认 `dp_shard=-1`，按 world size 解析，多卡直接运行 |
| HSDP | 上游模型支持，需在单独配方中设置 `dp_replicate` / `dp_shard` |
| Tensor Parallel（TP） | 通过构造函数 `tp` 参数设置 |
| Expert Parallel（EP，MoE） | 通过构造函数 `ep` 参数设置，EP>1 触发 all-to-all 派发 |
| Pipeline Parallel（PP） | 上游模型支持，需在单独配方中基于上游实现配置，建议启用激活重计算 |
| Context Parallel（CP） | 通过构造函数 `cp` 参数设置；GPT-OSS per-head sinks 不支持 Ulysses CP，需使用 FlexAttention 的 AllGather-KV 实现，仅 Ascend950（A5）可用；该路径不支持确定性训练 |
| 确定性训练 | 默认（`cp=1`）seed=42 + deterministic；`cp>1` 的 AllGather-KV CP 路径不支持确定性训练，使用默认 debug 配置，多卡确定性需配合 `HCCL_DETERMINISTIC=TRUE` |

## 2. 环境安装

参考[基础安装](../../INSTALL.md)完成环境搭建。启用激活重计算（如 PP 配置）依赖 `torch_npu` 的空指针占位 storage 支持，由基础安装中的 `storage.patch` 提供。

将 `torchtitan-ascend` 和 `torchtitan` 的实际项目路径设为环境变量。

```bash
export ASCEND_REPO=/absolute/path/to/torchtitan-ascend
export TORCHTITAN_SRC=/absolute/path/to/torchtitan
```

## 3. 数据集准备

使用 TorchTitan 源码中的 `tests/assets/c4_test`，无需另外下载数据。

## 4. Tokenizer 下载

使用 TorchTitan 自带的 `tests/assets/tokenizer`，无需下载生产模型 tokenizer。

## 5. 启动训练

### 5.1 单卡训练

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend_recipes.models.gpt_oss CONFIG=gpt_oss_debugmodel_npu \
    ./run_train.sh
```

默认训练 10 步，模型随机初始化，不保存检查点。默认使用 BF16、上游 VarlenAttention、AdamW 和 `DefaultActivationRemat`，关闭模型局部编译与 CUDA graphs。默认 `cp=1` 走 varlen attention，不依赖 FlexAttention，可在非 Ascend950（如 910B/A2）上运行；仅 `cp>1` 的上下文并行要求 Ascend950（A5）。

### 5.2 多卡训练

以**单机 8 卡纯 FSDP2 配置**为示例。

构建带有以下并行配置的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。

```python
from torchtitan_ascend_recipes.models.gpt_oss import gpt_oss_debugmodel_npu


def train_config():
    config = gpt_oss_debugmodel_npu()
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 1
    return config
```

运行 8 卡训练。

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 \
    ENTRYPOINT=torchtitan_ascend.train \
    MODULE=gpt_oss_fsdp2_config CONFIG=train_config \
    ./run_train.sh
```

| 配置项 | 值 |
|---|---|
| 总卡数 / 节点数 | 8 / 1 |
| dp_replicate / dp_shard（FSDP2） | 1 / 8，显式设置 |
| TP / PP / CP / EP | 1 / 1 / 1 / 1 |
| 单卡 micro-batch / gradient accumulation | 8 × max_context_length（16,384）tokens / 1 |
| 全局 tokens per step | 8 × 16,384 = 131,072 tokens |
| 其他必要环境变量 | `ASCEND_RT_VISIBLE_DEVICES` 中的设备数应与 `NGPU` 一致；多卡通信可设置 `HCCL_NPU_SOCKET_PORT_RANGE=auto` |

### 5.3 上下文并行

上下文并行要求 Ascend950/A5；`cp>1` 时配方自动改用 FlexAttention 的 AllGather-KV 后端，并关闭确定性训练（`debug` 使用默认配置，Ascend950 上该 CP 路径不支持确定性训练）。构建带有 `cp` 的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。

```python
from torchtitan_ascend_recipes.models.gpt_oss import gpt_oss_debugmodel_npu


def train_config():
    return gpt_oss_debugmodel_npu(cp=2)
```

运行 2 卡 CP=2 训练。

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1 NGPU=2 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=gpt_oss_cp2_config CONFIG=train_config \
    ./run_train.sh
```

`gpt_oss_debugmodel_npu()` 只接受 `tp`、`ep` 和 `cp` 参数，数据源固定为 C4 测试数据。自定义数据、其他并行组合、学习率计划和检查点需在单独的配方模块中配置。

## 支持范围

配方提供 debug 模型的文本训练，不提供 20B/120B 规模的完整配方。上游模型支持 FSDP/HSDP、TP、EP、CP、PP；本文档只描述通用构造函数的用法，不声明上述并行组合的验收结论。实际多卡运行需对应数量的 NPU。
