# Qwen3.8

## 1. 模型概述

Qwen3.8 复用 TorchTitan 的 Qwen3.5 模型主体，采用混合注意力结构：每 4 层为 1 层全注意力，其余为 Gated DeltaNet 线性注意力。本仓复用 TorchTitan 的模型和训练配方，GDN 算子由 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu) 提供 Ascend 实现。

### 1.1 模型规格与训练配置

| 模型规格 / flavor | 参数量 / 激活参数量 | 类型 | 上下文长度 / 输入尺寸 | MODULE | CONFIG | 默认数据 |
|---|---|---|---|---|---|---|
| `debugmodel` | 约 138M | Dense | 默认 4096 tokens | `torchtitan_ascend_recipes.models.qwen3_8` | `qwen38_debugmodel_npu` | 内置 C4 测试数据 |
| `debugmodel_moe` | 约 139M / 134M | MoE | 默认 4096 tokens | `torchtitan_ascend_recipes.models.qwen3_8` | `qwen38_debugmodel_moe_npu` | 内置 C4 测试数据 |

debugmodel 为验收用迷你模型，具体参数见 [CONFIG 设置](../../torchtitan_ascend_recipes/models/qwen3_8.py)。

### 1.2 特性支持

| 特性 | 说明 |
|---|---|
| BF16 训练 | 支持基于 BF16 低精度数据类型进行模型训练 |
| 分布式并行 | 支持 FSDP2、HSDP、DDP、TP、EP 训练 |
| 确定性训练 | 支持 HCCL_DETERMINISTIC=TRUE + 固定 seed |

## 2. 环境安装

参考[基础安装](../../INSTALL.md)完成环境搭建。GDN 使用 fla-npu 的 AscendC 算子，其构建方式见 [Qwen3.5 安装说明](qwen3_5.md#安装)。

将 `torchtitan-ascend` 和 `torchtitan` 的实际项目路径设为环境变量。

```bash
export ASCEND_REPO=/absolute/path/to/torchtitan-ascend
export TORCHTITAN_SRC=/absolute/path/to/torchtitan
```

## 3. 数据集准备

使用 TorchTitan 源码中的 `tests/assets/c4_test`，无需另外下载数据。

## 4. Tokenizer下载

使用 TorchTitan 自带的 `tests/assets/tokenizer`，无需下载生产模型 tokenizer。

## 5. 启动训练

### 5.1 单卡训练

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend_recipes.models.qwen3_8 CONFIG=qwen38_debugmodel_npu \
    ./run_train.sh
```

默认训练 10 步。MoE 模型将 `CONFIG` 改为 `qwen38_debugmodel_moe_npu`。

### 5.2 多卡训练

以**单机 8 卡纯 FSDP2 配置**为示例。

构建带有以下并行配置的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。

```python
from torchtitan_ascend_recipes.models.qwen3_8 import qwen38_debugmodel_npu


def train_config():
    config = qwen38_debugmodel_npu()
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
    MODULE=qwen3_8_fsdp2_config CONFIG=train_config \
    ./run_train.sh
```

| 配置项 | 值 |
|---|---|
| 总卡数 / 节点数 | 8 / 1 |
| dp_replicate / dp_shard（FSDP2） | 1 / 8，显式设置 |
| TP / PP / EP | 1 / 1 / 1 |
| 单卡 micro-batch / gradient accumulation | 512 tokens / 1 |
| 全局 tokens per step | 8 × 512 = 4,096 tokens |
| 其他必要环境变量 | `ASCEND_RT_VISIBLE_DEVICES` 中的设备数应与 `NGPU` 一致 |
