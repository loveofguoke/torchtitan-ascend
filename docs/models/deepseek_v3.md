# DeepSeek-V3

## 1. 模型概述

DeepSeek-V3 采用 Multi-head Latent Attention（MLA）注意力、细粒度 MoE 路由（含共享专家）、DeepSeekMoE 架构，可选支持 Multi-Token Prediction（MTP）。本仓复用 TorchTitan 的模型和训练配方，为部分算子提供 Ascend 适配。

### 1.1 模型规格与训练配置

| 模型规格 / flavor | 参数量 / 激活参数量 | 类型 | 上下文长度 / 输入尺寸 | MODULE | CONFIG | 默认数据 |
|---|---|---|---|---|---|---|
| `debugmodel` | 约 33M / 27.6M | MoE，8 专家、top-3、2 个共享专家 | 默认 16384 tokens | `torchtitan_ascend_recipes.models.deepseek_v3` | `deepseek_v3_debugmodel_npu` | 内置 C4 测试数据 |

debugmodel 为验收用迷你模型，具体参数见 [CONFIG 设置](../../torchtitan_ascend_recipes/models/deepseek_v3.py)。

### 1.2 特性支持

| 特性 | 说明 |
|---|---|
| BF16 训练 | 支持基于 BF16 低精度数据类型进行模型训练 |
| FSDP2 / HSDP | 支持完全分片数据并行和混合分片数据并行训练 |
| DDP | 支持多卡分布式数据并行训练 |
| Tensor Parallel（TP） | 支持多卡张量并行训练，MoE 模型要求 EP >= TP |
| Expert Parallel（EP，MoE） | 支持多卡混合专家并行训练 |
| Activation Checkpoint | 支持 selective 和 full 两种激活重计算模式 |
| `torch.compile` | varlen attention backend 下支持 Triton-Ascend 内核编译 |
| 确定性训练 | 支持 HCCL_DETERMINISTIC=TRUE + 固定 seed |

## 2. 环境安装

参考[基础安装](../../INSTALL.md)完成环境搭建。

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
    MODULE=torchtitan_ascend_recipes.models.deepseek_v3 CONFIG=deepseek_v3_debugmodel_npu \
    ./run_train.sh
```

默认训练 10 步。

### 5.2 多卡训练

以**单机 8 卡纯 FSDP2 配置**为示例。

构建带有以下并行配置的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。

```python
from torchtitan_ascend_recipes.models.deepseek_v3 import deepseek_v3_debugmodel_npu


def train_config():
    config = deepseek_v3_debugmodel_npu()
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
    MODULE=deepseek_v3_fsdp2_config CONFIG=train_config \
    ./run_train.sh
```

| 配置项 | 值 |
|---|---|
| 总卡数 / 节点数 | 8 / 1 |
| dp_replicate / dp_shard（FSDP2） | 1 / 8，显式设置 |
| TP / PP / CP / EP | 1 / 1 / 1 / 1 |
| 单卡 micro-batch / gradient accumulation | 8 × max_context_length tokens / 1 |
| 全局 tokens per step | 8 × 8 × 16384 = 1,048,576 tokens |
| 其他必要环境变量 | `ASCEND_RT_VISIBLE_DEVICES` 中的设备数应与 `NGPU` 一致 |
