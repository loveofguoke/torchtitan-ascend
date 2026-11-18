# DeepSeek V4

## 1. 模型概述

DeepSeek V4 采用 Multi-head Latent Attention（MLA）与 Mixture-of-Experts（MoE）结构，包含共享专家、路由专家、attn_sink、compressor 和 DSA 稀疏注意力等组件。本仓基于 TorchTitan 提供 Ascend 适配，当前提供 debugmodel 测试配方。

### 1.1 模型规格与训练配置

| 模型规格 / flavor | 参数量 / 激活参数量（如适用） | 类型 | 上下文长度 / 输入尺寸 | MODULE | CONFIG | 默认数据 |
|---|---|---|---|---|---|---|
| debugmodel | 10.6M 总参 / 9.3M 激活 | MoE | 默认 2048；flavor 上限 16,384；每 DP rank 每 microbatch 16,384 tokens | `torchtitan_ascend_recipes.models.deepseek_v4` | `deepseek_v4_debugmodel_npu` | c4_test |

### 1.2 特性支持

以下为千步验收结果；第 5 节默认命令执行 10 步，不加载或保存 checkpoint。

| 特性 | 说明 |
|---|---|
| BF16 训练 | 支持基于 BF16 低精度数据类型进行模型训练 |
| 分布式并行 | 支持FSDP2/HSDP、EP、TP、PP组合训练 |
| 确定性训练 | 支持确定性计算，需设置`export HCCL_DETERMINISTIC=TRUE` |
| `torch.compile` / 局部编译 | 验收关闭模型局部编译及图捕获/回放，编译开启态未验证 |

## 2. 环境安装

参考 [torchtitan-ascend 安装指南](https://gitcode.com/Ascend/torchtitan-ascend/blob/refactor%2Fnpu-override/INSTALL.md) 完成环境搭建。`TORCHTITAN_SRC` 为 TorchTitan 源码根目录。

## 3. 数据集准备

使用 TorchTitan 源码内置的 c4_test 测试数据，无需下载。

## 4. Tokenizer 准备

使用配方指定、随源码部署的测试 tokenizer，无需下载。

## 5. 启动训练

### 5.1 单卡训练

```bash
cd "$TORCHTITAN_SRC"
source /usr/local/Ascend/ascend-toolkit/cann/set_env.sh
NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend_recipes.models.deepseek_v4 CONFIG=deepseek_v4_debugmodel_npu \
    ./run_train.sh
```

固定 `seed=42`，默认训练 10 步，随机初始化，不加载或保存 checkpoint，可追加 --output-dir ./outputs/deepseekv4-debug 指定输出目录。

### 5.2 多卡训练

以下命令默认使用八卡纯 FSDP2（`dp_shard=8`，TP=EP=1）。

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 \
    ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend_recipes.models.deepseek_v4 CONFIG=deepseek_v4_debugmodel_npu \
    ./run_train.sh
```
