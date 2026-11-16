# Kimi K3.1

## 1. 模型概述

Kimi K3.1 使用 Kimi Delta Attention（KDA）和 DeepSeek Sparse Attention（DSA）混合解码器，并沿用 Kimi K3 的 LatentMoE、Attention Residuals 和视觉编码器，支持图文自回归训练。

### 1.1 模型规格与训练配置

| 模型规格 / flavor | 模型规模 | 文本层 / 注意力 | 默认上下文 | MODULE | CONFIG | 默认数据 |
|---|---|---|---|---|---|---|
| `debug` | 7168 维，64 个路由专家 | 8 层，6 层 KDA、2 层 DSA，96 头 | 16384 tokens | `torchtitan_ascend_recipes.models.kimi_k3_1` | `kimi_k3_1_reduced_npu` | `cc12m-test`（样本 packing） |
| `Kimi-K3.1-Pro` | 完整 Pro 结构 | 93 层，69 层 KDA、24 层 DSA，96 头 | 2048 tokens | 同上 | `kimi_k3_1_pro_npu` | `cc12m` |

两条配方每个 token 选择 16 个专家，词表大小为 163840，视觉编码器为 27 层。Pro 配方包含 896 个路由专家，模型结构以发布配置为准。 [Kimi K3.1 Pro 配置](https://huggingface.co/moonshotai/Kimi-K3.1-Pro/blob/main/config.json)

### 1.2 特性与配置约束

| 特性 | 说明 |
|---|---|
| BF16 图文训练 | 两个配方均使用图文输入和 BF16 训练，从随机初始化开始 |
| KDA | Ascend 配方启用 FLA NPU 的 KDA 和短卷积 |
| DSA | 默认 Indexer 为 32 头、每头 128 维，Top-K 为 2048。Pro 配方要求 TP=CP=1 |
| `torch.compile` | Ascend 配方的局部编译配置排除 `gated_rmsnorm` |

## 2. 环境安装

按[精度验证环境](#6-精度验证)中的 Ascend 实测版本准备环境。使用 Linux aarch64、Ascend 950DT、CANN `9.2.1`、驱动 `26.2.0.b020` 和配套固件。Python、PyTorch、TorchNPU 和 Triton-Ascend 版本见该节环境表。复现实测环境时，Triton-Ascend launcher 使用 C++20。 [C++20 launcher 补丁](../../fixes/triton-ascend/TA-3-cxx20-launcher.patch)

FLA NPU 按 Kimi K3 指南构建和配置 OPP，使用 `ascend950` 构建目标及实测提交 `7e39f9668eed285fc91cde0502f46cd48686a681`。精度验证使用 `FLA_NPU_OPS=causal_conv1d,causal_conv1d_bwd,chunk_kda_fwd,chunk_kda_bwd`。 [FLA NPU 安装](kimi_k3.md#安装-fla-npu)

## 3. 数据集准备

debug 配方使用 TorchTitan 的 `tests/assets/cc12m_test`，默认启用样本 packing。Pro 配方使用 `cc12m` 流式数据，数据源为 `pixparse/cc12m-wds`。

## 4. Tokenizer 准备

两个配方默认使用 TorchTitan 的 Kimi K3 多模态 tokenizer 和自带的 `tests/assets/tokenizer` 测试资产。Pro 配方可通过 `hf_assets_path` 指定兼容的 tokenizer 资产目录。官方 Pro 的 `tiktoken.model` 需要转换成当前加载器支持的格式后才能使用。

## 5. 启动训练

按 Kimi K3 指南设置 `ASCEND_CUSTOM_OPP_PATH` 和 NPU 编译环境变量。从所安装的 TorchTitan 源码根目录运行以下命令。 [训练环境配置](kimi_k3.md#5-启动训练)

### 5.1 debug 配方

以下命令以单机 8 卡为例。配方默认运行 10 步，关闭 checkpoint 保存。

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 \
    ENTRYPOINT=torchtitan_ascend.train MODULE=torchtitan_ascend_recipes.models.kimi_k3_1 \
    CONFIG=kimi_k3_1_reduced_npu ./run_train.sh
```

在自定义 Python 配方中，可调用 `npu_fusion_kimik31(config)` 启用 DSA 等融合算子。 [debug 配方融合验证](https://gitcode.com/Ascend/torchtitan-ascend/pull/195)

### 5.2 Pro 配方

`kimi_k3_1_pro_npu` 可在自定义 Python 配方中设置 `seq_len` 和 `parallelism`，保持 TP=CP=1。完整 Pro 训练、长上下文和官方量化权重加载待验证。 [配方定义](../../torchtitan_ascend_recipes/models/kimi_k3_1.py)

## 6. 精度验证

| 项目 | Ascend 950DT | NVIDIA H100 SXM |
|---|---|---|
| 硬件与架构 | 单卡 96 GiB，aarch64 | 单卡 80 GB，x86_64 |
| Python | `3.12.16` | `3.12.3` |
| PyTorch / Torchvision | `2.15.0+cpu` / `0.29.2+cpu` | `2.15.0+cu132` / `0.29.2` |
| 运行时 / 驱动 | CANN `9.2.1` / `26.2.0.b020` | CUDA `13.2`、cuDNN `92600` / `615.71.09` |
| TorchNPU | `2.15.0+gitUnknown` | — |
| Triton | Triton‑Ascend `3.6.0`，launcher 使用 C++20 | Triton `3.9.0` |
| FLA NPU | `7e39f9668eed285fc91cde0502f46cd48686a681` | — |
| 基础镜像 | — | `pytorch/pytorch:2.15.0-cuda13.2-cudnn9-runtime` |

验证使用从 Pro 配方缩小的 224,414,768 参数图文模型。用官方 Pro tokenizer 将 CC12M 图文前缀和 C4 文本准备成冻结输入，每步包含 2048 个有效 token。两端使用 DSA Top-K 128、BF16 和严格确定性（`warn_only=False`），加载同一份随机初态和冻结输入，各训练 1000 步。以 H100 loss 为分母，逐步相对差为 −0.159379% 至 +0.142395%，平均绝对相对差为 0.02472833%。[PR #178 实测记录](https://gitcode.com/Ascend/torchtitan-ascend/pull/178)

重建该对照需取得原实验的 TorchNPU wheel、Triton‑Ascend wheel 及 C++20 launcher 补丁、实验入口和冻结输入文件。GPU 镜像实测记录保留了标签，复现时核对表中的实际版本。
