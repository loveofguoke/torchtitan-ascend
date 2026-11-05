# DeepSeek V4.1

## 安装

完成[基础安装](../../INSTALL.md)。当前适配基于 TorchTitan `v0.4.0`（`45e33cd413b64cb37a7840cfd3c9765d3c639ae1`），无需额外安装模型专用算子包。

## 训练配置

配置模块：`torchtitan_ascend.recipes.deepseek_v4_1`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `deepseek_v4_1_debugmodel_npu` | 40 层 debug | 上游自带 tokenizer 和 C4 测试数据 |

默认使用 BF16 混合精度、AdamW 和 eager 执行，并启用 Ascend RoPE、RMSNorm 和 SwiGLU override。最大上下文长度为 512，每个 DP rank 每个 microbatch 128 tokens，共 10 步。模型随机初始化，不保存检查点。

配方保留上游 loss、优化器和并行实现。CSA2 稀疏注意力复用上游 `_gather_attn` 和 `SparseIndexerLoss`，归一化分母由训练引擎通过 `aux_loss_denominators` 传入。

## 数据准备

Debug 配方使用上游自带 tokenizer 和 C4 测试数据，无需下载。

## 启动

按[运行验证](../../INSTALL.md#运行验证)加载训练环境，从包含 `torchtitan` 的工作目录执行：

```bash
cd torchtitan
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.deepseek_v4_1 \
    CONFIG=deepseek_v4_1_debugmodel_npu \
    ./run_train.sh
```

## 自定义数据

`deepseek_v4_1_debugmodel_npu()` 只接受 `tp` 和 `ep` 参数，数据源固定为 C4 测试数据。自定义数据和其他训练参数需在单独的配方模块中配置。

默认 TP=1、EP=1，使用 FSDP2。TP 大于 1 时启用序列并行；TP 需满足 MoE 约束 `EP >= TP`。

## 支持范围

配方支持 packed 文本训练，不提供 Engram、图像或视频输入，不支持上下文并行（CP）和流水线并行（PP）。完整训练需自行配置数据、训练步数、学习率计划和检查点。

当前 v0.4.0 适配已通过 12 个测试（CPU 参考、NPU RoPE、候选池偏移和缺省 positions 对照），并成功应用 301 个 override 和构造 40 层 debug 模型。在 2 张 910B2 上完成 FSDP2（TP=1、EP=1、DP shard=2）10 步训练，loss 从 12.36851 降至 4.88288；另完成 TP=2、EP=2 的 10 步训练，loss 从 12.40856 降至 5.15806。两次训练的 grad_norm 均有限，进程正常退出。修改前后的确定性对照中 loss 不变，输出和梯度仅有浮点舍入级差异。旧 `edde21fb` 基线的训练记录不作为当前版本的验证结论。
