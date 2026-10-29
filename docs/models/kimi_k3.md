# Kimi K3

## 安装

完成[基础安装](../../INSTALL.md)，并按 [fla-npu 源码准备](qwen3_5.md#准备源码)准备算子源码和构建依赖。在工作目录执行：

```bash
cd flash-linear-attention-npu
FLA_NPU_SOC=ascend950 \
FLA_NPU_OPS=causal_conv1d,causal_conv1d_bwd,chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd,chunk_kda_fwd,chunk_kda_bwd \
    python scripts/build_wheel.py --no-progress
uv pip install dist/*.whl
cd ..
```

## 训练配置

配置模块：`torchtitan_ascend.recipes.kimi_k3`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `kimi_k3_debugmodel_npu` | debug | 上游自带 tokenizer 和 cc12m-test 图文数据 |

默认使用 BF16、DistMuon + AdamW、VarlenAttention 和 `DefaultActivationRematConfig`；最大上下文长度为 2048，每个 DP rank 每步 2048 tokens，共 10 步。模型随机初始化，不保存检查点。

## 数据准备

Debug 配方使用上游自带 tokenizer 和 cc12m-test 图文数据，无需下载。

## 启动

按[运行验证](../../INSTALL.md#运行验证)加载训练环境，并按 [Qwen3.5 启动说明](qwen3_5.md#启动)设置算子包路径。从包含 `torchtitan` 的工作目录执行：

```bash
cd torchtitan
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 \
    TORCHINDUCTOR_NPU_BACKEND=triton_experimental \
    TORCHINDUCTOR_COMPILE_THREADS=1 TORCHNPU_PRECOMPILE_THREADS=1 \
    ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.kimi_k3 CONFIG=kimi_k3_debugmodel_npu \
    ./run_train.sh
```

## 自定义数据

`kimi_k3_debugmodel_npu()` 不接受参数，只返回 `Trainer.Config`，数据源固定为 cc12m-test。自定义数据和训练参数需在单独的配方模块中配置。

## 支持范围

配方提供图文训练，不支持视频输入和上下文并行（CP）。完整训练需自行配置数据、训练步数、学习率计划和检查点。
