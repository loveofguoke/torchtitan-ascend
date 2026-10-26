# Kimi K3

## 安装

先完成[基础安装](../../INSTALL.md)，并按 [fla-npu 源码准备](qwen3_5.md#准备源码)准备固定版本的 fla-npu 源码和构建依赖。使用以下命令构建同时支持 GDN、KDA 的算子包：

```bash
cd "$FLA_NPU_SRC"
FLA_NPU_SOC=ascend950 \
FLA_NPU_OPS=causal_conv1d,causal_conv1d_bwd,chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd,chunk_kda_fwd,chunk_kda_bwd \
    python scripts/build_wheel.py --no-progress
uv pip install dist/*.whl
```

## 训练配置

配置模块：`torchtitan_ascend.recipes.kimi_k3`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `kimi_k3_debugmodel_npu` | debug | 上游自带 tokenizer 和 cc12m-test 测试数据 |

默认使用 BF16、DistMuon + AdamW、FlexAttention 和 SelectiveAC；最大上下文长度为 2048，每个 DP rank 2048 tokens，共 10 步。

## 数据准备

Debug 配方使用上游自带 tokenizer 和 cc12m-test 图文数据，无需下载。

## 启动

按[安装指南](../../INSTALL.md#验证安装)加载训练环境，并按 [Qwen3.5 启动说明](qwen3_5.md#启动)设置算子包路径，再选择编译后端：

```bash
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
```

在 TorchTitan 源码目录执行：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.kimi_k3 CONFIG=kimi_k3_debugmodel_npu \
    ./run_train.sh
```

## 集成测试

完成上述安装和环境配置后，使用两张 NPU 运行 loss 对照：

```bash
uv pip install -e "$ASCEND_REPO[test]"
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1 \
    python -m pytest "$ASCEND_REPO/tests/integration_tests/test_kimi_k3.py" -v
```

对比已有精度基线，覆盖开启确定性计算的双卡 FSDP2 的 20 步训练，逐步相对误差须小于 `0.2%`；要求至少两张可见 NPU，不足时自动跳过。

运行日志保存在仓库根目录 `outputs/kimi_k3/`，固定基线为 `tests/assets/losses/kimi_k3/kimi_k3_fsdp2.txt`。

## 自定义数据

`kimi_k3_debugmodel_npu()` 不接受参数，只返回 `Trainer.Config`；当前配方的数据源固定为 cc12m-test 数据集。自定义数据需在单独的配方模块中按固定 TorchTitan 接口实现 `Trainer.Config` 和 dataloader。

## 支持范围

| 场景 | 状态 |
|---|---|
| 单卡 BF16、图文数据 | A5 已完成 10 步训练验证 |
| 视频输入训练 | 不支持 |
| 双卡 FSDP2 + SelectiveAC | A5 已完成 20 步训练与 loss 基线对照 |
| 上下文并行（CP） | 不支持 |
| 张量、流水线并行（TP、PP） | 未验证 |
| 重计算策略（NoAC、FullAC、RegionAC、SelectiveAC） | A5 已完成 10 步训练验证 |
| 全图编译 | 未验证 |
| 预训练权重加载、导出和断点续训 | 未验证 |

长期收敛、稳定性和性能未验证。
