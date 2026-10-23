# Kimi K3

## 安装

先完成[基础安装](../../INSTALL.md)。配方通过 override 接入 Ascend KDA 实现。

## 训练配置

配置模块：`torchtitan_ascend.recipes.kimi_k3`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `kimi_k3_debugmodel_npu` | debug | 上游自带 tokenizer 和 cc12m-test 测试数据 |

Debug 配置使用 BF16、`max_context_length=2048`、每个 DP rank 2048 tokens、10 步训练，并禁用 CUDA Graph。Flex Attention 使用编译路径，activation checkpoint 沿用默认 SelectiveAC；配方保留 `gated_rmsnorm` 编译过滤，并将训练进程组超时下限设为 600 秒。

## 数据准备

Debug 配方无需下载数据。

可以通过以下命令确认 tokenizer 资源目录存在：

```bash
cd "$TORCHTITAN_SRC"
test -d ./tests/assets/tokenizer
```

## 启动

在当前 Python 环境中配置算子包路径和运行时变量；新建 shell 时先按[准备环境](../../INSTALL.md#准备环境)设置环境。

```bash
FLA_NPU_ROOT="$(TORCH_DEVICE_BACKEND_AUTOLOAD=0 python -S -c 'import sysconfig; print(sysconfig.get_path("purelib") + "/fla_npu")')"
export ASCEND_CUSTOM_OPP_PATH="$FLA_NPU_ROOT/opp/vendors/fla_npu_transformer${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=1
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

安装 `pytest` 并完成 TorchTitan 依赖安装后，在上游源码目录执行：

```bash
cd "$TORCHTITAN_SRC"
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
| 数据并行 | A5 已完成 10 步训练验证 |
| 上下文并行（CP） | 不支持 |
| 张量、流水线并行（TP、PP） | 未验证 |
| 重计算策略（NoAC、FullAC、RegionAC、SelectiveAC） | A5 已完成 10 步训练验证 |
| 编译、激活检查点 | 未验证 |
| 预训练权重加载、导出和断点续训 | 未验证 |

长期收敛、稳定性和性能未验证。
