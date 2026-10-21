# GLM-5

## 安装

先完成[基础安装](../../INSTALL.md)。GLM-5 NPU 配方复用 TorchTitan 上游模型实现，
不在本仓复制模型主体，也不引入模型专用的第三方 Python 依赖。

## 训练配置

配置模块：`torchtitan_ascend.recipes.glm5`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `glm5_2_debugmodel_npu` | GLM-5.2 debug | 上游自带 tokenizer 和 C4 测试数据 |
| `glm5_3_debugmodel_npu` | GLM-5.3 debug | 上游自带 tokenizer 和 C4 测试数据 |

`glm5_debugmodel_npu` 保留为 `glm5_2_debugmodel_npu` 的兼容名称。

默认配置为 BF16、eager、上下文长度 128、每 DP rank 的 microbatch 为 256 tokens，
共训练 10 步。模型随机初始化，不保存检查点。完整训练需自行配置数据、训练步数、
学习率计划和检查点。

## 启动

在 TorchTitan 源码目录执行：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.glm5 CONFIG=glm5_2_debugmodel_npu \
    ./run_train.sh
```

验证 GLM-5.3 时，将 `CONFIG` 改为 `glm5_3_debugmodel_npu`。

## Ascend 适配

- ComplexRoPE override 使用 `torch_npu.npu_rotary_mul`，并保留 TP/CP DTensor placements。
- DTensor 参数初始化使用 NPU 可执行的 truncated-normal 路径。
- TP vocabulary-parallel cross entropy 避免对本地词表范围外的 label 取值。
- 模型主体、flavor、MTP 和 checkpoint 映射由 TorchTitan 上游维护。

## 支持范围

| 场景 | 状态 |
|---|---|
| GLM-5.2/5.3 debug 配方构建 | CPU 测试覆盖 |
| 单卡 BF16、eager 文本训练 | NPU smoke 已验证 |
| 单卡 BF16 整网 loss 对照 | CUDA/NPU 相同 fixture 的历史单次运行已完成 150 步对照 |
| TP/CP 下 ComplexRoPE placements | 行为测试覆盖 |
| TP vocabulary-parallel loss | 行为测试覆盖 |
| FSDP、TP、CP、PP、EP 组合训练 | 本 PR 未形成正式精度证据 |
| 图模式和性能优化 | 不在本 PR 范围内 |

单卡 loss 对照为历史依赖基线数据，本次 PR 未重新运行。该结果只有一次配对运行，
不表示三次运行的均值或方差。长期收敛、稳定性和性能未验证。
