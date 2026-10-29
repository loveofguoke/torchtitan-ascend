# Qwen3

## 训练配置

配置模块：`torchtitan_ascend.recipes.qwen3`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `qwen3_debugmodel_npu` | debug | 上游自带 tokenizer 和 C4 测试数据 |

默认使用单卡 BF16、上游 VarlenAttention 和 `DefaultActivationRematConfig`，关闭模型的局部编译。最大上下文长度为 2048，每个 DP rank 的 microbatch 为 16384 tokens，共 10 步。模型随机初始化，不保存检查点。完整训练需自行配置数据、训练步数、学习率计划和检查点。

## 启动

完成[基础安装](../../INSTALL.md)，按[运行验证](../../INSTALL.md#运行验证)加载训练环境。从包含 `torchtitan` 的工作目录执行：

```bash
cd torchtitan
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```
