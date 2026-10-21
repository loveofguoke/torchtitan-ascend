# Qwen3

## 训练配置

配置模块：`torchtitan_ascend.recipes.qwen3`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `qwen3_debugmodel_npu` | debug | 上游自带 tokenizer 和 C4 测试数据 |

默认使用单卡 BF16、上游 FlexAttention 和 SelectiveAC，关闭模型的局部编译。最大上下文长度为 2048，每个 DP rank 的 microbatch 为 16384 tokens，共 10 步。模型随机初始化，不保存检查点。完整训练需自行配置数据、训练步数、学习率计划和检查点。

## 启动

完成[安装](../../INSTALL.md)后执行。新建 shell 时先按[准备环境](../../INSTALL.md#准备环境)设置环境。

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

## 支持范围

已验证上述默认配置的单卡文本训练 10 步。多卡并行、编译、其他激活检查点配置、预训练权重加载、导出和断点续训未验证。

长期收敛、稳定性和性能未验证。
