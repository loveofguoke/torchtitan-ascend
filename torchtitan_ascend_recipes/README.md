# Ascend 训练配方

本包组合 TorchTitan 模型、数据、优化器、并行配置与 Ascend 实现，提供返回完整训练配置的 Python 函数。模型实现和设备适配仍由 `torchtitan` 与 `torchtitan_ascend` 提供。

## 目录职责

- `models/<model>.py`：模型训练配方与共享配置，包括面向用户的最小运行和配置派生入口。
- `tests/models/`：按实际需要添加的可复用测试专用配置；单个集成测试专用配置仍随测试定义维护。
- `__init__.py`：包声明，不聚合导入各模型配方。

目录按用途划分，不按模型大小、运行步数或验证状态划分。模型文档、配方源码与用途统一见[模型与配方索引](../docs/models/README.md)，支持范围及验证记录见各模型文档。

## 启动

按[安装指南](../INSTALL.md)安装本项目并加载训练环境，从所选 TorchTitan 源码目录使用上游 `run_train.sh`：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend_recipes.models.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

`MODULE` 指定可导入的配方模块，`CONFIG` 指定返回训练配置的函数。自定义配方仍可放在任意可导入模块中，通过这两个参数选择。资产路径和工作目录要求见对应模型文档。

## 模块路径迁移

原 `torchtitan_ascend.recipes.<model>` 已迁至 `torchtitan_ascend_recipes.models.<model>`，适用于 `qwen3`、`qwen3_5`、`kimi_k3`、`kimi_k3_1` 和 `deepseek_v4_1`。请更新启动脚本中的 `MODULE` 和自定义配方的 Python 导入；旧路径不提供转发。

函数名、参数、默认配置以及 `ENTRYPOINT=torchtitan_ascend.train` 保持不变。新包随同一个 `torchtitan-ascend` 发行包安装；更新已有安装时按安装指南重新安装本项目，以确保新包可被导入。
