# 模型与配方索引

本索引列出本仓提供的模型使用文档和 Ascend 配方，不以是否具有本地模型实现目录判断模型可用性。配方用途不等于正式支持承诺；具体依赖、运行条件、限制和验证证据见各模型文档，支持状态遵循[维护策略](../maintenance.md#2-基线与支持状态)。

| 模型文档 | 实现来源 | 配方源码 | 配方用途 |
|---|---|---|---|
| [Qwen3](qwen3.md) | 复用 TorchTitan Qwen3 | [qwen3.py](../../torchtitan_ascend_recipes/models/qwen3.py) | debug 最小运行与配置派生 |
| [Qwen3.5](qwen3_5.md) | 复用 TorchTitan Qwen3.5 | [qwen3_5.py](../../torchtitan_ascend_recipes/models/qwen3_5.py) | debug、0.8B 文本短程运行与自定义数据配置 |
| [Qwen3.8](qwen3_8.md) | 复用 TorchTitan Qwen3.8 | [qwen3_8.py](../../torchtitan_ascend_recipes/models/qwen3_8.py) | debug 文本短程运行与配置派生 |
| [Kimi K3](kimi_k3.md) | 复用 TorchTitan Kimi K3 | [kimi_k3.py](../../torchtitan_ascend_recipes/models/kimi_k3.py) | debug、自定义 7B 规格及 MoonEP 配置 |
| [Kimi K3.1](kimi_k3_1.md) | 本地 Kimi K3.1 扩展，复用上游 Kimi 组件 | [kimi_k3_1.py](../../torchtitan_ascend_recipes/models/kimi_k3_1.py) | reduced 适配验证与 Pro 训练配置；完整 Pro 训练待验证 |
| [DeepSeek V3](deepseek_v3.md) | 复用 TorchTitan DeepSeek V3 | [deepseek_v3.py](../../torchtitan_ascend_recipes/models/deepseek_v3.py) | debug packed 文本短程运行及 FSDP/TP/EP 配置 |
| [DeepSeek V4](deepseek_v4.md) | 复用 TorchTitan DeepSeek V4 | [deepseek_v4.py](../../torchtitan_ascend_recipes/models/deepseek_v4.py) | debug packed 文本短程运行及 TP/EP/PP 配置 |
| [DeepSeek V4.1](deepseek_v4_1.md) | 本地 DeepSeek V4.1 实现，复用上游组件 | [deepseek_v4_1.py](../../torchtitan_ascend_recipes/models/deepseek_v4_1.py) | debug packed 文本短程运行及 TP/EP 配置 |
| [GPT-OSS](gpt-oss.md) | 本地 GPT-OSS 实现，复用上游组件 | [gpt_oss.py](../../torchtitan_ascend_recipes/models/gpt_oss.py) | debug packed 文本短程运行及 TP/EP 配置 |

先完成[基础安装](../../INSTALL.md)，再按模型文档准备专用依赖和数据。配方组织、启动接口及旧模块路径迁移见[配方说明](../../torchtitan_ascend_recipes/README.md)。
