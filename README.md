<div align="center">

# torchtitan-ascend

**面向 Ascend NPU 的 TorchTitan 训练扩展**

[安装指南](INSTALL.md) · [模型文档](#模型文档) · [设计方案](https://gitcode.com/Ascend/torchtitan-ascend/issues/5) · [问题反馈](https://gitcode.com/Ascend/torchtitan-ascend/issues)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Alpha-E34F26)

</div>

**torchtitan-ascend** 为 [TorchTitan](https://github.com/pytorch/torchtitan) 提供昇腾 NPU 支持，复用上游模型、训练流程和配置接口。

面向 A5（Ascend 950）的安装步骤和依赖版本见[安装指南](INSTALL.md)。

## 快速开始

完成[安装和环境配置](INSTALL.md)后，在包含 `torchtitan` 的工作目录执行：

```bash
cd torchtitan
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

示例使用内置测试数据和随机初始化的 Qwen3 debug 模型，预期完成 10 步并正常退出。更多配置见 [Qwen3 文档](docs/models/qwen3.md)。

## 模型文档

- [Qwen3](docs/models/qwen3.md)
- [Qwen3.5](docs/models/qwen3_5.md)
- [Kimi K3](docs/models/kimi_k3.md)
- [DeepSeek V4.1](docs/models/deepseek_v4_1.md)

## 开发

[安装补丁](fixes/README.md) · [运行时修复机制](torchtitan_ascend/patches/README.md) · [开发规范](AGENTS.md)

## 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。第三方组件归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)。
