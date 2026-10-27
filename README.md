<div align="center">

# torchtitan-ascend

**面向 Ascend NPU 的 TorchTitan 训练扩展**

[安装指南](INSTALL.md) · [模型文档](#模型文档) · [设计方案](https://gitcode.com/Ascend/torchtitan-ascend/issues/5) · [问题反馈](https://gitcode.com/Ascend/torchtitan-ascend/issues)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Alpha-E34F26)

</div>

**torchtitan-ascend** 为 [TorchTitan](https://github.com/pytorch/torchtitan) 提供昇腾 NPU 支持，复用上游模型、训练流程和配置接口。

## 快速开始

完成[安装](INSTALL.md)后，在 TorchTitan 源码目录执行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu ./run_train.sh
```

## 模型文档

- [Qwen3](docs/models/qwen3.md)
- [Qwen3.5](docs/models/qwen3_5.md)
- [Kimi K3](docs/models/kimi_k3.md)
- [GLM-5](docs/models/glm5.md)

## 开发

[安装补丁](fixes/README.md) · [运行时修复机制](torchtitan_ascend/patches/README.md) · [开发规范](AGENTS.md)

NPU 验证分别见 [Qwen3.5 模块与训练测试](docs/models/qwen3_5.md#验证)和 [Kimi K3 双卡集成测试](docs/models/kimi_k3.md#集成测试)。

## 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。第三方组件归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)。
