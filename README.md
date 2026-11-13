<div align="center">

# TorchTitan-Ascend

**面向 Ascend NPU 的 TorchTitan 训练扩展**

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Alpha-E34F26)

[快速开始](#快速开始) · [特性](#特性) · [模型与配方](#模型与配方) · [文档导航](#文档导航) · [社区交流](#社区交流) · [参与贡献](#参与贡献) · [问题反馈](https://gitcode.com/Ascend/torchtitan-ascend/issues)

</div>

**TorchTitan-Ascend** 为 [TorchTitan](https://github.com/pytorch/torchtitan) 提供昇腾 NPU 适配与优化，通过 Ascend 配方与组件 override 接入设备实现。

关于昇腾NPU：昇腾NPU（Neural Processing Unit，神经网络处理器）是华为开发的系列AI加速器。

- **PyTorch 原生训练**：复用 TorchTitan 的模型、训练流程与分布式并行能力，跟随上游持续演进。
- **昇腾亲和优化**：通过组件 override 接入 NPU 注意力、归一化等算子实现，沿用上游训练接口。
- **灵活的训练配方**：使用 Python 函数配置模型、数据和优化组件，从 debug 示例开始定制训练。

## 快速开始

### 1. 准备环境

按[安装指南](INSTALL.md)准备依赖，并完成[训练环境加载](INSTALL.md#运行验证)。当前安装基线为 Ascend 950PR&DT系列产品、Linux aarch64、Python 3.12 和 CANN 9.2.0，对齐 TorchTitan 0.4.0；完整版本与其他硬件的兼容性入口见安装指南。

### 2. 跑通单卡示例

在 **TorchTitan 源码目录**执行 Qwen3 debug 配方：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu \
    ./run_train.sh
```

示例使用内置 tokenizer 和测试数据，随机初始化模型，预期运行 **10 步并正常退出**。

### 3. 选择模型与数据

通过 `MODULE` 指定 Python 配方模块，通过 `CONFIG` 选择其中返回训练配置的函数。按下方模型文档准备专用依赖与数据；使用本地文本数据可参考 [Qwen3.5 0.8B 数据准备](docs/models/qwen3_5.md#数据准备)和[自定义数据示例](docs/models/qwen3_5.md#自定义数据)。

## 特性

<table>
<tr>
<td valign="top" width="50%">

**分布式并行**
- FSDP2、HSDP 和 DDP
- Tensor Parallel 和 Sequence Parallel
- FSDP/DDP 与 TP、CP 的组合
- Context Parallel：all-gather KV 和 Ulysses
- Expert Parallel 与 FSDP、TP 的组合

**显存与性能优化**
- FlexAttention 和 varlen attention
- Activation Checkpointing：Full、Selective 和 Regional
- Regional `torch.compile`（RoPE 区域）
- BF16 optimizer states

**训练能力**
- Grain data pipeline：packing、weighted mixing 和 checkpointable state
- Pretraining、SFT（多轮对话 loss mask）和 LoRA
- DistMuon、EMA 和 DeepSeek V3 Multi-token Prediction
- Warmup-Stable-Decay LR schedule 和 Validation

</td>
<td valign="top" width="50%">

**Checkpoint**
- Distributed Checkpoint 保存与同拓扑恢复
- 跨拓扑 model-only 恢复和异步保存
- Hugging Face safetensors 直接加载与保存
- HF 与 DCP 双向转换

**调试与观测**
- Metrics：loss、memory、throughput、TFLOPs 和 MFU
- TensorBoard 和 W&B
- Ascend Profiler 和 Memory Snapshot
- Structured Logging
- Deterministic Mode 和 Loss Comparison
- SDC Replay

**扩展机制**
- Python recipes
- Component Overrides
- GraphTrainer

</td>
</tr>
</table>

## 模型与配方

| 模型 | 现有配方 | 数据类型 | 额外依赖 |
|---|---|---|---|
| [Qwen3](docs/models/qwen3.md) | debug | 文本 | 基础环境 |
| [Qwen3.5](docs/models/qwen3_5.md) | debug、0.8B | 文本 | fla-npu 算子包 |
| [Kimi K3](docs/models/kimi_k3.md) | debug | 图文 | fla-npu 算子包 |
| [Kimi K3.1](docs/models/kimi_k3_1.md) | debug、Pro | 图文 | fla-npu 算子包 |
| [DeepSeek V4.1](docs/models/deepseek_v4_1.md) | debug | 文本 | 基础环境 |

具体运行条件、限制与验证证据见对应模型文档及公开评审记录。现有配方不代表完整训练或所有并行组合已完成验证；正式支持状态遵循[维护策略](docs/maintenance.md#2-基线与支持状态)。

## 文档导航

**使用与验证**

- [安装指南](INSTALL.md)：环境、依赖与最小运行验证。
- [模型与配方](#模型与配方)：模型依赖、数据准备与训练配置。
- [性能采集](docs/profiling.md)：性能采集与内存快照。
- [测试入口](CONTRIBUTING.md#现有测试入口) · [CI 镜像与用例运行](ci/OVERVIEW.zh.md)：检查前提、执行方式与覆盖范围。

**开发与扩展**

- [贡献指南](CONTRIBUTING.md) · [开发规范](AGENTS.md) · [维护策略](docs/maintenance.md)：开发步骤、适配约束与评审规则。

## 社区交流

欢迎参与 **[Ascend for PyTorch Distributed SIG](https://gitcode.com/Ascend/community/blob/master/AscendForPyTorch/sigs/distributed/README.md)**，交流使用经验，共同讨论模型适配、分布式训练与性能优化。

[例会与交流方式](https://gitcode.com/Ascend/community/blob/master/AscendForPyTorch/sigs/distributed/README.md#联系方式) · [问题与建议](https://gitcode.com/Ascend/torchtitan-ascend/issues)

## 参与贡献

欢迎改进文档、参与验证和提交代码。没有 NPU 也可以参与文档和满足依赖条件的配置检查。

从[贡献指南](CONTRIBUTING.md)开始，阅读[开发规范](AGENTS.md)，按改动范围完成[静态检查与相关测试](CONTRIBUTING.md#3-验证改动)，并在 [PR](CONTRIBUTING.md#4-提交-pr) 中记录实际验证结果。

## 免责声明

### 致使用者

- TorchTitan-Ascend 提供的所有内容仅供您用于非商业目的。
- 测试用例和示例中涉及的第三方模型和数据集用于功能验证。使用这些资源时，请遵守各自的许可证和使用条款。
- 本项目依赖的第三方开源软件由各自的社区维护。第三方软件的问题由相应社区跟进，本项目不保证能验证或修复其中的所有缺陷。
- 使用本项目时发现功能或合规问题，可通过[本仓 Issues](https://gitcode.com/Ascend/torchtitan-ascend/issues)反馈。

### 致模型和数据集权利人

如需更新或删除本项目中对您的模型或数据集的描述，请通过[本仓 Issues](https://gitcode.com/Ascend/torchtitan-ascend/issues)联系我们。

## 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。第三方组件归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)。
