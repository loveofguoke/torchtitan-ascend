<div align="center">

# TorchTitan-Ascend

**面向 Ascend NPU 的 TorchTitan 训练扩展**

[快速开始](#快速开始) · [特性](#特性) · [模型文档](#模型文档) · [文档](#文档) · [开发与贡献](#开发) · [问题反馈](https://gitcode.com/Ascend/torchtitan-ascend/issues)

[![Python](https://img.shields.io/badge/Python-3.12%2B-3776AB?logo=python&logoColor=white)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
![Status](https://img.shields.io/badge/Status-Alpha-E34F26)

</div>

**TorchTitan-Ascend** 为 [TorchTitan](https://github.com/pytorch/torchtitan) 提供昇腾 NPU 适配与优化，复用上游模型、训练流程和配置接口。训练框架继续由 TorchTitan 维护，本仓通过组件适配与 override 接入 Ascend 实现。

## 快速开始

1. 按[安装指南](INSTALL.md)准备面向 A5（Ascend 950）的依赖和训练环境。
2. 完成安装后，在 TorchTitan 源码目录运行 Qwen3 debug 配方：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train MODULE=torchtitan_ascend.recipes.qwen3 CONFIG=qwen3_debugmodel_npu ./run_train.sh
```

示例使用内置 tokenizer 和测试数据，随机初始化模型，预期运行 10 步并正常退出。

3. 其他模型的数据准备、配方与运行条件见下方模型文档。

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

## 模型文档

- [Qwen3](docs/models/qwen3.md)
- [Qwen3.5](docs/models/qwen3_5.md)
- [Kimi K3](docs/models/kimi_k3.md)
- [DeepSeek V4.1](docs/models/deepseek_v4_1.md)

模型文档中的局部验证不自动构成正式支持；具体组合与证据以相关模型文档及公开评审记录为准，支持状态遵循[维护策略](docs/maintenance.md#2-基线与支持状态)。

## 文档

| 内容 | 入口 |
|---|---|
| 环境准备与最小运行 | [安装指南](INSTALL.md) |
| 模型用法与前置条件 | [模型文档](#模型文档) |
| 维护、评审与验收规则 | [维护策略](docs/maintenance.md) |
| 贡献步骤与开发约束 | [贡献指南](CONTRIBUTING.md)、[开发规范](AGENTS.md) |

维护策略的采纳不表示相关实现或验收已经完成，也不自动赋予现有能力正式支持状态。实际使用步骤见安装及模型文档。

## 开发

欢迎报告问题、改进文档、参与验证和提交代码。按[贡献指南](CONTRIBUTING.md)准备环境、选择检查并提交 PR；没有 NPU 也可以参与文档和满足依赖条件的配置检查。

- [静态与文档检查](CONTRIBUTING.md#静态与文档检查)：工具准备、格式及链接检查。
- [测试入口](CONTRIBUTING.md#现有测试入口)：已有训练与算子用例的前提、命令和覆盖范围。
- [提交 PR](CONTRIBUTING.md#4-提交-pr)：变更说明、文档更新与验证证据。
- [架构设计讨论](https://gitcode.com/Ascend/torchtitan-ascend/issues/5)：已有方案讨论与背景。

## 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。第三方组件归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)。
