# 依赖补丁

上游状态、验证范围与移除条件见 [补丁跟踪 Issue #4](https://gitcode.com/Ascend/torchtitan-ascend/issues/4)。

| 组件 | 适用版本 |
|---|---|
| TorchTitan | `edde21fb648d446fecb5b78ef42d8c432274b935` |
| torch_npu | `2c8682df9271e545e7f50f8c45e2395563a05483` |
| op-plugin | `5ac12689b6a61ec9db3fc260eab17bbc5447567e` |
| Triton-Ascend | `3.6.0` |

| 补丁 | 用途 | 上游关联 |
|---|---|---|
| [FlashAttention](torch_npu/flash-attention.patch) | 为原生变长（varlen）注意力调用的 FlashAttention 前后向算子注册 NPU 内核。 | [torch_npu !45527](https://gitcode.com/Ascend/pytorch/merge_requests/45527)；[Issue #4439](https://gitcode.com/Ascend/pytorch/issues/4439)。 |
| [Storage](torch_npu/storage.patch) | 支持 RegionAC 保存区域使用的无 backing allocation placeholder storage；修复 NPU 工厂，通过 `PatchManager` 注册 Python 构造入口分流，并在正常初始化流程中只应用一次。 | [torch_npu !47944](https://gitcode.com/Ascend/pytorch/merge_requests/47944)；通用构造入口分流尚未提交上游。 |
| [Zero-uint64](op-plugin/NPU-6-zero-uint64.patch) | 支持变长注意力创建 uint64 RNG 占位张量所需的 `zero_`。 | [op-plugin !5801](https://gitcode.com/Ascend/op-plugin/merge_requests/5801)；[Issue #467](https://gitcode.com/Ascend/op-plugin/issues/467)。 |
| [TA-3](triton-ascend/TA-3-cxx20-launcher.patch) | 使用 C++20 编译 Triton launcher，兼容当前 PyTorch 头文件。 | 复用社区 [Triton-Ascend #1919](https://github.com/triton-lang/triton-ascend/pull/1919)；[Issue #1411](https://github.com/triton-lang/triton-ascend/issues/1411)。 |

应用与构建步骤见[安装指南](../INSTALL.md)。
