# Qwen3 最小后端修复

基线为 torch_npu `95c7e465426ab894ffdb43ae878ee111e090fb17` 及其 op-plugin 子模块 `0abe20992cf389c4e9615f96f7198205a155c769`。

| 补丁 | 原生 Qwen3 varlen 路径的依赖 | 上游来源 |
|---|---|---|
| `torch_npu/NPU-1-flash-attention-privateuse1.patch` | 为原生 `varlen_attn` 调用的 FlashAttention 前向、反向算子注册 NPU 内核。 | [torch_npu !45527](https://gitcode.com/Ascend/pytorch/merge_requests/45527) |
| `op-plugin/NPU-6-zero-uint64.patch` | 原生 `varlen_attn` 会创建 uint64 RNG 占位张量，需要 NPU `zero_` 支持。 | [op-plugin !5801](https://gitcode.com/Ascend/op-plugin/merge_requests/5801) |

配方使用真实单卡进程组、CosSinRoPE、eager 计算，不启用 activation checkpoint。fake 进程组、复数 RoPE、FlexAttention/Inductor、流水线 splitting、remat storage 及其他模型的修复不属于这条训练链。

本批补丁仅保留运行时修复，后端单元测试由对应上游 PR 维护；本仓只保留一个 Qwen3 训练验证，命令见项目 README。上游发布包含修复的兼容版本后，应重新验证并移除对应补丁。

## 移除对照

在同一上游版本、单张 910B2 和本仓配方下逐项验证：

| 后端内容 | 结果 |
|---|---|
| 仅 NPU-1，不含 NPU-6 | 第一步前向创建 RNG 张量时，`zero_` 报 `DT_UINT64` 不支持，错误码 `161002`。 |
| 保留 NPU-6，移除 NPU-1 注册 | `aten::_flash_attention_forward` 缺少 NPU 内核，CPU fallback 也不能执行该算子，训练失败。 |
| 仅 NPU-1 + NPU-6 | 完成 10 步原生训练，loss `7.66793 → 5.30473`。 |

对照使用从固定源码重新构建的 torch_npu，而非全量开发补丁环境。NPU-6 对照由无该补丁的完整构建开始，再增量构建清零内核并更新 wheel；NPU-1 对照仅恢复后端注册文件至上游版本。验证环境未安装 Triton-Ascend 或 fla-npu，上游 TorchTitan 源码未修改。
