# 安装补丁

固定版本与安装命令见 [INSTALL.md](../INSTALL.md)，上游状态见 [补丁跟踪 Issue #4](https://gitcode.com/Ascend/torchtitan-ascend/issues/4)。训练入口自动安装 `torchtitan_ascend/patches/` 中的运行时修复。

本目录仅存放固定依赖版本运行训练所必需、且无法由运行时修复覆盖的源码补丁，由安装指南显式应用。新增补丁须记录阻塞场景、适用版本和上游跟踪；仅保留必要的功能改动。升级到包含修复的依赖版本后，验证并移除对应补丁。

| 依赖 | 补丁 | 用途 |
| --- | --- | --- |
| torch_npu `2c8682df` | [storage.patch](torch_npu/storage.patch) | 支持 remat 使用的空指针、正逻辑大小、不可扩容的 storage 占位对象；跟踪 [!47944](https://gitcode.com/Ascend/pytorch/merge_requests/47944)。 |
| Triton-Ascend `3.6.0` | [TA-3-cxx20-launcher.patch](triton-ascend/TA-3-cxx20-launcher.patch) | 使用 C++20 编译 launcher，与 PyTorch 头文件一致；跟踪 [#1919](https://github.com/triton-lang/triton-ascend/pull/1919)。 |
| TorchTitan `edde21fb` | [optional-attn-gym-backends.patch](torchtitan/optional-attn-gym-backends.patch) | 允许未安装 Attention Gym 的 CUDA 后端时导入 GDN、KDA，并通过 override 使用 NPU 实现。 |
| TorchTitan `edde21fb` | [entrypoint.patch](torchtitan/entrypoint.patch) | 回移上游已合入的 [#5277](https://github.com/pytorch/torchtitan/pull/5277) 中的 `ENTRYPOINT` 支持，使 `run_train.sh` 调用 Ascend 训练入口。 |

TorchTitan 补丁不提供 Attention Gym reference 执行路径；调用缺失的 CUDA 后端仍会报错。
