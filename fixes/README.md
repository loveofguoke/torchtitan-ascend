# 安装补丁

依赖版本与安装命令见 [INSTALL.md](../INSTALL.md)。运行时修复见 [`torchtitan_ascend/patches/`](../torchtitan_ascend/patches/README.md)。

## 应用补丁

在包含各源码目录的工作目录中，于构建依赖前应用补丁：

```bash
git -C torch_npu apply ../torchtitan-ascend/fixes/torch_npu/storage.patch
git -C triton-ascend apply \
    ../torchtitan-ascend/fixes/triton-ascend/TA-3-cxx20-launcher.patch \
    ../torchtitan-ascend/fixes/triton-ascend/zero-stride-store.patch
```

补丁应用后分别构建、安装 torch_npu 和 Triton-Ascend。

## 补丁列表

| 依赖 | 补丁 | 用途 |
| --- | --- | --- |
| torch_npu `1a25e6be` | [storage.patch](torch_npu/storage.patch) | 支持 remat 使用的空指针、正逻辑大小、不可扩容的 storage 占位对象；跟踪 [!47944](https://gitcode.com/Ascend/pytorch/merge_requests/47944)。 |
| Triton-Ascend `3.6.0` | [TA-3-cxx20-launcher.patch](triton-ascend/TA-3-cxx20-launcher.patch) | 使用 C++20 编译 launcher，与 PyTorch 头文件一致；跟踪 [#1919](https://github.com/triton-lang/triton-ascend/pull/1919)。 |
| Triton-Ascend `3.6.0` | [zero-stride-store.patch](triton-ascend/zero-stride-store.patch) | 为 compiled BlockMask 的 store 保留零步长 pointer 轴和 mask，保持 pointer、value、mask 形状一致。跟踪 [#2783](https://github.com/triton-lang/triton-ascend/pull/2783)。 |

## 维护

本目录仅存放固定依赖版本运行训练所必需、且无法由运行时修复覆盖的源码补丁，由安装指南显式应用。补丁需列明适用版本、用途和上游跟踪。
