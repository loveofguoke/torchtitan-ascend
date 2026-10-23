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

## 开发

- [架构设计](https://gitcode.com/Ascend/torchtitan-ascend/issues/5)
- [依赖补丁](patches/README.md)

完成[安装](INSTALL.md)后，在同一 Python 环境和 shell 中安装测试依赖并运行 CPU 测试：

```bash
uv pip install -e "$ASCEND_REPO[test]"
cd "$ASCEND_REPO"
python -m pytest tests -m 'not npu' -q
```

NPU 测试需先完成 [Qwen3.5 安装与数据准备](docs/models/qwen3_5.md)，并按其[启动说明](docs/models/qwen3_5.md#启动)配置算子包路径：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
    python -m pytest "$ASCEND_REPO/tests" -m npu -q
```

## 许可证

本项目采用 [Apache-2.0](LICENSE) 许可证。第三方组件归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)。

## 静态检查

安装 `pre-commit>=4,<5`，并准备当前平台的 `lychee`（放在 `PATH`）和
`gitleaks`（放在仓库根目录并赋予执行权限）。macOS 可用 `brew install lychee gitleaks`，
再执行 `ln -s "$(command -v gitleaks)" ./gitleaks`。CI 同样需要预先提供这两个工具。

运行 `pre-commit run --all-files` 检查全仓，运行 `pre-commit install` 安装提交前检查。
Python 检查工具由 pre-commit 在隔离环境中安装，无需安装训练环境。
ufmt 统一格式化；Pyrefly 检查本仓类型，外部 Torch/NPU 运行时依赖按未知类型处理。
链接检查沿用上游策略，仅将 HTTP 404 判为失败，网络请求错误仍会报错。
Gitleaks hook 检查暂存区；全目录扫描使用 `./gitleaks dir --redact --config=pre-commit/.gitleaks.toml .`。
