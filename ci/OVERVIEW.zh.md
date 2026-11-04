# TorchTitan-Ascend CI Docker 镜像概述

[English](OVERVIEW.md) | [简体中文](OVERVIEW.zh.md)

## 镜像信息

| 目标 | 镜像名称 | 基础镜像 |
| ------ | ------ | ------ |
| CPU | `torchtitan-ascend-ci:cpu` | `python:3.12-slim-bookworm` |
| A2 NPU | `torchtitan-ascend-ci:a2` | `ascendai/cann:9.2.0-910b-ubuntu22.04-py3.12` |

构建入口为 `ci/build.sh`，Dockerfile 为 `ci/Dockerfile`，默认架构为 `linux/arm64`。镜像将 TorchTitan 安装到 `/opt/torchtitan`，将本仓安装到 `/opt/torchtitan-ascend`；测试在 `/opt/torchtitan` 下执行。

默认从[官方仓库](https://gitcode.com/Ascend/torchtitan-ascend)的 `refactor/npu-override` 分支克隆源码。容器使用镜像内的源码和用例清单。

| 组件 | 版本 |
| ------ | ------ |
| Python | 3.12 |
| PyTorch / torchvision | `2.15.0+cpu` / `0.29.2+cpu` |
| TorchTitan | v0.4.0，提交 `45e33cd413b64cb37a7840cfd3c9765d3c639ae1` |
| pytest | `8.3.5` |
| A2 依赖 | CANN `9.2.0`、torch_npu、Triton-Ascend、`fla_npu`（`ascend910b`） |

A2 依赖的固定提交和安装步骤见 [install.sh](install.sh)。

## 构建镜像

在 aarch64 构建机的仓库根目录执行，主机需已安装 Docker，并能访问远端源码仓库及依赖下载源。

| 构建变量 | 用途 | 默认值 |
| ------ | ------ | ------ |
| `ASCEND_REPO` | 本仓的远端 HTTPS 克隆地址 | `https://gitcode.com/Ascend/torchtitan-ascend.git` |
| `ASCEND_BRANCH` | 克隆源码使用的分支 | `refactor/npu-override` |
| `MAX_JOBS` | 源码编译并行度，按主机内存调整 | `32` |
| `CORPORATE_CA_FILE` | 企业 HTTPS 网关的 PEM 格式 CA 证书路径 | 自动查找宿主机证书 |

```bash
# 构建 CPU 测试镜像：torchtitan-ascend-ci:cpu
bash ci/build.sh cpu

# 构建 A2 NPU 测试镜像：torchtitan-ascend-ci:a2
bash ci/build.sh a2
```

默认使用 Docker 构建缓存。需要重新拉取分支代码时，追加 `--no-cache`，例如 `bash ci/build.sh cpu --no-cache`（A2 同理）。

## 运行用例

可在宿主机任意目录执行以下命令，启动容器并运行镜像内的对应清单：

| 目标 | 用例清单 |
| ------ | ------ |
| CPU | `ci/testlists/cpu.txt` |
| A2 | `ci/testlists/a2.txt` |

镜像默认入口会自动配置环境，并将镜像内的本仓适配用例复制到 TorchTitan 的 `tests/` 下：CPU 复制 `tests/unit_tests/cpu/`，A2 复制 `tests/`。无需手动运行 `entrypoint.sh`。

### CPU

```bash
docker run --rm torchtitan-ascend-ci:cpu \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu
```

### A2

完整 A2 清单需要 8 张可见 NPU。以下命令用于已安装昇腾驱动的 A2 主机，设备编号为 `0..7`，由 Bash 执行：

```bash
docker run --rm --network host --ipc host \
    --device=/dev/davinci{0..7} \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    torchtitan-ascend-ci:a2 \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh a2
```

如果已经通过镜像默认入口进入容器，直接运行所属目标的脚本即可：

```bash
# CPU 容器内
bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu

# A2 容器内
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2
```

分布式用例自行启动 worker，整份清单无需再套一层 `torchrun`。

## 脚本参数与结果

```bash
# 容器内仅收集 CPU 用例，每批 100 条
BATCH=100 bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu --collect-only
```

| 参数 | 含义 | 默认值 |
| ------ | ------ | ------ |
| `cpu` / `a2` | 执行对应清单 | `cpu` |
| `BATCH` | 每批条目数，必须为正整数 | `150` |
| `WORKDIR` | TorchTitan 测试工作目录 | `/opt/torchtitan` |
| pytest 参数 | 透传给每次 pytest，例如 `--collect-only` | 无 |

脚本跳过空行和注释。批次失败时，该批全部条目会逐条重跑；仍失败的条目以 `FAILED_OR_ERROR` 汇总。

退出码：`0` 表示最终没有报告失败，`1` 表示重跑后仍有失败，`2` 表示参数或清单无效。流水线需保留 pytest 日志；跳过的用例以 pytest 的 `skipped` 统计为准。

## 新增用例

### 1. 在个人分支准备测试文件和清单

| 目标 | 本仓测试文件位置 | 登记清单 |
| ------ | ------ | ------ |
| CPU | `tests/unit_tests/cpu/` 下的对应路径 | `ci/testlists/cpu.txt` |
| A2 | `tests/` 下与上游相同的相对路径，例如 `tests/unit_tests/gpu/` | `ci/testlists/a2.txt` |

在宿主机的本仓个人分支中修改。需要适配时，准备完整测试文件；直接复用固定版本 TorchTitan 已有且无需修改的测试时，只登记清单。CPU 用例应能在未安装 NPU 包的环境中运行。

将对应 node ID 加入 TXT 的测试文件分组，新文件则增加分组注释。每行只写一个 node ID，例如：

```text
tests/unit_tests/cpu/config/test_configurable.py::TestConfigurable::test_valid_config
```

以上条目已存在，仅用于展示格式。登记前检查重复；清单中不写 `pytest`、命令参数、引号或行尾注释。新增依赖时，同时更新安装配置。

### 2. 提交、推送并构建个人分支镜像

在宿主机仓库根目录确认本次测试和清单修改，提交并推送到个人分支：

```bash
git add tests/ ci/testlists/
# 如修改了安装或依赖配置，也先用 git add 暂存对应文件
git commit -m "test: add CPU or A2 CI cases"
git push origin HEAD
```

将下方 HTTPS 地址替换为该个人分支所在仓库的克隆地址，分支名替换为实际名称；构建机及容器需能访问该地址。所选分支需包含 CI 脚本和用例清单。

```bash
# CPU；验证 A2 时将最后的 cpu 改为 a2
ASCEND_REPO='https://gitcode.com/YOUR_ACCOUNT/torchtitan-ascend.git' \
ASCEND_BRANCH='your-test-branch' \
    bash ci/build.sh cpu
```

### 3. 启动验证容器

在宿主机任意目录按目标执行，镜像默认入口会自动初始化环境并复制测试文件。

```bash
# CPU
docker run --rm -it torchtitan-ascend-ci:cpu bash
```

```bash
# A2
docker run --rm -it --network host --ipc host \
    --device=/dev/davinci{0..7} \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    torchtitan-ascend-ci:a2 bash
```

### 4. 收集并单独运行用例

在容器内执行。以下用已有 CPU 用例演示，替换为自己负责的文件和 node ID：

```bash
cd /opt/torchtitan
pytest tests/unit_tests/cpu/config/test_configurable.py --collect-only -q
pytest 'tests/unit_tests/cpu/config/test_configurable.py::TestConfigurable::test_valid_config' -vv -rs
```

用收集结果核对清单中的精确 node ID。选择单个参数化用例时，保留完整的 `[参数]` 后缀，并在 shell 命令中加引号。若用例被跳过，先确认设备、数据和依赖是否满足要求。

本地修改或远端后续提交不会自动进入已有镜像。测试文件、清单或依赖需要调整时，回到步骤 1，重新提交、推送，使用 `--no-cache` 构建，再启动新容器验证。

### 5. 验证完整清单并提交 PR

在对应容器中运行所属目标的完整清单：

```bash
# CPU 容器内
bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu

# A2 容器内
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2
```

单独用例及完整清单验证通过后，从个人分支向官方目标分支提交 PR；已有 PR 继续使用原分支，并补充本次验证结果。PR 中注明镜像对应的源码提交、执行命令和 pytest 的通过、失败、跳过结果；仅有 skip 不能作为验证通过的依据。

## 许可证

本仓基于 Apache License 2.0，详见 [LICENSE](../LICENSE)。第三方许可见 [LICENSE-BSD-3-Clause](../LICENSE-BSD-3-Clause) 和 [Third_Party_Open_Source_Software_Notice](../Third_Party_Open_Source_Software_Notice)。
