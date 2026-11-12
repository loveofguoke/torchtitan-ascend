# TorchTitan-Ascend CI Docker 镜像概述

[English](OVERVIEW.md) | [简体中文](OVERVIEW.zh.md)

## 镜像信息

| 目标 | 镜像名称 | 基础镜像 |
| ------ | ------ | ------ |
| CPU | `torchtitan-ascend-ci:cpu` | `python:3.12-slim-bookworm` |
| A2 NPU | `torchtitan-ascend-ci:a2` | `ascendai/cann:9.2.0-910b-ubuntu22.04-py3.12` |

构建入口为 `ci/build.sh`，Dockerfile 为 `ci/Dockerfile`，默认架构为 `linux/arm64`。镜像将 TorchTitan 安装到 `/opt/torchtitan`，在 `/opt/torchtitan-ascend` 预装本仓及依赖，作为基础环境；测试在 `/opt/torchtitan` 下执行。

构建时默认从[官方仓库](https://gitcode.com/Ascend/torchtitan-ascend)的 `refactor/npu-override` 分支准备基础环境。镜像可重复使用；每次测试使用流水线检出的待测代码、脚本和清单。

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

日常代码和用例修改无需重建镜像。依赖或编译环境变化时更新镜像；需要完整重建时可追加 `--no-cache`。

## 运行用例

以下命令在容器内执行，使用 `/opt/torchtitan-ascend` 中的代码。流水线应运行本次检出目录中的 `ci/run-testlist.sh`。

| 目标 | 用例清单 |
| ------ | ------ |
| CPU | `ci/testlists/cpu.txt` |
| A2 单卡 | `ci/testlists/a2-1card.txt` |
| A2 双卡 | `ci/testlists/a2-2card.txt` |
| A2 8 卡 | `ci/testlists/a2-8card.txt` |

```bash
# CPU
bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu

# A2 单卡
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-1card

# A2 双卡
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-2card

# A2 8 卡
bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-8card
```

脚本安装所在仓库的代码、复制适配用例，并执行对应清单。A2 用例脚本自动加载 CANN 等运行环境，通过 `docker exec` 执行时也无需手动执行入口脚本。

### 本地验证 CPU

在宿主机执行，使用镜像内 `/opt/torchtitan-ascend` 的代码：

```bash
docker run --rm torchtitan-ascend-ci:cpu \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh cpu
```

### 本地验证 A2

以下以 8 卡清单为例。由 Bash 执行，要求宿主机已安装昇腾驱动，设备编号为 `0..7`：

```bash
docker run --rm --network host --ipc host \
    --device=/dev/davinci{0..7} \
    --device=/dev/davinci_manager --device=/dev/devmm_svm --device=/dev/hisi_hdc \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro \
    -v /usr/local/Ascend/firmware:/usr/local/Ascend/firmware:ro \
    torchtitan-ascend-ci:a2 \
    bash /opt/torchtitan-ascend/ci/run-testlist.sh a2-8card
```

分布式用例自行启动 worker，整份清单无需再套一层 `torchrun`。

## 脚本参数与结果

```bash
# 容器的待测仓库根目录：安装、复制并收集 CPU 用例
BATCH=100 bash ci/run-testlist.sh cpu --collect-only
```

| 参数 | 含义 | 默认值 |
| ------ | ------ | ------ |
| `cpu` / `a2-1card` / `a2-2card` / `a2-8card` | 执行对应清单；按组分配设备 | `cpu` |
| `BATCH` | 每批条目数，必须为正整数 | `150` |
| `WORKDIR` | TorchTitan 测试工作目录 | `/opt/torchtitan` |
| pytest 参数 | 透传给每次 pytest，例如 `--collect-only` | 无 |

清单脚本跳过空行和注释。每批只执行一次，失败后继续执行后续批次。pytest 输出各批的详细日志，脚本在末尾汇总失败批次和 pytest 的失败或错误信息。如果 pytest 找不到清单中的某个用例，汇总会标明该批未执行。

所有批次无错误时脚本返回 `0`，任一批失败时返回 `1`，脚本参数无效或清单缺失、为空时返回 `2`。跳过的用例以 pytest 的 `skipped` 统计为准。

## 新增上游用例

### 1. 在个人分支准备测试文件和清单

| 目标 | 本仓测试文件位置 | 登记清单 |
| ------ | ------ | ------ |
| CPU | `tests/unit_tests/cpu/` 下的对应路径 | `ci/testlists/cpu.txt` |
| A2 | `tests/` 下与上游相同的相对路径，例如 `tests/unit_tests/gpu/` | `ci/testlists/a2-{1,2,8}card.txt` |

在宿主机的本仓个人分支中修改。需要适配时，准备完整测试文件；直接复用固定版本 TorchTitan 已有且无需修改的测试时，只登记清单。CPU 用例应能在未安装 NPU 包的环境中运行。登记前确认测试文件和 node ID 存在于固定版本的上游仓库；可提交必要的同路径适配，不新增本仓自定义用例。

将对应 node ID 加入 TXT 的测试文件分组，新文件则增加分组注释。每行只写一个 node ID，例如：

```text
tests/unit_tests/cpu/config/test_configurable.py::TestConfigurable::test_valid_config
```

以上条目已存在，仅用于展示格式。登记前检查重复；清单中不写 `pytest`、命令参数、引号或行尾注释。新增依赖时，同时更新安装配置。

新增 NPU 用例须按实际所需卡数（分布式用例以 `world_size` 为准），登记到 `a2-1card.txt`、`a2-2card.txt` 或 `a2-8card.txt`。每条用例只登记到一个分组，避免重复。

### 2. 准备对应测试镜像

使用对应的 CPU 或 A2 镜像；依赖变更时更新镜像。

### 3. 启动验证容器

在宿主机执行。进入容器后，将 `/opt/torchtitan-ascend` 更新到已推送的待测提交，再执行后续验证。

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

### 4. 收集并核对新增用例

在容器内执行以下命令，A2 用例将 `cpu` 替换为对应的 A2 清单目标：

```bash
cd /opt/torchtitan-ascend
bash ci/run-testlist.sh cpu --collect-only
```

核对新增 node ID 能被收集。参数化用例需保留完整的 `[参数]` 后缀；清单中不加 shell 引号。只添加测试文件、未登记清单的用例不会自动运行。若用例被跳过，先确认设备、数据和依赖是否满足要求。

### 5. 验证完整清单并提交 PR

在对应容器中运行所属目标的完整清单：

```bash
# CPU 容器的待测仓库根目录
bash ci/run-testlist.sh cpu

# A2 容器的待测仓库根目录，按分配卡数执行对应清单
bash ci/run-testlist.sh a2-1card
bash ci/run-testlist.sh a2-2card
bash ci/run-testlist.sh a2-8card
```

完整清单验证通过后，提交测试文件、对应清单及必要的代码和依赖配置，从个人分支向官方目标分支提交 PR；已有 PR 继续使用原分支。PR 中注明待测提交、镜像名称、执行命令和 pytest 的通过、失败、跳过结果；仅有 skip 不能作为验证通过的依据。

## 许可证

本仓基于 Apache License 2.0，详见 [LICENSE](../LICENSE)。第三方许可见 [LICENSE-BSD-3-Clause](../LICENSE-BSD-3-Clause) 和 [Third_Party_Open_Source_Software_Notice](../Third_Party_Open_Source_Software_Notice)。
