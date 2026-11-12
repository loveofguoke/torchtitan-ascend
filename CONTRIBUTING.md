# 贡献指南

欢迎报告问题、改进文档、复现验证和提交代码。本指南说明贡献步骤；维护与验收规则集中在[维护策略](docs/maintenance.md)中。

## 1. 开始一项贡献

阅读 [README](README.md) 了解项目范围，在[社区 issue](https://gitcode.com/Ascend/torchtitan-ascend/issues)中查找相关讨论。参与已有任务时，留言说明拟处理的范围；已有贡献者参与时协商分工。修改代码前阅读 [AGENTS.md](AGENTS.md)，涉及模型或特性选型时参与相关社区 issue，并查阅相关模型文档与公开验证记录了解已有范围。

新问题按以下入口提交；普通修复和文档修改可以直接发 PR，无需等待任务认领或方案批准：

- 缺陷报告：说明预期与实际行为，提供已知版本、复现方式和日志；未知项可以注明，无需先定位根因。
- 功能或任务提议：说明使用场景和预期结果，已有方案或上游来源可以附上；实现方案和验收方法可在讨论中完善。

只报告问题或提出建议时，提交 issue 即可，无需准备开发环境或执行后续开发步骤。

RFC 提案在社区创建标题以 `[RFC]` 开头的 issue，说明背景、方案、取舍与验证计划，讨论后记录决议并关联后续 PR。涉及较大改动时，按 [RFC 使用范围](docs/maintenance.md#何时需要-rfc)判断是否需要设计提案；不确定实现归属或是否需要 RFC 时，可以先在 issue 中描述问题，请维护者协助判断。已有任务、方案或报告直接链接。

没有 NPU 时，也可以改进文档、报告问题或执行满足依赖条件的配置检查。需要协助验证的改动可以先提交 PR，具体方式见[参与评审并完成贡献](#5-参与评审并完成贡献)。

### 贡献归属与原则

本仓维护 Ascend 适配与优化，通用能力优先贡献所属上游；代码组织与设计原则见 [ARCHITECTURE.md](ARCHITECTURE.md)，编码与适配约束见 [AGENTS.md](AGENTS.md)，维护边界见[维护策略](docs/maintenance.md#1-项目边界)。

## 2. 开发改动

在 GitCode fork 仓库并将自己的 fork 克隆到本地的 `torchtitan-ascend/` 目录；已有仓库写入权限时可直接使用本仓。当前 Ascend 重构实现的贡献以 `refactor/npu-override` 为目标分支，从该分支创建个人开发分支；相关 issue 指定其他目标时以该说明为准，只修改本次贡献涉及的内容。

进入克隆后的仓库根目录，设置本仓路径：

```bash
export ASCEND_REPO="$PWD"
```

按贡献类型准备环境：

| 改动 | 所需准备 |
|---|---|
| 文档或静态检查 | 按下文准备检查工具，无需安装 TorchTitan、PyTorch 或 NPU 后端 |
| 配置、训练或算子实现 | 按 [INSTALL](INSTALL.md) 与相关模型文档准备对应检查的依赖，核对实际源码与环境组合，按具体检查准备依赖与设备 |

已有本仓 checkout 时，执行 INSTALL 的“准备环境”步骤应跳过克隆本仓的命令，从 `torchtitan-ascend/` 的父目录继续创建环境、安装依赖和本项目。安装步骤中的 `torchtitan-ascend/` 必须指向上述个人开发目录，与 `ASCEND_REPO` 一致；不要另行克隆官方仓库作为测试对象。

执行配置、训练或算子检查前，确认本仓与上游完整 SHA、依赖和补丁，专用前提见 [Qwen3](docs/models/qwen3.md)、[Qwen3.5](docs/models/qwen3_5.md)。当前文档不代替按维护策略完成的基线验收；遇到不一致时，在 issue 或 PR 中记录所用版本、命令和错误，请求协助。

## 3. 验证改动

按改动影响选择检查：文档修改检查链接、章节锚点、命令引用和前置条件；执行路径变化运行相关已有测试。涉及数值、功能或性能声明时，按[验收规则](docs/maintenance.md#5-验收规则)提前明确对照与通过标准。

### 静态与文档检查

在 Python 3.12 或以上版本的独立环境中安装 `pre-commit`：

```bash
python -m pip install 'pre-commit>=4,<5'
```

链接检查使用 PATH 中的 `lychee`，macOS 可用 `brew install lychee` 安装。缺少该工具时 hook 会提示并跳过，应将链接检查记录为未执行。当前配置未启用 Gitleaks hook，无需为执行 pre-commit 准备该工具。

在个人开发分支的仓库根目录，按本次修改的文件执行检查，替换示例中的文件列表：

```bash
cd "$ASCEND_REPO"
pre-commit run --files CONTRIBUTING.md
```

全仓检查使用 `pre-commit run --all-files`，未被 Git 跟踪的新文件需用 `--files` 显式选择。可选执行 `pre-commit install` 启用提交前自动检查。

Python 检查工具由 pre-commit 在隔离环境中安装；ufmt 统一格式化，Pyrefly 检查本仓类型，外部 Torch/NPU 运行时依赖按未知类型处理。具体检查项见[配置](.pre-commit-config.yaml)。静态检查不能证明真实运行兼容。

部分 hook 会修改文件，需复查 diff 并重跑相关检查。链接检查仅将 HTTP 404 判为失败，网络请求错误仍会报错。工具不可用或检查失败时记录实际情况，可先提交 PR 请求协助，合入要求见[合入条件](docs/maintenance.md#合入条件)。

### 现有测试入口

按改动影响选择用例；下表仅列常用测试示例及其覆盖范围，不是完整测试清单，也不能代替验收计划。其他用例按对应模型文档选择，例如 [Kimi K3 测试源码](tests/integration_tests/test_kimi_k3.py)；运行条件见 [Kimi K3 模型文档](docs/models/kimi_k3.md)。

完成对应依赖准备后，在同一 Python 环境和 shell 中安装本仓测试依赖，再设置实际使用的 TorchTitan 源码路径；`ASCEND_REPO` 沿用第 2 节的设置：

```bash
uv pip install -e "$ASCEND_REPO[test]"
export TORCHTITAN_SRC=/absolute/path/to/torchtitan
```

| 入口 | 前提与覆盖范围 |
|---|---|
| [test_training.py](tests/test_training.py) | 单卡真实进程组；Qwen3 debug、Qwen3.5 debug/0.8B 的规定步数、有限 loss/梯度和选定参数更新 |
| [test_gdn.py](tests/test_gdn.py) | Ascend 950 与 fla-npu；特定 BF16 形状下普通/打包短卷积的输出、输入梯度和权重梯度对照 |

`tests/conftest.py` 会导入 torch_npu、fla_npu 并执行 setup；不要将全目录测试视为普通 CPU 环境可运行的检查。下列单文件命令使用 `--noconftest`，仍需对应测试自身的依赖。新增上游移植用例位于 `tests/unit_tests/`，分布式配方位于 `tests/integration_tests/`，按具体用例选择设备、依赖和启动方式。

训练从固定 TorchTitan 源码目录运行，以读取测试资产。测试自行创建单进程进程组，无需再套分布式启动器。只检查 Qwen3 debug 的示例：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
    python -m pytest --noconftest "$ASCEND_REPO/tests/test_training.py" \
    -k qwen3_debugmodel_npu -q -rs
```

Qwen3.5 训练和卷积对照要求 Ascend 950、fla-npu 及算子路径；0.8B 还需对应 tokenizer 和 C4 数据。完成模型文档中的准备后，按需运行：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
    python -m pytest --noconftest "$ASCEND_REPO/tests/test_training.py" \
    -k qwen35 -q -rs
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
    python -m pytest --noconftest "$ASCEND_REPO/tests/test_gdn.py" -q -rs
```

设备编号仅作示例，实际设备须满足实现要求。`-k` 未选中的用例不计为已执行；结果状态按[结果与评审规则](docs/maintenance.md#结果与评审)记录。上述测试不证明完整 GDN、长期收敛、多卡或恢复能力。

### 说明变更价值

说明变更解决的问题、比较对象、实际结果和取舍。正确性证据按影响提供；性能声明说明吞吐、显存和相关代价，降低显存但牺牲吞吐等情况需解释适用场景。需要时附性能分析或内存记录，不将工具输出本身视为收益证明。

具体数值与性能方法见 [AGENTS.md](AGENTS.md#构建与验证)，检查范围和通过标准见[验收规则](docs/maintenance.md#5-验收规则)。分布式验证使用声明的真实组合，长期收敛证据按声明范围提供。缺少资源时列出待验证项并请求协助。

## 4. 提交 PR

提交修改并推送个人开发分支，在 GitCode 发起 PR：来源选择个人 fork 或本仓中的开发分支，目标选择本仓及前述目标分支。

使用 [PR 模板](.gitcode/PULL_REQUEST_TEMPLATE.md)，未自动填入时可复制其正文。说明问题、改后行为和验证结果；有相关 issue 时附链接，按实际内容勾选合入来源，简单修改可简要填写；资料变更不涉及时注明“不涉及”，兼容性无影响时注明“无”。CheckList 按实际完成情况勾选，不适用项注明原因。验证尚未完成时，列明待完成项或所需协助即可提交，报告选用与合入门槛见[合入条件](docs/maintenance.md#合入条件)。

同步受影响的接口和使用说明，检查同类调用与文档引用。按内容记录，已有记录直接链接：

| 内容 | 保存位置与说明 |
|---|---|
| 模型用法与支持范围 | 说明依赖、数据、配置与支持范围，保存为 `docs/models/<model>.md` |
| 特性用法与支持范围 | 说明用法、限制与支持范围，保存为 `docs/features/<feature>.md` |
| 验证证据 | 按[验收规则](docs/maintenance.md#5-验收规则)记录；可记录在 issue、PR 或可追溯报告中，不强制入仓 |
| 重要设计与策略提案 | 在社区创建 RFC issue，记录方案、讨论与决议，关联实现或文档 PR |

新增补丁按[补丁准入与退出](docs/maintenance.md#补丁准入与退出)准备来源、适用范围、验证与维护记录，并取得适用的 SIG 决策。当前[源码补丁](fixes/README.md)与[运行时补丁](torchtitan_ascend/patches/README.md)分别记录实现；目录迁移不代表获得交付批准。

RFC 决议形成后，长期维护规则与技术设计分别更新维护策略和架构或组件文档；任务、阶段计划及实验过程保留在 issue 或实验记录中。若复制模板正文到平台，将相对链接替换为所用分支或提交的完整 URL。

## 5. 参与评审并完成贡献

在 PR 中请求评审，无需事先认识维护者或加入群聊；评审角色与公开名单见[公开评审](docs/maintenance.md#公开评审)，维护者在公开讨论中协助确定相关评审和验证安排。尚无回复时，可在原 PR 留言说明待处理事项。

逐项回应评审意见并说明后续修改；需要澄清时在对应讨论中提问。寻求验证协助时附可执行的命令、环境要求和预期结果，协助者将实际结果回写 PR 或关联报告。

修改后按[合入条件](docs/maintenance.md#合入条件)补充受影响的检查，按[公开评审](docs/maintenance.md#公开评审)取得所需批准及适用的 SIG 决策，并将结论回写关联讨论。阻塞意见解决且必需检查通过后，由有权限的维护者合入。

合入后，在有关联 issue 时说明已完成的范围；仍有工作时列出后续事项。支持范围有变化时，相关说明随 PR 更新，并以评审结论为准。维护者另按[发布流程](docs/maintenance.md#6-发布与回退)处理发布，贡献者无需等待版本发布即可完成本次贡献，发布节奏与版本规则统一沿用维护策略。

## 许可证与第三方来源

本仓采用 [Apache-2.0](LICENSE) 许可证，第三方组件归属见 [Third_Party_Open_Source_Software_Notice](Third_Party_Open_Source_Software_Notice)。引入或修改第三方代码时说明来源，保留适用的版权与许可证声明。

## 参考

本文参考 [TorchTitan 贡献指南](https://github.com/pytorch/torchtitan/blob/main/CONTRIBUTING.md)，结合本仓的 Ascend 适配职责与 GitCode 协作流程组织。实际开发以所选依赖版本的接口与文档为准。
