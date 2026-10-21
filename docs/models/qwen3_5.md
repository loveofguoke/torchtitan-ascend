# Qwen3.5

## 安装

先完成[基础安装](../../INSTALL.md)，再在同一 Python 环境和 shell 中安装 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)。构建需要 GNU Make 和 gawk。

配方通过 override 接入 fla-npu AscendC 前后向算子，仅支持 Ascend 950。

```bash
export FLA_NPU_SRC=/absolute/path/to/flash-linear-attention-npu

git clone https://github.com/flashserve/flash-linear-attention-npu.git "$FLA_NPU_SRC"
cd "$FLA_NPU_SRC"
git checkout 54ae09b1243cd851d986c2624a57bbb5ee63dc27
uv pip install -r requirements.txt pip setuptools wheel packaging psutil
FLA_NPU_SOC=ascend950 \
FLA_NPU_OPS=causal_conv1d,causal_conv1d_bwd,chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd \
    python scripts/build_wheel.py --no-progress
uv pip install dist/*.whl
```

## 训练配置

配置模块：`torchtitan_ascend.recipes.qwen3_5`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `qwen35_debugmodel_npu` | debug | 上游自带 tokenizer 和 C4 测试数据 |
| `qwen35_0_8b_npu` | 0.8B | 本地 tokenizer 和文本数据 |

默认使用单卡 BF16、eager 执行，序列长度 256，每步 512 tokens，共 10 步。模型随机初始化，不保存检查点。完整训练需自行配置数据、训练步数、学习率计划和检查点。

## 数据准备

Debug 配方无需下载数据。

Qwen3.5 0.8B 需下载 tokenizer 和文本数据：

```bash
cd "$TORCHTITAN_SRC"
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.5-0.8B \
    --assets tokenizer --local_dir ./assets/hf
hf download allenai/c4 en/c4-train.00000-of-01024.json.gz \
    --repo-type dataset --local-dir ./assets/c4
```

## 启动

在当前 Python 环境中配置算子包路径；新建 shell 时先按[准备环境](../../INSTALL.md#准备环境)设置环境。

```bash
FLA_NPU_OPP=$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

print(Path(find_spec("fla_npu").origin).parent / "opp/vendors/fla_npu_transformer")
PY
)
export ASCEND_CUSTOM_OPP_PATH="$FLA_NPU_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

在 TorchTitan 源码目录执行：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3_5 CONFIG=qwen35_debugmodel_npu \
    ./run_train.sh
```

使用 0.8B 模型时，完成上述数据准备并设置 `CONFIG=qwen35_0_8b_npu`。

## 自定义数据

0.8B 配方接受包含字符串字段 `text` 的 JSON 或 JSONL 文件。将以下配方保存为 TorchTitan 源码目录下的 `my_config.py`：

```python
from torchtitan_ascend.recipes.qwen3_5 import qwen35_0_8b_npu


def train_config():
    return qwen35_0_8b_npu(
        hf_assets_path="/data/assets/hf/Qwen3.5-0.8B",
        data_path="/data/train.jsonl",
    )
```

将路径替换为本地文件位置，以 `MODULE=my_config CONFIG=train_config` 启动。

## 支持范围

| 场景 | 状态 |
|---|---|
| 单卡 BF16、eager 文本训练与序列打包（packing） | A5 已完成 10 步训练验证 |
| 0.8B 整网 loss 对照 | A5 与 A100 使用相同随机初始权重和数据，各完成 3 次 100 步训练对照 |
| 图像、视频训练 | 不支持 |
| 上下文并行（CP） | 不支持 |
| 数据、张量、流水线并行（DDP、FSDP、TP、PP） | 未验证 |
| 编译、激活检查点 | 未验证 |
| 预训练权重加载、导出和断点续训 | 未验证 |

长期收敛、稳定性和性能未验证。
