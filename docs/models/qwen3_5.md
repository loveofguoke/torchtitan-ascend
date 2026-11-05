# Qwen3.5

## 安装

完成[基础安装](../../INSTALL.md)。本模型使用 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu) 的 AscendC 算子，要求 Ascend 950、GNU Make 和 gawk。以下命令从包含 `torchtitan-ascend`、`torchtitan` 的工作目录执行。

### 准备源码

```bash
git clone https://github.com/flashserve/flash-linear-attention-npu.git
git -C flash-linear-attention-npu checkout 54ae09b1243cd851d986c2624a57bbb5ee63dc27
uv pip install -r flash-linear-attention-npu/requirements.txt pip setuptools wheel packaging
```

### 构建算子

```bash
cd flash-linear-attention-npu
FLA_NPU_SOC=ascend950 \
FLA_NPU_OPS=causal_conv1d,causal_conv1d_bwd,chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd \
    python scripts/build_wheel.py --no-progress
uv pip install dist/*.whl
cd ..
```

## 训练配置

配置模块：`torchtitan_ascend.recipes.qwen3_5`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `qwen35_debugmodel_npu` | debug | 上游自带 tokenizer 和 C4 测试数据 |
| `qwen35_0_8b_npu` | 0.8B | 本地 tokenizer 和文本数据 |

默认使用单卡 BF16 混合精度、eager 执行，序列长度 256，每步 512 tokens，共 10 步。模型随机初始化，不保存检查点。完整训练需自行配置数据、训练步数、学习率计划和检查点。

## 数据准备

Debug 配方无需下载数据。Qwen3.5 0.8B 需准备 tokenizer 和文本数据：

```bash
cd torchtitan
python scripts/download_hf_assets.py --repo_id Qwen/Qwen3.5-0.8B \
    --assets tokenizer --local_dir ./assets/hf
hf download allenai/c4 en/c4-train.00000-of-01024.json.gz \
    --repo-type dataset --local-dir ./assets/c4
cd ..
```

## 启动

按[运行验证](../../INSTALL.md#运行验证)加载训练环境，并设置算子包路径：

```bash
FLA_NPU_OPP=$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

print(Path(find_spec("fla_npu").origin).parent / "opp/vendors/fla_npu_transformer")
PY
)
export ASCEND_CUSTOM_OPP_PATH="$FLA_NPU_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
```

从包含 `torchtitan` 的工作目录执行：

```bash
cd torchtitan
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.qwen3_5 CONFIG=qwen35_debugmodel_npu \
    ./run_train.sh
```

使用 0.8B 模型时，完成数据准备并设置 `CONFIG=qwen35_0_8b_npu`。

## 自定义数据

0.8B 配方接受包含字符串字段 `text` 的 JSON 或 JSONL 文件。将以下配方保存为 `torchtitan/my_config.py`：

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

配方支持文本训练和序列打包（packing），不提供图像、视频输入，不支持上下文并行（CP）。默认关闭局部编译和激活重计算。
