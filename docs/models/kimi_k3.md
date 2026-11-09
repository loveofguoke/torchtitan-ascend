# Kimi K3

## 1. 模型概述

Kimi K3 使用 Kimi Delta Attention（KDA）与 Multi-head Latent Attention（MLA）混合解码器、LatentMoE、Attention Residuals 和 MoonViT-V2 视觉编码器，支持图文输入的自回归训练。本仓复用 TorchTitan 的模型和训练配方，为部分算子提供 Ascend 适配。

### 1.1 模型规格与训练配置

| 模型规格 / flavor | 参数量 / 激活参数量 | 类型 | 上下文长度 / 输入尺寸 | MODULE | CONFIG | 默认数据 |
|---|---|---|---|---|---|---|
| `debugmodel` | 约 26.88M / 18.34M，含视觉编码器 | MoE，8 专家、top-2 | 默认 2048 tokens，图像最多 256 个 token | `torchtitan_ascend.recipes.kimi_k3` | `kimi_k3_debugmodel_npu` | 内置 CC-12m 测试数据 |
| `7b` 基线 | 约 7.466B / 3.471B，含视觉编码器 | MoE，32 专家、top-2、2 个共享专家 | 同上 | `torchtitan_ascend.recipes.kimi_k3` | `kimi_k3_7b_npu` | 同上 |
| `7b` MoonEP | 与 7B 基线相同 | 相同模型，使用 MoonEP 专家 | 同上 | `torchtitan_ascend.recipes.kimi_k3` | `kimi_k3_7b_moonep` | 同上 |

7B 为本仓自定义的测试规格，具体参数见 [CONFIG 设置](../../torchtitan_ascend/recipes/kimi_k3.py)。

### 1.2 特性支持

| 特性 | 说明 |
|---|---|
| BF16 训练 | 支持基于 BF16 低精度数据类型进行模型训练 |
| FSDP2 / HSDP | 支持完全分片数据并行和混合分片数据并行训练 |
| DDP | 支持多卡分布式数据并行训练 |
| Pipeline Parallel（PP） | 支持多卡流水线并行训练 |
| Context Parallel（CP） | 支持多卡上下文并行训练 <br> MLA 支持 Ulysses 和 AllGather-KV 实现，KDA 支持 Ulysses 实现 |
| Expert Parallel（EP，MoE） | 支持多卡混合专家并行训练 |
| `torch.compile` | 支持图编译与局部编译，局部编译配置排除 `gated_rmsnorm` |

## 2. 环境安装

参考[基础安装](../../INSTALL.md)完成环境搭建。本模型需额外安装 [flash-linear-attention-npu](https://github.com/flashserve/flash-linear-attention-npu)、GNU Make 和 gawk。

将 `torchtitan-ascend` 和 `torchtitan` 的实际项目路径设为环境变量。

```bash
export ASCEND_REPO=/absolute/path/to/torchtitan-ascend
export TORCHTITAN_SRC=/absolute/path/to/torchtitan
```

### 安装 fla-npu

```bash
export FLA_NPU_SRC=/absolute/path/to/flash-linear-attention-npu

git clone https://github.com/flashserve/flash-linear-attention-npu.git "$FLA_NPU_SRC"
cd "$FLA_NPU_SRC"
git checkout 7e39f9668eed285fc91cde0502f46cd48686a681
uv pip install -r requirements.txt pip setuptools wheel packaging
```

- 构建算子

  ```bash
  cd "$FLA_NPU_SRC"
  # KDA 支持 Ascend 950 和 Ascend 920。按设备对应的 CANN SoC 设置下列变量
  export FLA_NPU_SOC=ascend950
  FLA_NPU_OPS=causal_conv1d,causal_conv1d_bwd,chunk_gated_delta_rule_fwd,chunk_gated_delta_rule_bwd,chunk_kda_fwd,chunk_kda_bwd \
      python scripts/build_wheel.py --no-progress
  uv pip install dist/*.whl
  ```

### 安装 MoonEP（可选）

使用 MoonEP 配方时，参考 [MoonEP 昇腾后端安装指南](https://gitcode.com/Ascend/MoonEP/blob/dev_npu/npu_patch/README.md)。

## 3. 数据集准备

使用 TorchTitan 源码中的 `tests/assets/cc12m_test`，无需另外下载数据。

## 4. Tokenizer下载

使用 TorchTitan 自带的 `tests/assets/tokenizer`，无需下载生产模型 tokenizer。

## 5. 启动训练

执行以下命令完成 fla-npu 和训练环境的配置。

```bash
FLA_NPU_OPP=$(python - <<'PY'
from importlib.util import find_spec
from pathlib import Path

print(Path(find_spec("fla_npu").origin).parent / "opp/vendors/fla_npu_transformer")
PY
)
export ASCEND_CUSTOM_OPP_PATH="$FLA_NPU_OPP${ASCEND_CUSTOM_OPP_PATH:+:$ASCEND_CUSTOM_OPP_PATH}"
export TORCHINDUCTOR_NPU_BACKEND=triton_experimental
export TORCHINDUCTOR_COMPILE_THREADS=1
export TORCHNPU_PRECOMPILE_THREADS=1
```

### 5.1 单卡训练

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.kimi_k3 CONFIG=kimi_k3_debugmodel_npu \
    ./run_train.sh
```

可追加 `--output-dir ./outputs/kimi-k3-debug` 指定输出目录，默认不保存 checkpoint。

- loss 曲线

  ![Kimi K3 debugmodel 未启用融合算子的单卡 2000 步 loss 曲线](https://raw.gitcode.com/user-images/assets/9763158/9f6a08bd-0541-4e39-8b0f-81b02953091d/image.png)

### 5.2 多卡训练

以**单机 8 卡纯 FSDP 配置**为示例。

构建带有以下并行配置的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。

```python
from torchtitan_ascend.recipes.kimi_k3 import kimi_k3_debugmodel_npu


def train_config():
    config = kimi_k3_debugmodel_npu()
    config.parallelism.data_parallel_replicate_degree = 1
    config.parallelism.data_parallel_shard_degree = 8
    config.parallelism.tensor_parallel_degree = 1
    config.parallelism.pipeline_parallel_degree = 1
    config.parallelism.context_parallel_degree = 1
    config.parallelism.expert_parallel_degree = 1
    return config
```

运行 8 卡训练。

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 \
    ENTRYPOINT=torchtitan_ascend.train \
    MODULE=kimi_k3_fsdp2_config CONFIG=train_config \
    ./run_train.sh
```

| 配置项 | 值 |
|---|---|
| 总卡数 / 节点数 | 8 / 1 |
| dp_replicate / dp_shard（FSDP2） | 1 / 8，显式设置 |
| TP / PP / CP / EP | 1 / 1 / 1 / 1 |
| 单卡 micro-batch / gradient accumulation | 2048 tokens / 1 |
| 全局 tokens per step | 16384 token slots，相当于 8 段满长 2048-token 序列。有效非 padding token 数取决于数据 |
| 其他必要环境变量 | `ASCEND_RT_VISIBLE_DEVICES` 中的设备数应与 `NGPU` 一致 |

以上 tokens 数按默认 `num_tokens_per_train_step=-1` 推导，调整配方后需重新核对。该八卡示例暂无对应的当前 loss 图、精度或性能结论。

### 5.3 可选训练特性

| 支持特性 | 启用方式 | 备注 |
|---|---|---|
| MoonEP | 使用 `kimi_k3_7b_moonep` 配方 | 要求 Ascend 950 且 EP 并行数 > 1 |
| Context Parallel（CP） | 设置 CP 度，构建新配方以调用 `apply_kimi_cp`。 | 要求连续序列分片，关闭 CP 负载均衡。KDA 的注意力头数须能被 TP × CP 整除。 |
| NPU 融合算子 | 构建新配方以调用 `npu_fusion`。 | 融合 attention 匹配 Flex 配置，融合 SiTU-GLU 需要 `cann_ops_nn`。 |
| 融合 permute/unpermute | 设置 EP 度，通过 `--override` 选择 dispatcher 或者 MC2。 | EP 度大于 1，all-to-all dispatcher 与 MC2 二选一，MC2 要求 `output_postprocess=None`。 |

#### 5.3.1 MoonEP

已提供带有 MoonEP 功能的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。配方默认使用 EP2 / DP shard2 和 MTE 同步。

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1 NGPU=2 \
    ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.kimi_k3 \
    CONFIG=kimi_k3_7b_moonep \
    ./run_train.sh
```

#### 5.3.2 上下文并行（Context Parallel）

构建带有 `apply_kimi_cp` 的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。函数默认采用 Ulysses MLA 和 Ulysses KDA。

```python
from torchtitan_ascend.recipes.kimi_k3 import apply_kimi_cp, kimi_k3_debugmodel_npu


def debug_cp2():
    config = kimi_k3_debugmodel_npu()
    config.parallelism.data_parallel_shard_degree = 1
    config.parallelism.context_parallel_degree = 2
    config.parallelism.context_parallel_load_balancer = None
    return apply_kimi_cp(config)
```

#### 5.3.3 NPU 融合算子

修改运行配置为 `CONFIG=debug_fusion`。

```python
from torchtitan_ascend.recipes.kimi_k3 import kimi_k3_debugmodel_npu, npu_fusion


def debug_fusion(attn_backend="varlen"):
    config = kimi_k3_debugmodel_npu(attn_backend=attn_backend)
    npu_fusion(config)
    return config
```

#### 5.3.4 融合 MoE 算子

构建带有 `EP并行` 的配方，修改命令配置 `MODULE` 和 `CONFIG` 以指向该配方。

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_ascend.recipes.kimi_k3 import kimi_k3_debugmodel_npu


def debug_ep8():
    return kimi_k3_debugmodel_npu(
        parallelism=ParallelismConfig(expert_parallel_degree=8)
    )
```

在 `./run_train.sh` 后追加所选实现的参数。

- 融合 permute/unpermute 的 all-to-all dispatcher 使用 `--override torchtitan_ascend.overrides.token_dispatcher.npu_token_dispatcher`。
- MC2 routed experts 使用 `--override torchtitan_ascend.overrides.mc2_routed_experts.npu_mc2_routed_experts`。
