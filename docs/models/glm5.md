# GLM-5

## 模型概述

GLM-5.2/5.3 的结构、配置、并行化和 checkpoint
映射位于 `torchtitan_ascend.models.glm5`，复用固定 TorchTitan
0.4.0（`45e33cd413b64cb37a7840cfd3c9765d3c639ae1`）的 MLA、router、MTP 和训练组件。
GLM 上游化工作见 [TorchTitan PR #5578](https://github.com/pytorch/torchtitan/pull/5578)。

NPU recipe 使用融合 DSA，实现位于 `torchtitan_ascend/overrides/glm5_dsa.py`。

## 训练配置

配置模块为 `torchtitan_ascend_recipes.models.glm5`。

| CONFIG | 模型 | 默认上下文长度 | 每 DP rank 的 microbatch tokens |
|---|---|---:|---:|
| `glm5_2_debugmodel_npu` | GLM-5.2 debug | 128 | 256 |
| `glm5_2_debugmodel_mtp_npu` | GLM-5.2 debug + MTP | 128 | 256 |
| `glm5_2_scaled_debugmodel_npu` | GLM-5.2 scaled debug | 4096 | 8192 |
| `glm5_3_debugmodel_npu` | GLM-5.3 debug | 128 | 256 |
| `glm5_3_debugmodel_mtp_npu` | GLM-5.3 debug + MTP | 128 | 256 |
| `glm5_3_scaled_debugmodel_npu` | GLM-5.3 scaled debug | 4096 | 8192 |

`glm5_debugmodel_npu` 是 `glm5_2_debugmodel_npu` 的兼容名称。默认使用 BF16、
上游 tokenizer 和 C4 测试数据，共训练 10 步。配置随机初始化模型且不保存检查点，完整训练需配置实际数据、
学习率计划、训练步数和 checkpoint。

Debug 配置保留小 hidden size 和层数，attention 使用融合算子要求的 512 维 latent KV、
64 维 RoPE、128 维 indexer 和 32 个 attention heads。

## 支持范围

| 特性 | 配置 |
|---|---|
| 混合精度 | BF16 参数与前向计算，FP32 梯度归约 |
| 数据并行 | DDP、FSDP2、HSDP |
| 模型并行 | TP/SP、CP、PP、EP 及组合拓扑，见下方精度表 |
| 执行模式 | Eager、Inductor + Ascend-Triton、Inductor + DVM |
| 重计算 | 默认重计算、全重计算 |
| Indexer 训练 | warmup、sparse、frozen |
| Checkpoint | Hugging Face 权重映射、warmup 到 sparse 的模型权重交接 |

## 安装、数据与 Tokenizer

先完成[基础安装](../../INSTALL.md)及[训练环境加载](../../INSTALL.md#运行验证)。
Debug 配方使用 TorchTitan 源码自带的 C4 测试数据和 tokenizer，无需额外下载。

## 两阶段训练

各 recipe 接受 `indexer_stage` 和 `indexer_loss_coeff`：

| indexer_stage | 主 attention | 参数更新 |
|---|---|---|
| `warmup` | 全部因果 key | 冻结主模型，仅训练 indexer 的 KL 目标 |
| `sparse`（默认） | indexer 选择的 top-k key | 训练主模型和 indexer，后者使用独立 KL 目标 |
| `frozen` | indexer 选择的 top-k key | 固定预训练 indexer，仅训练主模型 |

Indexer 输入及教师分布从主 attention 的梯度图分离。KL 梯度通过 TorchTitan
`AuxLoss` 注入，并使用 trainer 的全局有效 token 数归一化。主 attention 的反向由
TorchNPU 的 SparseFlashAttention autograd 提供。

在 TorchTitan 源码目录创建 `glm5_stages.py`：

```python
from torchtitan_ascend_recipes.models.glm5 import glm5_2_debugmodel_npu


def glm5_warmup():
    return glm5_2_debugmodel_npu(indexer_stage="warmup", indexer_loss_coeff=1.0)


def glm5_sparse():
    return glm5_2_debugmodel_npu(indexer_stage="sparse", indexer_loss_coeff=1.0)
```

配置实际 checkpoint 路径后，将 warmup 得到的 indexer 权重用于 sparse 阶段；两个阶段
使用相同的参数命名和 Hugging Face checkpoint 映射。阶段切换时仅加载模型权重，
重新创建 optimizer 和学习率计划，不恢复 warmup 的 optimizer 状态。

## 启动与并行配置

在 TorchTitan 源码目录执行：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend_recipes.models.glm5 CONFIG=glm5_2_debugmodel_npu \
    ./run_train.sh
```

运行上面的 warmup 配置时，设置 `MODULE=glm5_stages CONFIG=glm5_warmup`。

并行拓扑通过 Python recipe 配置。例如：

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_ascend_recipes.models.glm5 import glm5_2_debugmodel_npu


def glm5_tp2():
    return glm5_2_debugmodel_npu(
        parallelism=ParallelismConfig(
            tensor_parallel_degree=2,
            expert_parallel_degree=2,
        )
    )
```

启动时设置 `NGPU=2` 并选择对应模块和配置。MoE 配置中的 EP 度数须满足上游 TP/EP
约束。MLA 的升维权重按 head 切分，输出使用上游 row-parallel projection。Indexer
在 TP 维复制，KL 教师分布使用全部 attention heads。CP 收集共享 KV，按文档和本地
query 区间生成 TND 元数据；CP 重排和 padding 不得改变文档隔离或因果关系。

Recipe 使用上游 `DefaultActivationRematConfig` 和模型的局部编译默认值（当前为 `local_compile_regions=[]`）。
`npu_changes` 保留调用方设置的重计算策略和局部编译区域。需要局部编译时，在构建
Trainer 前显式设置 `config.model.local_compile_regions = ["loss", "fused_binary_activation"]`。
需要全重计算时，从 `torchtitan.distributed.activation_remat` 导入 `FullActivationRematConfig`，
在构建 Trainer 前设置 `config.activation_remat = FullActivationRematConfig()`；设为 `None` 关闭重计算。

## Ascend 融合算子

| 环节 | TorchNPU 接口 |
|---|---|
| Indexer 选择 | `npu_lightning_indexer` |
| 稀疏 attention | `npu_sparse_flash_attention` |
| 128 维 NoPE warmup 的 indexer 统计量和 KL | `npu_dense_lightning_indexer_softmax_lse`、`npu_dense_lightning_indexer_grad_kl_loss` |
| Sparse 阶段及 192 维 NoPE warmup 的 indexer KL | `npu_sparse_lightning_indexer_grad_kl_loss` |

MLA 使用吸收形式：将 K 的升维权重作用到 query，将 V 的升维权重作用到 attention
输出，保持原有 attention scale 和 checkpoint 权重结构。Indexer 的缩放合入 mixing
weights。模型执行不构造 Q×K 的 dense legality mask 或 FlexAttention BlockMask，也不
保留本地 DSA eager 计算后端。

当前固定 CANN 栈的约束：

- 算子构造允许 Ascend 910B、Ascend 910_93 和 Ascend 950；下方精度数据来自 Ascend 950。
- latent KV 维度为 512，RoPE 维度为 64，indexer head dim 为 128。
- Indexer KL 使用 32、64 或 128 个全局 attention heads，以及 8、16、32 或 64 个
  indexer heads。TP 局部 head 数不作为完整教师分布。
- 128 维 NoPE warmup 使用 Dense KL，全因果 key 的 attention 选择宽度按 1024
  对齐，不受 Sparse KL 的 8192 上限约束。GLM released 配置的 192 维 NoPE
  warmup 使用全部因果 key 的融合 sparse attention/KL，受 Sparse KL 选择宽度
  限制，每个文档最多 8192 tokens。
- Sparse 阶段的 `index_topk` 必须是 1024 到 8192 之间的 1024 整数倍，直接使用
  算子支持的选择宽度。不能将较小的 top-k 用 `-1` 补齐后交给 Sparse KL；这种
  补齐不保持 loss 和梯度等价。因果边界以外的无效槽位仍使用 `-1`。
- NPU scaled debug recipe 使用 `index_topk=1024`；普通 debug 和 released 配置使用 2048。
- Packed-document 元数据表达有效 token 和累计序列长度；padding query 的输出为零。

接口规格见 [OpPlugin 文档](https://gitcode.com/Ascend/op-plugin/tree/master/docs/zh/custom_APIs/torch_npu)。

主 MLA 和 indexer 的 RoPE 直接沿用上游 `ComplexRoPE`。A5 的 LightningIndexer
要求 Q、K 和 mixing weights 使用相同 dtype，因此仅在离散 top-k 选择调用前转换
weights；参数投影和 KL 目标继续使用 FP32 weights。A2/A3 保留 FP32 选择输入。
选择权重的量化可能改变接近的 top-k 边界，不保证与 FP32 选择逐项一致。

## 精度对齐

GLM-5.2 debug，BF16 eager，上下文长度 128，每步 8192 tokens，seed 61。
CUDA reference 与 Ascend 950 candidate 使用相同 fixture，各拓扑训练 500 步。
下表为 loss 对照，`diff = NPU loss - GPU loss`；曲线见 [PR #190](https://gitcode.com/Ascend/torchtitan-ascend/pull/190)。

| 拓扑 | steps | GPU final loss | NPU final loss | final diff | mean abs diff | max abs diff | mean rel error | max rel error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cp2 | 500 | 2.672269 | 2.684941 | 0.012672 | 0.004574 | 0.028400 | 0.167742% | 1.018862% |
| cp4 | 500 | 2.680279 | 2.683335 | 0.003056 | 0.004106 | 0.042507 | 0.149218% | 1.381605% |
| cp8 | 500 | 2.685600 | 2.670897 | -0.014703 | 0.007774 | 0.025120 | 0.285745% | 0.949958% |
| ddp2 | 500 | 2.681354 | 2.674269 | -0.007084 | 0.007475 | 0.028124 | 0.275058% | 1.056959% |
| ddp8 | 500 | 2.689227 | 2.681564 | -0.007663 | 0.004185 | 0.018610 | 0.153159% | 0.713506% |
| ep8 | 500 | 2.664427 | 2.686571 | 0.022143 | 0.006043 | 0.024619 | 0.222537% | 0.863515% |
| fsdp2-cp4 | 500 | 2.669189 | 2.690247 | 0.021058 | 0.006200 | 0.021058 | 0.228631% | 0.788914% |
| fsdp2-pp4 | 500 | 2.644457 | 2.664284 | 0.019827 | 0.012681 | 0.042209 | 0.472692% | 1.653069% |
| fsdp2-tp2-pp2 | 500 | 2.652910 | 2.628390 | -0.024519 | 0.006831 | 0.024519 | 0.253843% | 0.924247% |
| fsdp2-tp4 | 500 | 2.680554 | 2.681143 | 0.000588 | 0.003868 | 0.016674 | 0.140648% | 0.630088% |
| fsdp2-tp4-ep8 | 500 | 2.696242 | 2.687403 | -0.008838 | 0.004365 | 0.019500 | 0.158281% | 0.639538% |
| fsdp4-tp2 | 500 | 2.682959 | 2.681753 | -0.001207 | 0.003475 | 0.016113 | 0.126091% | 0.571566% |
| fsdp8 | 500 | 2.684637 | 2.679167 | -0.005470 | 0.003644 | 0.025432 | 0.132146% | 0.901988% |
| hsdp2x4 | 500 | 2.688979 | 2.680238 | -0.008742 | 0.004095 | 0.027303 | 0.149189% | 0.976326% |
| hsdp4x2 | 500 | 2.684929 | 2.677582 | -0.007348 | 0.004834 | 0.024941 | 0.176578% | 0.883491% |
| pp8 | 500 | 2.647608 | 2.631306 | -0.016302 | 0.009544 | 0.032145 | 0.354263% | 1.164133% |
| single | 500 | 2.678719 | 2.681564 | 0.002845 | 0.005308 | 0.027203 | 0.194638% | 0.985560% |
| tp2-cp4 | 500 | 2.684263 | 2.668397 | -0.015867 | 0.008689 | 0.033775 | 0.319523% | 1.266334% |
| tp8 | 500 | 2.681278 | 2.683371 | 0.002093 | 0.003725 | 0.017052 | 0.135749% | 0.663948% |

## 测试入口

CPU 检查覆盖配置、checkpoint roundtrip、文档边界、padding 和 CP 重排元数据：

```bash
python -m pytest -q tests/test_glm5_model_config.py tests/test_glm5_state_dict_adapter.py \
    tests/test_glm5_dsa.py -m "not npu"
```

在已安装固定依赖且可访问 Ascend NPU 的环境运行融合 attention 和 dense/sparse KL
的 loss、梯度对照，以及 warmup 的全 key 选择宽度边界检查：

```bash
python -m pytest -q tests/test_glm5_dsa.py -m npu
```

从 TorchTitan 源码目录运行完整模型短程训练测试：

```bash
python -m pytest -q "$ASCEND_REPO/tests/test_glm5.py"
```

使用两张 NPU 单独检查原生 DTensor 参数初始化和 TP vocabulary-parallel loss：

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --standalone --nproc-per-node=2 \
    -m pytest -q "$ASCEND_REPO/tests/integration_tests/test_glm5_native_ops.py"
```

该检查覆盖 FP32/BF16、复制与分片 DTensor 的初始化分布及 seed 复现性，
以及均匀/不均匀词表分片、跨 shard label、`IGNORE_INDEX`、两种 reduction 的
loss 和 logits 梯度。

短程训练测试使用单卡 BF16、上下文长度 32，每个用例运行两步，覆盖无重计算、
默认重计算、全重计算，以及 `loss`、`fused_binary_activation` 局部编译区域的单独和联合启用。
同时检查 sparse、warmup、frozen 阶段和 GLM-5.3 MTP 的参数更新行为。
