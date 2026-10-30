# GLM-5

## 安装

先完成[基础安装](../../INSTALL.md)。GLM-5.2/5.3 的结构、配置、并行化和 checkpoint
映射位于 `torchtitan_ascend.models.glm5`，复用固定 TorchTitan
`edde21fb648d446fecb5b78ef42d8c432274b935` 的 MLA、router、MTP 和训练组件。
GLM 上游化工作见 [TorchTitan PR #5578](https://github.com/pytorch/torchtitan/pull/5578)。

NPU DSA 实现位于 `torchtitan_ascend/overrides/glm5_dsa.py`，由 GLM NPU recipe
显式启用。模型构建前替换 GLM attention Config，并在工厂内组装所需的 RoPE 和 indexer。
不修改 PyTorch、TorchNPU 或其他模型的全局调用路径。

## 训练配置

配置模块为 `torchtitan_ascend.recipes.glm5`。

| CONFIG | 模型 |
|---|---|
| `glm5_2_debugmodel_npu` | GLM-5.2 debug |
| `glm5_2_debugmodel_mtp_npu` | GLM-5.2 debug + MTP |
| `glm5_2_scaled_debugmodel_npu` | GLM-5.2 scaled debug |
| `glm5_3_debugmodel_npu` | GLM-5.3 debug |
| `glm5_3_debugmodel_mtp_npu` | GLM-5.3 debug + MTP |
| `glm5_3_scaled_debugmodel_npu` | GLM-5.3 scaled debug |

`glm5_debugmodel_npu` 是 `glm5_2_debugmodel_npu` 的兼容名称。默认使用 BF16、
上游 tokenizer 和 C4 测试数据，上下文长度 128，每 DP rank 的 microbatch 为
256 tokens，共训练 10 步。配置随机初始化模型且不保存检查点，完整训练需配置实际数据、
学习率计划、训练步数和 checkpoint。

Debug 配置保留小 hidden size 和层数，attention 使用融合算子要求的 512 维 latent KV、
64 维 RoPE、128 维 indexer 和 32 个 attention heads。

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
from torchtitan_ascend.recipes.glm5 import glm5_2_debugmodel_npu


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
    MODULE=torchtitan_ascend.recipes.glm5 CONFIG=glm5_2_debugmodel_npu \
    ./run_train.sh
```

运行上面的 warmup 配置时，设置 `MODULE=glm5_stages CONFIG=glm5_warmup`。

并行拓扑通过 Python recipe 配置。例如：

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_ascend.recipes.glm5 import glm5_2_debugmodel_npu


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

Recipe 默认沿用上游 SelectiveAC，并启用 `loss` 和 `swiglu` 局部编译区域。
`npu_changes` 保留调用方设置的重计算策略和局部编译区域。需要 FullAC 时，从
`torchtitan.distributed.activation_checkpoint` 导入 `FullAC`，在构建 Trainer 前设置
`config.activation_checkpoint = FullAC.Config()`。
融合算子的图编译和各并行组合需分别验证，短程训练通过不代表长期收敛或所有拓扑通过。

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

- 硬件为 Ascend 910B / Atlas A2 或 Ascend 910_93 / Atlas A3。
- latent KV 维度为 512，RoPE 维度为 64，indexer head dim 为 128。
- Indexer KL 使用 32、64 或 128 个全局 attention heads，以及 8、16、32 或 64 个
  indexer heads。TP 局部 head 数不作为完整教师分布。
- Dense KL 的主 Q/K 内容维度为 128。GLM released 配置的 192 维 NoPE warmup
  使用全部因果 key 的融合 sparse attention/KL，表达稠密目标。当前两条 warmup
  路径都受全 key 选择宽度限制，每个文档最多 8192 tokens。
- Sparse KL 所需的索引宽度按算子支持值补齐 `-1`，不改变模型实际选择的 top-k。
- Packed-document 元数据表达有效 token 和累计序列长度；padding query 的输出为零。

接口规格见 [OpPlugin 文档](https://gitcode.com/Ascend/op-plugin/tree/master/docs/zh/custom_APIs/torch_npu)。
固定栈的实际前后向验证优先于其他版本接口文档中的能力说明。

ComplexRoPE 使用已有的 `npu_rotary_mul` override。参数初始化沿用上游
`torch.nn.init.trunc_normal_` 等配置。

## 验证

CPU 检查覆盖配置、checkpoint roundtrip、文档边界、padding 和 CP 重排元数据：

```bash
python -m pytest -q tests/test_glm5_model_config.py tests/test_glm5_state_dict_adapter.py \
    tests/test_glm5_dsa.py -m "not npu"
```

在已安装固定依赖且可访问 Ascend NPU 的环境运行融合 attention 和 dense/sparse KL
的 loss、梯度对照：

```bash
python -m pytest -q tests/test_glm5_dsa.py -m npu
```

从 TorchTitan 源码目录运行完整模型短程训练测试：

```bash
python -m pytest -q "$ASCEND_REPO/tests/test_glm5.py"
```

短程训练测试使用单卡 BF16、上下文长度 32，每个用例运行两步，覆盖无重计算、
SelectiveAC、FullAC，以及 `loss`、`swiglu` 局部编译区域的单独和联合启用。
同时检查 sparse、warmup、frozen 阶段和 GLM-5.3 MTP 的参数更新行为。
这些测试不作为 released 模型精度、长期收敛或所有并行与编译组合的验证结论。
