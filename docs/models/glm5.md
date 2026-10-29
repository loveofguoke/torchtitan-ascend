# GLM-5

## 安装

先完成[基础安装](../../INSTALL.md)。本仓提供完整的 GLM-5 模型定义、DSA、MTP、
并行化和 checkpoint 映射，代码位于 `torchtitan_ascend.models.glm5`。模型通过
TorchTitan 的公共配置、训练、并行化和数据接口接入，不复制 Trainer 或通用训练组件。
模型实现与 [TorchTitan PR #5578](https://github.com/pytorch/torchtitan/pull/5578)
中的 GLM-5 实现保持一致，并适配本仓安装指南固定的 TorchTitan
`edde21fb648d446fecb5b78ef42d8c432274b935` 公共接口。上游正式发布后可直接切换到
对应公共模型包。

## 训练配置

配置模块：`torchtitan_ascend.recipes.glm5`。

| CONFIG | 模型 | 数据 |
|---|---|---|
| `glm5_2_debugmodel_npu` | GLM-5.2 debug | 上游自带 tokenizer 和 C4 测试数据 |
| `glm5_2_debugmodel_mtp_npu` | GLM-5.2 debug + MTP | 上游自带 tokenizer 和 C4 测试数据 |
| `glm5_2_scaled_debugmodel_npu` | GLM-5.2 scaled debug | 上游自带 tokenizer 和 C4 测试数据 |
| `glm5_3_debugmodel_npu` | GLM-5.3 debug | 上游自带 tokenizer 和 C4 测试数据 |
| `glm5_3_debugmodel_mtp_npu` | GLM-5.3 debug + MTP | 上游自带 tokenizer 和 C4 测试数据 |
| `glm5_3_scaled_debugmodel_npu` | GLM-5.3 scaled debug | 上游自带 tokenizer 和 C4 测试数据 |

`glm5_debugmodel_npu` 保留为 `glm5_2_debugmodel_npu` 的兼容名称。

默认配置为 BF16、eager、上下文长度 128、每 DP rank 的 microbatch 为 256 tokens，
共训练 10 步。模型随机初始化，不保存检查点。完整训练需自行配置数据、训练步数、
学习率计划和检查点。

## 启动

在 TorchTitan 源码目录执行：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 NGPU=1 ENTRYPOINT=torchtitan_ascend.train \
    MODULE=torchtitan_ascend.recipes.glm5 CONFIG=glm5_2_debugmodel_npu \
    ./run_train.sh
```

验证 GLM-5.3 时，将 `CONFIG` 改为 `glm5_3_debugmodel_npu`。

## Ascend 适配

- ComplexRoPE 使用 `torch_npu.npu_rotary_mul` 执行相邻维旋转，并保留 TP/CP DTensor
  placements。固定的 TorchTitan 基线仍在 eager 路径使用 complex tensor；上游后续加入的
  compile 实数旋转路径不能替代该 eager NPU override。
- DTensor truncated-normal 兼容实现用于已验证的 Atlas A2、PyTorch 2.14 nightly、
  TorchNPU 2.14 nightly 与 CANN 9.1 组合：上游 rejection-sampling 路径在分片参数初始化时
  可能使各 rank 以不同次序进入 lazy HCCL subgroup，表现为初始化卡死。该实现只替换 GLM
  config 中的 DTensor initializer；普通 Tensor 继续调用 PyTorch 原实现。
- DSA 使用 TorchTitan FlexAttention 接口表达动态 top-k 稀疏掩码；模型代码不固定 NPU
  attention 后端。
- 完整模型实现暂存于本仓 `torchtitan_ascend.models.glm5`，仅调用固定 TorchTitan
  基线提供的模型配置、并行化、训练和 checkpoint 公共接口，不修改 TorchTitan 源码。

上述初始化适配属于 PyTorch DTensor 初始化与旧 TorchNPU/HCCL 组合的跨层兼容缺口，
不是 GLM 模型数学修改，也不能归因于单一 TorchTitan 组件。当前证据来自 Atlas A2 固定
软件栈，因此它是该已验证基线上的必要兼容实现，不代表 Ascend 950（A5）及后续
TorchNPU/CANN 永久需要相同适配。A5 是否可删除必须在冻结设备、驱动、CANN、PyTorch
和 TorchNPU 版本后关闭对应适配，验证参数初始化 hash/统计量/首步 loss；在这些 A/B 证据完成前
不按硬件名称猜测并删除。

DTensor initializer 由 GLM recipe 显式改写自身 config，保留在
`torchtitan_ascend/overrides/glm5.py`，不注册成全局 PyTorch patch，以免影响 Qwen、Kimi 等
无关模型。

## 支持范围

模型支持 GLM-5.2/5.3、DSA、MTP、Hugging Face checkpoint adapter，以及
FSDP、TP/SP、CP、PP、EP 的声明式 sharding。训练使用 BF16 参数与前向计算、FP32
梯度归约。

| 拓扑 | 并行配置 | Eager | Inductor + Ascend-Triton | Inductor + DVM |
|---|---|---|---|---|
| `single` | 单卡 | 通过 | 通过 | 通过 |
| `ddp2` | DDP2 | 通过 | 通过 | 通过 |
| `ddp8` | DDP8 | 通过 | 通过 | 通过 |
| `fsdp8` | FSDP8 | 通过 | 通过 | 通过 |
| `hsdp2x4` | HSDP：replicate 2，shard 4 | 通过 | 通过 | 通过 |
| `hsdp4x2` | HSDP：replicate 4，shard 2 | 通过 | 通过 | 通过 |
| `tp8` | TP8 + SP + EP8 | 通过 | 通过 | 通过 |
| `cp2` | CP2 | 通过 | 通过 | 通过 |
| `cp8` | CP8 | 通过 | 通过 | 通过 |
| `pp8` | PP8 | 通过 | 通过 | 通过 |
| `ep8` | FSDP8 + EP8 | 通过 | 通过 | 通过 |
| `fsdp2-tp4` | FSDP2 + TP4 + SP + EP4 | 通过 | 通过 | 通过 |
| `fsdp2-cp4` | FSDP2 + CP4 | 通过 | 通过 | 通过 |
| `tp2-cp4` | TP2 + SP + CP4 + EP2 | 通过 | 通过 | 通过 |
| `fsdp4-tp2` | FSDP4 + TP2 + SP + EP2 | 通过 | 通过 | 通过 |
| `fsdp2-pp4` | FSDP2 + PP4 | 通过 | 通过 | 通过 |
| `fsdp2-tp2-pp2` | FSDP2 + TP2 + SP + PP2 + EP2 | 通过 | 通过 | 通过 |
| `fsdp2-tp4-ep8` | FSDP2 + TP4 + SP + EP8 | 通过 | 通过 | 通过 |

### 启动并行拓扑

TorchTitan 通过 Python recipe 选择并行拓扑。表中的拓扑名是便于阅读的验证标识，
不是 `run_train.sh` 参数。例如，在 TorchTitan 源码目录创建
`glm5_topology.py`：

```python
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan_ascend.recipes.glm5 import glm5_2_debugmodel_npu


def glm5_2_debugmodel_fsdp2_tp4_npu():
    return glm5_2_debugmodel_npu(
        parallelism=ParallelismConfig(
            data_parallel_shard_degree=2,
            tensor_parallel_degree=4,
            expert_parallel_degree=4,
        )
    )
```

然后使用 TorchTitan 官方启动入口运行 8 卡 FSDP2 + TP4 + SP + EP4：

```bash
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NGPU=8 \
    ENTRYPOINT=torchtitan_ascend.train \
    MODULE=glm5_topology CONFIG=glm5_2_debugmodel_fsdp2_tp4_npu \
    ./run_train.sh
```

其他拓扑按表格设置 `ParallelismConfig` 中的 `data_parallel_replicate_degree`、
`data_parallel_shard_degree`、`tensor_parallel_degree`、
`context_parallel_degree`、`pipeline_parallel_degree` 和
`expert_parallel_degree`。`NGPU` 等于 DP replicate、DP shard、TP、CP 和 PP 度数的
乘积；EP 在该设备网格上划分专家，不再额外乘入 `NGPU`。

## 精度对齐

GLM-5.2 debug model 使用 BF16 eager、上下文长度 128、每步 8192 tokens 和 seed 61。
CUDA reference 与 NPU candidate 使用相同 fixture，各拓扑完成 500 个严格对齐 step；loss
差值为 `NPU - GPU`。

| 拓扑 | steps | GPU final loss | NPU final loss | final diff | mean abs diff | max abs diff | mean rel error | max rel error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `single` | 500 | 2.561436 | 2.551692 | -0.009744 | 0.008077 | 0.032962 | 0.299196% | 1.180570% |
| `ddp2` | 500 | 2.560761 | 2.556582 | -0.004179 | 0.006863 | 0.029731 | 0.254237% | 1.138627% |
| `ddp8` | 500 | 2.562466 | 2.558473 | -0.003993 | 0.006372 | 0.020852 | 0.235024% | 0.705714% |
| `fsdp8` | 500 | 2.554454 | 2.550097 | -0.004357 | 0.005108 | 0.028708 | 0.187740% | 1.052928% |
| `hsdp2x4` | 500 | 2.549722 | 2.562037 | 0.012315 | 0.005511 | 0.021544 | 0.203145% | 0.808338% |
| `hsdp4x2` | 500 | 2.539855 | 2.544791 | 0.004936 | 0.004446 | 0.027079 | 0.163891% | 0.898216% |
| `tp8` | 500 | 2.545842 | 2.548984 | 0.003142 | 0.004311 | 0.019085 | 0.158397% | 0.661760% |
| `cp2` | 500 | 2.558650 | 2.556951 | -0.001699 | 0.004701 | 0.027690 | 0.173289% | 0.999725% |
| `cp4` | 500 | 2.560246 | 2.554687 | -0.005559 | 0.007164 | 0.023609 | 0.264935% | 0.864944% |
| `cp8` | 500 | 2.550963 | 2.544420 | -0.006543 | 0.005006 | 0.028952 | 0.185188% | 1.089829% |
| `pp8` | 500 | 2.519800 | 2.500758 | -0.019042 | 0.010461 | 0.033368 | 0.391709% | 1.288559% |
| `ep8` | 500 | 2.558728 | 2.560805 | 0.002076 | 0.004944 | 0.024495 | 0.181653% | 0.923190% |
| `fsdp2-tp4` | 500 | 2.546308 | 2.563389 | 0.017081 | 0.007982 | 0.044550 | 0.295235% | 1.475921% |
| `fsdp2-cp4` | 500 | 2.546153 | 2.550308 | 0.004155 | 0.005145 | 0.030890 | 0.189206% | 1.014473% |
| `tp2-cp4` | 500 | 2.546159 | 2.551347 | 0.005188 | 0.005919 | 0.026460 | 0.219574% | 1.034385% |
| `fsdp4-tp2` | 500 | 2.558816 | 2.557643 | -0.001173 | 0.005555 | 0.024331 | 0.204831% | 0.879901% |
| `fsdp2-pp4` | 500 | 2.505396 | 2.512713 | 0.007317 | 0.006704 | 0.031359 | 0.247639% | 1.076978% |
| `fsdp2-tp2-pp2` | 500 | 2.507141 | 2.494877 | -0.012264 | 0.006131 | 0.045049 | 0.228384% | 1.502309% |
| `fsdp2-tp4-ep8` | 500 | 2.536863 | 2.555841 | 0.018978 | 0.007502 | 0.031108 | 0.279143% | 1.214954% |

19 个拓扑的 loss mean relative error 为 `0.158397%–0.391709%`，max relative error
为 `0.661760%–1.502309%`。
