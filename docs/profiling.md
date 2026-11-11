# 性能采集

性能采集用于查看训练时 CPU、NPU 的执行时间线，帮助定位耗时。本仓通过 `NPUProfiler` 接入 `torch_npu.profiler`，复用 TorchTitan 的训练循环和 profiler 生命周期。在现有 Python recipe（训练配置函数）中设置采集参数即可。

## 快速开始

1. 完成[安装与环境配置](../INSTALL.md)，准备对应[模型](../README.md#模型文档)的数据和运行条件。
2. 在现有 Python recipe 的原 `return config` 前，选择以下一种方式开启采集。示例沿用原配置变量名 `config`。

   **方式一：直接调用配置函数。** 导入 `npu_profiler`，将返回的 NPU profiler 配置赋给 `config.profiler`。

   ```python
   from torchtitan_ascend.overrides.profiler import npu_profiler

   config.profiler = npu_profiler(config.profiler, enable_profiling=True)
   ```

   **方式二：通过 override 配置。** 将配置函数的完整路径和参数加入 `config.override.imports`，由 TorchTitan 应用 override 时替换 profiler 配置。

   ```python
   config.override.imports.append(
       (
           "torchtitan_ascend.overrides.profiler.npu_profiler",
           {"enable_profiling": True},
       )
   )
   ```
3. 按对应模型文档加载环境，从 TorchTitan 源码目录使用原 `MODULE`、`CONFIG` 和训练启动命令运行。

## 采集配置

原有 `config.profiler` 已包含公共采集配置，`npu_profiler` 在此基础上扩展 NPU 采集选项，并允许通过其支持的参数覆盖采集设置。下文列出新增参数及默认值。

### 采集内容

| 字段                 | 默认值                | 用途                                                                                  |
| -------------------- | --------------------- | ------------------------------------------------------------------------------------- |
| `enable_profiling` | `None`              | 沿用原配置中的采集开关；显式传 `True` 或 `False` 覆盖原值，上游配置默认关闭。     |
| `with_cpu`         | `True`              | 同时采集 CPU 和 NPU；设为 `False` 时仅采集 NPU。                                    |
| `record_shapes`    | `False`             | 记录 PyTorch 算子输入张量的形状（例如批大小、序列长度）时，需要同时设置 `with_cpu=True`（默认已开启）；开启会增加性能与内存开销。         |
| `with_stack`       | `False`             | 记录算子调用栈，帮助查找对应代码；需要 `with_cpu=True`（默认已开启），开启会增加性能与内存开销。 |
| `profile_memory`   | `False`             | 记录 profiler 内存事件，与独立内存快照分别控制。                                      |
| `level`            | `"level1"`          | 可选 `level0`、`level1`、`level2`，信息量逐级增加，开销通常也增大。 |
| `aic_metrics_type` | `"PipeUtilization"` | 默认查看 AI Core 各流水线的利用率；`ArithmeticUtilization` 查看算术单元利用率。`AiCoreNone` 表示不采这些指标，当前接口尚不支持该值。 |
| `ranks`            | `(0,)`              | 默认仅采集全局 rank 0；可传 `[0, 1]` 等序列，`[-1]` 或 `None` 表示采集全部 rank。   |

### 采集周期

以下参数用于设置采集时机和记录步数，支持公共周期模式和相对窗口模式两种采集模式，分别说明如下。

#### 一、公共周期模式

不提供 `start_step/end_step` 时，使用原 `config.profiler` 的公共调度参数。

| 字段                         | 默认值   | 用途                                                             |
| ---------------------------- | -------- | ---------------------------------------------------------------- |
| `profile_freq`             | `10`   | 每个周期的总步数，包含等待、预热和有效记录。                     |
| `profiler_warmup`          | `3`    | 每周期采集器预热步数，预热数据不作为正式记录结果。               |
| `profiler_active`          | `1`    | 每周期有效记录步数。                                             |
| `profiler_repeat`          | `None` | `None` 或 `0` 表示不限周期数；正整数限制采集周期数，`1` 表示只采一次。 |
| `profiler_skip_first`      | `None` | `None` 或 `0` 表示不跳过；正整数表示跳过最初多少个全局训练步。 |
| `profiler_skip_first_wait` | `None` | `None` 或 `0` 表示首周期正常等待；非零非负整数表示跳过首次等待，之后周期仍正常等待。 |

周期模式要求 `profile_freq >= profiler_warmup + profiler_active`。频率与有效记录步数应为正整数，预热、重复次数及跳过步数应为非负整数。默认周期如下：

```text
等待步数 = profile_freq - profiler_warmup - profiler_active
默认周期：10 = 6 + 3 + 1

训练步数   1  2  3  4  5  6 | 7  8  9 | 10
采集阶段         等待        |   预热   | 记录
```

训练步指完整训练迭代，不是梯度累积中的单个 microbatch。自定义周期时，先设置公共字段，再调用配置函数，例如：

```python
config.profiler.profile_freq = 10
config.profiler.profiler_warmup = 3
config.profiler.profiler_active = 1
config.profiler.profiler_repeat = 1
config.profiler = npu_profiler(config.profiler, enable_profiling=True)
```

此例仅记录全局第 10 步一次。周期模式续训时跟随全局训练进度：从第 5 步恢复，下一次仍记录全局第 10 步；若该采集周期已经结束，则不会重新采集。在有效窗口中途恢复不恢复上次进程的采集缓冲。

#### 二、相对窗口模式

同时提供 `start_step` 和 `end_step` 时，以本次启动或恢复位置为起点，只采集一次。

| 字段           | 默认值   | 用途                                               |
| -------------- | -------- | -------------------------------------------------- |
| `start_step` | `None` | 相对本次启动或恢复位置，先跳过的训练步数。         |
| `end_step`   | `None` | 相对本次启动或恢复位置，记录的最后一步，包含该步。 |

例如：

```python
config.profiler = npu_profiler(
    config.profiler,
    enable_profiling=True,
    start_step=5,
    end_step=6,
    ranks=[0],
    save_dir="./profiling",
)
```

两个边界必须同时提供，要求 `0 <= start_step < end_step`，共记录 `end_step - start_step` 步。窗口模式固定等待与预热为 `0`、重复次数为 `1`，先跳过 `start_step` 步；公共调度字段不再控制时间线窗口，其原值及内存快照频率保持不变。

上述示例跳过前 5 步，仅记录 rank 0 的第 6 步。从头训练时记录全局第 6 步；从第 100 步恢复时记录全局第 106 步。全局记录范围为 `(global_step + start_step, global_step + end_step]`，其中 `global_step` 由训练器传入。

窗口模式在每次启动或恢复后重新计数，与公共周期模式不同。窗口内没有采集器预热，跳过训练步也不等于采集器预热；选择窗口时应考虑初始化、编译与训练预热。

### 输出设置

| 字段 | 默认值 | 用途 |
|---|---|---|
| `save_traces_folder` | `"profiling/traces"` | 未指定 `save_dir` 时，与训练输出根目录 `base_folder` 拼接；绝对路径保持原样。 |
| `save_dir` | `None` | 显式指定时直接作为时间线输出目录，不拼接 `base_folder`；相对路径以启动时工作目录为基准。 |
| `analyse_flag` | `True` | 采集结束后自动解析结果。 |
| `data_simplification` | `False` | 是否启用底层采集数据精简。 |

默认采集结果输出目录为 `base_folder/profiling/traces`；正常训练使用默认输出目录时通常为 `./outputs/profiling/traces`。当前 NPU 采集配置不拼接 `leaf_folder`。各 worker 使用 `rank{全局编号}` 标识，实际子目录和文件格式由 `torch_npu` 导出处理器决定。

## 内存快照

独立内存快照沿用 TorchTitan 的实现和配置，与时间线中的 `profile_memory` 内存事件分别控制。时间线开关 `enable_profiling`、rank 筛选 `ranks` 和输出覆盖 `save_dir` 不控制公共内存快照。

| 字段                            | 默认值                          | 用途                                                         |
| ------------------------------- | ------------------------------- | ------------------------------------------------------------ |
| `enable_memory_snapshot`      | `False`                       | 开启独立的内存快照；目标 NPU 环境的 API 支持需单独验证。 |
| `memory_snapshot_freq`        | `None`                        | 默认使用 `profile_freq` 作为快照间隔，也可单独设置正整数。 |
| `save_memory_snapshot_folder` | `"profiling/memory_snapshot"` | 内存快照输出子目录，相对于 `base_folder`。                 |
| `memory_snapshot_max_entries` | `1000000`                     | 内存历史事件保留上限，应为正整数；缓冲满后淘汰最旧事件。     |

## 查看结果与排查

按启动日志给出的路径查看结果。没有生成时间线时，先确认现有 recipe 已直接赋值 NPU profiler 配置或在 `override.imports` 中配置 NPU profiler，且已开启采集、选中了当前 rank，并且训练确实执行到记录窗口；再检查依赖、输出目录权限、可用空间及底层解析日志。

## 使用限制

- `cuspy` 是 CUDA 专用选项，NPU 必须保持 `None`。
- 采集会增加运行开销，不要用采集时的吞吐代表正常训练性能。
- 分享结果前检查其中的路径、调用栈和模型信息。
