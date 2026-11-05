# Runtime patches

## 加载与应用

训练入口在导入 TorchTitan 前调用 `torchtitan_ascend.setup()`，由 `apply_all()` 发现并安装修复。仅导入本包不会安装修复。

自动发现按模块名排序，仅扫描本目录的公开模块和包；跳过 `registry.py` 及以 `_` 开头的模块。注册时不导入目标模块，安装时才解析目标。修复必须先于调用方保存原函数引用。

`deferred=True` 在目标模块加载完成、返回调用方前安装修复；已加载的目标立即安装。此选项用于避免提前加载可选编译后端，不向独立 Python 子进程传播修复。

## 声明修复

在公开模块或包的 `__init__.py` 中使用 `@patch(target="module:function", kind="replace")` 声明目标，无需修改注册表。被装饰的函数接收原函数并返回替代函数，示例见 [varlen_attention](varlen_attention/__init__.py)。

- `wrap`：返回调用 `original` 的包装函数，原函数必须存在。
- `replace`：返回替代实现，原函数必须存在。
- `polyfill`：仅为缺失目标提供实现，此时 `original` 为 `None`。

## 约束

- 目标格式为 `module:function` 或 `module:Class.method`，保留静态方法和类方法的绑定方式。
- 每个目标仅注册一次；重复安装同一修复时跳过，避免多层包装。
- 即时修复按注册顺序安装，延迟修复按模块导入时机安装；不提供依赖排序。
- 导入、解析和安装失败直接抛出异常；模块顶层不得初始化硬件或执行计算。

原生依赖要求见[安装补丁](../../fixes/README.md)，构建步骤见[安装指南](../../INSTALL.md)。

## 当前修复

| 修复 | 当前行为 |
| --- | --- |
| [attention_gym.py](attention_gym.py) | 对可选 CUDA 后端缺失复用 Attention Gym 的占位实现；实际构造缺失内核及基础依赖缺失时仍报错。 |
| [causal_conv1d_spmd.py](causal_conv1d_spmd.py) | 延迟注册本地因果卷积的 SPMD 类型规则。 |
| [fused_swiglu.py](fused_swiglu.py) | 为上游 SwiGLU 前后向算子补充 PrivateUse1 派发。 |
| [storage.py](storage.py) | 使用 NPUStorageImpl 构造外部 storage，包括 remat 的空数据占位对象；依赖安装阶段的 storage 补丁。 |
| [inductor.py](inductor.py) | 复用 NPU scheduler 构造，限制 sub-parent epilogue，并适配 reduction 接口。 |
| [inductor_determinism.py](inductor_determinism.py) | 对确定性归约筛选配置并禁用计时选优缓存。 |
| [sort_layout.py](sort_layout.py) | 在 fallback 创建时约束 NPU 稳定排序输入连续，覆盖普通 lowering 及节点级强制 fallback，避免错误 BlockMask 反向索引。 |
| [flex_attention_determinism.py](flex_attention_determinism.py) | 启用严格确定性时，Flex 从首次调用尝试动态形状编译，并使用固定的缓存版本标记隔离修复前的图。 |
| [checkpoint_rng.py](checkpoint_rng.py) | 初始化 Python、NumPy RNG，并在原生完整 DCP 检查点保存和恢复逐全局 rank 的 Python、NumPy、Torch CPU 和当前 NPU RNG；模型/HF 导出沿用上游策略。 |
| [varlen_attention](varlen_attention/__init__.py) | 将公共 varlen API 连接到 NPU 前后向，按 Q/K 配对过滤空文档槽，保留文档边界及参数校验。 |

### 因果卷积 SPMD 类型规则

目标模块加载后，为 `torchtitan_ascend.models.common.linear_attention._CausalConv1d`
调用 `spmd_types.register_local_autograd_function`。该注册供 SPMD 类型检查使用，
返回原自动求导类，保留其前后向实现和调用方导入路径。

### Fused SwiGLU 派发

上游 `torchtitan_recipes.overrides.fused_swiglu` 加载后，为 `silu_and_mul_op`
和 `silu_and_mul_backward_op` 注册 PrivateUse1 内核，分别复用上游 Triton 前后向实现。
沿用 packed `gate_up [..., 2, F]` 接口及已有的 fake、autograd 和 SPMD 规则。

### NPU 编译确定性

Triton experimental 的 reduction 和 persistent reduction 在 metadata 请求确定性时，使用 PyTorch 的配置筛选函数并禁用计时选优缓存。普通归约和 pointwise 保持原生逻辑。

### NPU 稳定排序布局

Flex Attention 使用 BlockMask 记录哪些 Q 块和 KV 块需要计算。反向过程会产生 KV 块到 Q 块的索引，但是编译器记录的 stride 与 NPU 实际返回的 stride 不一致，导致后续编译算子会错误读取。
为解决这一问题，在 fallback 创建时约束 NPU 稳定排序使用连续输入，使其与实际输出的布局一致，覆盖普通 lowering 及编译节点的强制 fallback。

### Flex Attention 确定性

`torch.compile()` 默认使用 `dynamic=None`，通常先按静态形状编译，形状变化触发重编译时再尝试动态图。NPU Flex 的静态与动态内核可能有不同的浮点舍入结果。
在同一个训练步，连续训练可能已切换到动态内核，续训进程却可能因首次处理该形状而使用静态内核，导致相同输入得到不同的结果。
启用 `debug.deterministic` 时，`flex_attention_determinism.py` 设置 `dynamic=True`，让连续训练与续训都从首次调用起尝试动态形状编译，避免依赖此前见过的输入形状。缓存条件匹配时可跨进程复用已编译产物。

`sort_layout.py` 的运行时修复不会自动改变图缓存键。Flex 补丁向 `torch.compiler.config.cache_key_tag` 追加固定版本标记，隔离修复前的错误图并保留调用方已有标记。

### 完整检查点的 RNG 状态

全局 rank 的 Python、NumPy、Torch CPU 和 NPU RNG（随机流）会影响模型训练的确定性。而原生 TorchTitan 在断点续训时不会记录/加载 Python 和 NumPy RNG，导致断点续训确定性不一致。
因此，`checkpoint_rng.py` 在训练引擎构造完成后、模型和数据加载器创建前，为显式指定 seed 的 CPU/NPU 配置初始化 Python、NumPy 全局 RNG，采用 Torch 最终 seed 保留 mesh 轴偏移，NumPy 使用其低 32 位。未指定 seed 时保持原生行为。
`checkpoint_rng.py` 还会在 worker 已绑定设备后，为完整 DCP 注册每个全局 rank 的 RNG，保存为 `rng.rank_<global_rank>`。

限制：严格续训要求相同 rank 拓扑；缺失的随机流无法恢复，不构成严格零误差续训。旧检查点若没有 `rng`，可显式设置 `checkpointer.exclude_from_loading=["rng"]` 兼容加载。
