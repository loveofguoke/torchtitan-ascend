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
| [storage.py](storage.py) | 使用 NPUStorageImpl 构造外部 storage，包括 remat 的空数据占位对象；依赖安装阶段的 storage 补丁。 |
| [inductor.py](inductor.py) | 复用 NPU scheduler 构造，限制 sub-parent epilogue，并适配 reduction 接口。 |
| [inductor_determinism.py](inductor_determinism.py) | 对确定性归约筛选配置并禁用计时选优缓存。 |
| [varlen_attention](varlen_attention/__init__.py) | 将公共 varlen API 连接到 NPU 前后向，按 Q/K 配对过滤空文档槽，保留文档边界及参数校验。 |

### NPU 编译确定性

Triton experimental 的 reduction 和 persistent reduction 在 metadata 请求确定性时，使用 PyTorch 的配置筛选函数并禁用计时选优缓存。普通归约和 pointwise 保持原生逻辑。
