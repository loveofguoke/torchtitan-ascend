# TorchTitan Ascend

在 Ascend NPU 上运行上游 TorchTitan 的 Qwen3 最小训练配方。

本次重构移除旧的 `torchtitanturbo` 实现。模型、数据加载、Trainer 和优化器均使用上游代码；本仓仅选择原生 varlen attention、关闭局部编译与 activation checkpoint，并缩小冒烟配置。

## 核心结构

```text
torchtitan_ascend/
├── __init__.py        # 惰性公开 setup；导入包不加载 torch 或修改上游
├── _setup.py          # 按选项应用已注册 fixes，相同选项只执行一次
├── train.py           # setup 后委托给 torchtitan.train.main
├── fixes/
│   ├── __init__.py
│   └── registry.py    # 延迟注册和应用 wrap / replace / polyfill
└── recipes/
    ├── __init__.py
    └── qwen3.py       # 唯一配方：上游 Qwen3 单卡最小训练
```

上游训练入口导入 `torch` 时会自动加载已安装的 `torch_npu`；本仓不提前导入两者。

`fixes` 只提供核心机制，本批没有默认注册的运行时修复。后端所需的两项源码补丁仍通过 [安装说明](INSTALL.md) 构建进 `torch_npu`，不会由 `setup()` 自动安装。自定义修复模块需在 `setup()` 前导入完成注册；初始化后新增的注册需显式调用 `fixes.apply_all()`。初始化采用相同选项时只执行一次，改变 `apply_fixes` 选项会报错。

## 运行

按 [安装说明](INSTALL.md) 准备固定版本的依赖，并设置 `TORCHTITAN_SRC` 为对应源码目录。从上游目录运行，以读取其自带的 tokenizer 和 C4 测试数据：

```bash
source /usr/local/Ascend/ascend-toolkit/set_env.sh
cd "$TORCHTITAN_SRC"
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m torch.distributed.run \
  --nproc_per_node=1 --master_addr=127.0.0.1 --master_port=29500 \
  -m torchtitan_ascend.train \
  --module torchtitan_ascend.recipes.qwen3 --config qwen3_npu \
  --output-dir ./outputs/qwen3-npu
```

配方使用上游 `debugmodel`（约 3200 万参数）、BF16、序列长度 256、每步 512 tokens，运行 10 步。它验证单卡训练链，不代表 Qwen3-8B/14B、多卡、收敛、性能或 checkpoint 能力已经验证。默认 FlexAttention 和局部编译不在本批范围内。

## 验证

`ASCEND_REPO` 指向本仓源码目录。在相同环境、上游工作目录中运行：

```bash
ASCEND_RT_VISIBLE_DEVICES=0 OMP_NUM_THREADS=4 \
python -m pytest --noconftest "$ASCEND_REPO/tests/test_qwen3.py" -q
```

仅保留一个核心功能测试，检查完成 10 次训练更新、loss 和梯度范数有限以及模型参数实际变化。所需后端修复及移除对照见 [补丁说明](patches/README.md)。

设计讨论见 [RFC #5](https://gitcode.com/Ascend/torchtitan-ascend/issues/5)。项目代码采用 [Apache-2.0](LICENSE)；第三方归属见 [Third Party Notice](Third_Party_Open_Source_Software_Notice)。
