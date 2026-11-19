# SPDX-License-Identifier: Apache-2.0

"""GPT-OSS training recipe with Ascend runtime settings."""

from torchtitan.components.data import ConcatThenSplitPackingConfig, GrainDataLoader
from torchtitan.components.data.text_datasets import DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import (
    AdamW,
    LRSchedulersContainer,
    Optim,
    OptimizersContainer,
)
from torchtitan.config import DebugConfig, ParallelismConfig, TrainingConfig
from torchtitan.distributed.activation_remat import DefaultActivationRematConfig
from torchtitan.models.common.config_utils import (
    decoder_vocab_size,
    DEFAULT_DEBUG_MODEL_SEQ_LEN,
)
from torchtitan.models.gpt_oss import build_model_config
from torchtitan.observability.metrics import MetricsProcessor
from torchtitan.trainer import Trainer


def npu_changes(config: Trainer.Config) -> None:
    """Apply Ascend runtime settings in place."""
    config.model.local_compile_regions = []
    config.comm.train_timeout_seconds = max(config.comm.train_timeout_seconds, 600)


def _apply_gpt_oss_cp(config: Trainer.Config) -> Trainer.Config:
    """Replace flex attention with the AllGather-KV variants for CP.

    GPT-OSS per-head sinks do not support Ulysses CP, so context parallelism
    uses AllGatherKV FlexAttention where each rank holds local queries and all
    keys/values.
    """
    from torchtitan.config.transform import apply_transforms, ContextParallelTransform
    from torchtitan.models.common.attention import FlexInnerAttention
    from torchtitan.models.common.attention.cp_attention import (
        AllGatherKVFlexInnerAttention,
        AllGatherKVSlidingWindowFlexInnerAttention,
    )
    from torchtitan.models.gpt_oss.flavors import SlidingWindowFlexInnerAttention

    return apply_transforms(
        config,
        [
            ContextParallelTransform(
                inner_attention_map={
                    FlexInnerAttention: AllGatherKVFlexInnerAttention,
                    SlidingWindowFlexInnerAttention: (
                        AllGatherKVSlidingWindowFlexInnerAttention
                    ),
                }
            )
        ],
    )


def gpt_oss_debugmodel_npu(
    *,
    tp: int = 1,
    ep: int = 1,
    cp: int = 1,
) -> Trainer.Config:
    """Validate the GPT-OSS debug model with ten steps on C4 test data.

    ``cp`` enables context parallelism through AllGather-KV FlexAttention,
    which requires Ascend950/A5 and does not support deterministic training.
    NPU-specific adaptations:
    - model local compilation disabled
    - CUDA graphs disabled
    - seed=42 with deterministic training (cp=1 only)
    - widened communication timeout
    """
    if cp < 1:
        raise ValueError(f"context_parallel_degree must be >= 1, got {cp}")
    parallelism = ParallelismConfig(
        tensor_parallel_degree=tp,
        expert_parallel_degree=ep,
        enable_sequence_parallel=(tp > 1),
    )
    model = build_model_config(
        "debugmodel",
        seq_len=DEFAULT_DEBUG_MODEL_SEQ_LEN,
        attn_backend="flex" if cp > 1 else "varlen",
        parallelism=parallelism,
    )
    config = Trainer.Config(
        model=model,
        hf_assets_path="./tests/assets/tokenizer",
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(
                global_vocab_size=decoder_vocab_size(model),
            ),
        ),
        dataloader=GrainDataLoader.Config(
            dataset=ConcatThenSplitPackingConfig(dataset=DATASETS["c4_test"]),
            max_num_documents=None,
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)],
            ),
            lr_scheduler=LRSchedulersContainer.Config(
                warmup_steps=2,
                decay_ratio=0.8,
                decay_type="linear",
                min_lr_factor=0.0,
            ),
        ),
        training=TrainingConfig(
            num_tokens_per_microbatch_per_dp_rank=8 * model.max_context_length,
            max_context_length=model.max_context_length,
            steps=10,
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        checkpointer=None,
        activation_remat=DefaultActivationRematConfig(),
        metrics=MetricsProcessor.Config(log_freq=1),
        # Deterministic training is not supported on the AllGather-KV CP path,
        # so only pin seed/determinism when context parallelism is disabled.
        debug=(DebugConfig(seed=42, deterministic=True) if cp == 1 else DebugConfig()),
    )
    npu_changes(config)
    if cp > 1:
        config.parallelism.context_parallel_degree = cp
        config = _apply_gpt_oss_cp(config)
    return config
