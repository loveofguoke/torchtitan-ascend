# SPDX-License-Identifier: Apache-2.0

from dataclasses import replace

from torchtitan.components.data.multimodal.mm_datasets import MM_DATASETS
from torchtitan.components.loss import ChunkedLossWrapper, CrossEntropyLoss
from torchtitan.components.optim import AdamW, Optim, OptimizersContainer
from torchtitan.components.tokenizer import MultiModalTokenizer
from torchtitan.config import derive, TrainingConfig
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common.config_utils import decoder_vocab_size
from torchtitan.models.kimi_k3 import KIMI_K3_SPECIAL_TOKENS
from torchtitan.trainer import Trainer
from torchtitan_recipes.tests.models.kimi_k3 import (
    _kimi_k3_multimodal_dataloader,
    kimi_k3_debugmodel,
)

from torchtitan_ascend.models.kimi_k3_1 import build_model_config
from torchtitan_ascend.recipes.kimi_k3 import npu_changes


def kimi_k3_1_multimodal_debugmodel_npu(*, seq_len: int = 256) -> Trainer.Config:
    parallelism = ParallelismConfig(enable_sequence_parallel=False)
    model = build_model_config(
        "debugmodel", parallelism=parallelism, seq_len=seq_len, multimodal=True
    )
    k3 = kimi_k3_debugmodel(seq_len=seq_len, parallelism=parallelism)
    config = derive(
        k3,
        Trainer.Config,
        model=model,
        optim=replace(
            k3.optim,
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=8e-4)],
            ),
        ),
        activation_remat=None,
        debug=replace(k3.debug, seed=42),
    )
    npu_changes(config)
    return config


def kimi_k3_1_pro_npu(
    *,
    hf_assets_path: str = "./tests/assets/tokenizer",
    seq_len: int = 2048,
    parallelism: ParallelismConfig | None = None,
) -> Trainer.Config:
    """Build a BF16 training recipe for the released K3.1 Pro architecture."""
    if parallelism is None:
        parallelism = ParallelismConfig(enable_sequence_parallel=False)
    model = build_model_config(
        "Kimi-K3.1-Pro", parallelism=parallelism, seq_len=seq_len, multimodal=True
    )
    vocab_size = decoder_vocab_size(model)
    config = Trainer.Config(
        model=model,
        hf_assets_path=hf_assets_path,
        tokenizer=MultiModalTokenizer.Config(**KIMI_K3_SPECIAL_TOKENS),
        dataloader=_kimi_k3_multimodal_dataloader(MM_DATASETS["cc12m"]),
        loss=ChunkedLossWrapper.Config(
            loss_fn=CrossEntropyLoss.Config(global_vocab_size=vocab_size),
        ),
        optim=Optim.Config(
            optimizer=OptimizersContainer.Config(
                optimizers=[AdamW.Config(pattern=r".*", lr=1e-5)],
            ),
        ),
        training=TrainingConfig(
            max_context_length=seq_len,
            num_tokens_per_microbatch_per_dp_rank=seq_len,
            dtype="bfloat16",
            disable_cuda_graphs=True,
        ),
        parallelism=parallelism,
        activation_remat=None,
    )
    npu_changes(config)
    return config
