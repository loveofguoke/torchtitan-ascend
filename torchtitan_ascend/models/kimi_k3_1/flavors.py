# Copyright (c) Meta Platforms, Inc. and affiliates.
# SPDX-License-Identifier: BSD-3-Clause

from functools import partial

from torch import finfo, float32, nn
from torchtitan.config import derive
from torchtitan.config.parallelism import ParallelismConfig
from torchtitan.models.common import LayerNorm, Linear
from torchtitan.models.common.decoder_sharding import replicated_state_sharding_config
from torchtitan.models.deepseek_v4.compressor import SparseIndexerLoss
from torchtitan.models.kimi_k3.attention import KimiMLAAttention
from torchtitan.models.kimi_k3.flavors import (
    _kimi_k3,
    _kimi_k3_config,
    _vision_encoder_config,
)
from torchtitan.models.kimi_k3.model import KimiK3Model

from .dsa import Indexer, KimiDSAAttention, KimiDSALatentAttention
from .model import KimiK31Model, KimiK31TransformerBlock

_INDEXER_LINEAR_INIT = {"weight": partial(nn.init.trunc_normal_, std=0.02)}
_INDEXER_NORM_INIT = {"weight": nn.init.ones_, "bias": nn.init.zeros_}


def _indexer_linear(*, in_features: int, out_features: int) -> Linear.Config:
    return Linear.Config(
        in_features=in_features,
        out_features=out_features,
        bias=False,
        param_init=_INDEXER_LINEAR_INIT,
        sharding_config=replicated_state_sharding_config("weight"),
    )


def _make_dsa_config(
    attention: KimiMLAAttention.Config,
    *,
    num_index_heads: int,
    index_head_dim: int,
    index_topk: int,
) -> KimiDSAAttention.Config:
    indexer = Indexer.Config(
        num_index_heads=num_index_heads,
        index_head_dim=index_head_dim,
        index_topk=index_topk,
        wq_b=_indexer_linear(
            in_features=attention.wq_a.out_features,
            out_features=num_index_heads * index_head_dim,
        ),
        wk=_indexer_linear(in_features=attention.dim, out_features=index_head_dim),
        k_norm=LayerNorm.Config(
            normalized_shape=index_head_dim,
            eps=1e-5,
            param_init=_INDEXER_NORM_INIT,
            sharding_config=replicated_state_sharding_config("weight", "bias"),
        ),
        weights_proj=_indexer_linear(
            in_features=attention.dim, out_features=num_index_heads
        ),
    )
    inner_attention = KimiDSALatentAttention.Config(
        n_heads=attention.n_heads,
        kv_lora_rank=attention.kv_lora_rank,
        qk_nope_head_dim=attention.qk_nope_head_dim,
        qk_rope_head_dim=attention.qk_rope_head_dim,
        v_head_dim=attention.v_head_dim,
        wkv_b=attention.inner_attention.wkv_b,
    )
    inner_attention.aux_loss = SparseIndexerLoss.Config(
        coeff=0.01,
        reduce_mesh="loss",
        softmax_scale=inner_attention.softmax_scale,
        num_heads=attention.n_heads,
        compress_ratio=1,
    )
    return derive(
        attention,
        KimiDSAAttention.Config,
        inner_attention=inner_attention,
        indexer=indexer,
    )


def _debug_backbone(
    seq_len: int, *, parallelism: ParallelismConfig, multimodal: bool
) -> KimiK3Model.Config:
    dim = 7168
    vision_encoder = None
    if multimodal:
        vision_encoder = _vision_encoder_config(
            text_dim=dim,
            dim=256,
            qkv_dim=512,
            hidden_dim=512,
            num_layers=2,
            num_heads=4,
            init_pos_emb_height=32,
            init_pos_emb_width=32,
        )
    return _kimi_k3_config(
        max_context_length=seq_len,
        dim=dim,
        vocab_size=2048,
        num_layers=17,
        full_attention_layers={3, 7, 11, 15, 16},
        attn_res_block_size=12,
        num_heads=96,
        q_lora_rank=1536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        kda_head_dim=128,
        conv_kernel_size=4,
        dense_hidden_dim=33792,
        latent_dim=3584,
        expert_hidden_dim=256,
        num_experts=32,
        top_k=16,
        num_shared_experts=2,
        vision_encoder=vision_encoder,
        attn_backend="varlen",
        parallelism=parallelism,
    )


def build_model_config(
    flavor: str,
    *,
    parallelism: ParallelismConfig,
    seq_len: int,
    num_index_heads: int = 32,
    index_head_dim: int = 128,
    index_topk: int = 2048,
    multimodal: bool = False,
) -> KimiK31Model.Config:
    if (
        parallelism.tensor_parallel_degree != 1
        or parallelism.context_parallel_degree != 1
    ):
        raise ValueError("Kimi K3.1 DSA reference attention requires TP=CP=1.")
    if seq_len <= 0:
        raise ValueError("seq_len must be positive.")
    if flavor == "debugmodel":
        base = _debug_backbone(seq_len, parallelism=parallelism, multimodal=multimodal)
    elif flavor == "Kimi-K3.1-Pro":
        if not multimodal:
            raise ValueError("Kimi K3.1 Pro requires its vision encoder.")
        if seq_len > 1_048_576:
            raise ValueError("Kimi K3.1 Pro context cannot exceed 1048576 tokens.")
        base = _kimi_k3("varlen", seq_len=seq_len, parallelism=parallelism)
    else:
        raise ValueError(f"Unknown Kimi K3.1 flavor: {flavor}")
    vision = base.vision_encoder
    if vision is not None:
        # Match the vLLM vision RMSNorm default for BF16.
        for norm in (vision.block.norm1, vision.block.norm2, vision.final_norm):
            norm.eps = finfo(float32).eps
    layers = list(base.layers)
    for index, layer in enumerate(layers):
        if isinstance(layer.attention, KimiMLAAttention.Config):
            attention = _make_dsa_config(
                layer.attention,
                num_index_heads=num_index_heads,
                index_head_dim=index_head_dim,
                index_topk=index_topk,
            )
            layers[index] = derive(
                layer, KimiK31TransformerBlock.Config, attention=attention
            )
    return derive(base, KimiK31Model.Config, layers=layers)
