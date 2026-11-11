# SPDX-License-Identifier: Apache-2.0
"""Native MOSS test models and attention-pool helpers."""

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

import torch
from sglang.srt.layers.attention.torch_native_backend import TorchNativeAttnBackend
from sglang.srt.mem_cache.memory_pool import MHATokenToKVPool, ReqToTokenPool
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, ForwardMode
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.runtime_context import get_context, get_parallel

from sglang_omni.models.moss_speech.hf_config import MossSpeechConfig
from sglang_omni.models.moss_speech.sglang_model import (
    MossSpeechModelOutput,
    MossSpeechSGLangModel,
)


@dataclass(kw_only=True)
class AttentionPools:
    device: str
    req_to_token_pool: ReqToTokenPool
    token_to_kv_pool: MHATokenToKVPool


@contextmanager
def native_runtime(device: torch.device) -> Iterator[None]:
    with (
        get_context().override_server_args(
            device=device.type,
            dtype="bfloat16",
            attention_backend="torch_native",
            disable_cuda_graph=True,
            enable_torch_compile=False,
            disable_radix_cache=True,
            tp_size=1,
        ),
        get_parallel().override(
            tp_rank=0,
            pp_rank=0,
            attn_tp_rank=0,
            attn_dp_rank=0,
            attn_cp_rank=0,
            moe_tp_rank=0,
            moe_dp_rank=0,
            moe_ep_rank=0,
            tp_group=None,
            attn_tp_group=None,
        ),
        torch.device(device),
        torch.random.fork_rng(
            devices=[device.index or 0] if device.type == "cuda" else []
        ),
    ):
        previous_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.bfloat16)
        try:
            yield
        finally:
            torch.set_default_dtype(previous_dtype)


def tiny_model(
    device: torch.device, *, audio_vocab_size: int = 64
) -> MossSpeechSGLangModel:
    configuration = MossSpeechConfig(
        vocab_size=256,
        audio_vocab_size=audio_vocab_size,
        hidden_size=128,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=64,
        num_shared_layers=2,
        num_modality_layers=1,
        num_hidden_layers=3,
        max_position_embeddings=16,
        moss_decode_buffer_bs=2,
    )
    model = MossSpeechSGLangModel(configuration, init_device=device)
    generator = torch.Generator(device=device).manual_seed(183)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(mean=0.0, std=0.2, generator=generator)
    return model


def checkpoint_weights(
    configuration: MossSpeechConfig,
) -> list[tuple[str, torch.Tensor]]:
    """Construct the external checkpoint schema without the native name mapper."""
    assert configuration.audio_vocab_size is not None
    hidden_size = configuration.hidden_size
    query_rows = configuration.num_attention_heads * configuration.head_dim
    key_value_rows = configuration.num_key_value_heads * configuration.head_dim
    shapes: list[tuple[str, tuple[int, ...]]] = [
        ("model.embed_tokens.weight", (configuration.vocab_size, hidden_size)),
        ("model.audio_embed.weight", (configuration.audio_vocab_size, hidden_size)),
        ("model.text_norm.weight", (hidden_size,)),
        ("model.audio_norm.weight", (hidden_size,)),
        ("text_lm_head.weight", (configuration.vocab_size, hidden_size)),
        ("audio_lm_head.weight", (configuration.audio_vocab_size, hidden_size)),
    ]
    layer_shapes = [
        ("self_attn.q_proj.weight", (query_rows, hidden_size)),
        ("self_attn.k_proj.weight", (key_value_rows, hidden_size)),
        ("self_attn.v_proj.weight", (key_value_rows, hidden_size)),
        ("self_attn.o_proj.weight", (hidden_size, query_rows)),
        ("self_attn.q_norm.weight", (configuration.head_dim,)),
        ("self_attn.k_norm.weight", (configuration.head_dim,)),
        ("mlp.gate_proj.weight", (configuration.intermediate_size, hidden_size)),
        ("mlp.up_proj.weight", (configuration.intermediate_size, hidden_size)),
        ("mlp.down_proj.weight", (hidden_size, configuration.intermediate_size)),
        ("input_layernorm.weight", (hidden_size,)),
        ("post_attention_layernorm.weight", (hidden_size,)),
    ]
    for block, depth in (
        ("shared_block", configuration.num_shared_layers),
        ("text_block", configuration.num_modality_layers),
        ("audio_block", configuration.num_modality_layers),
    ):
        for layer_index in range(depth):
            shapes.extend(
                (f"model.{block}.layers.{layer_index}.{leaf}", shape)
                for leaf, shape in layer_shapes
            )
    return [
        (name, torch.full(shape, index + 1, dtype=torch.bfloat16))
        for index, (name, shape) in enumerate(shapes)
    ]


def prefill(
    model: MossSpeechSGLangModel,
    prompt_embeddings: list[torch.Tensor],
    request_ids: list[str],
) -> tuple[MossSpeechModelOutput, TorchNativeAttnBackend]:
    device = model.embed_tokens.weight.device
    token_lengths = [embeddings.shape[0] for embeddings in prompt_embeddings]
    token_count = sum(token_lengths)
    request_pool = ReqToTokenPool(
        size=len(token_lengths),
        max_context_len=16,
        device=str(device),
        enable_memory_saver=False,
    )
    kv_pool = MHATokenToKVPool(
        size=32,
        page_size=1,
        dtype=torch.bfloat16,
        head_num=model.config.num_key_value_heads,
        head_dim=model.head_dim,
        layer_num=model.total_attention_layers,
        device=str(device),
        enable_memory_saver=False,
        enable_alt_stream=False,
    )
    next_slot = 1
    for request_index, token_length in enumerate(token_lengths, 1):
        request_pool.req_to_token[request_index, :token_length] = torch.arange(
            next_slot, next_slot + token_length, device=device
        )
        next_slot += token_length
    backend = TorchNativeAttnBackend(
        AttentionPools(
            device=str(device),
            req_to_token_pool=request_pool,
            token_to_kv_pool=kv_pool,
        )
    )
    batch = ForwardBatch(
        forward_mode=ForwardMode.EXTEND,
        batch_size=len(token_lengths),
        input_ids=torch.zeros(token_count, dtype=torch.long, device=device),
        req_pool_indices=torch.arange(1, len(token_lengths) + 1, device=device),
        seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
        out_cache_loc=torch.arange(1, token_count + 1, device=device),
        seq_lens_sum=token_count,
        positions=torch.cat(
            [torch.arange(length, device=device) for length in token_lengths]
        ),
        extend_seq_lens=torch.tensor(token_lengths, dtype=torch.int32, device=device),
        extend_prefix_lens=torch.zeros(
            len(token_lengths), dtype=torch.int32, device=device
        ),
        extend_seq_lens_cpu=token_lengths,
        extend_prefix_lens_cpu=[0] * len(token_lengths),
        extend_num_tokens=token_count,
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        output = model(
            batch.input_ids,
            batch.positions,
            batch,
            input_embeds=torch.cat(prompt_embeddings),
            omni_prefill_rids=request_ids,
        )
    return output, backend


def decode(
    model: MossSpeechSGLangModel,
    backend: TorchNativeAttnBackend,
    prompt_lengths: list[int],
    feedback_rows: torch.Tensor,
    request_ids: list[str],
) -> MossSpeechModelOutput:
    device = model.embed_tokens.weight.device
    batch_size = len(prompt_lengths)
    token_slots = torch.arange(
        sum(prompt_lengths) + 1,
        sum(prompt_lengths) + 1 + batch_size,
        device=device,
    )
    for request_index, token_length in enumerate(prompt_lengths, 1):
        backend.req_to_token_pool.req_to_token[request_index, token_length] = (
            token_slots[request_index - 1]
        )
    batch = ForwardBatch(
        forward_mode=ForwardMode.DECODE,
        batch_size=batch_size,
        input_ids=feedback_rows[:, 0] * model.audio_vocab_size + feedback_rows[:, 1],
        req_pool_indices=torch.arange(1, batch_size + 1, device=device),
        seq_lens=torch.tensor(
            [length + 1 for length in prompt_lengths], dtype=torch.int32, device=device
        ),
        out_cache_loc=token_slots,
        seq_lens_sum=sum(prompt_lengths) + batch_size,
        positions=torch.tensor(prompt_lengths, device=device),
        rids=request_ids,
    )
    backend.init_forward_metadata(batch)
    with forward_context(ForwardContext(attn_backend=backend)):
        return model(batch.input_ids, batch.positions, batch)
