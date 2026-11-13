# SPDX-License-Identifier: Apache-2.0
"""Whole-decoder CUDA graphs for uniform, uncached Whisper prompts."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.mem_cache.memory_pool import KVWriteLoc, MHATokenToKVPool
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)

from sglang_omni.models.whisper_asr.sglang_model import (
    WhisperDecoder,
    WhisperDecoderLayer,
    WhisperForConditionalGeneration,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class DensePrefillGraph:
    graph: torch.cuda.CUDAGraph
    logits: torch.Tensor


class WhisperDensePrefillRunner:
    """Replay whole-decoder prefill graphs keyed by (requests, prompt length)."""

    def __init__(
        self,
        model: WhisperForConditionalGeneration,
        token_to_kv_pool: MHATokenToKVPool,
        *,
        encoder_token_count: int,
        max_requests: int,
        prompt_lengths: tuple[int, ...],
    ) -> None:
        self.model: WhisperForConditionalGeneration = model
        self.token_to_kv_pool: MHATokenToKVPool = token_to_kv_pool
        self.encoder_token_count: int = int(encoder_token_count)
        self.max_requests: int = int(max_requests)
        self.prompt_lengths: tuple[int, ...] = tuple(sorted(set(prompt_lengths)))
        decoder = model.model.decoder
        self.decoder: WhisperDecoder = decoder
        reference = decoder.embed_tokens.weight
        self.device: torch.device = reference.device
        self.dtype: torch.dtype = reference.dtype
        self.hidden_size: int = int(model.config.d_model)
        self.num_heads: int = int(model.config.decoder_attention_heads)
        self.head_dim: int = self.hidden_size // self.num_heads
        max_prompt_tokens = self.max_requests * max(self.prompt_lengths)
        max_encoder_tokens = self.max_requests * self.encoder_token_count
        self.input_ids: torch.Tensor = torch.zeros(
            max_prompt_tokens, dtype=torch.int64, device=self.device
        )
        self.positions: torch.Tensor = torch.zeros(
            max_prompt_tokens, dtype=torch.int64, device=self.device
        )
        self.prompt_cache_loc: torch.Tensor = torch.zeros(
            max_prompt_tokens, dtype=torch.int64, device=self.device
        )
        self.encoder_cache_loc: torch.Tensor = torch.zeros(
            max_encoder_tokens, dtype=torch.int64, device=self.device
        )
        self.encoder_states: torch.Tensor = torch.zeros(
            max_encoder_tokens, self.hidden_size, dtype=self.dtype, device=self.device
        )
        self.graphs: dict[tuple[int, int], DensePrefillGraph] = {}
        self.logged_shapes: set[tuple[int, int]] = set()

    def attention_heads(self, states: torch.Tensor, rows: int) -> torch.Tensor:
        """View [rows * tokens, hidden] as [rows, heads, tokens, head_dim]."""
        return states.view(rows, -1, self.num_heads, self.head_dim).transpose(1, 2)

    def decoder_layer(
        self,
        layer: WhisperDecoderLayer,
        hidden_states: torch.Tensor,
        encoder_states: torch.Tensor,
        *,
        request_count: int,
        prompt_cache_loc: torch.Tensor,
        encoder_cache_loc: torch.Tensor,
    ) -> torch.Tensor:
        cross_attention = layer.encoder_attn
        cross_key, cross_value = cross_attention.kv_proj(encoder_states).chunk(
            2, dim=-1
        )
        self.token_to_kv_pool.set_kv_buffer(
            cross_attention.attn,
            KVWriteLoc(encoder_cache_loc),
            cross_key.view(-1, self.num_heads, self.head_dim),
            cross_value.view(-1, self.num_heads, self.head_dim),
        )

        self_attention = layer.self_attn
        residual = hidden_states
        normed = layer.self_attn_layer_norm(hidden_states)
        query, key, value = self_attention.qkv_proj(normed).chunk(3, dim=-1)
        self.token_to_kv_pool.set_kv_buffer(
            self_attention.attn,
            KVWriteLoc(prompt_cache_loc),
            key.view(-1, self.num_heads, self.head_dim),
            value.view(-1, self.num_heads, self.head_dim),
        )
        attention_output = F.scaled_dot_product_attention(
            self.attention_heads(query, request_count),
            self.attention_heads(key, request_count),
            self.attention_heads(value, request_count),
            is_causal=True,
            scale=self_attention.scaling,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            -1, self.hidden_size
        )
        hidden_states = residual + self_attention.out_proj(attention_output)

        residual = hidden_states
        normed = layer.encoder_attn_layer_norm(hidden_states)
        query = cross_attention.q_proj(normed)
        attention_output = F.scaled_dot_product_attention(
            self.attention_heads(query, request_count),
            self.attention_heads(cross_key, request_count),
            self.attention_heads(cross_value, request_count),
            scale=cross_attention.scaling,
        )
        attention_output = attention_output.transpose(1, 2).reshape(
            -1, self.hidden_size
        )
        hidden_states = residual + cross_attention.out_proj(attention_output)

        residual = hidden_states
        normed = layer.final_layer_norm(hidden_states)
        return residual + layer.fc2(layer.activation_fn(layer.fc1(normed)))

    def forward_static(self, request_count: int, prompt_length: int) -> torch.Tensor:
        prompt_tokens = request_count * prompt_length
        encoder_tokens = request_count * self.encoder_token_count
        hidden_states = self.decoder.embed_input_ids(
            self.input_ids[:prompt_tokens], self.positions[:prompt_tokens]
        )
        encoder_states = self.encoder_states[:encoder_tokens]
        prompt_cache_loc = self.prompt_cache_loc[:prompt_tokens]
        encoder_cache_loc = self.encoder_cache_loc[:encoder_tokens]
        for layer in self.decoder.layers:
            hidden_states = self.decoder_layer(
                layer,
                hidden_states,
                encoder_states,
                request_count=request_count,
                prompt_cache_loc=prompt_cache_loc,
                encoder_cache_loc=encoder_cache_loc,
            )
        hidden_states = self.decoder.layer_norm(hidden_states)
        last_hidden_states = hidden_states.view(
            request_count, prompt_length, self.hidden_size
        )[:, -1]
        return torch.matmul(last_hidden_states, self.model.lm_head.weight.T).float()

    @torch.inference_mode()
    def capture(self) -> None:
        """Capture every (requests, prompt length) shape, largest first."""
        memory_pool = torch.cuda.graph_pool_handle()
        stream = torch.cuda.Stream(device=self.device)
        # Capture writes K/V into slot 0 of the pool, which never holds a request.
        self.prompt_cache_loc.zero_()
        self.encoder_cache_loc.zero_()
        for request_count in range(self.max_requests, 0, -1):
            for prompt_length in self.prompt_lengths:
                stream.wait_stream(torch.cuda.current_stream(self.device))
                with torch.cuda.stream(stream):
                    for _ in range(2):
                        self.forward_static(request_count, prompt_length)
                torch.cuda.current_stream(self.device).wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=memory_pool, stream=stream):
                    logits = self.forward_static(request_count, prompt_length)
                self.graphs[(request_count, prompt_length)] = DensePrefillGraph(
                    graph=graph, logits=logits
                )
        torch.cuda.synchronize(self.device)
        logger.info(
            f"Captured Whisper dense prefill graphs for up to {self.max_requests} "
            f"requests, prompt lengths {self.prompt_lengths}"
        )

    def prompt_length_of(self, forward_batch: ForwardBatch) -> int | None:
        """Return the shared prompt length when this batch has a captured graph."""
        extend_lengths = forward_batch.extend_seq_lens_cpu
        prefix_lengths = forward_batch.extend_prefix_lens_cpu
        request_count = forward_batch.batch_size
        if (
            forward_batch.forward_mode != ForwardMode.EXTEND
            or not 1 <= request_count <= self.max_requests
            or forward_batch.return_logprob
            or forward_batch.capture_hidden_mode != CaptureHiddenMode.NULL
            or extend_lengths is None
            or len(extend_lengths) != request_count
            or prefix_lengths is None
            or len(prefix_lengths) != request_count
            or any(prefix_lengths)
            or forward_batch.encoder_cached is None
            or len(forward_batch.encoder_cached) != request_count
            or any(forward_batch.encoder_cached)
            or forward_batch.encoder_lens_cpu is None
            or len(forward_batch.encoder_lens_cpu) != request_count
            or any(
                encoder_length != self.encoder_token_count
                for encoder_length in forward_batch.encoder_lens_cpu
            )
        ):
            return None
        else:
            pass
        prompt_length = int(extend_lengths[0])
        if (
            prompt_length not in self.prompt_lengths
            or any(int(length) != prompt_length for length in extend_lengths)
            or (request_count, prompt_length) not in self.graphs
        ):
            return None
        else:
            return prompt_length

    def run(
        self, forward_batch: ForwardBatch, prompt_length: int
    ) -> LogitsProcessorOutput:
        request_count = forward_batch.batch_size
        prompt_tokens = request_count * prompt_length
        encoder_tokens = request_count * self.encoder_token_count
        encoder_states = self.model.batch_precomputed_encoder_states(forward_batch)
        assert encoder_states is not None and encoder_states.shape[0] == encoder_tokens
        self.input_ids[:prompt_tokens].copy_(forward_batch.input_ids)
        self.positions[:prompt_tokens].copy_(forward_batch.positions)
        self.prompt_cache_loc[:prompt_tokens].copy_(forward_batch.out_cache_loc)
        self.encoder_cache_loc[:encoder_tokens].copy_(
            forward_batch.encoder_out_cache_loc
        )
        self.encoder_states[:encoder_tokens].copy_(encoder_states)
        captured = self.graphs[(request_count, prompt_length)]
        shape = (request_count, prompt_length)
        if shape not in self.logged_shapes:
            logger.info(
                f"Replaying Whisper dense prefill graph requests={request_count} "
                f"prompt_length={prompt_length}"
            )
            self.logged_shapes.add(shape)
        else:
            pass
        captured.graph.replay()
        return LogitsProcessorOutput(next_token_logits=captured.logits.clone())


__all__ = ["WhisperDensePrefillRunner"]
