# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import bisect

import torch
import torch.nn.functional as F
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from torch import nn

from sglang_omni.platforms.device_graph import DeviceGraphBackend, ReplayableGraph

REQUESTS_PER_ATTENTION_CALL = 8


class WhisperDecodeAttention(AttentionBackend):

    def __init__(self, token_to_kv_pool: KVCache) -> None:
        super().__init__()
        self.token_to_kv_pool = token_to_kv_pool
        self.encoder_slots = torch.empty(0, dtype=torch.int64)
        self.decoder_slots = torch.empty(0, dtype=torch.int64)
        self.decoder_attention_mask = torch.empty(0, dtype=torch.bool)

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        assert save_kv_cache, "every decode step writes its K/V"
        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if layer.is_cross_attention:
            slots = self.encoder_slots
            attention_mask = None
        else:
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(forward_batch.out_cache_loc), k, v
            )
            slots = self.decoder_slots
            attention_mask = self.decoder_attention_mask
        batch_size = q.shape[0]
        queries = q.view(batch_size, layer.tp_q_head_num, 1, layer.head_dim)
        attention_outputs = []
        for start in range(0, batch_size, REQUESTS_PER_ATTENTION_CALL):
            rows = slice(start, start + REQUESTS_PER_ATTENTION_CALL)
            attention_outputs.append(
                F.scaled_dot_product_attention(
                    queries[rows],
                    key_cache[slots[rows]].transpose(1, 2),
                    value_cache[slots[rows]].transpose(1, 2),
                    attn_mask=None if attention_mask is None else attention_mask[rows],
                    scale=layer.scaling,
                )
            )
        attention_output = (
            attention_outputs[0]
            if len(attention_outputs) == 1
            else torch.cat(attention_outputs)
        )
        return attention_output.reshape(
            batch_size, layer.tp_q_head_num * layer.head_dim
        )


class WhisperDecodeGraphRunner:

    def __init__(
        self,
        model: nn.Module,
        *,
        request_token_slots: torch.Tensor,
        token_to_kv_pool: KVCache,
        graph_backend: DeviceGraphBackend,
        captured_request_counts: tuple[int, ...],
        allow_padding: bool,
        encoder_token_count: int,
        decoder_token_count: int,
    ) -> None:
        self.model = model
        self.request_token_slots = request_token_slots
        self.graph_backend = graph_backend
        self.captured_request_counts = tuple(sorted(set(captured_request_counts)))
        self.allow_padding = allow_padding
        self.encoder_token_count = int(encoder_token_count)
        self.decoder_token_count = min(
            int(decoder_token_count),
            int(request_token_slots.shape[1]) - self.encoder_token_count,
        )
        self.attention = WhisperDecodeAttention(token_to_kv_pool)
        device = request_token_slots.device
        max_request_count = self.captured_request_counts[-1]
        self.decoder_columns = torch.arange(self.decoder_token_count, device=device)
        self.input_ids = torch.zeros(
            max_request_count, dtype=torch.int64, device=device
        )
        self.positions = torch.zeros(
            max_request_count, dtype=torch.int64, device=device
        )
        self.req_pool_indices = torch.zeros(
            max_request_count, dtype=torch.int64, device=device
        )
        self.seq_lens = torch.ones(max_request_count, dtype=torch.int64, device=device)
        self.out_cache_loc = torch.zeros(
            max_request_count, dtype=torch.int64, device=device
        )
        self.encoder_lens = torch.full(
            (max_request_count,),
            self.encoder_token_count,
            dtype=torch.int64,
            device=device,
        )
        self.graphs: dict[int, ReplayableGraph] = {}
        self.logits: dict[int, torch.Tensor] = {}

    def decode_step(self, request_count: int) -> torch.Tensor:
        request_slots = self.request_token_slots[
            self.req_pool_indices[:request_count]
        ].long()
        self.attention.encoder_slots = request_slots[:, : self.encoder_token_count]
        decoder_slots = request_slots[
            :,
            self.encoder_token_count : self.encoder_token_count
            + self.decoder_token_count,
        ]
        decoder_columns_in_use = (
            self.decoder_columns[None, :] < self.seq_lens[:request_count, None]
        )
        self.attention.decoder_slots = torch.where(
            decoder_columns_in_use, decoder_slots, decoder_slots[:, :1]
        )
        self.attention.decoder_attention_mask = decoder_columns_in_use.view(
            request_count, 1, 1, -1
        )
        forward_batch = ForwardBatch(
            forward_mode=ForwardMode.DECODE,
            batch_size=request_count,
            input_ids=self.input_ids[:request_count],
            req_pool_indices=self.req_pool_indices[:request_count],
            seq_lens=self.seq_lens[:request_count],
            out_cache_loc=self.out_cache_loc[:request_count],
            seq_lens_sum=0,
            encoder_lens=self.encoder_lens[:request_count],
            positions=self.positions[:request_count],
            capture_hidden_mode=CaptureHiddenMode.NULL,
        )
        with forward_context(
            ForwardContext(attn_backend=self.attention), forward_batch=forward_batch
        ):
            return self.model.forward(
                forward_batch.input_ids, forward_batch.positions, forward_batch
            ).next_token_logits

    @torch.inference_mode()
    def capture(self) -> None:
        memory_pool = self.graph_backend.graph_pool_handle()
        for request_count in reversed(self.captured_request_counts):
            self.decode_step(request_count)
            with self.graph_backend.capture(pool=memory_pool) as graph:
                logits = self.decode_step(request_count)
            self.graphs[request_count] = graph
            self.logits[request_count] = logits

    def can_run(self, forward_batch: ForwardBatch) -> bool:
        return (
            forward_batch.forward_mode.is_decode()
            and (
                forward_batch.batch_size <= self.captured_request_counts[-1]
                if self.allow_padding
                else forward_batch.batch_size in self.captured_request_counts
            )
            and forward_batch.spec_info is None
            and forward_batch.capture_hidden_mode in (None, CaptureHiddenMode.NULL)
            and forward_batch.encoder_lens_cpu is not None
            and all(
                encoder_length == self.encoder_token_count
                for encoder_length in forward_batch.encoder_lens_cpu
            )
        )

    def replay(self, forward_batch: ForwardBatch) -> LogitsProcessorOutput:
        request_count = forward_batch.batch_size
        graph_request_count = self.captured_request_counts[
            bisect.bisect_left(self.captured_request_counts, request_count)
        ]
        self.input_ids[:request_count].copy_(forward_batch.input_ids)
        self.positions[:request_count].copy_(forward_batch.positions)
        self.req_pool_indices[:request_count].copy_(forward_batch.req_pool_indices)
        self.seq_lens[:request_count].copy_(forward_batch.seq_lens)
        self.out_cache_loc[:request_count].copy_(forward_batch.out_cache_loc)
        if request_count < graph_request_count:
            self.req_pool_indices[request_count:graph_request_count].zero_()
            self.seq_lens[request_count:graph_request_count].fill_(1)
            self.out_cache_loc[request_count:graph_request_count].zero_()
        else:
            pass
        self.graphs[graph_request_count].replay()
        return LogitsProcessorOutput(
            next_token_logits=self.logits[graph_request_count][:request_count].clone()
        )


__all__ = ["WhisperDecodeAttention", "WhisperDecodeGraphRunner"]
