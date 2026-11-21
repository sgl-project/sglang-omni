# SPDX-License-Identifier: Apache-2.0
"""CUDA graphs of the Whisper draft's decode steps in a small greedy speculative round."""

from __future__ import annotations

import torch
from sglang.srt.layers.attention.base_attn_backend import AttentionBackend
from sglang.srt.layers.radix_attention import RadixAttention
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.mem_cache.memory_pool import KVCache, KVWriteLoc
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.model_executor.forward_context import ForwardContext, forward_context
from sglang.srt.model_executor.model_runner import ModelRunner
from transformers import WhisperConfig


class DraftDecodeAttention(AttentionBackend):
    """Decode attention of a fixed number of requests with no host-side plan, so one CUDA graph can record several decode steps.

    Each layer gathers every request's K/V through a fixed-width row of KV slots and masks the columns past the request's length. Cross-attention gathers its encoder K/V once per round and reuses it for every step.
    """

    def __init__(self, token_to_kv_pool: KVCache) -> None:
        super().__init__()
        self.token_to_kv_pool = token_to_kv_pool
        self.decoder_slots = torch.empty(0, dtype=torch.int64)
        self.decoder_attention_mask = torch.empty(0, dtype=torch.bool)
        self.encoder_slots = torch.empty(0, dtype=torch.int64)
        self.encoder_attention_mask = torch.empty(0, dtype=torch.bool)
        self.encoder_keys_values: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    def forward_decode(
        self,
        q: torch.Tensor,
        k: torch.Tensor | None,
        v: torch.Tensor | None,
        layer: RadixAttention,
        forward_batch: ForwardBatch,
        save_kv_cache: bool = True,
    ) -> torch.Tensor:
        key_cache, value_cache = self.token_to_kv_pool.get_kv_buffer(layer.layer_id)
        if layer.is_cross_attention:
            if layer.layer_id not in self.encoder_keys_values:
                self.encoder_keys_values[layer.layer_id] = (
                    key_cache[self.encoder_slots],
                    value_cache[self.encoder_slots],
                )
            else:
                pass
            keys, values = self.encoder_keys_values[layer.layer_id]
            attention_mask = self.encoder_attention_mask
        else:
            assert save_kv_cache, "every draft decode step writes its K/V"
            self.token_to_kv_pool.set_kv_buffer(
                layer, KVWriteLoc(forward_batch.out_cache_loc), k, v
            )
            keys = key_cache[self.decoder_slots]
            values = value_cache[self.decoder_slots]
            attention_mask = self.decoder_attention_mask
        batch_size = q.shape[0]
        attention_output = torch.nn.functional.scaled_dot_product_attention(
            q.view(batch_size, layer.tp_q_head_num, 1, layer.head_dim),
            keys.transpose(1, 2),
            values.transpose(1, 2),
            attn_mask=attention_mask,
            scale=layer.scaling,
        )
        return attention_output.reshape(
            batch_size, layer.tp_q_head_num * layer.head_dim
        )


class DraftDecodeGraph:
    """The draft decode steps after a greedy round's first draft, recorded as one CUDA graph for batch_size requests.

    Each step writes its K/V, takes the argmax of its biased logits and feeds it to the next step on the device, so a round replays one graph instead of building, planning and replaying every step from the host. A smaller round fills the leading rows.
    """

    def __init__(
        self, draft_runner: ModelRunner, num_decode_steps: int, batch_size: int
    ) -> None:
        self.draft_runner = draft_runner
        self.num_decode_steps = num_decode_steps
        self.batch_size = batch_size
        self.req_to_token = draft_runner.req_to_token_pool.req_to_token
        self.attention_backend = DraftDecodeAttention(draft_runner.token_to_kv_pool)
        config: WhisperConfig = draft_runner.model.config
        device = self.req_to_token.device
        self.decoder_columns = torch.arange(config.max_target_positions, device=device)
        self.encoder_columns = torch.arange(config.max_source_positions, device=device)
        # note (Haoling Pu): the capture reads request pool row 0, SGLang's padding row, whose KV slots are all the padding slot.
        self.request_pool_indices = torch.zeros(
            batch_size, dtype=torch.int64, device=device
        )
        self.encoder_lengths = torch.full(
            (batch_size,), config.max_source_positions, dtype=torch.int64, device=device
        )
        self.first_step_sequence_lengths = torch.ones(
            batch_size, dtype=torch.int64, device=device
        )
        self.first_draft_token_ids = torch.zeros(
            batch_size, dtype=torch.int64, device=device
        )
        self.logit_bias = torch.zeros(
            batch_size, config.vocab_size, dtype=torch.float32, device=device
        )
        self.graph: torch.cuda.CUDAGraph | None = None
        self.draft_token_ids = torch.empty(0, dtype=torch.int64, device=device)

    def load_inputs(
        self,
        batch: ScheduleBatch,
        first_draft_token_ids: torch.Tensor,
        logit_bias: torch.Tensor | None,
    ) -> None:
        round_size = len(batch.reqs)
        self.request_pool_indices[:round_size].copy_(batch.req_pool_indices)
        # note (Haoling Pu): a row left from a larger round would write its request's KV slots, so the rows past this round go back to the padding row.
        self.request_pool_indices[round_size:].zero_()
        self.encoder_lengths[:round_size].copy_(batch.encoder_lens)
        # note (Haoling Pu): the round's chain forward wrote the previous and newest tokens, so the first decode step attends two more.
        self.first_step_sequence_lengths[:round_size].copy_(batch.seq_lens + 2)
        self.first_draft_token_ids[:round_size].copy_(first_draft_token_ids)
        if logit_bias is None:
            self.logit_bias.zero_()
        else:
            self.logit_bias[:round_size].copy_(logit_bias)

    def run_steps(self) -> torch.Tensor:
        """Decode num_decode_steps drafts from the loaded inputs; under capture this records them."""
        request_slots = self.req_to_token[self.request_pool_indices].long()
        self.attention_backend.encoder_slots = request_slots[:, self.encoder_columns]
        self.attention_backend.encoder_attention_mask = (
            self.encoder_columns[None, :] < self.encoder_lengths[:, None]
        ).view(self.batch_size, 1, 1, -1)
        self.attention_backend.decoder_slots = request_slots.gather(
            1,
            (self.encoder_lengths[:, None] + self.decoder_columns[None, :]).clamp(
                max=request_slots.shape[1] - 1
            ),
        )
        self.attention_backend.encoder_keys_values = {}
        token_ids = self.first_draft_token_ids
        step_token_ids = []
        for step in range(self.num_decode_steps):
            sequence_lengths = self.first_step_sequence_lengths + step
            positions = sequence_lengths - 1
            self.attention_backend.decoder_attention_mask = (
                self.decoder_columns[None, :] < sequence_lengths[:, None]
            ).view(self.batch_size, 1, 1, -1)
            forward_batch = ForwardBatch(
                forward_mode=ForwardMode.DECODE,
                batch_size=self.batch_size,
                input_ids=token_ids,
                req_pool_indices=self.request_pool_indices,
                seq_lens=sequence_lengths,
                out_cache_loc=request_slots.gather(
                    1, (self.encoder_lengths + positions)[:, None]
                ).squeeze(1),
                # note (Haoling Pu): one graph serves every length, so no host-side length exists, and this attention plans nothing from one.
                seq_lens_sum=0,
                encoder_lens=self.encoder_lengths,
                positions=positions,
                capture_hidden_mode=CaptureHiddenMode.NULL,
            )
            with forward_context(
                ForwardContext(attn_backend=self.attention_backend),
                forward_batch=forward_batch,
            ):
                logits = self.draft_runner.model.forward(
                    token_ids, positions, forward_batch
                ).next_token_logits
            token_ids = (logits + self.logit_bias).argmax(dim=-1)
            step_token_ids.append(token_ids)
        return torch.stack(step_token_ids, dim=1)

    def capture(
        self, memory_pool: tuple[int, int], capture_stream: torch.cuda.Stream
    ) -> None:
        # note (Haoling Pu): the encoder's graphs replay from their own thread with the cuBLAS workspace of torch.cuda.graph's default stream, which a graph captured on that stream would write too.
        if capture_stream == torch.cuda.graph.default_capture_stream:
            raise RuntimeError(
                "the Whisper draft graph would capture on torch.cuda.graph's default stream, "
                "whose cuBLAS workspace the encoder graphs write from another thread"
            )
        else:
            pass
        self.run_steps()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(
            graph,
            pool=memory_pool,
            stream=capture_stream,
            capture_error_mode="thread_local",
        ):
            self.draft_token_ids = self.run_steps()
        self.graph = graph
        # note (Haoling Pu): every replay gathers its slots, masks and encoder K/V again, so dropping the ones gathered under capture lets the next, smaller graph reuse their memory in the shared pool.
        self.attention_backend = DraftDecodeAttention(
            self.draft_runner.token_to_kv_pool
        )

    def replay(self) -> torch.Tensor:
        """The draft chain of every row from the loaded inputs: the first draft, then num_decode_steps decoded drafts."""
        self.graph.replay()
        return torch.cat(
            [self.first_draft_token_ids[:, None], self.draft_token_ids], dim=1
        )
