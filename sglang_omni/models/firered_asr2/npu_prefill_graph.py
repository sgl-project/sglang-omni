# SPDX-License-Identifier: Apache-2.0
"""NPU prefill capture with request-local embeddings and KV write addresses."""

import copy
import logging
from collections import OrderedDict
from dataclasses import dataclass, replace

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.model_executor.forward_context import (
    ForwardContext,
    forward_context,
    get_attn_backend,
    get_forward_context,
)
from sglang.srt.models.qwen2 import Qwen2ForCausalLM
from sglang.srt.runtime_context import get_parallel
from torch import nn

from sglang_omni.platforms.device_graph import ReplayableGraph

logger = logging.getLogger(__name__)
PREFILL_GRAPH_MAX_ENTRIES = 32


@dataclass(kw_only=True)
class PrefillGraph:
    graph: ReplayableGraph
    batch: ForwardBatch
    context: ForwardContext
    hidden_states: torch.Tensor


class FireRedNpuPrefillGraph:
    def __init__(self, language_model: Qwen2ForCausalLM) -> None:
        if get_parallel().tp_size != 1 or language_model.pp_group.world_size != 1:
            raise ValueError(
                "FireRed NPU prefill graphs require TP=1 and PP=1; disable_prefill_cuda_graph enables eager prefill"
            )
        else:
            pass
        self.language_model: Qwen2ForCausalLM = language_model
        self.graphs: OrderedDict[tuple[int, tuple[int, ...]], PrefillGraph] = (
            OrderedDict()
        )
        self.pool: tuple[int, int] = torch.npu.graph_pool_handle()
        self.replay_count: int = 0

    def get_input_embeddings(self) -> nn.Embedding:
        return self.language_model.get_input_embeddings()

    @torch.no_grad()
    def __call__(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor,
    ) -> LogitsProcessorOutput:
        if get_attn_backend().use_fia or get_attn_backend().use_fa:
            raise ValueError(
                "FireRed NPU prefill graphs require the default Ascend attention kernels"
            )
        else:
            pass
        query_token_lengths = forward_batch.extend_seq_lens_cpu
        query_bucket_tokens = 1 << (max(query_token_lengths) - 1).bit_length()
        prefix_token_lengths = tuple(forward_batch.extend_prefix_lens_cpu)
        # note (cocoa00): Ascend captures host lengths; keep prefixes exact and pad only causal query tails.
        capture_shape = (query_bucket_tokens, prefix_token_lengths)
        captured = self.graphs.get(capture_shape)
        if captured is None:
            if len(self.graphs) == PREFILL_GRAPH_MAX_ENTRIES:
                torch.npu.synchronize()
                _, expired = self.graphs.popitem(last=False)
                expired.graph.reset()
            else:
                pass
            batch = copy.copy(forward_batch)
            token_count = forward_batch.batch_size * query_bucket_tokens
            batch.input_embeds = input_embeds.new_zeros(
                token_count, input_embeds.shape[1]
            )
            batch.positions = positions.new_zeros(token_count)
            batch.out_cache_loc = forward_batch.out_cache_loc.new_zeros(token_count)
            batch.extend_seq_lens_cpu = [query_bucket_tokens] * batch.batch_size
            batch.extend_seq_lens = forward_batch.extend_seq_lens.new_full(
                (batch.batch_size,), query_bucket_tokens
            )
            batch.extend_start_loc = (
                torch.arange(
                    batch.batch_size,
                    device=positions.device,
                    dtype=forward_batch.extend_start_loc.dtype,
                )
                * query_bucket_tokens
            )
            batch.extend_num_tokens = token_count
            batch.global_num_token_non_padded_cpu = token_count
            batch.seq_lens_cpu = torch.tensor(
                [prefix + query_bucket_tokens for prefix in prefix_token_lengths],
                dtype=forward_batch.seq_lens_cpu.dtype,
            )
            batch.seq_lens = batch.seq_lens_cpu.to(positions.device)
            batch.seq_lens_sum = int(batch.seq_lens_cpu.sum())
            context = replace(
                get_forward_context(), attn_backend=copy.copy(get_attn_backend())
            )
            metadata = copy.copy(get_attn_backend().forward_metadata)
            context_tokens = max(prefix_token_lengths) + query_bucket_tokens
            page_size = context.attn_backend.page_size
            metadata.block_tables = metadata.block_tables.new_zeros(
                batch.batch_size, (context_tokens + page_size - 1) // page_size
            )
            metadata.extend_seq_lens_cpu_int = torch.tensor(
                batch.extend_seq_lens_cpu, dtype=torch.int32
            )
            metadata.seq_lens_cpu_int = batch.seq_lens_cpu.int()
            context.attn_backend.forward_metadata = metadata
        else:
            batch = captured.batch
            context = captured.context
            self.graphs.move_to_end(capture_shape)

        batch.input_embeds.zero_()
        batch.positions.zero_()
        # note (cocoa00): Padded queries write to the reserved KV slot and follow all real queries.
        batch.out_cache_loc.zero_()
        context.attn_backend.forward_metadata.block_tables.zero_()
        live_block_tables = get_attn_backend().forward_metadata.block_tables
        context.attn_backend.forward_metadata.block_tables[
            :, : live_block_tables.shape[1]
        ].copy_(live_block_tables)
        offset = 0
        for index, query_tokens in enumerate(query_token_lengths):
            destination = slice(
                index * query_bucket_tokens, index * query_bucket_tokens + query_tokens
            )
            source = slice(offset, offset + query_tokens)
            batch.input_embeds[destination].copy_(input_embeds[source])
            batch.positions[destination].copy_(positions[source])
            batch.out_cache_loc[destination].copy_(forward_batch.out_cache_loc[source])
            offset += query_tokens

        if captured is None:
            with forward_context(context):
                for _ in range(2):
                    self.language_model.model(
                        None, batch.positions, batch, batch.input_embeds
                    )
                torch.npu.synchronize()
                graph = torch.npu.NPUGraph()
                with torch.npu.graph(graph, pool=self.pool):
                    hidden_states = self.language_model.model(
                        None, batch.positions, batch, batch.input_embeds
                    )
            captured = PrefillGraph(
                graph=graph, batch=batch, context=context, hidden_states=hidden_states
            )
            self.graphs[capture_shape] = captured
            logger.info(
                f"Captured FireRed NPU prefill graph: query_tokens={query_bucket_tokens}, prefix_tokens={prefix_token_lengths}"
            )
        else:
            pass
        captured.graph.replay()
        self.replay_count += 1
        logger.debug(f"Replayed FireRed NPU prefill graph: count={self.replay_count}")
        hidden_states = torch.cat(
            [
                captured.hidden_states[
                    index * query_bucket_tokens : index * query_bucket_tokens
                    + query_tokens
                ]
                for index, query_tokens in enumerate(query_token_lengths)
            ]
        )
        return self.language_model.logits_processor(
            input_ids, hidden_states, self.language_model.lm_head, forward_batch
        )
