# SPDX-License-Identifier: Apache-2.0
"""Encode streaming windows at any request progress in one batch."""

from __future__ import annotations

import logging
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import torch
from torch.nn import functional as F

from sglang_omni.models.nemotron3_5_asr.encoder_state_pool import (
    EncoderStateSlot,
    NemotronEncoderStatePool,
    PooledAttentionCache,
    PooledPaddingCache,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)

logger = logging.getLogger(__name__)
CAPTURE_WARMUP_STEPS = 3


def encode_pooled_streaming_batch(
    model: Nemotron3_5AsrForRNNT,
    input_features: Sequence[torch.Tensor],
    prompt_ids: torch.Tensor,
    *,
    encoder_slots: Sequence[EncoderStateSlot],
    num_lookahead_tokens: int,
    graph_runner: NemotronStreamingEncoderGraphRunner | None = None,
) -> torch.Tensor:
    """Encode mixed first/subsequent windows while updating persistent slots."""
    assert not model.training
    pool = encoder_slots[0].pool
    groups: dict[tuple[bool, int], list[int]] = defaultdict(list)
    for index, (features, slot) in enumerate(
        zip(input_features, encoder_slots, strict=True)
    ):
        groups[(slot.seen_frames == 0, features.shape[1])].append(index)
    encoded_by_row: dict[int, torch.Tensor] = {}
    for (is_first_chunk, mel_frames), indices in groups.items():
        features = torch.cat([input_features[index] for index in indices])
        slot_ids = torch.tensor(
            [encoder_slots[index].slot_id for index in indices],
            device=pool.layout.device,
        )
        captured_batch = (
            graph_runner.captured_batches.get(len(indices))
            if graph_runner is not None
            and not is_first_chunk
            and mel_frames == graph_runner.subsequent_mel_frames
            else None
        )
        if captured_batch is None:
            encoded_frames = encode_pooled_windows(
                model,
                features,
                prompt_ids[indices],
                slot_ids,
                pool,
                is_first_chunk=is_first_chunk,
                num_lookahead_tokens=num_lookahead_tokens,
            )
        else:
            captured_batch.input_features.copy_(features)
            captured_batch.prompt_ids.copy_(prompt_ids[indices])
            captured_batch.slot_ids.copy_(slot_ids)
            captured_batch.graph.replay()
            encoded_frames = captured_batch.encoded_frames
        encoded_by_row.update(zip(indices, encoded_frames.split(1), strict=True))
        for index in indices:
            encoder_slots[index].seen_frames += encoded_frames.shape[1]
    # note (wirybeaver): Only one group can replay; cat owns its output before the next batch can reuse graph storage.
    return torch.cat([encoded_by_row[index] for index in range(len(encoder_slots))])


def encode_pooled_windows(
    model: Nemotron3_5AsrForRNNT,
    input_features: torch.Tensor,
    prompt_ids: torch.Tensor,
    slot_ids: torch.Tensor,
    pool: NemotronEncoderStatePool,
    *,
    is_first_chunk: bool,
    num_lookahead_tokens: int,
) -> torch.Tensor:
    """Encode equal-shaped windows and update pool state through tensor slot IDs."""
    padding_cache = PooledPaddingCache(
        pool=pool, slot_ids=slot_ids, is_first_chunk=is_first_chunk
    )
    hidden_states = model.encoder.subsampling(
        input_features, attention_mask=None, padding_cache=padding_cache
    )
    hidden_states *= model.encoder.input_scale
    attention_cache = PooledAttentionCache(pool=pool, slot_ids=slot_ids)
    chunk_frames = hidden_states.shape[1]
    attention_mask = attention_cache.create_mask(chunk_frames, num_lookahead_tokens)
    position_embeddings = model.encoder.encode_positions(
        hidden_states, cached_frames=attention_mask.shape[-1] - hidden_states.shape[1]
    )
    all_masked_rows = torch.all(~attention_mask, dim=-1)
    for encoder_layer in model.encoder.layers:
        hidden_states = encoder_layer(
            hidden_states,
            attention_mask=attention_mask,
            all_masked_rows=all_masked_rows,
            position_embeddings=position_embeddings,
            past_key_values=attention_cache,
            padding_cache=padding_cache,
            use_cache=True,
        )

    one_hot = F.one_hot(
        prompt_ids.to(hidden_states.device), num_classes=model.config.num_prompts
    ).to(hidden_states.dtype)
    one_hot = one_hot[:, None, :].expand(-1, hidden_states.shape[1], -1)
    fused = model.prompt_projector(torch.cat([hidden_states, one_hot], dim=-1))
    encoded_frames = model.encoder_projector(fused)
    pool.seen_frames.index_add_(0, slot_ids, torch.full_like(slot_ids, chunk_frames))
    return encoded_frames


@dataclass(kw_only=True)
class CapturedEncoderBatch:
    input_features: torch.Tensor
    prompt_ids: torch.Tensor
    slot_ids: torch.Tensor
    encoded_frames: torch.Tensor
    graph: torch.cuda.CUDAGraph


class NemotronStreamingEncoderGraphRunner:
    """Capture once at startup; the model owner serializes all batch execution."""

    @torch.inference_mode()
    def __init__(
        self,
        model: Nemotron3_5AsrForRNNT,
        pool: NemotronEncoderStatePool,
        *,
        subsequent_mel_frames: int,
        num_lookahead_tokens: int,
        max_batch_size: int,
    ) -> None:
        self.device: torch.device = pool.layout.device
        self.subsequent_mel_frames: int = subsequent_mel_frames
        self.captured_batches: dict[int, CapturedEncoderBatch] = {}
        graph_pool = torch.cuda.graph_pool_handle()
        capture_stream = torch.cuda.Stream(device=self.device)
        max_batch_size = min(max_batch_size, pool.capacity_slots)
        capture_slots = [pool.acquire() for _ in range(max_batch_size)]
        try:
            for batch_size in range(max_batch_size, 0, -1):
                input_features = torch.zeros(
                    batch_size,
                    subsequent_mel_frames,
                    model.config.encoder_config.num_mel_bins,
                    device=self.device,
                    dtype=pool.layout.dtype,
                )
                prompt_ids = torch.zeros(
                    batch_size, dtype=torch.long, device=self.device
                )
                slot_ids = torch.tensor(
                    [slot.slot_id for slot in capture_slots[:batch_size]],
                    device=self.device,
                )
                current_stream = torch.cuda.current_stream(self.device)
                capture_stream.wait_stream(current_stream)
                with torch.cuda.stream(capture_stream):
                    for _ in range(CAPTURE_WARMUP_STEPS):
                        encode_pooled_windows(
                            model,
                            input_features,
                            prompt_ids,
                            slot_ids,
                            pool,
                            is_first_chunk=False,
                            num_lookahead_tokens=num_lookahead_tokens,
                        )
                current_stream.wait_stream(capture_stream)
                torch.cuda.synchronize(self.device)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(
                    graph,
                    pool=graph_pool,
                    stream=capture_stream,
                    capture_error_mode="thread_local",
                ):
                    encoded_frames = encode_pooled_windows(
                        model,
                        input_features,
                        prompt_ids,
                        slot_ids,
                        pool,
                        is_first_chunk=False,
                        num_lookahead_tokens=num_lookahead_tokens,
                    )
                self.captured_batches[batch_size] = CapturedEncoderBatch(
                    input_features=input_features,
                    prompt_ids=prompt_ids,
                    slot_ids=slot_ids,
                    encoded_frames=encoded_frames,
                    graph=graph,
                )
                logger.info(f"Captured Nemotron streaming encoder batch={batch_size}")
        finally:
            torch.cuda.synchronize(self.device)
            for slot in capture_slots:
                slot.release()

    def close(self) -> None:
        torch.cuda.synchronize(self.device)
        self.captured_batches.clear()
