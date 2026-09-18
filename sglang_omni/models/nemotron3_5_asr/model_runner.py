# SPDX-License-Identifier: Apache-2.0
"""Model-owned batched RNN-T inference for Nemotron 3.5 ASR."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto import StagePayload
from sglang_omni.utils.checkpoint import resolve_checkpoint

from .hf_compat import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrProcessor,
    Nemotron3_5AsrRNNTDecoderCache,
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
    NemotronAsrStreamingEncoderModelOutput,
)
from .request_builders import (
    NEMOTRON_ASR_SAMPLE_RATE,
    Nemotron3_5ASRRequest,
    build_nemotron3_5_asr_result,
)
from .text import clean_nemotron_text, resolve_nemotron_locale


@dataclass(slots=True)
class Nemotron3_5ASRPreparedChunk:
    input_features: torch.Tensor
    prompt_ids: torch.Tensor


@dataclass(slots=True)
class Nemotron3_5ASRDecodeState:
    tokens: list[int]
    durations: list[int]
    symbols_at_frame: int = 0
    encoder_frames: int = 0
    decoder_steps: int = 0
    attention_cache: DynamicCache | None = None
    padding_cache: NemotronAsrStreamingEncoderCausalConvPaddingCache | None = None
    decoder_cache: Nemotron3_5AsrRNNTDecoderCache | None = None


@dataclass(slots=True)
class Nemotron3_5ASRStreamingBatchResult:
    elapsed_s: float
    raw_texts: list[str]
    clean_texts: list[str]
    languages: list[str | None]


class Nemotron3_5ASRModelRunner:
    """Serialize model inference across threads; callers own request state."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        dtype: str | torch.dtype = "float32",
        num_lookahead_tokens: int = 3,
    ) -> None:
        checkpoint = str(Path(resolve_checkpoint(model_path)).resolve())
        resolved_dtype = resolve_dtype(dtype)
        if resolved_dtype is None:
            raise ValueError("dtype must resolve to a concrete torch dtype")

        self.device = torch.device(device)
        self.dtype = resolved_dtype
        self.processor = Nemotron3_5AsrProcessor.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        self.processor.set_num_lookahead_tokens(int(num_lookahead_tokens))
        config = Nemotron3_5AsrConfig.from_pretrained(
            checkpoint,
            local_files_only=True,
        )
        self.model = Nemotron3_5AsrForRNNT.from_pretrained(
            checkpoint,
            config=config,
            dtype=resolved_dtype,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        # Note (Li Gang): generate mutates model-owned decoder progress.
        self.model_lock = threading.Lock()

    @property
    def prompt_dictionary(self) -> dict[str, int]:
        return dict(self.processor.prompt_dictionary)

    @property
    def streaming_chunk_spec(self) -> dict[str, int]:
        feature_extractor = self.processor.feature_extractor
        return {
            "sample_rate": int(feature_extractor.sampling_rate),
            "first_samples": int(self.processor.num_samples_first_audio_chunk),
            "subsequent_samples": int(self.processor.num_samples_per_audio_chunk),
            "first_frames": int(self.processor.num_mel_frames_first_audio_chunk),
            "subsequent_frames": int(self.processor.num_mel_frames_per_audio_chunk),
            "hop_length": int(feature_extractor.hop_length),
            "n_fft": int(feature_extractor.n_fft),
            "streaming_latency_ms": int(self.processor.streaming_latency_ms),
        }

    def new_streaming_decode_state(self) -> Nemotron3_5ASRDecodeState:
        blank_token_id = int(self.model.config.blank_token_id)
        return Nemotron3_5ASRDecodeState(tokens=[blank_token_id], durations=[0])

    def prepare_streaming_chunk(
        self,
        waveform: np.ndarray,
        *,
        language: str,
        is_first: bool,
    ) -> Nemotron3_5ASRPreparedChunk:
        processor_inputs = self.processor(
            waveform,
            sampling_rate=NEMOTRON_ASR_SAMPLE_RATE,
            is_streaming=True,
            is_first_audio_chunk=is_first,
            language=language,
            return_tensors="pt",
        )
        input_features = processor_inputs.input_features
        required_frames = (
            self.processor.num_mel_frames_first_audio_chunk
            if is_first
            else self.processor.num_mel_frames_per_audio_chunk
        )
        if is_first:
            assert input_features.shape[1] >= required_frames
            input_features = input_features[:, :required_frames, :]
        else:
            assert input_features.shape[1] == required_frames
        return Nemotron3_5ASRPreparedChunk(
            input_features=input_features.to(device=self.device, dtype=self.dtype),
            prompt_ids=processor_inputs.prompt_ids.to(device=self.device),
        )

    @staticmethod
    def merge_attention_caches(
        caches: Sequence[DynamicCache | None],
    ) -> DynamicCache | None:
        if all(cache is None for cache in caches):
            return None
        assert all(cache is not None for cache in caches)
        layer_count = len(caches[0].layers)
        assert all(len(cache.layers) == layer_count for cache in caches)
        key_value_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []
        for layer_index in range(layer_count):
            layers = [cache.layers[layer_index] for cache in caches]
            sequence_lengths = {layer.get_seq_length() for layer in layers}
            assert len(sequence_lengths) == 1
            key_value_pairs.append(
                (
                    torch.cat([layer.keys for layer in layers], dim=0),
                    torch.cat([layer.values for layer in layers], dim=0),
                )
            )
        return DynamicCache(ddp_cache_data=key_value_pairs)

    @staticmethod
    def split_attention_cache(
        cache: DynamicCache, batch_size: int
    ) -> list[DynamicCache]:
        request_caches: list[DynamicCache] = []
        for batch_index in range(batch_size):
            key_value_pairs = [
                (
                    layer.keys[batch_index : batch_index + 1].clone(),
                    layer.values[batch_index : batch_index + 1].clone(),
                )
                for layer in cache.layers
            ]
            request_caches.append(DynamicCache(ddp_cache_data=key_value_pairs))
        return request_caches

    @staticmethod
    def merge_padding_caches(
        caches: Sequence[NemotronAsrStreamingEncoderCausalConvPaddingCache | None],
    ) -> NemotronAsrStreamingEncoderCausalConvPaddingCache | None:
        if all(cache is None for cache in caches):
            return None
        assert all(cache is not None for cache in caches)
        keys = list(caches[0].layers)
        assert all(list(cache.layers) == keys for cache in caches)
        merged = NemotronAsrStreamingEncoderCausalConvPaddingCache()
        for key in keys:
            source_layers = [cache.layers[key] for cache in caches]
            layer = copy(source_layers[0])
            layer.cache = torch.cat([source.cache for source in source_layers], dim=0)
            merged.layers[key] = layer
        return merged

    @staticmethod
    def split_padding_cache(
        cache: NemotronAsrStreamingEncoderCausalConvPaddingCache, batch_size: int
    ) -> list[NemotronAsrStreamingEncoderCausalConvPaddingCache]:
        request_caches = [
            NemotronAsrStreamingEncoderCausalConvPaddingCache()
            for _ in range(batch_size)
        ]
        for key, source in cache.layers.items():
            for batch_index, target in enumerate(request_caches):
                layer = copy(source)
                layer.cache = source.cache[batch_index : batch_index + 1].clone()
                target.layers[key] = layer
        return request_caches

    def merge_decoder_caches(
        self, caches: Sequence[Nemotron3_5AsrRNNTDecoderCache | None]
    ) -> Nemotron3_5AsrRNNTDecoderCache:
        is_initialized = [
            cache is not None and cache.is_initialized for cache in caches
        ]
        merged = Nemotron3_5AsrRNNTDecoderCache(self.model.config)
        if not any(is_initialized):
            return merged
        assert all(is_initialized)
        merged.cache = torch.cat([cache.cache for cache in caches], dim=0)
        merged.hidden_state = torch.cat([cache.hidden_state for cache in caches], dim=1)
        merged.cell_state = torch.cat([cache.cell_state for cache in caches], dim=1)
        merged.is_initialized = True
        return merged

    def split_decoder_cache(
        self, cache: Nemotron3_5AsrRNNTDecoderCache, batch_size: int
    ) -> list[Nemotron3_5AsrRNNTDecoderCache]:
        request_caches = [
            Nemotron3_5AsrRNNTDecoderCache(self.model.config) for _ in range(batch_size)
        ]
        if not cache.is_initialized:
            return request_caches
        for batch_index, request_cache in enumerate(request_caches):
            request_cache.cache = cache.cache[batch_index : batch_index + 1].clone()
            request_cache.hidden_state = cache.hidden_state[
                :, batch_index : batch_index + 1
            ].clone()
            request_cache.cell_state = cache.cell_state[
                :, batch_index : batch_index + 1
            ].clone()
            request_cache.is_initialized = True
        return request_caches

    def run_streaming_batch(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        assert states and len(states) == len(chunks) == len(requested_languages)
        token_limits = (
            [None] * len(states) if max_new_tokens is None else max_new_tokens
        )
        assert len(token_limits) == len(states)
        assert all(
            limit is None or state.decoder_steps < limit
            for state, limit in zip(states, token_limits)
        )

        input_features = torch.cat([chunk.input_features for chunk in chunks], dim=0)
        prompt_ids = torch.cat([chunk.prompt_ids.reshape(-1) for chunk in chunks])
        attention_cache = self.merge_attention_caches(
            [state.attention_cache for state in states]
        )
        padding_cache = self.merge_padding_caches(
            [state.padding_cache for state in states]
        )
        self._synchronize_device()
        started_at_s = time.perf_counter()
        with self.model_lock, torch.inference_mode():
            encoder_outputs = self.model.get_audio_features(
                input_features=input_features,
                prompt_ids=prompt_ids,
                past_key_values=attention_cache,
                padding_cache=padding_cache,
                num_lookahead_tokens=self.processor.default_num_lookahead_tokens,
                use_cache=True,
                output_attention_mask=False,
            )
            split_attention = self.split_attention_cache(
                encoder_outputs.past_key_values, len(states)
            )
            split_padding = self.split_padding_cache(
                encoder_outputs.padding_cache, len(states)
            )
            for state, cache in zip(states, split_attention):
                state.attention_cache = cache
            for state, cache in zip(states, split_padding):
                state.padding_cache = cache

            encoded_frame_chunks = [
                encoder_outputs.pooler_output[index : index + 1]
                for index in range(len(states))
            ]
            local_frame_indices = [0] * len(states)
            for state, encoded_frames in zip(states, encoded_frame_chunks):
                state.encoder_frames += int(encoded_frames.shape[1])

            active_indices = list(range(len(states)))
            while active_indices:
                current_frames = torch.cat(
                    [
                        encoded_frame_chunks[index][
                            :,
                            local_frame_indices[index] : local_frame_indices[index] + 1,
                        ]
                        for index in active_indices
                    ],
                    dim=0,
                )
                decoder_input_ids = torch.tensor(
                    [[states[index].tokens[-1]] for index in active_indices],
                    dtype=torch.long,
                    device=self.device,
                )
                decoder_cache = self.merge_decoder_caches(
                    [states[index].decoder_cache for index in active_indices]
                )
                decoder_outputs = self.model(
                    encoder_outputs=NemotronAsrStreamingEncoderModelOutput(
                        pooler_output=current_frames
                    ),
                    decoder_input_ids=decoder_input_ids,
                    decoder_cache=decoder_cache,
                    use_decoder_cache=True,
                )
                # Note (Li Gang): blank decisions advance each request's frame independently.
                predicted_token_ids = (
                    decoder_outputs.logits[:, -1, :].argmax(dim=-1).tolist()
                )
                split_decoder = self.split_decoder_cache(
                    decoder_outputs.decoder_cache, len(active_indices)
                )
                next_active_indices: list[int] = []
                for row, state_index in enumerate(active_indices):
                    state = states[state_index]
                    state.decoder_cache = split_decoder[row]
                    token_id = int(predicted_token_ids[row])
                    state.tokens.append(token_id)
                    is_blank = token_id == int(self.model.config.blank_token_id)
                    symbols_at_frame = 0 if is_blank else state.symbols_at_frame + 1
                    should_advance_frame = is_blank or symbols_at_frame >= int(
                        self.model.max_symbols_per_step
                    )
                    state.symbols_at_frame = (
                        0 if should_advance_frame else symbols_at_frame
                    )
                    frame_advance = int(should_advance_frame)
                    state.durations.append(frame_advance)
                    state.decoder_steps += 1
                    local_frame_indices[state_index] += frame_advance
                    token_limit = token_limits[state_index]
                    if (
                        local_frame_indices[state_index]
                        >= encoded_frame_chunks[state_index].shape[1]
                    ):
                        continue
                    if token_limit is not None and state.decoder_steps >= token_limit:
                        continue
                    next_active_indices.append(state_index)
                active_indices = next_active_indices
        self._synchronize_device()
        elapsed_s = time.perf_counter() - started_at_s

        token_tensors = [
            torch.tensor(state.tokens, dtype=torch.long) for state in states
        ]
        raw_texts = [
            self.processor.batch_decode(tokens[None], skip_special_tokens=False)[0]
            for tokens in token_tensors
        ]
        clean_texts = [clean_nemotron_text(text) for text in raw_texts]
        languages = [
            resolve_nemotron_locale(raw_text, requested)
            for raw_text, requested in zip(raw_texts, requested_languages)
        ]
        return Nemotron3_5ASRStreamingBatchResult(
            elapsed_s=elapsed_s,
            raw_texts=raw_texts,
            clean_texts=clean_texts,
            languages=languages,
        )

    def generate_compatible_batch(
        self,
        requests: Sequence[Nemotron3_5ASRRequest],
        *,
        max_new_tokens: int | None,
    ) -> list[StagePayload]:
        processor_inputs = self.processor(
            [request.waveform for request in requests],
            sampling_rate=NEMOTRON_ASR_SAMPLE_RATE,
            language=[request.language for request in requests],
            padding="longest",
            return_tensors="pt",
        )
        started_at_s = time.perf_counter()
        with self.model_lock:
            sequences = self._generate_sequences(
                dict(processor_inputs), max_new_tokens=max_new_tokens
            )
        elapsed_s = time.perf_counter() - started_at_s
        raw_texts = self.processor.batch_decode(
            sequences,
            skip_special_tokens=False,
        )
        results: list[StagePayload] = []
        for request, raw_text in zip(requests, raw_texts, strict=True):
            stage_latency_s = (
                time.perf_counter() - request.started_at_s
                if request.started_at_s
                else elapsed_s
            )
            results.append(
                build_nemotron3_5_asr_result(
                    request.stage_payload,
                    raw_text=raw_text,
                    requested_language=request.language,
                    duration_s=request.duration_s,
                    asr_latency_s=stage_latency_s,
                    model_latency_s=elapsed_s,
                    extra_data={
                        "batch_size": len(requests),
                    },
                )
            )
        return results

    def _synchronize_device(self) -> None:
        # Include asynchronous accelerator work in streaming compute timings.
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()

    def _generate_sequences(self, processor_inputs, *, max_new_tokens):
        from transformers.feature_extraction_utils import BatchFeature

        model_inputs = BatchFeature(processor_inputs).to(
            device=self.device, dtype=self.dtype
        )
        kwargs = {"return_dict_in_generate": True}
        if max_new_tokens is not None:
            kwargs["max_new_tokens"] = max_new_tokens
        with torch.inference_mode():
            generated = self.model.generate(**model_inputs, **kwargs)
        return generated.sequences.detach().to("cpu")

    def run_batch(
        self, requests: Sequence[Nemotron3_5ASRRequest]
    ) -> list[StagePayload]:
        if not requests:
            return []

        # Note (Li Gang): generate applies one token limit to the entire batch.
        groups: dict[int | None, list[tuple[int, Nemotron3_5ASRRequest]]] = defaultdict(
            list
        )
        for index, request in enumerate(requests):
            groups[request.max_new_tokens].append((index, request))

        ordered_results: dict[int, StagePayload] = {}
        for max_new_tokens, indexed_requests in groups.items():
            compatible = [request for _, request in indexed_requests]
            batch_results = self.generate_compatible_batch(
                compatible,
                max_new_tokens=max_new_tokens,
            )
            for (index, _), result in zip(indexed_requests, batch_results, strict=True):
                ordered_results[index] = result
        return [ordered_results[index] for index in range(len(requests))]

    def close(self) -> None:
        del self.model, self.processor


__all__ = [
    "Nemotron3_5ASRDecodeState",
    "Nemotron3_5ASRModelRunner",
    "Nemotron3_5ASRPreparedChunk",
    "Nemotron3_5ASRStreamingBatchResult",
]
