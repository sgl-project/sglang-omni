# SPDX-License-Identifier: Apache-2.0
"""Model-owned batched RNN-T inference for Nemotron 3.5 ASR."""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Sequence
from copy import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
)
from .hf_compat.generation_nemotron3_5_asr import Nemotron3_5AsrRNNTDecoderCache
from .hf_compat.modeling_nemotron_asr_streaming import (
    NemotronAsrStreamingEncoderCausalConvPaddingCache,
    NemotronAsrStreamingEncoderModelOutput,
)
from .request_builders import NEMOTRON_ASR_SAMPLE_RATE, Nemotron3_5ASRRequest
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
    attention_cache: Any = None
    padding_cache: Any = None
    decoder_cache: Any = None


@dataclass(slots=True)
class Nemotron3_5ASRStreamingBatchResult:
    elapsed_s: float
    raw_texts: list[str]
    clean_texts: list[str]
    languages: list[str | None]
    emitted_token_counts: list[int]
    encoder_frame_counts: list[int]


def _move_model_inputs(
    inputs: dict[str, Any],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Any]:
    moved: dict[str, Any] = {}
    for name, value in inputs.items():
        if not isinstance(value, torch.Tensor):
            moved[name] = value
        elif value.is_floating_point():
            moved[name] = value.to(device=device, dtype=dtype)
        else:
            moved[name] = value.to(device=device)
    return moved


class Nemotron3_5ASRModelRunner:
    """Own one processor/model pair and serialize its mutable generate path."""

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
        # Upstream generate stores encoder/decoder progress on the model. Even
        # callers outside SimpleScheduler must never overlap calls.
        self._model_lock = threading.Lock()

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
        if input_features.shape[1] < required_frames:
            raise ValueError(
                f"Nemotron processor returned {input_features.shape[1]} frames; "
                f"the {'first' if is_first else 'subsequent'} streaming chunk "
                f"requires {required_frames}"
            )
        if is_first:
            input_features = input_features[:, :required_frames, :]
        elif input_features.shape[1] != required_frames:
            raise ValueError(
                f"Nemotron processor returned {input_features.shape[1]} frames; "
                f"the subsequent streaming chunk requires exactly {required_frames}"
            )
        return Nemotron3_5ASRPreparedChunk(
            input_features=input_features.to(device=self.device, dtype=self.dtype),
            prompt_ids=processor_inputs.prompt_ids.to(device=self.device),
        )

    @staticmethod
    def _merge_attention_caches(caches: Sequence[Any]) -> Any:
        if all(cache is None for cache in caches):
            return None
        if any(cache is None for cache in caches):
            raise RuntimeError(
                "Cannot batch initialized and uninitialized attention caches"
            )
        layer_count = len(caches[0].layers)
        if any(len(cache.layers) != layer_count for cache in caches):
            raise RuntimeError("Nemotron attention cache layer counts do not match")
        data = []
        for layer_index in range(layer_count):
            layers = [cache.layers[layer_index] for cache in caches]
            sequence_lengths = {layer.get_seq_length() for layer in layers}
            if len(sequence_lengths) != 1:
                raise RuntimeError(
                    "Nemotron attention caches can only batch at the same chunk index"
                )
            data.append(
                (
                    torch.cat([layer.keys for layer in layers], dim=0),
                    torch.cat([layer.values for layer in layers], dim=0),
                )
            )
        return DynamicCache(ddp_cache_data=data)

    @staticmethod
    def _split_attention_cache(cache: Any, batch_size: int) -> list[Any]:
        split: list[Any] = []
        for batch_index in range(batch_size):
            data = [
                (
                    layer.keys[batch_index : batch_index + 1].clone(),
                    layer.values[batch_index : batch_index + 1].clone(),
                )
                for layer in cache.layers
            ]
            split.append(DynamicCache(ddp_cache_data=data))
        return split

    @staticmethod
    def _merge_padding_caches(caches: Sequence[Any]) -> Any:
        if all(cache is None for cache in caches):
            return None
        if any(cache is None for cache in caches):
            raise RuntimeError("Cannot batch initialized and uninitialized conv caches")
        keys = list(caches[0].layers)
        if any(list(cache.layers) != keys for cache in caches):
            raise RuntimeError("Nemotron conv cache layouts do not match")
        merged = NemotronAsrStreamingEncoderCausalConvPaddingCache()
        for key in keys:
            source_layers = [cache.layers[key] for cache in caches]
            layer = copy(source_layers[0])
            layer.cache = torch.cat([source.cache for source in source_layers], dim=0)
            merged.layers[key] = layer
        return merged

    @staticmethod
    def _split_padding_cache(cache: Any, batch_size: int) -> list[Any]:
        split = [
            NemotronAsrStreamingEncoderCausalConvPaddingCache()
            for _ in range(batch_size)
        ]
        for key, source in cache.layers.items():
            for batch_index, target in enumerate(split):
                layer = copy(source)
                layer.cache = source.cache[batch_index : batch_index + 1].clone()
                target.layers[key] = layer
        return split

    def _merge_decoder_caches(self, caches: Sequence[Any]) -> Any:
        initialized = [cache is not None and cache.is_initialized for cache in caches]
        merged = Nemotron3_5AsrRNNTDecoderCache(self.model.config)
        if not any(initialized):
            return merged
        if not all(initialized):
            raise RuntimeError("Cannot batch initialized and uninitialized RNNT caches")
        merged.cache = torch.cat([cache.cache for cache in caches], dim=0)
        merged.hidden_state = torch.cat([cache.hidden_state for cache in caches], dim=1)
        merged.cell_state = torch.cat([cache.cell_state for cache in caches], dim=1)
        merged.is_initialized = True
        return merged

    def _split_decoder_cache(self, cache: Any, batch_size: int) -> list[Any]:
        split = []
        for batch_index in range(batch_size):
            item = Nemotron3_5AsrRNNTDecoderCache(self.model.config)
            if cache.is_initialized:
                item.cache = cache.cache[batch_index : batch_index + 1].clone()
                item.hidden_state = cache.hidden_state[
                    :, batch_index : batch_index + 1
                ].clone()
                item.cell_state = cache.cell_state[
                    :, batch_index : batch_index + 1
                ].clone()
                item.is_initialized = True
            split.append(item)
        return split

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def run_streaming_batch(
        self,
        states: Sequence[Nemotron3_5ASRDecodeState],
        chunks: Sequence[Nemotron3_5ASRPreparedChunk],
        *,
        requested_languages: Sequence[str],
        max_new_tokens: Sequence[int | None] | None = None,
    ) -> Nemotron3_5ASRStreamingBatchResult:
        if not states or len(states) != len(chunks):
            raise ValueError(
                "Nemotron streaming batch inputs must be non-empty and aligned"
            )
        if len(requested_languages) != len(states):
            raise ValueError("Nemotron streaming language batch is not aligned")
        if max_new_tokens is None:
            max_new_tokens = [None] * len(states)
        if len(max_new_tokens) != len(states):
            raise ValueError("Nemotron streaming token limits are not aligned")
        if any(
            limit is not None and state.decoder_steps >= limit
            for state, limit in zip(states, max_new_tokens)
        ):
            raise ValueError(
                "Nemotron streaming batch contains an exhausted token limit"
            )

        features = torch.cat([chunk.input_features for chunk in chunks], dim=0)
        prompt_ids = torch.cat([chunk.prompt_ids.reshape(-1) for chunk in chunks])
        attention_cache = self._merge_attention_caches(
            [state.attention_cache for state in states]
        )
        padding_cache = self._merge_padding_caches(
            [state.padding_cache for state in states]
        )
        token_counts_before = [len(state.tokens) for state in states]
        self._synchronize()
        started_at_s = time.perf_counter()
        with self._model_lock, torch.inference_mode():
            encoded = self.model.get_audio_features(
                input_features=features,
                prompt_ids=prompt_ids,
                past_key_values=attention_cache,
                padding_cache=padding_cache,
                num_lookahead_tokens=self.processor.default_num_lookahead_tokens,
                use_cache=True,
                output_attention_mask=False,
            )
            split_attention = self._split_attention_cache(
                encoded.past_key_values, len(states)
            )
            split_padding = self._split_padding_cache(
                encoded.padding_cache, len(states)
            )
            for state, cache in zip(states, split_attention):
                state.attention_cache = cache
            for state, cache in zip(states, split_padding):
                state.padding_cache = cache

            pooler_chunks = [
                encoded.pooler_output[index : index + 1] for index in range(len(states))
            ]
            local_frame_indices = [0] * len(states)
            for state, pooler in zip(states, pooler_chunks):
                state.encoder_frames += int(pooler.shape[1])

            active = list(range(len(states)))
            max_steps = sum(int(pooler.shape[1]) for pooler in pooler_chunks) * int(
                self.model.max_symbols_per_step
            ) + len(states)
            batch_steps = 0
            while active:
                if batch_steps >= max_steps:
                    raise RuntimeError(
                        f"Nemotron streaming decoder exceeded {max_steps} batched steps"
                    )
                current_frames = torch.cat(
                    [
                        pooler_chunks[index][
                            :,
                            local_frame_indices[index] : local_frame_indices[index] + 1,
                        ]
                        for index in active
                    ],
                    dim=0,
                )
                decoder_input_ids = torch.tensor(
                    [[states[index].tokens[-1]] for index in active],
                    dtype=torch.long,
                    device=self.device,
                )
                decoder_cache = self._merge_decoder_caches(
                    [states[index].decoder_cache for index in active]
                )
                outputs = self.model(
                    encoder_outputs=NemotronAsrStreamingEncoderModelOutput(
                        pooler_output=current_frames
                    ),
                    decoder_input_ids=decoder_input_ids,
                    decoder_cache=decoder_cache,
                    use_decoder_cache=True,
                )
                predicted = outputs.logits[:, -1, :].argmax(dim=-1).tolist()
                split_decoder = self._split_decoder_cache(
                    outputs.decoder_cache, len(active)
                )
                next_active: list[int] = []
                for row, state_index in enumerate(active):
                    state = states[state_index]
                    state.decoder_cache = split_decoder[row]
                    token = int(predicted[row])
                    state.tokens.append(token)
                    is_blank = token == int(self.model.config.blank_token_id)
                    symbols = 0 if is_blank else state.symbols_at_frame + 1
                    force_advance = symbols >= int(self.model.max_symbols_per_step)
                    state.symbols_at_frame = 0 if is_blank or force_advance else symbols
                    advance = int(is_blank or force_advance)
                    state.durations.append(advance)
                    state.decoder_steps += 1
                    local_frame_indices[state_index] += advance
                    token_limit = max_new_tokens[state_index]
                    if local_frame_indices[state_index] < pooler_chunks[
                        state_index
                    ].shape[1] and (
                        token_limit is None or state.decoder_steps < token_limit
                    ):
                        next_active.append(state_index)
                active = next_active
                batch_steps += 1
        self._synchronize()
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
            emitted_token_counts=[
                len(state.tokens) - before
                for state, before in zip(states, token_counts_before)
            ],
            encoder_frame_counts=[int(pooler.shape[1]) for pooler in pooler_chunks],
        )

    def _generate_compatible_batch(
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
        model_inputs = _move_model_inputs(
            dict(processor_inputs), device=self.device, dtype=self.dtype
        )
        generate_kwargs: dict[str, Any] = {"return_dict_in_generate": True}
        if max_new_tokens is not None:
            generate_kwargs["max_new_tokens"] = max_new_tokens

        started_at_s = time.perf_counter()
        with self._model_lock, torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generate_kwargs)
        elapsed_s = time.perf_counter() - started_at_s
        sequences = generated.sequences.detach().to("cpu")
        raw_texts = self.processor.batch_decode(
            sequences,
            skip_special_tokens=False,
        )
        if len(raw_texts) != len(requests):
            raise RuntimeError(
                "Nemotron processor returned "
                f"{len(raw_texts)} transcripts for {len(requests)} requests"
            )

        results: list[StagePayload] = []
        for request, raw_text in zip(requests, raw_texts):
            payload = request.stage_payload
            stage_latency_s = (
                time.perf_counter() - request.started_at_s
                if request.started_at_s
                else elapsed_s
            )
            results.append(
                StagePayload(
                    request_id=payload.request_id,
                    request=payload.request,
                    data={
                        "text": str(raw_text).strip(),
                        "language": request.language,
                        "duration_s": request.duration_s,
                        "asr_latency_s": stage_latency_s,
                        "model_latency_s": elapsed_s,
                        "batch_size": len(requests),
                        "usage": {"engine_time_s": stage_latency_s},
                        "modality": "text",
                    },
                )
            )
        return results

    def run_batch(
        self, requests: Sequence[Nemotron3_5ASRRequest]
    ) -> list[StagePayload]:
        if not requests:
            return []

        # GenerationConfig has one output cap for a whole tensor batch. Keep
        # explicit per-request caps exact by batching only compatible requests;
        # the normal endpoint path (no override) remains one true generate call.
        groups: OrderedDict[int | None, list[tuple[int, Nemotron3_5ASRRequest]]] = (
            OrderedDict()
        )
        for index, request in enumerate(requests):
            groups.setdefault(request.max_new_tokens, []).append((index, request))

        ordered_results: list[StagePayload | None] = [None] * len(requests)
        for max_new_tokens, indexed_requests in groups.items():
            compatible = [request for _, request in indexed_requests]
            batch_results = self._generate_compatible_batch(
                compatible,
                max_new_tokens=max_new_tokens,
            )
            for (index, _), result in zip(indexed_requests, batch_results):
                ordered_results[index] = result
        if any(result is None for result in ordered_results):
            raise RuntimeError("Nemotron batch result ordering was incomplete")
        return [result for result in ordered_results if result is not None]

    def run_one(self, request: Nemotron3_5ASRRequest) -> StagePayload:
        return self.run_batch([request])[0]

    def close(self) -> None:
        # The worker process normally exits after shutdown; dropping references
        # here also makes explicit scheduler teardown release model ownership.
        self.model = None  # type: ignore[assignment]
        self.processor = None  # type: ignore[assignment]


__all__ = [
    "Nemotron3_5ASRDecodeState",
    "Nemotron3_5ASRModelRunner",
    "Nemotron3_5ASRPreparedChunk",
    "Nemotron3_5ASRStreamingBatchResult",
]
