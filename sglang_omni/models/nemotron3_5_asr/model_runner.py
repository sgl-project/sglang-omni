# SPDX-License-Identifier: Apache-2.0
# ruff: noqa: N801  # Keep the Nemotron 3.5 API spelling.
"""Model-owned batched RNN-T inference for Nemotron 3.5 ASR."""

from __future__ import annotations

import threading
import time
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from transformers.cache_utils import DynamicCache

from sglang_omni.models.nemotron3_5_asr.decoder import (
    Nemotron3_5ASRDecodeState,
    decode_streaming_batch,
)
from sglang_omni.models.nemotron3_5_asr.encoder import encode_streaming_batch
from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
    Nemotron3_5AsrProcessor,
    Nemotron3_5AsrRNNTDecoderCache,
)
from sglang_omni.models.nemotron3_5_asr.request_builders import (
    NEMOTRON_ASR_SAMPLE_RATE,
    Nemotron3_5ASRRequest,
    build_nemotron3_5_asr_result,
)
from sglang_omni.models.nemotron3_5_asr.text import (
    clean_nemotron_text,
    resolve_nemotron_locale,
)
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.proto import StagePayload
from sglang_omni.utils.checkpoint import resolve_checkpoint


@dataclass(slots=True)
class Nemotron3_5ASRPreparedChunk:
    input_features: torch.Tensor
    prompt_ids: torch.Tensor


@dataclass(slots=True)
class Nemotron3_5ASRStreamingBatchResult:
    elapsed_s: float
    raw_texts: list[str]
    clean_texts: list[str]
    languages: list[str | None]
    errors: list[Exception | None] | None = None


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
        else:
            pass

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

    @property
    def streaming_state_budget_bytes(self) -> int:
        # note (Li Gang): Reserve cache growth before creating a session, including convolution state.
        config = self.model.config.encoder_config
        cache_frames = config.sliding_window + self.processor.num_mel_frames_per_audio_chunk + 1
        attention_bytes = 2 * config.num_hidden_layers * config.hidden_size * cache_frames * self.dtype.itemsize
        return attention_bytes + 16 * 1024 * 1024

    def new_streaming_decode_state(self) -> Nemotron3_5ASRDecodeState:
        blank_token_id = int(self.model.config.blank_token_id)
        return Nemotron3_5ASRDecodeState(
            tokens=[blank_token_id],
            durations=[0],
            attention_cache=DynamicCache(config=self.model.config.encoder_config),
            decoder_cache=Nemotron3_5AsrRNNTDecoderCache(self.model.config),
        )

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

        with self.model_lock, torch.inference_mode():
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            else:
                pass
            started_at_s = time.perf_counter()
            encoded_frames = encode_streaming_batch(
                self.model,
                [chunk.input_features for chunk in chunks],
                torch.cat([chunk.prompt_ids.reshape(-1) for chunk in chunks]),
                attention_caches=[state.attention_cache for state in states],
                padding_caches=[state.padding_cache for state in states],
                num_lookahead_tokens=self.processor.default_num_lookahead_tokens,
            )
            decode_streaming_batch(self.model, states, encoded_frames, token_limits)
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
            else:
                pass
            elapsed_s = time.perf_counter() - started_at_s

        raw_texts: list[str] = []
        clean_texts: list[str] = []
        languages: list[str | None] = []
        errors: list[Exception | None] = []
        for state, requested in zip(states, requested_languages, strict=True):
            try:
                raw_text = self.processor.batch_decode(
                    [torch.tensor(state.tokens, dtype=torch.long)],
                    skip_special_tokens=False,
                )[0]
                clean_text = clean_nemotron_text(raw_text)
                language = resolve_nemotron_locale(raw_text, requested)
            except Exception as exc:
                raw_texts.append("")
                clean_texts.append("")
                languages.append(None)
                errors.append(exc)
            else:
                raw_texts.append(raw_text)
                clean_texts.append(clean_text)
                languages.append(language)
                errors.append(None)
        return Nemotron3_5ASRStreamingBatchResult(
            elapsed_s=elapsed_s, raw_texts=raw_texts, clean_texts=clean_texts,
            languages=languages, errors=errors,
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
        model_inputs = processor_inputs.to(device=self.device, dtype=self.dtype)
        if max_new_tokens is not None:
            generate_kwargs = {
                "return_dict_in_generate": True,
                "max_new_tokens": max_new_tokens,
            }
        else:
            generate_kwargs = {"return_dict_in_generate": True}

        started_at_s = time.perf_counter()
        with self.model_lock, torch.inference_mode():
            generated = self.model.generate(**model_inputs, **generate_kwargs)
        elapsed_s = time.perf_counter() - started_at_s
        sequences = generated.sequences.detach().to("cpu")
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

    def run_batch(
        self, requests: Sequence[Nemotron3_5ASRRequest]
    ) -> list[StagePayload]:
        if not requests:
            return []
        else:
            pass

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
    "Nemotron3_5ASRModelRunner",
    "Nemotron3_5ASRPreparedChunk",
    "Nemotron3_5ASRStreamingBatchResult",
]
