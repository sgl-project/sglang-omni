# SPDX-License-Identifier: Apache-2.0
"""Reference-audio encoding for MOSS-TTS-Realtime."""

from __future__ import annotations

import concurrent.futures
import logging
import queue
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse

import numpy as np
import torch

from sglang_omni.preprocessing.cache_key import (
    hash_bytes,
    hash_media_item,
    reference_path_cache_key,
)
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeKey,
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.utils.audio import decode_audio_data_uri, load_audio

logger = logging.getLogger(__name__)

_MAX_AUDIO_SECONDS = 100.0


def _normalize_audio_source(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if not isinstance(value, Mapping):
        return value
    for key in ("audio", "audio_path", "path", "bytes"):
        candidate = value.get(key)
        if candidate is not None:
            return candidate
    encoded = value.get("base64") or value.get("data")
    if encoded is not None:
        if isinstance(encoded, str) and encoded.startswith("data:"):
            return encoded
        media_type = value.get("media_type") or "audio/wav"
        return f"data:{media_type};base64,{encoded}"
    raise ValueError("audio reference mapping contains no supported audio source")


def _local_audio_path(source: str) -> str | None:
    if source.startswith(("http://", "https://", "data:")):
        return None
    if source.startswith("file://"):
        return unquote(urlparse(source).path)
    return source


def _reference_input_key(source: Any) -> str | None:
    if isinstance(source, str):
        path = _local_audio_path(source)
        return None if path is None else reference_path_cache_key(path)
    try:
        return hash_media_item(source)
    except (TypeError, ValueError):
        return None


@dataclass(frozen=True)
class _MossTTSRealtimeReferenceInput:
    source: Any
    input_key: str | None


def _normalize_reference_input(value: Any) -> _MossTTSRealtimeReferenceInput:
    source = _normalize_audio_source(value)
    if isinstance(source, Path):
        source = str(source)
    if isinstance(source, memoryview):
        source = source.tobytes()
    elif isinstance(source, bytearray):
        source = bytes(source)
    if isinstance(source, str):
        decoded = decode_audio_data_uri(source)
        if decoded is not None:
            source = decoded
        else:
            path = _local_audio_path(source)
            if path is not None:
                source = path
    return _MossTTSRealtimeReferenceInput(
        source=source,
        input_key=_reference_input_key(source),
    )


def _reference_description(value: Any) -> str:
    source = _normalize_audio_source(value)
    if isinstance(source, str):
        if source.startswith("data:"):
            return "data-URI"
        if source.startswith(("http://", "https://")):
            parsed = urlparse(source)
            source = parsed._replace(query="", fragment="").geturl()
        return repr(source if len(source) <= 160 else f"{source[:157]}...")
    if isinstance(source, (bytes, bytearray, memoryview)):
        return f"{type(source).__name__}({len(source)} bytes)"
    if isinstance(source, torch.Tensor):
        return f"tensor(shape={tuple(source.shape)}, dtype={source.dtype})"
    if isinstance(source, np.ndarray):
        return f"ndarray(shape={source.shape}, dtype={source.dtype})"
    return type(source).__name__


class MossTTSRealtimeAudioEncoder:
    """Normalize audio and invoke the checkpoint codec encoder."""

    def __init__(
        self,
        codec: Any,
        *,
        device: str,
        num_quantizers: int | None = None,
    ) -> None:
        self.codec = codec
        self.device = torch.device(device)
        self.num_quantizers = None if num_quantizers is None else int(num_quantizers)
        if self.num_quantizers is not None and self.num_quantizers < 1:
            raise ValueError("num_quantizers must be positive")
        self._lock = threading.Lock()
        config = getattr(codec, "config", None)
        if config is None:
            raise ValueError("MOSS-TTS-Realtime codec must expose config")
        self.sample_rate = int(
            getattr(config, "sampling_rate", 0) or getattr(config, "sample_rate", 0)
        )
        if self.sample_rate < 1:
            raise ValueError("MOSS-TTS-Realtime codec sample rate must be positive")
        self.downsample_rate = int(getattr(config, "downsample_rate", 0) or 0)
        if self.downsample_rate < 1:
            raise ValueError("MOSS-TTS-Realtime codec downsample rate must be positive")

    def prepare_waveform(self, value: Any) -> torch.Tensor:
        source = _normalize_audio_source(value)
        if isinstance(source, torch.Tensor):
            waveform = source.detach().to(dtype=torch.float32, device="cpu")
        elif isinstance(source, np.ndarray):
            waveform = torch.from_numpy(np.ascontiguousarray(source, dtype=np.float32))
        else:
            waveform = torch.from_numpy(
                load_audio(
                    source,
                    source_name="MOSS-TTS-Realtime audio",
                    target_sample_rate=self.sample_rate,
                    mono=True,
                )
            )
        if waveform.ndim == 2:
            waveform = (
                waveform.mean(dim=0) if waveform.shape[0] > 1 else waveform.squeeze(0)
            )
        if waveform.ndim != 1:
            raise ValueError("audio waveform must normalize to mono rank 1")
        waveform = waveform.contiguous().to(dtype=torch.float32)
        if waveform.numel() == 0:
            raise ValueError("audio waveform must not be empty")
        duration_s = waveform.numel() / self.sample_rate
        if duration_s > _MAX_AUDIO_SECONDS:
            raise ValueError(
                f"audio input is {duration_s:.1f}s long, limit is "
                f"{_MAX_AUDIO_SECONDS:.0f}s"
            )
        return waveform

    def _quantizer_kwargs(self) -> dict[str, int]:
        if self.num_quantizers is None:
            return {}
        return {"num_quantizers": self.num_quantizers}

    def _prompt_code_lengths(self, waveforms: list[torch.Tensor]) -> list[int]:
        # Upstream MOSS-TTS-Realtime uses every raw codec frame after the codec
        # pads a waveform to downsample_rate. Derive that per-item ceil length
        # from the normalized waveform instead of the codec's floor-divided
        # audio_codes_lengths.
        return [
            (int(waveform.numel()) + self.downsample_rate - 1) // self.downsample_rate
            for waveform in waveforms
        ]

    def encode(self, value: Any) -> torch.Tensor:
        waveform = self.prepare_waveform(value).to(self.device)
        prompt_lengths = self._prompt_code_lengths([waveform])
        with self._lock, torch.inference_mode():
            output = self.codec.encode(
                waveform.unsqueeze(0),
                return_dict=True,
                **self._quantizer_kwargs(),
            )
        return self._split_codec_output(
            output,
            prompt_lengths=prompt_lengths,
        )[0]

    def encode_waveforms(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        if not waveforms:
            raise ValueError("waveforms must contain at least one waveform")
        prepared = [waveform.to(self.device) for waveform in waveforms]
        prompt_lengths = self._prompt_code_lengths(prepared)
        batch_encode = getattr(self.codec, "batch_encode", None)
        if callable(batch_encode):
            with self._lock, torch.inference_mode():
                output = batch_encode(prepared, **self._quantizer_kwargs())
            return self._split_codec_output(
                output,
                prompt_lengths=prompt_lengths,
            )

        results: list[torch.Tensor] = []
        with self._lock, torch.inference_mode():
            for waveform, prompt_length in zip(
                prepared,
                prompt_lengths,
                strict=True,
            ):
                output = self.codec.encode(
                    waveform.unsqueeze(0),
                    return_dict=True,
                    **self._quantizer_kwargs(),
                )
                results.extend(
                    self._split_codec_output(
                        output,
                        prompt_lengths=[prompt_length],
                    )
                )
        return results

    def _split_codec_output(
        self,
        output: Any,
        *,
        prompt_lengths: list[int],
    ) -> list[torch.Tensor]:
        batch_size = len(prompt_lengths)
        if isinstance(output, Mapping):
            audio_codes = output.get("audio_codes")
        else:
            audio_codes = getattr(output, "audio_codes", None)
        if not isinstance(audio_codes, torch.Tensor):
            raise RuntimeError(
                "MOSS-TTS-Realtime codec encode returned no audio_codes tensor"
            )
        if audio_codes.ndim != 3 or int(audio_codes.shape[1]) != batch_size:
            raise RuntimeError(
                "MOSS-TTS-Realtime codec audio_codes must have shape [Q, B, T]"
            )
        if self.num_quantizers is not None:
            if int(audio_codes.shape[0]) < self.num_quantizers:
                raise RuntimeError(
                    "MOSS-TTS-Realtime codec returned fewer quantizers than requested"
                )
            audio_codes = audio_codes[: self.num_quantizers]

        codes = audio_codes.detach().to(device="cpu", dtype=torch.long)
        results: list[torch.Tensor] = []
        raw_frames = int(codes.shape[-1])
        for index, length in enumerate(prompt_lengths):
            if length < 1:
                raise RuntimeError(
                    "MOSS-TTS-Realtime reference produced an invalid prompt length"
                )
            if length > raw_frames:
                raise RuntimeError(
                    "MOSS-TTS-Realtime codec returned fewer raw frames than "
                    "required by the reference waveform"
                )
            results.append(codes[:, index, :length].transpose(0, 1).contiguous())
        return results


class BatchedMossTTSRealtimeAudioEncoder:
    """Coalesce concurrent audio encodes into short codec microbatches."""

    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        encoder: MossTTSRealtimeAudioEncoder,
        *,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
    ) -> None:
        self._encoder = encoder
        self.sample_rate = encoder.sample_rate
        self.num_quantizers = encoder.num_quantizers
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._queue: queue.Queue[
            tuple[torch.Tensor, concurrent.futures.Future[torch.Tensor]]
        ] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker,
            name="moss-realtime-audio-encode",
            daemon=True,
        )
        self._thread.start()

    def encode(self, value: Any) -> torch.Tensor:
        waveform = self._encoder.prepare_waveform(value)
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        self._queue.put((waveform, future))
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _drain_batch(
        self,
    ) -> list[tuple[torch.Tensor, concurrent.futures.Future[torch.Tensor]]]:
        batch = [self._queue.get()]
        deadline = time.monotonic() + self._max_wait_s
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    batch.append(self._queue.get(timeout=remaining))
                else:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            results = self._encode_batch([waveform for waveform, _ in batch])
            for index, (_, future) in enumerate(batch):
                outcome = results.get(index)
                if isinstance(outcome, BaseException):
                    future.set_exception(
                        RuntimeError(f"audio encode failed: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(RuntimeError("audio encode produced no codes"))
                else:
                    future.set_result(outcome)

    def _encode_batch(self, waveforms: list[torch.Tensor]) -> dict[int, Any]:
        try:
            encoded = self._encoder.encode_waveforms(waveforms)
            if len(encoded) != len(waveforms):
                raise RuntimeError("codec batch result count does not match inputs")
            return dict(enumerate(encoded))
        except Exception:
            logger.exception(
                "MOSS-TTS-Realtime batched audio encode failed; retrying per item"
            )

        results: dict[int, Any] = {}
        for index, waveform in enumerate(waveforms):
            try:
                results[index] = self._encoder.encode_waveforms([waveform])[0]
            except Exception as exc:
                results[index] = exc
        return results


class _MossTTSRealtimeReferenceEncodeHook(
    TensorReferenceEncodeHook[_MossTTSRealtimeReferenceInput]
):
    model_id = "moss_tts_realtime"
    encoder_id = "moss_audio_tokenizer"
    artifact_kind = "moss_tts_realtime_reference_codes"
    storage_dtype = torch.int32
    output_dtype = torch.long

    def __init__(
        self,
        encoder: BatchedMossTTSRealtimeAudioEncoder,
        *,
        model_revision: str,
        num_quantizers: int,
    ) -> None:
        self._encoder = encoder
        self.model_revision = str(model_revision)
        self.encoder_config_hash = hash_bytes(
            (
                f"sample_rate:{encoder.sample_rate}:"
                f"num_quantizers:{int(num_quantizers)}"
            ).encode("utf-8")
        )

    def normalize_input(self, raw_input: Any) -> _MossTTSRealtimeReferenceInput:
        if isinstance(raw_input, _MossTTSRealtimeReferenceInput):
            return raw_input
        return _normalize_reference_input(raw_input)

    def encode_one(self, item: _MossTTSRealtimeReferenceInput) -> torch.Tensor:
        return self._encoder.encode(item.source)

    def input_key(self, item: _MossTTSRealtimeReferenceInput) -> str | None:
        return item.input_key

    def revalidate(
        self,
        item: _MossTTSRealtimeReferenceInput,
        key: ReferenceEncodeKey,
    ) -> bool:
        return _reference_input_key(item.source) == key.input_key


class MossTTSRealtimeReferenceEncoder:
    """Bounded CPU cache and single-flight wrapper for voice references."""

    def __init__(
        self,
        encoder: BatchedMossTTSRealtimeAudioEncoder,
        *,
        model_revision: str,
        num_quantizers: int,
        max_items: int | None = 8192,
        max_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        self._service = ReferenceEncodeService(
            _MossTTSRealtimeReferenceEncodeHook(
                encoder,
                model_revision=model_revision,
                num_quantizers=num_quantizers,
            ),
            max_items=max_items,
            max_bytes=max_bytes,
            timeout_s=BatchedMossTTSRealtimeAudioEncoder.ENCODE_TIMEOUT_S + 10,
            log_prefix="MOSS-TTS-Realtime ref cache",
        )

    def encode(self, value: Any) -> torch.Tensor:
        return self._service.get_or_encode(
            value,
            desc=_reference_description(value),
        )

    def stats(self) -> dict[str, int]:
        return self._service.stats()
