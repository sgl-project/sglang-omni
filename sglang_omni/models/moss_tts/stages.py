# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Delay pipeline."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any, TypeAlias, cast

import torch
from transformers import AutoConfig, AutoTokenizer

from sglang_omni.models.moss_tts.audio_tokenizer import (
    DEFAULT_MOSS_TTS_AUDIO_TOKENIZER,
    MossTTSAudioTokenizer,
    load_moss_tts_audio_tokenizer,
)
from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
)
from sglang_omni.models.moss_tts.payload_types import moss_tts_special_token_defaults
from sglang_omni.models.moss_tts.request_builders import (
    cleanup_prepared_moss_tts_request,
    preprocess_moss_tts_payload,
    set_moss_tts_preprocessing_context,
)
from sglang_omni.models.moss_tts.streaming_vocoder import MossStreamingVocoderScheduler
from sglang_omni.models.moss_tts.vocoder import MossTTSVocoder
from sglang_omni.preprocessing.cache_key import hash_bytes
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeService,
    TensorReferenceEncodeHook,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.audio import audio_fingerprint, load_audio

logger = logging.getLogger(__name__)

_MOSS_TTS_INSTALL_HINT = (
    "MOSS-TTS support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer."
)
_MAX_REFERENCE_SECONDS = 100.0
_MOSS_TTS_REFERENCE_ENCODE_STOP = object()


@dataclass(frozen=True, eq=False)
class _LoadedReferenceWaveform:
    waveform: torch.Tensor
    sample_rate: int
    content_key: str


_ReferenceEncodeQueueEntry: TypeAlias = tuple[
    _LoadedReferenceWaveform,
    concurrent.futures.Future[torch.Tensor],
]


_COMPUTE_DTYPES: dict[str, torch.dtype] = {
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


def _resolve_compute_dtype(
    dtype: str | torch.dtype | None,
) -> torch.dtype | None:
    if dtype is None:
        return None
    if isinstance(dtype, str):
        try:
            return _COMPUTE_DTYPES[dtype.lower()]
        except KeyError:
            pass
    elif isinstance(dtype, torch.dtype) and dtype in (torch.float32, torch.bfloat16):
        return dtype
    raise ValueError(f"compute_dtype must be float32, bfloat16, or null; got {dtype!r}")


def _normalize_moss_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _audio_tokenizer_model_path_from_processor_dict(
    processor_dict: dict[str, Any],
) -> str | None:
    model_path = processor_dict.get("audio_tokenizer_name_or_path")
    audio_tokenizer_dict = processor_dict.get("audio_tokenizer")
    if isinstance(audio_tokenizer_dict, dict):
        model_path = (
            audio_tokenizer_dict.get("audio_tokenizer_name_or_path") or model_path
        )
    return str(model_path) if model_path else None


def _load_moss_processor(
    model_path: str,
) -> Any:
    logger.info(f"Loading MOSS-TTS processor from {model_path} without codec")
    try:
        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(model_path)
            processor_dict, _ = processor_cls.get_processor_dict(model_path)
            audio_tokenizer_model_path = (
                _audio_tokenizer_model_path_from_processor_dict(processor_dict)
            )
            model_config = AutoConfig.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            if audio_tokenizer_model_path:
                # processor_cls.from_pretrained normally resolves this metadata
                # before loading the codec. Preserve the same selection while
                # constructing the processor without a codec instance.
                model_config.audio_tokenizer_name_or_path = audio_tokenizer_model_path
            tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
            )
            processor = processor_cls(
                tokenizer=tokenizer,
                audio_tokenizer=None,
                model_config=model_config,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_INSTALL_HINT) from exc

    _normalize_moss_processor_config(processor)
    return processor


def _resolve_audio_tokenizer_model_path(
    processor: Any,
    codec_model_path: str | None,
) -> str:
    return str(
        codec_model_path
        or getattr(processor.model_config, "audio_tokenizer_name_or_path", None)
        or DEFAULT_MOSS_TTS_AUDIO_TOKENIZER
    )


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


class _BatchedReferenceEncoder:
    """Coalesce concurrent Delay reference encodes into codec batches."""

    MAX_REFERENCE_SECONDS = _MAX_REFERENCE_SECONDS
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        audio_tokenizer: MossTTSAudioTokenizer,
        *,
        n_vq: int,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
    ) -> None:
        self._audio_tokenizer = audio_tokenizer
        self._n_vq = int(n_vq)
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._queue: queue.Queue[object] = queue.Queue()
        self._lifecycle_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._worker,
            name="moss-tts-ref-encode",
            daemon=True,
        )
        self._thread.start()

    def close(self) -> None:
        """Stop the codec worker after all already-queued jobs finish."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_MOSS_TTS_REFERENCE_ENCODE_STOP)
        self._thread.join(timeout=5.0)

    def load(self, source: str | os.PathLike[str]) -> _LoadedReferenceWaveform:
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MOSS-TTS reference encoder is closed")
        return _load_reference_waveform(self._audio_tokenizer, source)

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        return self.encode_input(self.load(source))

    def encode_input(self, item: _LoadedReferenceWaveform) -> torch.Tensor:
        future: concurrent.futures.Future[torch.Tensor] = concurrent.futures.Future()
        with self._lifecycle_lock:
            if self._closed:
                raise RuntimeError("MOSS-TTS reference encoder is closed")
            self._queue.put((item, future))
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _drain_batch(
        self,
    ) -> tuple[list[_ReferenceEncodeQueueEntry], bool]:
        first = self._queue.get()
        if first is _MOSS_TTS_REFERENCE_ENCODE_STOP:
            return [], True
        batch = [cast(_ReferenceEncodeQueueEntry, first)]
        deadline = time.monotonic() + self._max_wait_s
        shutdown = False
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        break
                    queued = self._queue.get(timeout=remaining)
                else:
                    queued = self._queue.get_nowait()
            except queue.Empty:
                break
            if queued is _MOSS_TTS_REFERENCE_ENCODE_STOP:
                shutdown = True
                break
            batch.append(cast(_ReferenceEncodeQueueEntry, queued))
        return batch, shutdown

    def _worker(self) -> None:
        while True:
            batch, shutdown = self._drain_batch()
            if not batch:
                return
            try:
                results = self._encode_batch(batch)
            except BaseException as exc:
                logger.exception("MOSS-TTS reference encode worker failed")
                results = {index: exc for index in range(len(batch))}
            for index, (_, future) in enumerate(batch):
                outcome = results.get(index)
                if isinstance(outcome, BaseException):
                    future.set_exception(
                        RuntimeError(f"reference encode failed: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError("reference encode produced no codes")
                    )
                else:
                    future.set_result(outcome)
            if shutdown:
                return

    def _encode_batch(
        self,
        batch: list[_ReferenceEncodeQueueEntry],
    ) -> dict[int, torch.Tensor | BaseException]:
        results: dict[int, torch.Tensor | BaseException] = {}
        group_indices: list[list[int]] = []
        waveforms: list[tuple[torch.Tensor, int]] = []
        content_to_group: dict[str, int] = {}
        for index, (job, _) in enumerate(batch):
            group = content_to_group.get(job.content_key)
            if group is None:
                group = len(waveforms)
                waveforms.append((job.waveform, job.sample_rate))
                group_indices.append([])
                content_to_group[job.content_key] = group
            group_indices[group].append(index)

        try:
            encoded = self._audio_tokenizer.encode_waveforms(
                waveforms,
                num_quantizers=self._n_vq,
            )
            if len(encoded) != len(waveforms):
                raise RuntimeError(
                    "MOSS-TTS audio tokenizer returned an unexpected batch size: "
                    f"{len(encoded)} != {len(waveforms)}"
                )
            for indices, codes in zip(group_indices, encoded):
                for index in indices:
                    results[index] = codes
            return results
        except Exception:
            logger.exception(
                "MOSS-TTS batched reference encode failed; retrying per item"
            )

        for indices, waveform in zip(group_indices, waveforms):
            try:
                codes = self._audio_tokenizer.encode_waveforms(
                    [waveform],
                    num_quantizers=self._n_vq,
                )[0]
            except Exception as exc:
                codes = exc
            for index in indices:
                results[index] = codes
        return results


def _load_reference_waveform(
    audio_tokenizer: MossTTSAudioTokenizer,
    source: str | os.PathLike[str],
) -> _LoadedReferenceWaveform:
    """Load once through the shared resolver and key the exact codec input."""

    waveform = load_audio(
        os.fsdecode(source),
        source_name="MOSS-TTS reference",
        target_sample_rate=audio_tokenizer.sample_rate,
        mono=True,
    )
    duration = len(waveform) / max(audio_tokenizer.sample_rate, 1)
    if duration > _MAX_REFERENCE_SECONDS:
        raise ValueError(
            f"reference audio is {duration:.1f}s long, limit is "
            f"{_MAX_REFERENCE_SECONDS:.0f}s"
        )
    return _LoadedReferenceWaveform(
        torch.from_numpy(waveform),
        audio_tokenizer.sample_rate,
        f"waveform:{audio_fingerprint(waveform)}",
    )


class _MossTTSReferenceEncodeHook(TensorReferenceEncodeHook[_LoadedReferenceWaveform]):
    model_id = "moss_tts_delay"
    encoder_id = "moss_audio_tokenizer"
    artifact_kind = "moss_tts_reference_codes"
    storage_dtype = torch.int32
    output_dtype = torch.long

    def __init__(
        self,
        encoder: _BatchedReferenceEncoder,
        *,
        codec_model_path: str,
        n_vq: int,
    ) -> None:
        self._encoder = encoder
        self.model_revision = str(codec_model_path)
        model = getattr(encoder._audio_tokenizer, "model", None)
        try:
            parameter_dtype = str(next(model.parameters()).dtype)
        except (AttributeError, StopIteration):
            parameter_dtype = "unknown"
        config = (
            f"n_vq:{int(n_vq)}|sample_rate:"
            f"{getattr(encoder._audio_tokenizer, 'sample_rate', 'unknown')}|device:"
            f"{getattr(encoder._audio_tokenizer, 'device', 'unknown')}|dtype:"
            f"{parameter_dtype}"
        )
        self.encoder_config_hash = hash_bytes(config.encode("utf-8"))

    def normalize_input(self, raw_input: Any) -> _LoadedReferenceWaveform:
        if isinstance(raw_input, _LoadedReferenceWaveform):
            return raw_input
        # The service needs content identity before lookup; derive it only after
        # load_audio has resolved and normalized the source.
        return self._encoder.load(raw_input)

    def input_key(self, item: _LoadedReferenceWaveform) -> str:
        return item.content_key

    def encode_one(self, item: _LoadedReferenceWaveform) -> torch.Tensor:
        return self._encoder.encode_input(item)

    def close(self) -> None:
        close = getattr(self._encoder, "close", None)
        if callable(close):
            close()


class _MossTTSReferenceEncoder:
    """Load each source, then cache/merge codec work by waveform identity."""

    def __init__(
        self,
        encoder: _BatchedReferenceEncoder,
        *,
        codec_model_path: str,
        n_vq: int,
        max_items: int | None = 8192,
        max_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        self._service = ReferenceEncodeService(
            _MossTTSReferenceEncodeHook(
                encoder,
                codec_model_path=codec_model_path,
                n_vq=n_vq,
            ),
            max_items=max_items,
            max_bytes=max_bytes,
            timeout_s=_BatchedReferenceEncoder.ENCODE_TIMEOUT_S + 10,
            log_prefix="MOSS-TTS ref cache",
        )

    def encode(self, source: str | os.PathLike[str]) -> torch.Tensor:
        source = os.fsdecode(source)
        return self._service.get_or_encode(
            source,
            desc="data-URI" if source.startswith("data:") else repr(source),
        )

    def stats(self) -> dict[str, int]:
        return self._service.stats()

    def close(self) -> None:
        self._service.close()


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    for name, value in (
        ("ref_audio_cache_max_items", ref_audio_cache_max_items),
        ("ref_audio_cache_max_bytes", ref_audio_cache_max_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"{name} must be an integer >= 1; got {value!r}")

    env_toggle = os.environ.get("MOSS_REF_AUDIO_CACHE")
    if env_toggle is not None:
        ref_audio_cache = env_toggle.strip().lower() not in (
            "0",
            "false",
            "no",
            "off",
            "",
        )
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_processor(model_path)
    resolved_codec_model_path = _resolve_audio_tokenizer_model_path(
        processor,
        codec_model_path,
    )
    audio_tokenizer = load_moss_tts_audio_tokenizer(
        resolved_codec_model_path,
        device=device,
        dtype=dtype,
    )
    reference_encoder: Any = _BatchedReferenceEncoder(
        audio_tokenizer,
        n_vq=int(processor.model_config.n_vq),
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
    )
    if ref_audio_cache:
        reference_encoder = _MossTTSReferenceEncoder(
            reference_encoder,
            codec_model_path=resolved_codec_model_path,
            n_vq=int(processor.model_config.n_vq),
            max_items=ref_audio_cache_max_items,
            max_bytes=ref_audio_cache_max_bytes,
        )
    set_moss_tts_preprocessing_context(
        processor=processor,
        reference_encoder=reference_encoder,
    )
    # note (Zhang Yiyang): Every device uses the same batch queue; there is no
    # device-specific fallback.
    return SimpleScheduler(
        preprocess_moss_tts_payload,
        abort_callback=cleanup_prepared_moss_tts_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
    return MossTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


create_tts_engine_executor = create_sglang_tts_engine_executor


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    dtype: str = "float32",
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 8,
    stream_followup_stride: int = 8,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 1,
    initial_chunk_frames: int = 0,
    compute_dtype: str | torch.dtype | None = "bfloat16",
) -> MossStreamingVocoderScheduler:
    # An explicit device is a model policy/user override; gpu_id is only the
    # placement-derived fallback. This matches preprocessing resolution and
    # permits a CPU-vocoder escape hatch on especially constrained hardware.
    device = _resolve_codec_device(device, gpu_id)
    resolved_compute_dtype = _resolve_compute_dtype(compute_dtype)
    processor = _load_moss_processor(model_path)
    audio_tokenizer = load_moss_tts_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
        dtype=dtype,
    )

    vocoder = MossTTSVocoder(
        processor,
        audio_tokenizer,
        device,
        compute_dtype=resolved_compute_dtype,
        max_segment_batch_size=max_batch_size,
    )
    return MossStreamingVocoderScheduler(
        vocoder,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
