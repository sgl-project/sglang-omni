# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Local (v1.5) pipeline."""

from __future__ import annotations

import base64
import concurrent.futures
import io
import logging
import os
import queue
import threading
from dataclasses import dataclass
from typing import Any, TypeAlias

import torch

from sglang_omni.models.moss_tts.hf_loading import (
    load_moss_processor_class,
    moss_transformers_processor_compat,
    resolve_moss_checkpoint,
)
from sglang_omni.models.moss_tts.request_builders import _DATA_URI_RE
from sglang_omni.models.moss_tts_local.audio_tokenizer import (
    DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER,
    load_moss_tts_local_audio_tokenizer,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    cleanup_prepared_moss_tts_local_request,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.preprocessing.cache_key import hash_bytes as _hash_bytes
from sglang_omni.preprocessing.cache_key import (
    reference_path_cache_key as _reference_path_cache_key,
)
from sglang_omni.scheduling.reference_encoder import (
    ReferenceEncodeHook,
    ReferenceEncodeKey,
    ReferenceEncodeService,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

logger = logging.getLogger(__name__)

_MOSS_TTS_LOCAL_INSTALL_HINT = (
    "MOSS-TTS Local support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2."
)
_MAX_REFERENCE_SECONDS = 100.0

# NOTE: preprocessing and vocoder stages each load their own codec instance:
# `model.streaming()` flips codec state, so a decode on a shared instance would
# corrupt a concurrent reference encode (see streaming_vocoder.py).


@dataclass(frozen=True)
class _ArMemoryBudget:
    effective_total_gpu_memory_fraction: float | None
    applied_codec_mem_reserve: float


@dataclass(frozen=True)
class _PathReferenceJob:
    path: str


@dataclass(frozen=True)
class _WaveformReferenceJob:
    wav: torch.Tensor
    sample_rate: int


_ReferenceEncodeJob: TypeAlias = _PathReferenceJob | _WaveformReferenceJob


def _apply_colocated_ar_memory_budget(
    overrides: dict[str, Any],
    *,
    total_gpu_memory_fraction: float | None,
    codec_mem_reserve: float,
) -> _ArMemoryBudget:
    if total_gpu_memory_fraction is None:
        return _ArMemoryBudget(
            effective_total_gpu_memory_fraction=None,
            applied_codec_mem_reserve=0.0,
        )
    if not 0.0 <= codec_mem_reserve < 1.0:
        raise ValueError("codec_mem_reserve must be in [0, 1)")

    effective_total_gpu_memory_fraction = round(
        total_gpu_memory_fraction - codec_mem_reserve,
        3,
    )
    if effective_total_gpu_memory_fraction < 0.1:
        raise ValueError(
            f"colocated total_gpu_memory_fraction {total_gpu_memory_fraction:.3f} "
            f"minus codec_mem_reserve {codec_mem_reserve:.3f} = "
            f"{effective_total_gpu_memory_fraction:.3f} is below the safe floor "
            f"0.1; lower codec_mem_reserve or increase the tts_engine stage budget."
        )

    explicit_mem_fraction = overrides.get("mem_fraction_static")
    applied_codec_mem_reserve = codec_mem_reserve
    if explicit_mem_fraction is not None:
        explicit_mem_fraction = float(explicit_mem_fraction)
        if not 0.0 < explicit_mem_fraction < 1.0:
            raise ValueError(
                f"mem_fraction_static must be > 0 and < 1, got {explicit_mem_fraction}"
            )
        if explicit_mem_fraction > total_gpu_memory_fraction:
            raise ValueError(
                f"MOSS-TTS Local tts_engine mem_fraction_static cannot exceed "
                f"runtime.resources.total_gpu_memory_fraction: "
                f"{explicit_mem_fraction:.3f} > {total_gpu_memory_fraction:.3f}"
            )
        effective_total_gpu_memory_fraction = explicit_mem_fraction
        applied_codec_mem_reserve = round(
            total_gpu_memory_fraction - effective_total_gpu_memory_fraction,
            3,
        )
    else:
        overrides["mem_fraction_static"] = effective_total_gpu_memory_fraction

    return _ArMemoryBudget(
        effective_total_gpu_memory_fraction=effective_total_gpu_memory_fraction,
        applied_codec_mem_reserve=applied_codec_mem_reserve,
    )


def _normalize_processor_config(processor: Any) -> None:
    model_config = getattr(processor, "model_config", None)
    if model_config is None:
        return
    audio_vocab_size = int(getattr(model_config, "audio_vocab_size", 1024) or 1024)
    for attr, default in moss_tts_local_special_token_defaults(audio_vocab_size):
        if getattr(model_config, attr, None) is None:
            setattr(model_config, attr, default)


def _resolve_codec_device(device: str | None, gpu_id: int | None) -> str:
    """Pick the codec GPU for the preprocessing/vocoder stages.

    The ~1B-param codec encoder costs ~0.25 GPU-seconds per reference, which
    at concurrency 16 starves the AR engine when both share one device.
    The default config passes an explicit ``device`` so the second-GPU codec
    placement is visible in the pipeline config. ``gpu_id`` remains a fallback
    for custom colocated configs and launcher-injected runtime defaults.
    """
    if device:
        return device
    if gpu_id is not None:
        return f"cuda:{int(gpu_id)}"
    return "cuda:0"


def _load_moss_tts_local_processor(model_path: str) -> Any:
    checkpoint_dir = resolve_moss_checkpoint(model_path)
    logger.info(f"Loading MOSS-TTS Local processor from {checkpoint_dir} without codec")
    try:
        from transformers import AutoConfig, AutoTokenizer

        with moss_transformers_processor_compat():
            processor_cls = load_moss_processor_class(checkpoint_dir)
            model_config = AutoConfig.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
            processor = processor_cls(
                tokenizer=tokenizer,
                audio_tokenizer=None,
                model_config=model_config,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_LOCAL_INSTALL_HINT) from exc

    _normalize_processor_config(processor)
    return processor


def _resolve_audio_tokenizer_model_path(
    processor: Any,
    codec_model_path: str | None,
) -> str:
    if codec_model_path is not None:
        return codec_model_path
    return str(
        getattr(
            processor.model_config,
            "audio_tokenizer_name_or_path",
            DEFAULT_MOSS_TTS_LOCAL_AUDIO_TOKENIZER,
        )
    )


class _BatchedReferenceEncoder:
    """Coalesces concurrent reference-audio encodes into batched codec calls.

    Each request needs its reference run through the ~1B-param codec encoder
    (~0.25 GPU-seconds). The preprocessing workers call :meth:`encode`
    concurrently; a single daemon thread drains the queue and encodes up to
    ``max_batch_size`` files in one ``batch_encode`` forward, which costs
    barely more than a single encode. Failures fall back to per-item encodes
    so one bad file only fails its own request.
    """

    # Mirrors the Higgs reference-audio cap: bounds both encoder runtime and
    # the batch-padding memory amplification.
    MAX_REFERENCE_SECONDS = _MAX_REFERENCE_SECONDS
    # An encode batch takes well under a second; a result this late means the
    # worker died or wedged, so fail the request instead of hanging the slot.
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        audio_tokenizer: Any,
        *,
        n_vq: int,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
    ) -> None:
        self._audio_tokenizer = audio_tokenizer
        self._n_vq = int(n_vq)
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._queue: queue.Queue[
            tuple[_ReferenceEncodeJob, concurrent.futures.Future]
        ] = queue.Queue()
        self._thread = threading.Thread(
            target=self._worker, name="moss-local-ref-encode", daemon=True
        )
        self._thread.start()

    @classmethod
    def _check_reference_duration(cls, path: str) -> None:
        try:
            import torchaudio

            info = torchaudio.info(path)
            duration = info.num_frames / max(int(info.sample_rate), 1)
        except Exception:
            return  # unreadable files fail with a clearer error in the codec
        if duration > cls.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio is {duration:.1f}s long, limit is "
                f"{cls.MAX_REFERENCE_SECONDS:.0f}s"
            )

    @staticmethod
    def _data_uri_audio_bytes(ref_audio: str) -> bytes:
        match = _DATA_URI_RE.match(ref_audio)
        if match is None:
            raise ValueError(f"encode_data_uri: not a data URI ({ref_audio[:40]!r}...)")
        return base64.b64decode(match.group("data"))

    @staticmethod
    def _decode_data_uri_audio(raw: bytes) -> tuple[torch.Tensor, int]:
        import soundfile as sf

        audio, sample_rate = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        duration = audio.shape[0] / max(int(sample_rate), 1)
        if duration > _BatchedReferenceEncoder.MAX_REFERENCE_SECONDS:
            raise ValueError(
                f"reference audio is {duration:.1f}s long, limit is "
                f"{_BatchedReferenceEncoder.MAX_REFERENCE_SECONDS:.0f}s"
            )
        return torch.from_numpy(audio.T), int(sample_rate)

    def encode(self, path: str) -> torch.Tensor:
        """Encode one reference file; blocks until its batch completes."""
        path = str(path)
        self._check_reference_duration(path)
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((_PathReferenceJob(path), future))
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def encode_wav(self, wav: torch.Tensor, sample_rate: int) -> torch.Tensor:
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((_WaveformReferenceJob(wav, int(sample_rate)), future))
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def encode_data_uri(self, ref_audio: str) -> torch.Tensor:
        raw = self._data_uri_audio_bytes(ref_audio)
        wav, sample_rate = self._decode_data_uri_audio(raw)
        return self.encode_wav(wav, sample_rate)

    def _drain_batch(
        self,
    ) -> list[tuple[_ReferenceEncodeJob, concurrent.futures.Future]]:
        batch = [self._queue.get()]
        while len(batch) < self._max_batch_size:
            try:
                if self._max_wait_s > 0:
                    batch.append(self._queue.get(timeout=self._max_wait_s))
                else:
                    batch.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return batch

    def _worker(self) -> None:
        while True:
            batch = self._drain_batch()
            results = self._encode_batch(batch)
            for index, (_, future) in enumerate(batch):
                outcome = results.get(index)
                if isinstance(outcome, Exception):
                    # Fresh exception per future: a shared instance would be
                    # mutated concurrently by every waiter's traceback raise.
                    future.set_exception(
                        RuntimeError(f"reference encode failed: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError("reference encode produced no codes")
                    )
                else:
                    future.set_result(outcome)

    def _encode_batch(
        self, batch: list[tuple[_ReferenceEncodeJob, concurrent.futures.Future]]
    ) -> dict[int, Any]:
        results: dict[int, Any] = {}
        path_to_indices: dict[str, list[int]] = {}
        waveforms: list[tuple[torch.Tensor, int]] = []
        waveform_indices: list[int] = []
        for index, (job, _) in enumerate(batch):
            if isinstance(job, _PathReferenceJob):
                path_to_indices.setdefault(job.path, []).append(index)
            elif isinstance(job, _WaveformReferenceJob):
                waveform_indices.append(index)
                waveforms.append((job.wav, job.sample_rate))
            else:
                raise TypeError(f"unknown reference encode job: {type(job).__name__}")

        unique_paths = list(path_to_indices)
        try:
            path_waveforms = (
                self._audio_tokenizer.load_paths(unique_paths) if unique_paths else []
            )
            encoded = self._audio_tokenizer.encode_waveforms(
                path_waveforms + waveforms,
                num_quantizers=self._n_vq,
            )
            path_count = len(unique_paths)
            for path, codes in zip(unique_paths, encoded[:path_count]):
                for index in path_to_indices[path]:
                    results[index] = codes
            for index, codes in zip(waveform_indices, encoded[path_count:]):
                results[index] = codes
        except Exception:
            logger.exception(
                "MOSS-TTS Local batched reference encode failed; retrying per item"
            )
            for path, indices in path_to_indices.items():
                try:
                    codes = self._audio_tokenizer.encode_paths(
                        [path],
                        num_quantizers=self._n_vq,
                    )[0]
                except Exception as exc:
                    codes = exc
                for index in indices:
                    results[index] = codes
            for index, waveform in zip(waveform_indices, waveforms):
                try:
                    results[index] = self._audio_tokenizer.encode_waveforms(
                        [waveform],
                        num_quantizers=self._n_vq,
                    )[0]
                except Exception as exc:
                    results[index] = exc
        return results


@dataclass(frozen=True)
class _MossLocalReferenceInput:
    source_kind: str
    source: str
    raw: bytes | None = None


class _MossLocalReferenceEncodeHook(
    ReferenceEncodeHook[_MossLocalReferenceInput, torch.Tensor, torch.Tensor]
):
    def __init__(
        self,
        encoder: _BatchedReferenceEncoder,
        *,
        n_vq: int,
    ) -> None:
        self._encoder = encoder
        self._n_vq = int(n_vq)
        self._encoder_config_hash = _hash_bytes(f"n_vq:{self._n_vq}".encode("utf-8"))

    def normalize_input(self, raw_input: Any) -> _MossLocalReferenceInput:
        if isinstance(raw_input, _MossLocalReferenceInput):
            return raw_input
        return _MossLocalReferenceInput("path", str(raw_input))

    def cache_key(self, item: _MossLocalReferenceInput) -> ReferenceEncodeKey | None:
        input_key = self._input_key(item)
        if input_key is None:
            return None
        return ReferenceEncodeKey(
            model_id="moss_tts_local",
            model_revision="local_audio_tokenizer",
            encoder_id="moss_tts_local_audio_tokenizer",
            encoder_config_hash=self._encoder_config_hash,
            artifact_kind="moss_tts_local_reference_codes",
            input_key=input_key,
        )

    def encode_one(self, item: _MossLocalReferenceInput) -> torch.Tensor:
        if item.source_kind == "path":
            return self._encoder.encode(item.source)
        if item.source_kind == "data_uri":
            raw = item.raw
            if raw is None:
                raw = _BatchedReferenceEncoder._data_uri_audio_bytes(item.source)
            wav, sample_rate = _BatchedReferenceEncoder._decode_data_uri_audio(raw)
            return self._encoder.encode_wav(wav, sample_rate)
        raise TypeError(f"unknown MOSS-local reference source: {item.source_kind}")

    def store_artifact(self, artifact: torch.Tensor) -> torch.Tensor:
        return artifact.detach().to("cpu", dtype=torch.int32).clone()

    def load_artifact(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.detach().clone().to(torch.long)

    def revalidate(
        self,
        item: _MossLocalReferenceInput,
        key: ReferenceEncodeKey,
    ) -> bool:
        if item.source_kind != "path":
            return True
        return _reference_path_cache_key(item.source) == key.input_key

    def _input_key(self, item: _MossLocalReferenceInput) -> str | None:
        if item.source_kind == "path":
            _BatchedReferenceEncoder._check_reference_duration(item.source)
            return _reference_path_cache_key(item.source)
        if item.source_kind == "data_uri":
            raw = item.raw
            if raw is None:
                raw = _BatchedReferenceEncoder._data_uri_audio_bytes(item.source)
            return f"bytes:{_hash_bytes(raw)}"
        return None


class _MossLocalReferenceEncoder:
    def __init__(
        self,
        encoder: _BatchedReferenceEncoder,
        *,
        n_vq: int,
        max_items: int | None = 256,
        max_bytes: int | None = 64 * 1024 * 1024,
    ) -> None:
        self._service = ReferenceEncodeService(
            _MossLocalReferenceEncodeHook(encoder, n_vq=n_vq),
            max_items=max_items,
            max_bytes=max_bytes,
            timeout_s=_BatchedReferenceEncoder.ENCODE_TIMEOUT_S + 10,
            log_prefix="MOSS-TTS Local ref cache",
        )

    def encode(self, path: str) -> torch.Tensor:
        return self._service.get_or_encode(
            _MossLocalReferenceInput("path", str(path)),
            desc=repr(str(path)),
        )

    def encode_data_uri(self, ref_audio: str) -> torch.Tensor:
        raw = _BatchedReferenceEncoder._data_uri_audio_bytes(ref_audio)
        return self._service.get_or_encode(
            _MossLocalReferenceInput("data_uri", str(ref_audio), raw),
            desc="data-URI",
        )

    def stats(self) -> dict[str, int]:
        return self._service.stats()


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    # MOSS_REF_AUDIO_CACHE=0 disables the cache at startup (ops kill switch / A-B
    # toggle) without a config edit; unset => kwarg/config default.
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
    processor = _load_moss_tts_local_processor(model_path)
    audio_tokenizer = load_moss_tts_local_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
    )
    reference_encoder: Any = _BatchedReferenceEncoder(
        audio_tokenizer,
        n_vq=int(processor.model_config.n_vq),
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
    )
    if ref_audio_cache:
        reference_encoder = _MossLocalReferenceEncoder(
            reference_encoder,
            n_vq=int(processor.model_config.n_vq),
            max_items=ref_audio_cache_max_items,
            max_bytes=ref_audio_cache_max_bytes,
        )
    set_moss_tts_local_preprocessing_context(
        processor=processor, reference_encoder=reference_encoder
    )
    # Reference encoding runs through the ~1B-param causal codec encoder, so
    # unlike MOSS Delay the audio tokenizer must live on the GPU; threads
    # release the GIL during the codec forward, keeping the AR engine fed.
    return SimpleScheduler(
        preprocess_moss_tts_local_payload,
        abort_callback=cleanup_prepared_moss_tts_local_request,
        max_concurrency=max_concurrency,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    dtype: str = "bfloat16",
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
    total_gpu_memory_fraction: float | None = None,
    codec_mem_reserve: float = 0.0,
) -> Any:
    from sglang_omni.models.moss_tts_local.engine_builder import (
        MossTtsLocalEngineBuilder,
    )

    return MossTtsLocalEngineBuilder(
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        codec_mem_reserve=codec_mem_reserve,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_model_path: str | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_slots: int = 8,
    stream_chunk_frames: int = 25,
    # Note(Jiaxin): serve-aligned streaming defaults. The first chunk is small (1) for low TTFC,
    # while coalesce_floor_frames (5) is the separate steady-state coalescing join floor.
    initial_chunk_frames: int = 1,
    coalesce_floor_frames: int = 5,
    cuda_graph: bool = True,
    cuda_graph_frames: list[int] | None = None,
    cuda_graph_min_free_gb: float = 3.0,
) -> MossTTSLocalStreamingVocoderScheduler:
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_tts_local_processor(model_path)
    audio_tokenizer = load_moss_tts_local_audio_tokenizer(
        _resolve_audio_tokenizer_model_path(processor, codec_model_path),
        device=device,
    )
    scheduler = MossTTSLocalStreamingVocoderScheduler(
        audio_tokenizer.model,
        n_vq=int(processor.model_config.n_vq),
        sample_rate=audio_tokenizer.sample_rate,
        stream_slots=stream_slots,
        stream_chunk_frames=stream_chunk_frames,
        initial_chunk_frames=initial_chunk_frames,
        coalesce_floor_frames=coalesce_floor_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        cuda_graph=cuda_graph,
        cuda_graph_frames=cuda_graph_frames,
        cuda_graph_min_free_gb=cuda_graph_min_free_gb,
    )
    # Capture graphs in the factory: it runs before the process is marked ready, so serving never
    # races a half-captured graph. Same-process guarantee (each colocate/split stage warms its own).
    scheduler.warmup_now()
    return scheduler
