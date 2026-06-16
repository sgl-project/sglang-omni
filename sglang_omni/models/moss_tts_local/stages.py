# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the MOSS-TTS Local (v1.5) pipeline."""

from __future__ import annotations

import concurrent.futures
import logging
import os
import queue
import threading
import time
from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.models.moss_tts.stages import (
    _load_moss_processor_class,
    _moss_transformers_processor_compat,
    _resolve_checkpoint,
)
from sglang_omni.models.moss_tts_local.payload_types import (
    moss_tts_local_special_token_defaults,
)
from sglang_omni.models.moss_tts_local.request_builders import (
    cleanup_prepared_moss_tts_local_request,
    make_moss_tts_local_scheduler_adapters,
    preprocess_moss_tts_local_payload,
    set_moss_tts_local_preprocessing_context,
)
from sglang_omni.models.moss_tts_local.streaming_vocoder import (
    MossTTSLocalStreamingVocoderScheduler,
)
from sglang_omni.preprocessing.cache_key import (
    reference_path_cache_key as _reference_path_cache_key,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

_MOSS_TTS_LOCAL_INSTALL_HINT = (
    "MOSS-TTS Local support requires the upstream custom Transformers code. "
    "Launch with trust_remote_code=True and make sure the checkpoint can load "
    "OpenMOSS-Team/MOSS-Audio-Tokenizer-v2."
)

# NOTE: preprocessing and vocoder stages each load their own processor (and
# ~4.3 GB bf16 codec instance): `model.streaming()` flips module-global codec
# state, so a decode on a shared instance would corrupt a concurrent
# reference encode (see streaming_vocoder.py).


@dataclass(frozen=True)
class _ArMemoryBudget:
    effective_total_gpu_memory_fraction: float | None
    applied_codec_mem_reserve: float


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


def _load_moss_tts_local_processor(model_path: str, *, device: str) -> Any:
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info(
        "Loading MOSS-TTS Local processor from %s on %s", checkpoint_dir, device
    )
    try:
        with _moss_transformers_processor_compat():
            processor_cls = _load_moss_processor_class(checkpoint_dir)
            processor = processor_cls.from_pretrained(
                checkpoint_dir,
                trust_remote_code=True,
            )
    except Exception as exc:
        raise RuntimeError(_MOSS_TTS_LOCAL_INSTALL_HINT) from exc

    _normalize_processor_config(processor)
    audio_tokenizer = getattr(processor, "audio_tokenizer", None)
    if audio_tokenizer is not None:
        if hasattr(audio_tokenizer, "eval"):
            audio_tokenizer.eval()
        if hasattr(audio_tokenizer, "to"):
            # Device move only: the v2 codec manages its own dtypes (bf16
            # encoder/decoder with an fp32 quantizer); a blanket dtype cast
            # would corrupt the quantizer codebooks.
            audio_tokenizer.to(device)
    return processor


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
    MAX_REFERENCE_SECONDS = 100.0
    # An encode batch takes well under a second; a result this late means the
    # worker died or wedged, so fail the request instead of hanging the slot.
    ENCODE_TIMEOUT_S = 120.0

    def __init__(
        self,
        processor: Any,
        *,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 4,
    ) -> None:
        self._processor = processor
        self._max_batch_size = max(int(max_batch_size), 1)
        self._max_wait_s = max(float(max_batch_wait_ms), 0.0) / 1000.0
        self._queue: queue.Queue[tuple[str, concurrent.futures.Future]] = queue.Queue()
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
                f"reference audio is {duration:.1f}s long; the limit is "
                f"{cls.MAX_REFERENCE_SECONDS:.0f}s"
            )

    def encode(self, path: str) -> torch.Tensor:
        """Encode one reference file; blocks until its batch completes."""
        path = str(path)
        self._check_reference_duration(path)
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put((path, future))
        return future.result(timeout=self.ENCODE_TIMEOUT_S)

    def _drain_batch(self) -> list[tuple[str, concurrent.futures.Future]]:
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
            unique_paths = list(dict.fromkeys(path for path, _ in batch))
            results: dict[str, Any] = {}
            try:
                encoded = self._processor.encode_audios_from_path(unique_paths)
                results = dict(zip(unique_paths, encoded))
            except Exception:
                logger.exception(
                    "MOSS-TTS Local batched reference encode failed; "
                    "retrying per item"
                )
                for path in unique_paths:
                    try:
                        results[path] = self._processor.encode_audios_from_path([path])[
                            0
                        ]
                    except Exception as exc:
                        results[path] = exc
            for path, future in batch:
                outcome = results.get(path)
                if isinstance(outcome, Exception):
                    # Fresh exception per future: a shared instance would be
                    # mutated concurrently by every waiter's traceback raise.
                    future.set_exception(
                        RuntimeError(f"reference encode failed for {path}: {outcome}")
                    )
                elif outcome is None:
                    future.set_exception(
                        RuntimeError(f"reference encode produced no codes: {path}")
                    )
                else:
                    future.set_result(outcome)


class CachedReferenceEncoder:
    """Content-addressed LRU cache + single-flight dedup in front of _BatchedReferenceEncoder.

    Every path (miss, hit, follower) returns an independent CPU long tensor, so
    downstream sees one device/dtype regardless of cache temperature.
    Stores codes as int32 on CPU (lossless for codebook values in [0, 1023]).
    """

    # Cadence for the periodic stats log; class attr so it is easy to tune.
    LOG_INTERVAL_S = 60.0

    def __init__(
        self,
        encoder: _BatchedReferenceEncoder,
        *,
        max_items: int = 256,
        max_bytes: int = 64 * 1024 * 1024,
    ) -> None:
        # Fail fast on non-positive capacities: a negative max_items makes
        # StageOutputCache evict from an empty dict and KeyError at request time.
        if max_items < 1:
            raise ValueError(f"ref_audio_cache_max_items must be >= 1, got {max_items}")
        if max_bytes < 1:
            raise ValueError(f"ref_audio_cache_max_bytes must be >= 1, got {max_bytes}")
        self._encoder = encoder
        self._cache = StageOutputCache(
            max_size=max_items,
            max_bytes=max_bytes,
            cache_device="cpu",
        )
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._last_log_time: float = 0.0

    def encode(self, path: str) -> torch.Tensor:
        path = str(path)
        # Note(Jiaxin): duration gate runs first — a >100 s ref must never reach
        # the cache or the inflight dict.
        _BatchedReferenceEncoder._check_reference_duration(path)
        # trust_stat left False (review feedback): keep the sentinel byte-read so a
        # same-size+mtime+ctime overwrite cannot stale-hit. The flag stays available
        # in reference_path_cache_key for deployments that guarantee immutable refs.
        key = _reference_path_cache_key(path)
        if key is None:
            return self._encoder.encode(path)  # uncacheable (URL/missing) -> bypass
        return self._cached_encode(
            key,
            lambda: self._encoder.encode(path),
            desc=repr(path),
            # TOCTOU re-stat: skip the put if the file changed during the encode.
            revalidate=lambda: _reference_path_cache_key(path) == key,
        )

    def _cached_encode(
        self, key: str, encode_fn, *, desc: str, revalidate=None
    ) -> torch.Tensor:
        """Single-flight skeleton shared by encode() and encode_data_uri().

        All paths return an independent CPU long tensor. revalidate(), if given,
        is evaluated outside the lock and gates the put (TOCTOU guard for file
        paths).
        """
        leader_fut: concurrent.futures.Future | None = None
        follower_fut: concurrent.futures.Future | None = None

        with self._lock:
            stored = self._cache.get(key)
            if stored is not None:
                self._hits += 1
            elif key in self._inflight:
                self._merged += 1
                follower_fut = self._inflight[key]
            else:
                self._misses += 1
                leader_fut = concurrent.futures.Future()
                self._inflight[key] = leader_fut

        if stored is not None:
            # Note(Jiaxin): clone on hit so callers can't mutate the shared entry.
            self._maybe_log()
            return stored.clone().to(torch.long)

        if follower_fut is not None:
            # Note(Jiaxin): each follower raises a FRESH RuntimeError — sharing one
            # exception instance lets concurrent re-raises corrupt its traceback
            # (same lesson as _BatchedReferenceEncoder._worker).
            timeout = _BatchedReferenceEncoder.ENCODE_TIMEOUT_S + 10
            try:
                stored = follower_fut.result(timeout=timeout)
            except Exception as cause:
                raise RuntimeError(
                    f"reference encode failed for {desc}: {cause}"
                ) from cause
            return stored.clone().to(torch.long)

        assert leader_fut is not None
        try:
            result = encode_fn()
        except BaseException as exc:
            with self._lock:
                self._inflight.pop(key, None)
            leader_fut.set_exception(exc)
            raise

        do_put = revalidate() if revalidate is not None else True
        stored = result.detach().to("cpu", dtype=torch.int32)
        with self._lock:
            if do_put:
                self._cache.put(key, stored)
            self._inflight.pop(key, None)
        leader_fut.set_result(stored)
        self._maybe_log()
        # CPU long like the hit path.
        return stored.to(torch.long)

    def _maybe_log(self) -> None:
        now = time.monotonic()
        if now - self._last_log_time < self.LOG_INTERVAL_S:
            return
        with self._lock:
            if now - self._last_log_time < self.LOG_INTERVAL_S:
                return
            self._last_log_time = now
            snapshot = (
                self._hits,
                self._misses,
                self._merged,
                len(self._cache._cache),
                self._cache.current_bytes,
            )
        logger.info(
            "MOSS-TTS Local ref cache: hits=%d misses=%d merged=%d entries=%d bytes=%d",
            *snapshot,
        )

    def encode_data_uri(self, ref_audio: str, *, processor: Any) -> torch.Tensor:
        """Cache-aware encode for data-URI refs through the same LRU + single-flight
        as file paths (adds the duration check _reference_for_processor lacks).

        Note(Jiaxin): file: and bytes: keyspaces never collide — the two decode
        chains differ, so codes aren't guaranteed identical for the "same" audio.
        """
        import base64
        import io

        from sglang_omni.models.moss_tts.request_builders import _DATA_URI_RE
        from sglang_omni.preprocessing.cache_key import hash_bytes as _hash_bytes

        match = _DATA_URI_RE.match(ref_audio)
        if match is None:
            raise ValueError(f"encode_data_uri: not a data URI ({ref_audio[:40]!r}...)")

        raw = base64.b64decode(match.group("data"))
        key = f"bytes:{_hash_bytes(raw)}"

        def _encode() -> torch.Tensor:
            import soundfile as sf

            audio, sample_rate = sf.read(
                io.BytesIO(raw), dtype="float32", always_2d=True
            )
            # Note(Jiaxin): the duration check runs inside the leader (not before
            # inflight registration like the file path) so concurrent same-payload
            # requests share one sf.read of a potentially large decoded buffer.
            duration = audio.shape[0] / max(int(sample_rate), 1)
            if duration > _BatchedReferenceEncoder.MAX_REFERENCE_SECONDS:
                raise ValueError(
                    f"reference audio is {duration:.1f}s long; the limit is "
                    f"{_BatchedReferenceEncoder.MAX_REFERENCE_SECONDS:.0f}s"
                )
            wav = torch.from_numpy(audio.T)
            return processor.encode_audios_from_wav([wav], int(sample_rate))[0]

        return self._cached_encode(key, _encode, desc="data-URI")

    def stats(self) -> dict:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache._cache),
                "bytes": self._cache.current_bytes,
            }


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int = 16,
    encode_batch_size: int = 8,
    encode_batch_wait_ms: int = 4,
    ref_audio_cache: bool = True,
    ref_audio_cache_max_items: int = 8192,
    ref_audio_cache_max_bytes: int = 64 * 1024 * 1024,
) -> SimpleScheduler:
    # MOSS_REF_AUDIO_CACHE=0 disables the cache at startup (ops kill switch / A-B
    # toggle) without a config edit; unset => kwarg default.
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
    processor = _load_moss_tts_local_processor(model_path, device=device)
    reference_encoder: Any = _BatchedReferenceEncoder(
        processor,
        max_batch_size=encode_batch_size,
        max_batch_wait_ms=encode_batch_wait_ms,
    )
    if ref_audio_cache:
        reference_encoder = CachedReferenceEncoder(
            reference_encoder,
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
    from sglang_omni.models.moss_tts_local.model_runner import MossTTSLocalModelRunner
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangOutputProcessor,
        build_sglang_server_args,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "dtype": dtype,
        "cuda_graph_bs": [1, 2, 4, 8, 16],
        "cuda_graph_max_bs": 16,
        "disable_cuda_graph": False,
        "disable_overlap_schedule": True,
        "enable_torch_compile": False,
        "max_prefill_tokens": 8192,
        "max_running_requests": 16,
        "sampling_backend": "pytorch",
        "torch_compile_max_bs": 16,
        "trust_remote_code": True,
    }
    if total_gpu_memory_fraction is None:
        # without a typed stage budget, this path cannot use process-scoped
        # colocated profiling
        # keep the legacy static fraction for split/custom deployments
        overrides["mem_fraction_static"] = 0.6 if torch.cuda.device_count() > 1 else 0.5
    if server_args_overrides:
        overrides.update(server_args_overrides)
    memory_budget = _apply_colocated_ar_memory_budget(
        overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
        codec_mem_reserve=codec_mem_reserve,
    )
    profile_total_gpu_memory_fraction = (
        memory_budget.effective_total_gpu_memory_fraction
    )
    if profile_total_gpu_memory_fraction is not None:
        from sglang_omni.utils.gpu_memory import get_process_gpu_memory_bytes

        if get_process_gpu_memory_bytes(gpu_id) is None:
            logger.warning(
                f"MOSS-TTS Local colocated process memory accounting is unavailable; "
                f"falling back to upstream SGLang free-memory profiling. "
                f"effective_total_gpu_memory_fraction={profile_total_gpu_memory_fraction}"
            )
            profile_total_gpu_memory_fraction = None

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=8192,
        **overrides,
    )

    want_cuda_graph = not bool(getattr(server_args, "disable_cuda_graph", False))
    if want_cuda_graph:
        server_args.disable_cuda_graph = True

    logger.info(
        f"MOSS-TTS Local SGLang startup: gpu_id={gpu_id} "
        f"total_gpu_memory_fraction={total_gpu_memory_fraction} "
        f"effective_total_gpu_memory_fraction="
        f"{memory_budget.effective_total_gpu_memory_fraction} "
        f"codec_mem_reserve={memory_budget.applied_codec_mem_reserve:.3f} "
        f"mem_fraction_static={server_args.mem_fraction_static} "
        f"profile_total_gpu_memory_fraction={profile_total_gpu_memory_fraction}"
    )

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(
        server_args,
        gpu_id,
        model_arch_override="MossTTSLocalSGLangModel",
        total_gpu_memory_fraction=profile_total_gpu_memory_fraction,
    )

    if want_cuda_graph:
        server_args.disable_cuda_graph = False

    model = model_worker.model_runner.model
    if want_cuda_graph:
        model_worker.model_runner.init_device_graphs()
        # Also graph the per-frame local-transformer decode (1 + n_vq
        # micro-steps and 13 seeded sampling passes per frame): eager it is
        # kernel-launch-bound at ~22 ms/frame independent of batch size.
        model.init_frame_decode_graphs(
            list(overrides.get("cuda_graph_bs") or [1, 2, 4, 8, 16])
        )

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model,
    )
    request_builder, result_adapter = make_moss_tts_local_scheduler_adapters(
        model=model
    )

    def abort_request(request_id: str) -> None:
        # Drop any prepared handoff and release any held pool row; both are
        # idempotent no-ops if the request never reached them.
        cleanup_prepared_moss_tts_local_request(request_id)
        model.reset_request(request_id)

    model_runner = MossTTSLocalModelRunner(model_worker, output_proc)
    scheduler = OmniScheduler(
        tp_worker=model_worker,
        tree_cache=tree_cache,
        req_to_token_pool=req_to_token_pool,
        token_to_kv_pool_allocator=token_to_kv_pool_allocator,
        server_args=server_args,
        model_config=model_config,
        prefill_manager=prefill_mgr,
        decode_manager=decode_mgr,
        model_runner=model_runner,
        request_builder=request_builder,
        result_adapter=result_adapter,
        abort_callback=abort_request,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )
    model_runner.set_stream_outbox(scheduler.outbox)
    return scheduler


def create_tts_engine_executor(*args, **kwargs) -> Any:
    return create_sglang_tts_engine_executor(*args, **kwargs)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
    stream_slots: int = 8,
    stream_chunk_frames: int = 25,
    initial_chunk_frames: int = 5,
) -> MossTTSLocalStreamingVocoderScheduler:
    device = _resolve_codec_device(device, gpu_id)
    processor = _load_moss_tts_local_processor(model_path, device=device)
    return MossTTSLocalStreamingVocoderScheduler(
        processor,
        stream_slots=stream_slots,
        stream_chunk_frames=stream_chunk_frames,
        initial_chunk_frames=initial_chunk_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
