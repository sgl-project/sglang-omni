# SPDX-License-Identifier: Apache-2.0
# Follows the sglang-omni TTS stage pattern (cf. Higgs / Qwen3-TTS;
# docs/developer_reference/tts_model_integration.md).
"""Stage factories for the CSM TTS pipeline:
``preprocessing → audio_encoder (Mimi encode, no-op fast path) → tts_engine →
vocoder (Mimi decode)``.

Referenced by dotted path from
:class:`sglang_omni.models.csm_tts.config.CsmTtsPipelineConfig` — this module
imports sglang and is therefore NEVER imported by ``config.py`` (the
registry-silent-skip gotcha).


"""

from __future__ import annotations

import base64
import logging
import threading
from collections import OrderedDict
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as F_audio

from sglang_omni.models.csm_tts.model_runner import CsmTTSModelRunner
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.request_builders import make_csm_scheduler_adapters
from sglang_omni.models.csm_tts.text_tokenizer import CsmPromptBuilder
from sglang_omni.models.csm_tts.utils import (
    BACKBONE_CTX,
    DEFAULT_MAX_FRAMES,
    FRAME_RATE,
    NUM_CODEBOOKS,
    SAMPLE_RATE,
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    to_codes_F32,
)
from sglang_omni.models.csm_tts.vocoder_scheduler import CsmStreamingVocoderScheduler
from sglang_omni.preprocessing.cache_key import hash_bytes, hash_media_item
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend import (
    SGLangOutputProcessor,
    build_sglang_server_args,
)
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.stage_cache import StageOutputCache
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

logger = logging.getLogger(__name__)

# The binding constraint on the 3060 is COMPUTE (frame step must beat 80 ms
# real time), not VRAM — mrr/concurrency default to 8.
DEFAULT_MAX_CONCURRENCY = 8

# Context-audio budget guard: 60 s = 750 backbone positions at 12.5 Hz
# (+1 <|audio_eos|> per segment); text capped ~1500 tokens after reserving
# frames.
MAX_CONTEXT_AUDIO_SECONDS = 60
_MAX_CONTEXT_FRAMES = int(MAX_CONTEXT_AUDIO_SECONDS * FRAME_RATE)  # 750
_MAX_TEXT_TOKENS = 1500

# Three-tier context-audio caching (higgs copy): path-hash memo /
# waveform LRU / encoded-codes LRU (as CPU int32 so the cache can byte-bound).
_CTX_CODE_CACHE_MAX_ITEMS = 256
_CTX_CODE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_CTX_WAVEFORM_CACHE_MAX_ITEMS = 256
_CTX_WAVEFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024
_CTX_PATH_HASH_MEMO_MAX_ITEMS = 1024
_CTX_PATH_HASH_SENTINEL_BYTES = 8192
_CTX_PATH_HASH_MEMO: OrderedDict[str, tuple[str, str]] = OrderedDict()
_CTX_PATH_HASH_MEMO_LOCK = threading.Lock()


def _context_path_hash_memo_key(path: Path) -> tuple[str, int] | None:
    try:
        if not path.is_file():
            return None
        stat_result = path.stat()
        memo_key = (
            f"{path.resolve()}:"
            f"{stat_result.st_size}:"
            f"{stat_result.st_mtime_ns}:"
            f"{stat_result.st_ctime_ns}"
        )
        return memo_key, int(stat_result.st_size)
    except OSError:
        return None


def _context_path_sentinel(path: Path, file_size: int) -> str | None:
    try:
        chunk_size = min(_CTX_PATH_HASH_SENTINEL_BYTES, file_size)
        with path.open("rb") as f:
            chunks = [f.read(chunk_size)]
            if file_size > _CTX_PATH_HASH_SENTINEL_BYTES:
                middle_offset = max((file_size - chunk_size) // 2, 0)
                f.seek(middle_offset)
                chunks.append(f.read(chunk_size))
            if file_size > 2 * _CTX_PATH_HASH_SENTINEL_BYTES:
                f.seek(max(file_size - chunk_size, 0))
                chunks.append(f.read(chunk_size))
        return hash_bytes(b"".join(chunks) + f"|size:{file_size}".encode())
    except OSError:
        return None


def _get_context_path_hash(memo_key: str, sentinel: str) -> str | None:
    with _CTX_PATH_HASH_MEMO_LOCK:
        cached = _CTX_PATH_HASH_MEMO.get(memo_key)
        if cached is None:
            return None
        cached_sentinel, digest = cached
        if cached_sentinel != sentinel:
            _CTX_PATH_HASH_MEMO.pop(memo_key, None)
            return None
        _CTX_PATH_HASH_MEMO.move_to_end(memo_key)
        return digest


def _put_context_path_hash(memo_key: str, sentinel: str, digest: str) -> None:
    with _CTX_PATH_HASH_MEMO_LOCK:
        _CTX_PATH_HASH_MEMO[memo_key] = (sentinel, digest)
        _CTX_PATH_HASH_MEMO.move_to_end(memo_key)
        while len(_CTX_PATH_HASH_MEMO) > _CTX_PATH_HASH_MEMO_MAX_ITEMS:
            _CTX_PATH_HASH_MEMO.popitem(last=False)


def _context_path_cache_key(path_like: str | Path) -> str | None:
    # Memoized full-content hash. The stat tuple avoids rereading stable local
    # files while still invalidating normal and rapid same-size replacements.
    path = Path(str(path_like)).expanduser()
    memo = _context_path_hash_memo_key(path)
    if memo is None:
        return None
    memo_key, file_size = memo
    sentinel = _context_path_sentinel(path, file_size)
    if sentinel is None:
        return None
    digest = _get_context_path_hash(memo_key, sentinel)
    if digest is not None:
        return f"file:{digest}"
    try:
        digest = hash_bytes(path.read_bytes())
    except OSError:
        return None
    if _context_path_hash_memo_key(path) == memo:
        _put_context_path_hash(memo_key, sentinel, digest)
    return f"file:{digest}"


def _context_audio_cache_key(context_audio: Any) -> str | None:
    """Stable cache key for one context-audio input (higgs copy)."""
    if isinstance(context_audio, (str, Path)):
        return _context_path_cache_key(context_audio)
    if not isinstance(context_audio, dict):
        return None
    path = context_audio.get("audio_path") or context_audio.get("path")
    if path:
        return _context_path_cache_key(path)
    if "bytes" in context_audio:
        data = context_audio["bytes"]
        if isinstance(data, str):
            data = data.encode()
        return hash_media_item(data)
    encoded = context_audio.get("base64") or context_audio.get("data")
    if encoded is None:
        return None
    raw = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
    return hash_media_item(raw)


def _normalize_context_inputs(
    inputs: dict[str, Any], default_speaker: int
) -> list[dict[str, Any]]:
    """``context`` list, or higgs-compatible ``references`` /
    ``reference_audio``+``reference_text`` aliased into context segments."""
    context = inputs.get("context")
    if context:
        if not isinstance(context, list):
            raise TypeError(f"context must be a list, got {type(context).__name__}")
        return [dict(seg) for seg in context]

    segments: list[dict[str, Any]] = []
    refs = inputs.get("references")
    if refs and isinstance(refs, list):
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            seg: dict[str, Any] = {
                "speaker": ref.get("speaker", default_speaker),
                "text": ref.get("text") or "",
            }
            if "bytes" in ref or "base64" in ref or "data" in ref:
                seg["audio"] = ref
            else:
                seg["audio"] = ref.get("audio_path") or ref.get("path")
            segments.append(seg)
        return segments

    reference_audio = inputs.get("reference_audio")
    if reference_audio is not None:
        segments.append(
            {
                "speaker": inputs.get("reference_speaker", default_speaker),
                "text": inputs.get("reference_text") or "",
                "audio": reference_audio,
            }
        )
    return segments


def create_preprocessing_executor(
    model_path: str,
    *,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> ThreadedSimpleScheduler:
    """CPU preprocessing: parse inputs, tokenize, route the audio path.

    Parses ``payload.request.inputs`` (str | dict with ``input|text``,
    ``speaker``, ``context: [{text, speaker, audio|codes}]``,
    higgs-compatible ``references`` aliasing). Three branches:

    (a) pre-encoded codes → full prompt NOW (F = ``codes.shape[0]``);
    (b) no context → prompt now;
    (c) raw waveform → load/resample 24 kHz mono and DEFER prompt assembly to
        the audio_encoder — the placeholder count must come from the actual
        Mimi encode (kills the off-by-one class; do not optimize away).

    Limits: context audio <= 60 s; text <= ~1500 tokens. Three-tier caching
    copied from higgs (path-hash memo / waveform LRU / encoded-codes LRU as
    CPU int32).

    Returns:
        ``ThreadedSimpleScheduler`` wrapping the preprocess fn.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    builder = CsmPromptBuilder(checkpoint_dir)
    waveform_cache = StageOutputCache(
        max_size=_CTX_WAVEFORM_CACHE_MAX_ITEMS,
        max_bytes=_CTX_WAVEFORM_CACHE_MAX_BYTES,
    )
    waveform_cache_lock = threading.Lock()

    def _load_context_waveform(audio: Any) -> tuple[torch.Tensor, str | None]:
        cache_key = _context_audio_cache_key(audio)
        with waveform_cache_lock:
            cached = waveform_cache.get(cache_key)
        if cached is not None:
            return cached.clone(), cache_key
        waveform_np, sample_rate = load_audio_to_24k(audio)
        wav = torch.from_numpy(waveform_np)
        if sample_rate != SAMPLE_RATE:
            wav = F_audio.resample(wav, sample_rate, SAMPLE_RATE)
        if wav.shape[-1] > MAX_CONTEXT_AUDIO_SECONDS * SAMPLE_RATE:
            raise ValueError(
                f"context audio is too long ({wav.shape[-1] / SAMPLE_RATE:.1f}s); "
                f"cap at {MAX_CONTEXT_AUDIO_SECONDS}s."
            )
        wav = wav.reshape(1, 1, -1).contiguous().float()
        with waveform_cache_lock:
            waveform_cache.put(cache_key, wav.clone())
        return wav, cache_key

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = inputs.get("input") or inputs.get("text") or ""
        speaker_id = int(inputs.get("speaker", inputs.get("speaker_id", 0)) or 0)
        text_tokens = builder.tokenize_segment_text(speaker_id, text)
        if len(text_tokens) > _MAX_TEXT_TOKENS:
            raise ValueError(
                f"input text is too long ({len(text_tokens)} tokens); cap at "
                f"{_MAX_TEXT_TOKENS} tokens after reserving frame budget."
            )

        context_entries: list[dict[str, Any]] = []
        context_codes: list[torch.Tensor | None] = []
        has_raw_audio = False
        for seg in _normalize_context_inputs(inputs, speaker_id):
            seg_speaker = int(
                seg.get("speaker", seg.get("speaker_id", speaker_id)) or 0
            )
            seg_text = seg.get("text") or ""
            entry: dict[str, Any] = {"speaker_id": seg_speaker, "text": seg_text}
            codes = to_codes_F32(seg.get("codes"))
            if codes is not None:
                # (a) pre-encoded fast path: F is just codes.shape[0].
                if codes.shape[0] > _MAX_CONTEXT_FRAMES:
                    raise ValueError(
                        f"context codes are too long ({int(codes.shape[0])} frames); "
                        f"cap at {MAX_CONTEXT_AUDIO_SECONDS}s "
                        f"(~{_MAX_CONTEXT_FRAMES} frames at {FRAME_RATE} Hz)."
                    )
                context_codes.append(codes.to(torch.int32).cpu())
            else:
                audio = seg.get("audio")
                if audio is None:
                    raise ValueError(
                        "each context segment needs either pre-encoded 'codes' "
                        "or raw 'audio'"
                    )
                # (c) raw waveform: defer prompt assembly to audio_encoder so
                # the placeholder count comes from the ACTUAL encode.
                wav, cache_key = _load_context_waveform(audio)
                entry["waveform"] = wav
                entry["cache_key"] = cache_key
                context_codes.append(None)
                has_raw_audio = True
            context_entries.append(entry)

        if has_raw_audio:
            prompt_ids: list[int] = []  # assembled by audio_encoder
        else:
            prompt_ids, _spans = builder.build_prompt(
                text=text,
                speaker_id=speaker_id,
                context=context_entries,
                context_frame_counts=[
                    int(c.shape[0]) for c in context_codes if c is not None
                ],
            )

        max_new_tokens = params.get("max_new_tokens")
        temperature = params.get("temperature")
        top_k = params.get("top_k")
        state = CsmTtsState(
            text=text,
            speaker_id=speaker_id,
            context=context_entries,
            prompt_ids=prompt_ids,
            context_codes=context_codes if context_entries else None,
            max_new_tokens=(
                int(max_new_tokens) if max_new_tokens else DEFAULT_MAX_FRAMES
            ),
            temperature=float(temperature) if temperature is not None else 0.9,
            top_k=int(top_k) if top_k is not None else 50,
            top_p=params.get("top_p"),
            depth_temperature=float(params.get("depth_temperature", 0.9)),
            depth_top_k=int(params.get("depth_top_k", 50)),
            seed=params.get("seed"),
            stream=bool(params.get("stream", False)),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
) -> SimpleScheduler:
    """Mimi ENCODE stage (optional fast-path no-op).

    No-op on the pre-encoded fast path; else ``codec.encode_reference`` →
    ``[F, 32]`` → build the prompt with exactly F ``128002`` placeholders +
    ``128003`` per segment → null the waveform. Startup warmup encode of 1 s
    of zeros. Shares the process-wide Mimi via ``get_or_load_codec``.

    Returns:
        ``SimpleScheduler`` with batched encode.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    builder = CsmPromptBuilder(checkpoint_dir)
    codec = get_or_load_codec(checkpoint_dir, device, "float32")
    # Warmup so the first request doesn't pay kernel autotune/lazy init.
    codec.encode_reference(torch.zeros(1, 1, codec.SAMPLE_RATE), codec.SAMPLE_RATE)
    # Single-threaded SimpleScheduler stage, so no lock needed. Cache CPU int32
    # tensors so StageOutputCache can byte-bound them (higgs pattern).
    code_cache = StageOutputCache(
        max_size=_CTX_CODE_CACHE_MAX_ITEMS,
        max_bytes=_CTX_CODE_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode(payload: StagePayload) -> StagePayload:
        state = CsmTtsState.from_dict(payload.data)
        if state.prompt_ids:
            return payload  # fast path: prompt already assembled upstream

        context_codes = list(state.context_codes or [None] * len(state.context))
        for i, entry in enumerate(state.context):
            if context_codes[i] is not None:
                continue
            waveform = entry.get("waveform")
            if waveform is None:
                raise ValueError(
                    f"context segment {i} has neither codes nor a loaded waveform"
                )
            cache_key = entry.get("cache_key")
            codes = code_cache.get(cache_key)
            if codes is None:
                wav = torch.as_tensor(waveform, dtype=torch.float32)
                codes_F32 = codec.encode_reference(wav, codec.SAMPLE_RATE)
                if codes_F32.ndim != 2 or codes_F32.shape[1] != NUM_CODEBOOKS:
                    raise ValueError(
                        f"Mimi encode must return [F, {NUM_CODEBOOKS}], got "
                        f"{tuple(codes_F32.shape)}"
                    )
                codes = codes_F32.to(torch.int32).cpu()
                code_cache.put(cache_key, codes)
            context_codes[i] = codes
            entry.pop("waveform", None)
            entry.pop("cache_key", None)

        # F per segment from the ACTUAL encode (or codes.shape[0] on the
        # pre-encoded segments) — never the closed-form estimate.
        prompt_ids, _spans = builder.build_prompt(
            text=state.text or "",
            speaker_id=state.speaker_id,
            context=state.context,
            context_frame_counts=[int(c.shape[0]) for c in context_codes],
        )
        state.prompt_ids = prompt_ids
        state.context_codes = context_codes
        payload.data = state.to_dict()
        return payload

    return SimpleScheduler(
        _encode,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int | None = DEFAULT_MAX_FRAMES,
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
) -> OmniScheduler:
    """SGLang-backed frame-AR engine (the-contract recipe).

    Recipe: ``resolve_checkpoint`` →
    ``build_sglang_server_args(checkpoint_dir, context_length=2048,
    disable_cuda_graph=True (eager; set False to enable CUDA graphs), cuda_graph_max_bs=8,
    mem_fraction_static=0.5, max_running_requests=8,
    chunked_prefill_size=2048, dtype="bfloat16", **overrides)`` →
    ``server_args.disable_overlap_schedule = True`` (contract-required) →
    ``create_sglang_infrastructure`` → ``CsmTTSModelRunner(model_worker,
    SGLangOutputProcessor(capture_hidden=False, ...))`` →
    ``make_csm_scheduler_adapters`` → ``OmniScheduler(...,
    abort_callback=model.reset_request)`` →
    ``model_runner.set_stream_outbox(scheduler.outbox)``.

    Deliberate deltas vs higgs: NO ``truncate_rope_to_bf16`` (CSM ckpt is
    fp32-trained — leave SGLang's fp32 cos/sin cache alone); the
    ``on_retract`` callable passed to bootstrap must ABORT with an actionable
    error instead of silently re-prefilling (sampler pool has no rollback).

    Returns:
        Configured ``OmniScheduler``.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        # Eager by default (correctness baseline); set disable_cuda_graph=False to enable CUDA graphs with
        # the captured backbone+cb0+depth step.
        "disable_cuda_graph": True,
        "cuda_graph_max_bs": DEFAULT_MAX_CONCURRENCY,
        "mem_fraction_static": 0.5,
        "max_running_requests": DEFAULT_MAX_CONCURRENCY,
        "chunked_prefill_size": 2048,
        "dtype": "bfloat16",
        # Radix cache is namespaced per context audio via Req.extra_key (set
        # in build_sglang_csm_request); identical 128002xF placeholder
        # prefixes from different voices can't cross-contaminate the KV tree.
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=BACKBONE_CTX,
        **overrides,
    )
    server_args.disable_overlap_schedule = True

    (
        model_worker,
        tree_cache,
        req_to_token_pool,
        token_to_kv_pool_allocator,
        prefill_mgr,
        decode_mgr,
        model_config,
    ) = create_sglang_infrastructure(server_args, gpu_id)

    # NO truncate_rope_to_bf16 here: CSM is fp32-trained, the higgs
    # bf16-training-parity rationale doesn't apply.

    def _abort_on_retract(req: Any) -> None:
        # retraction would replay the prompt and sample a NEW
        # frame 0 while output_frames/the vocoder already consumed the old
        # stream — the sampler pool has no rollback. Admission checks + the
        # 45x-overprovisioned KV pool make this structurally unreachable;
        # fail loudly if it ever fires.
        raise RuntimeError(
            "CSM tts_engine cannot retract request "
            f"{getattr(req, 'rid', '<unknown>')!r}: the frame sampler/vocoder "
            "stream has no rollback. Retraction means the KV "
            "pool is overcommitted — raise mem_fraction_static or lower "
            "max_running_requests."
        )

    decode_mgr.on_retract = _abort_on_retract

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = CsmTTSModelRunner(model_worker, output_proc)
    model = model_worker.model_runner.model
    request_builder, result_adapter = make_csm_scheduler_adapters(
        model,
        max_new_tokens_cap=max_new_tokens,
    )

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
        abort_callback=model.reset_request,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    )
    model_runner.set_stream_outbox(scheduler.outbox)
    return scheduler


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "float32",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 13,
    stream_followup_stride: int = 6,
    stream_overlap_frames: int = 4,
    stream_holdback_frames: int = 2,
) -> CsmStreamingVocoderScheduler:
    """Mimi DECODE stage (fp32 default — conv-transpose stability).

    ``resolve_checkpoint`` → ``get_or_load_codec`` (shared with the encoder)
    → :class:`CsmStreamingVocoderScheduler` with the 12.5 Hz framing knobs.

    Returns:
        ``CsmStreamingVocoderScheduler``.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    codec = get_or_load_codec(checkpoint_dir, device, dtype)

    return CsmStreamingVocoderScheduler(
        codec,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_frames=stream_overlap_frames,
        stream_holdback_frames=stream_holdback_frames,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "MAX_CONTEXT_AUDIO_SECONDS",
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
