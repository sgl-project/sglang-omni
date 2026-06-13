# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Higgs TTS pipeline.

Pipeline shape::

    preprocessing → audio_encoder → tts_engine → vocoder

- ``create_preprocessing_executor``: text tokenize + (if raw audio path)
  load waveform; fast path also delay-encodes client-supplied
  ``reference_codes`` and builds the prompt. Returns a
  :class:`ThreadedSimpleScheduler` for CPU-heavy work.
- ``create_audio_encoder_executor``: GPU codec encode for the raw-audio
  path → delayed ref codes + prompt assembly. No-op on the fast path.
- ``create_sglang_tts_engine_executor``: runs :class:`HiggsTTSModel` under
  sglang's worker; the model runner computes the fused multi-codebook
  embedding inline in prefill from ``reference_codes_delayed`` and overlays
  it at ``-100`` placeholder positions. Returns a :class:`OmniScheduler`.
- ``create_vocoder_executor``: creates the Higgs vocoder scheduler, preserving
  batched non-streaming decode and incremental streaming audio chunks.
"""

from __future__ import annotations

import base64
import logging
import os
import threading
from pathlib import Path
from typing import Any

import torch
import torchaudio.functional as F_audio
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.request_builders import make_higgs_scheduler_adapters
from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter
from sglang_omni.models.higgs_tts.utils import (
    apply_delay_pattern,
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    to_codes_TN,
    truncate_rope_to_bf16,
)
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    HiggsStreamingVocoderScheduler,
)

# _REF_PATH_HASH_MEMO is the shared memo object, re-exported so tests can
# reset it; the underscored alias keeps this module's historical API.
from sglang_omni.preprocessing.cache_key import _REF_PATH_HASH_MEMO  # noqa: F401
from sglang_omni.preprocessing.cache_key import hash_bytes, hash_media_item
from sglang_omni.preprocessing.cache_key import (
    reference_path_cache_key as _reference_path_cache_key,
)
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


# Codec runs at 75 Hz; chunked prefill of the multi-codebook prompt is unsafe
# (sampler state machine has no rollback) so reject inputs past chunked_prefill_size.
_MAX_REF_AUDIO_SEC = 100
_REF_CODE_CACHE_MAX_ITEMS = 256
_REF_CODE_CACHE_MAX_BYTES = 256 * 1024 * 1024
_REF_WAVEFORM_CACHE_MAX_ITEMS = 256
_REF_WAVEFORM_CACHE_MAX_BYTES = 512 * 1024 * 1024

# Saturates near c=16 on H100/H200; higher client concurrency only queues.
DEFAULT_MAX_CONCURRENCY = 16


def _reference_audio_cache_key(reference_audio: Any) -> str | None:
    """Safe source key for preprocessing waveform-cache lookup."""
    if isinstance(reference_audio, (str, Path)):
        return _reference_path_cache_key(reference_audio)
    if not isinstance(reference_audio, dict):
        return None
    path = reference_audio.get("audio_path") or reference_audio.get("path")
    if path:
        return _reference_path_cache_key(path)
    if "bytes" in reference_audio:
        data = reference_audio["bytes"]
        if isinstance(data, str):
            data = data.encode()
        return hash_media_item(data)
    encoded = reference_audio.get("base64") or reference_audio.get("data")
    if encoded is None:
        return None
    raw = base64.b64decode(encoded) if isinstance(encoded, str) else bytes(encoded)
    return hash_media_item(raw)


def _reference_code_cache_key_from_waveform(
    waveform: torch.Tensor, sample_rate: int
) -> str:
    """Content key for the reference-code cache after audio decode/resample.

    Hashing the waveform consumed by the codec keeps cache reuse tied to actual
    audio content across local files, bytes/base64 payloads, and URL refs.
    """
    wav = waveform.detach().cpu().contiguous().float()
    meta = f"sr:{int(sample_rate)}|shape:{tuple(wav.shape)}"
    return f"waveform:{meta}:{hash_bytes(wav.numpy().tobytes())}"


def create_preprocessing_executor(
    model_path: str,
    *,
    num_codebooks: int = 8,
    codebook_size: int = 1026,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
):
    """CPU stage: text tokenize + optional ref-audio file IO.

    Builds the full prompt + delays the codes when the client supplied
    pre-encoded ``reference_codes``. When raw audio is supplied, defers
    codec encoding (and prompt assembly) to the audio_encoder stage —
    only the loaded waveform is shipped forward.
    """
    checkpoint_dir = resolve_checkpoint(model_path)

    # Note:(Chenchen Hong) Load tokenizer.json directly to avoid checkpoint metadata drift.
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)
    # Runs on a ThreadedSimpleScheduler pool for preprocessing;
    reference_waveform_cache = StageOutputCache(
        max_size=_REF_WAVEFORM_CACHE_MAX_ITEMS,
        max_bytes=_REF_WAVEFORM_CACHE_MAX_BYTES,
    )
    reference_waveform_cache_lock = threading.Lock()

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        raw_refs = inputs.get("references")
        if raw_refs and isinstance(raw_refs, list):
            first = raw_refs[0]
            if isinstance(first, dict):
                inputs = dict(inputs)
                if first.get("text") and not inputs.get("reference_text"):
                    inputs["reference_text"] = first["text"]
                if inputs.get("reference_audio") is None:
                    if "bytes" in first or "base64" in first or "data" in first:
                        inputs["reference_audio"] = first
                    else:
                        inputs["reference_audio"] = first.get(
                            "audio_path"
                        ) or first.get("path")

        text = inputs.get("input") or inputs.get("text") or ""
        reference_text = inputs.get("reference_text") or None
        ref_codes_TN = to_codes_TN(inputs.get("reference_codes"), num_codebooks)
        if ref_codes_TN is not None and ref_codes_TN.shape[0] > _MAX_REF_AUDIO_SEC * 75:
            raise ValueError(
                f"reference_codes is too long ({ref_codes_TN.shape[0]} frames); "
                f"cap at {_MAX_REF_AUDIO_SEC}s of audio "
                f"(~{_MAX_REF_AUDIO_SEC * 75} frames at 75 Hz)."
            )

        waveform_tensor = None
        reference_code_cache_key = None
        if ref_codes_TN is None and inputs.get("reference_audio") is not None:
            reference_audio = inputs["reference_audio"]
            reference_source_key = _reference_audio_cache_key(reference_audio)
            with reference_waveform_cache_lock:
                cached_reference = reference_waveform_cache.get(reference_source_key)
            if cached_reference is not None:
                cached_waveform, reference_code_cache_key = cached_reference
                waveform_tensor = cached_waveform.clone()
            if waveform_tensor is None:
                waveform_np, sample_rate = load_audio_to_24k(reference_audio)
                wav = torch.from_numpy(waveform_np)
                if sample_rate != 24000:
                    wav = F_audio.resample(wav, sample_rate, 24000)
                if wav.shape[-1] > _MAX_REF_AUDIO_SEC * 24000:
                    raise ValueError(
                        f"reference_audio is too long "
                        f"({wav.shape[-1] / 24000:.1f}s); cap at {_MAX_REF_AUDIO_SEC}s."
                    )
                waveform_tensor = wav.view(1, 1, -1).contiguous().float()
                reference_code_cache_key = _reference_code_cache_key_from_waveform(
                    waveform_tensor, 24000
                )
                if reference_source_key is not None:
                    with reference_waveform_cache_lock:
                        reference_waveform_cache.put(
                            reference_source_key,
                            (waveform_tensor.clone(), reference_code_cache_key),
                        )

        if ref_codes_TN is not None:
            delayed = apply_delay_pattern(ref_codes_TN)
            prompt_ids = adapter.build_prompt(
                text,
                num_ref_tokens=delayed.shape[0],
                reference_text=reference_text,
            )
            ref_codes_delayed: list[list[int]] | None = delayed.tolist()
            target_text_for_encoder = None
            reference_text_for_encoder = None
        elif waveform_tensor is None:
            prompt_ids = adapter.build_prompt(
                text, num_ref_tokens=0, reference_text=reference_text
            )
            ref_codes_delayed = None
            target_text_for_encoder = None
            reference_text_for_encoder = None
        else:
            prompt_ids = []
            ref_codes_delayed = None
            target_text_for_encoder = text
            reference_text_for_encoder = reference_text

        state = HiggsTtsState(
            prompt_token_ids=prompt_ids,
            reference_codes_delayed=ref_codes_delayed,
            reference_waveform=waveform_tensor,
            reference_code_cache_key=reference_code_cache_key,
            target_text=target_text_for_encoder,
            reference_text=reference_text_for_encoder,
            num_codebooks=num_codebooks,
            codebook_size=codebook_size,
            max_new_tokens=int(params.get("max_new_tokens", 2048)),
            temperature=float(params.get("temperature", 1.0)),
            top_p=params.get("top_p"),
            top_k=params.get("top_k"),
            seed=params.get("seed"),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=max_concurrency)


def create_audio_encoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    dtype: str = "bfloat16",
    num_codebooks: int = 8,
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
):
    """GPU stage: codec-encode raw ref audio → delayed codes + prompt assembly.

    No-op when preprocessing already produced ``reference_codes_delayed`` (the
    client-supplied pre-encoded fast path). Codec weights are extracted from
    the TTS checkpoint itself (bundled at ``tied.embedding.modality_embeddings``).
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)

    codec = get_or_load_codec(checkpoint_dir, device, dtype)
    codec.model.acoustic_encoder = torch.compile(
        codec.model.acoustic_encoder, mode="default", dynamic=True
    )
    codec.encode_reference(
        torch.zeros(codec.SAMPLE_RATE), sample_rate=codec.SAMPLE_RATE
    )
    # Single-threaded SimpleScheduler stage, so no lock needed. Cache a CPU
    # tensor (not list[list[int]]) so StageOutputCache can byte-bound it.
    reference_code_cache = StageOutputCache(
        max_size=_REF_CODE_CACHE_MAX_ITEMS,
        max_bytes=_REF_CODE_CACHE_MAX_BYTES,
        cache_device="cpu",
    )

    def _encode(payload: StagePayload) -> StagePayload:
        state = HiggsTtsState.from_dict(payload.data)
        waveform = state.reference_waveform
        if waveform is None:
            return payload

        cached_delayed = reference_code_cache.get(state.reference_code_cache_key)
        if cached_delayed is not None:
            delayed_rows = cached_delayed.tolist()
        else:
            ref_codes_TN = codec.encode_reference(waveform, sample_rate=24000).to(
                torch.long
            )
            if ref_codes_TN.ndim != 2 or ref_codes_TN.shape[1] != num_codebooks:
                raise ValueError(
                    f"codec output must be [T, {num_codebooks}], got "
                    f"{tuple(ref_codes_TN.shape)}"
                )
            delayed = apply_delay_pattern(ref_codes_TN)
            delayed_rows = delayed.tolist()
            reference_code_cache.put(
                state.reference_code_cache_key, delayed.to("cpu", torch.int32)
            )
        state.reference_codes_delayed = delayed_rows
        state.prompt_token_ids = adapter.build_prompt(
            state.target_text or "",
            num_ref_tokens=len(delayed_rows),
            reference_text=state.reference_text,
        )
        state.reference_waveform = None
        state.reference_code_cache_key = None
        state.target_text = None
        state.reference_text = None
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
    max_new_tokens: int | None = 2048,
    server_args_overrides: dict[str, Any] | None = None,
    enable_async_decode: bool = False,
    async_decode_min_batch_size: int = 2,
):
    """sglang-backed AR engine for Higgs TTS."""
    checkpoint_dir = resolve_checkpoint(model_path)
    gpu_id = int(device.split(":")[-1]) if ":" in device else 0

    overrides: dict[str, Any] = {
        "disable_cuda_graph": False,
        "cuda_graph_max_bs": DEFAULT_MAX_CONCURRENCY,
        "mem_fraction_static": 0.85,
        "max_running_requests": DEFAULT_MAX_CONCURRENCY,
        "chunked_prefill_size": 8192,
        "dtype": "bfloat16",
        # Radix cache is namespaced per ref-audio via Req.extra_key (set in
        # build_sglang_higgs_request); shared -100 placeholder prefixes from
        # different ref audios can't cross-contaminate the KV tree.
    }
    if server_args_overrides:
        overrides.update(server_args_overrides)

    server_args = build_sglang_server_args(
        checkpoint_dir,
        context_length=4096,
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

    truncate_rope_to_bf16(model_worker.model_runner.model)

    output_proc = SGLangOutputProcessor(
        capture_hidden=False,
        capture_hidden_layers=None,
        model=model_worker.model_runner.model,
    )
    model_runner = HiggsTTSModelRunner(model_worker, output_proc)
    model = model_worker.model_runner.model
    request_builder, result_adapter = make_higgs_scheduler_adapters(
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
    dtype: str = "bfloat16",
    max_batch_size: int = DEFAULT_MAX_CONCURRENCY,
    max_batch_wait_ms: int = 2,
    stream_stride: int = 75,
    stream_followup_stride: int = 75,
    stream_overlap_tokens: int = 8,
    stream_holdback_tokens: int = 4,
):
    """Decode Higgs delayed codes to a mono 24 kHz waveform.

    Codec weights are extracted from the TTS checkpoint itself.
    """
    checkpoint_dir = resolve_checkpoint(model_path)
    codec = get_or_load_codec(checkpoint_dir, device, dtype)

    return HiggsStreamingVocoderScheduler(
        codec,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        stream_stride=stream_stride,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_holdback_tokens=stream_holdback_tokens,
    )


__all__ = [
    "DEFAULT_MAX_CONCURRENCY",
    "create_audio_encoder_executor",
    "create_preprocessing_executor",
    "create_sglang_tts_engine_executor",
    "create_vocoder_executor",
]
