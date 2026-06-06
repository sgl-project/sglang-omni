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

import logging
import os
import threading
from typing import Any

import torch
from tokenizers import Tokenizer
from transformers import PreTrainedTokenizerFast

from sglang_omni.models.higgs_tts.model_runner import HiggsTTSModelRunner
from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
from sglang_omni.models.higgs_tts.preprocessing import (
    HiggsPreprocessingConfig,
    build_higgs_preprocessed_state,
    encode_higgs_reference_audio,
    normalize_higgs_inputs,
    prepare_higgs_reference,
)
from sglang_omni.models.higgs_tts.request_builders import make_higgs_scheduler_adapters
from sglang_omni.models.higgs_tts.text_tokenizer import HiggsTokenizerAdapter
from sglang_omni.models.higgs_tts.utils import (
    get_or_load_codec,
    load_audio_to_24k,
    resolve_checkpoint,
    truncate_rope_to_bf16,
)
from sglang_omni.models.higgs_tts.vocoder_scheduler import (
    HiggsStreamingVocoderScheduler,
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

    # Higgs ckpt tokenizer_config.json uses transformers v5 metadata and crashes
    # transformers<5's from_pretrained; load tokenizer.json directly to avoid it.
    raw = Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json"))
    tokenizer = PreTrainedTokenizerFast(tokenizer_object=raw)
    adapter = HiggsTokenizerAdapter(tokenizer)
    config = HiggsPreprocessingConfig(
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        max_ref_audio_sec=_MAX_REF_AUDIO_SEC,
    )
    reference_waveform_cache = StageOutputCache(
        max_size=_REF_WAVEFORM_CACHE_MAX_ITEMS,
        max_bytes=_REF_WAVEFORM_CACHE_MAX_BYTES,
    )
    reference_waveform_cache_lock = threading.Lock()

    def _preprocess(payload: StagePayload) -> StagePayload:
        params = payload.request.params or {}
        inputs = normalize_higgs_inputs(payload.request.inputs)
        prepared = prepare_higgs_reference(
            inputs,
            adapter=adapter,
            reference_waveform_cache=reference_waveform_cache,
            reference_waveform_cache_lock=reference_waveform_cache_lock,
            load_audio_fn=load_audio_to_24k,
            config=config,
        )
        state = build_higgs_preprocessed_state(
            prepared,
            params=params,
            config=config,
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
        if state.reference_waveform is None:
            return payload

        state = encode_higgs_reference_audio(
            state,
            codec=codec,
            adapter=adapter,
            reference_code_cache=reference_code_cache,
            num_codebooks=num_codebooks,
        )
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
