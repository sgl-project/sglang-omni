# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Fish Audio Drama 3 pipeline."""

from __future__ import annotations

import gc
import logging
import math
import os
import re
from collections.abc import Mapping
from typing import Literal

import torch
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.config import (
    DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE,
    DEFAULT_CONTEXT_LENGTH,
    DEFAULT_ENABLE_ASYNC_DECODE,
    DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS,
    DEFAULT_FIRST_FRAME_CUDA_GRAPHS,
    DEFAULT_MAX_NEW_TOKENS_CAP,
    DEFAULT_MAX_PROMPT_ROWS,
    DEFAULT_MAX_RUNNING_REQUESTS,
    DEFAULT_REFERENCE_SAMPLING,
    DEFAULT_REFERENCE_SAMPLING_FALLBACK,
    DEFAULT_STREAM_FOLLOWUP_STRIDE,
    DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
    DEFAULT_STREAM_STRIDE,
    DEFAULT_TOWER_FUSED_LAYERS,
    DEFAULT_TRITON_SAMPLING,
    ReferenceSamplingFallback,
)
from sglang_omni.models.fishaudio_drama3.engine_builder import Drama3EngineBuilder
from sglang_omni.models.fishaudio_drama3.hf_config import FishOmniConfig
from sglang_omni.models.fishaudio_drama3.incremental_decoder import (
    IncrementalDecoder,
    IncrementalWindowDecoder,
    incremental_window_lengths,
)
from sglang_omni.models.fishaudio_drama3.request_builders import Drama3RequestData
from sglang_omni.models.fishaudio_drama3.speech_request import Drama3SpeechPreprocessor
from sglang_omni.models.fishaudio_drama3.stream_windows import stream_window_lengths
from sglang_omni.models.fishaudio_drama3.vocoder import (
    Drama3IncrementalVocoderScheduler,
    Drama3VocoderScheduler,
    WindowGraphDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.stages import (
    FishReferenceEncodeHook,
    load_codec,
)
from sglang_omni.scheduling.audio_stream import WindowDecoder
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.reference_encoder import ReferenceEncodeService
from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)

CODEC_FILE_NAME = "codec.pth"
REFERENCE_ENCODE_WARMUP_SECONDS = 1.0
MEMORY_PROBE_ROW_COUNT = 2
MEMORY_PROBE_FRAME_COUNT = 128
# sglang 0.5.23 FA3 paged-KV kernel: SM 8.x/9.x, CUDA >= 12.3.
FA3_COMPUTE_CAPABILITY_MAJORS = frozenset({8, 9})
MINIMUM_FA3_CUDA_VERSION = (12, 3)
# A hub codec names a full commit, so preprocessing and the vocoder load one file.
HUB_CODEC_PATTERN = re.compile(
    r"(?P<repo_id>[^@\s]+/[^@\s]+)@(?P<revision>[0-9a-f]{40})"
)


def cuda_compute_capability(device: torch.device) -> tuple[int, int] | None:
    try:
        major, minor = torch.cuda.get_device_capability(device)
        return major, minor
    except (AssertionError, AttributeError, RuntimeError, ValueError):
        return None


def cuda_runtime_version() -> tuple[int, ...] | None:
    version = getattr(torch.version, "cuda", None)
    if not version:
        return None
    else:
        pass
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None


def fa3_available(device: torch.device) -> bool:
    """FA3's paged KV attention is available only on SM 8.x/9.x (sglang 0.5.23)."""
    if device.type != "cuda":
        return False
    else:
        pass
    capability = cuda_compute_capability(device)
    if capability is None or capability[0] not in FA3_COMPUTE_CAPABILITY_MAJORS:
        return False
    else:
        pass
    version = cuda_runtime_version()
    return version is not None and version >= MINIMUM_FA3_CUDA_VERSION


def incremental_decode_enabled(
    *,
    stream_decode: str,
    decode_dtype: str,
    device: torch.device,
    fa3_supported: bool,
) -> bool:
    """Incremental decoding needs CUDA fp16 and FA3's paged KV attention."""
    return (
        stream_decode == "incremental"
        and decode_dtype == "float16"
        and device.type == "cuda"
        and fa3_supported
    )


def resolve_codec_directory(codec_path: str) -> str:
    """The local directory holding the codec.pth codec_path names: that file, a
    directory holding it, or <repo_id>@<40-hex commit>, of which only codec.pth
    downloads."""
    hub_codec = HUB_CODEC_PATTERN.fullmatch(codec_path)
    if os.path.isfile(codec_path) and os.path.basename(codec_path) == CODEC_FILE_NAME:
        return os.path.dirname(os.path.abspath(codec_path))
    elif os.path.isfile(os.path.join(codec_path, CODEC_FILE_NAME)):
        return codec_path
    elif hub_codec is not None:
        return os.path.dirname(
            hf_hub_download(
                hub_codec["repo_id"], CODEC_FILE_NAME, revision=hub_codec["revision"]
            )
        )
    else:
        raise ValueError(
            f"codec_path {codec_path!r} names no {CODEC_FILE_NAME} file, no "
            f"directory holding one and no <repo_id>@<40-hex commit> on the Hub"
        )


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_concurrency: int,
    max_prompt_rows: int,
    codec_path: str,
    reference_encode_dtype: Literal["float32", "bfloat16"],
    max_reference_audio_seconds: float,
) -> ThreadedSimpleScheduler:
    """Speech requests to prompt rows; reference audio encodes with the vocoder's codec."""
    resolved_device = str(resolve_concrete_device(device, gpu_id))
    is_on_cpu = torch.device(resolved_device).type == "cpu"
    if is_on_cpu and reference_encode_dtype == "bfloat16":
        raise ValueError(
            "Drama 3 preprocessing encodes references in bfloat16 only on an "
            "accelerator; on the CPU set "
            "stages.preprocessing.factory.reference_encode_dtype to float32"
        )
    else:
        pass
    checkpoint_dir = resolve_checkpoint(model_path)
    codec_directory = resolve_codec_directory(codec_path)
    codec = load_codec(codec_directory, resolved_device)
    reference_encode_hook = FishReferenceEncodeHook(
        codec=codec,
        checkpoint_id=os.path.join(codec_directory, CODEC_FILE_NAME),
        device=resolved_device,
        autocast_dtype=torch.bfloat16 if reference_encode_dtype == "bfloat16" else None,
        max_reference_audio_seconds=max_reference_audio_seconds,
    )
    if not is_on_cpu:
        warmup_seconds = min(
            REFERENCE_ENCODE_WARMUP_SECONDS, max_reference_audio_seconds
        )
        reference_encode_hook.encode_reference_waveform(
            torch.zeros((1, int(codec.sample_rate * warmup_seconds))),
            int(codec.sample_rate),
        )
    else:
        pass
    preprocessor = Drama3SpeechPreprocessor(
        tokenizer=Tokenizer.from_file(os.path.join(checkpoint_dir, "tokenizer.json")),
        config=FishOmniConfig.from_pretrained(checkpoint_dir),
        max_prompt_rows=max_prompt_rows,
        reference_encoder=ReferenceEncodeService(
            reference_encode_hook, log_prefix="Fish Audio Drama 3"
        ),
    )
    return ThreadedSimpleScheduler(preprocessor, max_concurrency=max_concurrency)


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    context_length: int = DEFAULT_CONTEXT_LENGTH,
    max_running_requests: int = DEFAULT_MAX_RUNNING_REQUESTS,
    max_new_tokens: int = DEFAULT_MAX_NEW_TOKENS_CAP,
    max_prompt_rows: int = DEFAULT_MAX_PROMPT_ROWS,
    server_args_overrides: Mapping[str, object] | None = None,
    stream_stride: int = DEFAULT_STREAM_STRIDE,
    stream_followup_stride_ramp: tuple[int, ...] = DEFAULT_STREAM_FOLLOWUP_STRIDE_RAMP,
    stream_followup_stride: int = DEFAULT_STREAM_FOLLOWUP_STRIDE,
    first_frame_cuda_graphs: bool = DEFAULT_FIRST_FRAME_CUDA_GRAPHS,
    first_frame_cuda_graph_max_requests: int = (
        DEFAULT_FIRST_FRAME_CUDA_GRAPH_MAX_REQUESTS
    ),
    triton_sampling: bool = DEFAULT_TRITON_SAMPLING,
    reference_sampling: bool = DEFAULT_REFERENCE_SAMPLING,
    reference_sampling_fallback: ReferenceSamplingFallback = (
        DEFAULT_REFERENCE_SAMPLING_FALLBACK
    ),
    tower_fused_layers: bool = DEFAULT_TOWER_FUSED_LAYERS,
    enable_async_decode: bool = DEFAULT_ENABLE_ASYNC_DECODE,
    async_decode_min_batch_size: int = DEFAULT_ASYNC_DECODE_MIN_BATCH_SIZE,
) -> OmniScheduler[Drama3RequestData]:
    overrides = dict(server_args_overrides or {})
    mem_fraction_static = overrides.get("mem_fraction_static")
    if total_gpu_memory_fraction is None:
        pass
    elif mem_fraction_static is None:
        overrides["mem_fraction_static"] = total_gpu_memory_fraction
    elif mem_fraction_static != total_gpu_memory_fraction:
        raise ValueError(
            f"Drama 3 sizes its engine from the tts_engine stage's "
            f"gpu_memory_fraction ({total_gpu_memory_fraction:g}), so "
            f"mem_fraction_static={mem_fraction_static} would be ignored; set "
            f"--tts_engine.gpu_memory_fraction instead of mem_fraction_static"
        )
    else:
        pass
    return Drama3EngineBuilder(
        context_length=context_length,
        max_running_requests=max_running_requests,
        stream_stride=stream_stride,
        stream_followup_stride_ramp=stream_followup_stride_ramp,
        stream_followup_stride=stream_followup_stride,
        max_prompt_rows=max_prompt_rows,
        max_new_tokens_cap=max_new_tokens,
        first_frame_cuda_graphs=first_frame_cuda_graphs,
        first_frame_cuda_graph_max_requests=first_frame_cuda_graph_max_requests,
        triton_sampling=triton_sampling,
        reference_sampling=reference_sampling,
        reference_sampling_fallback=reference_sampling_fallback,
        tower_fused_layers=tower_fused_layers,
        enable_async_decode=enable_async_decode,
        async_decode_min_batch_size=async_decode_min_batch_size,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=overrides or None,
        total_gpu_memory_fraction=(
            process_total_gpu_memory_fraction
            if process_total_gpu_memory_fraction is not None
            else total_gpu_memory_fraction
        ),
    )


def measure_max_padded_frames_per_decode(
    decode: WindowDecoder,
    *,
    device: torch.device,
    num_codebooks: int,
    total_gpu_memory_fraction: float | None,
    max_new_tokens: int,
) -> int:
    device_module = torch.get_device_module(device)
    device_module.empty_cache()
    held_bytes = device_module.memory_reserved(device)
    probe_codes = torch.zeros(
        (MEMORY_PROBE_ROW_COUNT, num_codebooks, MEMORY_PROBE_FRAME_COUNT),
        dtype=torch.long,
        device=device,
    )
    allocated_bytes = device_module.memory_allocated(device)
    device_module.reset_peak_memory_stats(device)
    with torch.no_grad():
        decode(probe_codes)
    probe_peak_bytes = device_module.max_memory_allocated(device) - allocated_bytes
    device_module.empty_cache()
    bytes_per_padded_frame = math.ceil(
        probe_peak_bytes / (MEMORY_PROBE_ROW_COUNT * MEMORY_PROBE_FRAME_COUNT)
    )
    device_bytes = device_module.get_device_properties(device).total_memory
    if total_gpu_memory_fraction is None:
        share_bytes = device_bytes
    else:
        share_bytes = int(total_gpu_memory_fraction * device_bytes)
    max_padded_frames_per_decode = (share_bytes - held_bytes) // bytes_per_padded_frame
    if max_padded_frames_per_decode < max_new_tokens:
        raise ValueError(
            f"the vocoder's GPU memory share of {share_bytes / 2**30:.2f} GiB holds "
            f"{held_bytes / 2**30:.2f} GiB of codec weights and CUDA graphs and "
            f"leaves room for {max_padded_frames_per_decode} padded frames per "
            f"decode, fewer than one request of max_new_tokens={max_new_tokens} "
            f"frames; raise --vocoder.gpu_memory_fraction or lower "
            f"--tts_engine.factory.max_new_tokens"
        )
    else:
        pass
    logger.info(
        f"Drama 3 vocoder: a non-streaming decode takes up to "
        f"{max_padded_frames_per_decode} padded frames of {bytes_per_padded_frame} "
        f"bytes each; share {share_bytes / 2**30:.2f} GiB, held "
        f"{held_bytes / 2**30:.2f} GiB"
    )
    return max_padded_frames_per_decode


def create_vocoder_executor(
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    codec_path: str,
    stream_stride: int,
    stream_followup_stride_ramp: tuple[int, ...],
    stream_followup_stride: int,
    stream_overlap_tokens: int,
    stream_crossfade_samples: int,
    max_batch_size: int,
    max_batch_wait_ms: int,
    window_cuda_graphs: bool,
    decode_dtype: Literal["float32", "float16"],
    max_new_tokens: int,
    total_gpu_memory_fraction: float | None = None,
    stream_decode: Literal["incremental", "overlap"] = "overlap",
    max_stream_batch_size: int = 16,
    stream_slot_count: int = 256,
    decode_step_message_budget: int = 8,
) -> Drama3VocoderScheduler:
    resolved_device = str(resolve_concrete_device(device, gpu_id))
    torch_device = torch.device(resolved_device)
    codec = load_codec(resolve_codec_directory(codec_path), resolved_device)
    codec.fold_weight_norm()
    num_codebooks = (
        codec.quantizer.semantic_quantizer.n_codebooks
        + codec.quantizer.quantizer.n_codebooks
    )
    if decode_dtype == "float16":
        decode: WindowDecoder = ChannelsLastDecoder(codec, dtype=torch.float16)
    else:
        decode = codec.from_indices
    is_incremental = incremental_decode_enabled(
        stream_decode=stream_decode,
        decode_dtype=decode_dtype,
        device=torch_device,
        fa3_supported=fa3_available(torch_device),
    )
    if (
        stream_decode == "incremental"
        and not is_incremental
        and torch_device.type == "cuda"
    ):
        logger.warning(
            "Drama 3 vocoder: incremental decoding needs FA3 (SM 8.x/9.x); "
            "falling back to overlap decoding on %s",
            torch_device,
        )
    else:
        pass

    def overlap_window() -> WindowDecoder:
        if window_cuda_graphs:
            return WindowGraphDecoder(
                decode,
                num_codebooks=num_codebooks,
                frame_length=int(codec.frame_length),
                device=torch_device,
                window_lengths=stream_window_lengths(
                    stream_stride=stream_stride,
                    stream_followup_stride_ramp=stream_followup_stride_ramp,
                    stream_followup_stride=stream_followup_stride,
                    stream_overlap_tokens=stream_overlap_tokens,
                ),
            )
        else:
            return decode

    window_decoder: IncrementalWindowDecoder | None = None
    if is_incremental:
        window_decoder = build_incremental_window_decoder(
            decode,
            num_codebooks=num_codebooks,
            device=torch_device,
            strides=(
                stream_stride,
                *stream_followup_stride_ramp,
                stream_followup_stride,
            ),
            max_stream_batch_size=max_stream_batch_size,
            stream_slot_count=stream_slot_count,
            use_graphs=window_cuda_graphs,
        )
        decode_window = decode
        logger.info(
            f"Drama 3 vocoder: incremental streams in up to {stream_slot_count} slots, "
            f"grouped up to {max_stream_batch_size} windows of "
            f"{window_decoder.window_lengths} frames"
        )
    else:
        decode_window = overlap_window()

    if torch_device.type == "cpu":
        max_padded_frames_per_decode = None
    else:
        try:
            max_padded_frames_per_decode = measure_max_padded_frames_per_decode(
                decode,
                device=torch_device,
                num_codebooks=num_codebooks,
                total_gpu_memory_fraction=total_gpu_memory_fraction,
                max_new_tokens=max_new_tokens,
            )
        except ValueError as budget_error:
            if window_decoder is None:
                raise
            else:
                pass
            logger.warning(
                "Drama 3 vocoder: incremental decoding does not fit the vocoder's "
                "GPU memory share (%s); falling back to overlap decoding. Raise "
                "--vocoder.gpu_memory_fraction or lower "
                "--vocoder.factory.stream_slot_count to keep incremental streams.",
                budget_error,
            )
            del window_decoder
            gc.collect()
            device_module = torch.get_device_module(torch_device)
            device_module.synchronize(torch_device)
            device_module.empty_cache()
            window_decoder = None
            decode_window = overlap_window()
            max_padded_frames_per_decode = measure_max_padded_frames_per_decode(
                decode,
                device=torch_device,
                num_codebooks=num_codebooks,
                total_gpu_memory_fraction=total_gpu_memory_fraction,
                max_new_tokens=max_new_tokens,
            )
    scheduler_args = dict(
        device=resolved_device,
        stream_stride=stream_stride,
        stream_followup_stride_ramp=stream_followup_stride_ramp,
        stream_followup_stride=stream_followup_stride,
        stream_overlap_tokens=stream_overlap_tokens,
        stream_crossfade_samples=stream_crossfade_samples,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        decode_window=decode_window,
        max_padded_frames_per_decode=max_padded_frames_per_decode,
    )
    if window_decoder is None:
        return Drama3VocoderScheduler(codec, **scheduler_args)
    else:
        return Drama3IncrementalVocoderScheduler(
            codec,
            window_decoder=window_decoder,
            max_stream_batch_size=max_stream_batch_size,
            stream_slot_count=stream_slot_count,
            max_stream_frames=int(codec.quantizer.post_module.freqs_cis.shape[0]),
            decode_step_message_budget=decode_step_message_budget,
            **scheduler_args,
        )


def build_incremental_window_decoder(
    decode: ChannelsLastDecoder,
    *,
    num_codebooks: int,
    device: torch.device,
    strides: tuple[int, ...],
    max_stream_batch_size: int,
    stream_slot_count: int,
    use_graphs: bool,
) -> IncrementalWindowDecoder:
    batch_sizes = {max_stream_batch_size}
    batch_size = 1
    while batch_size < max_stream_batch_size:
        batch_sizes.add(batch_size)
        batch_size *= 2
    return IncrementalWindowDecoder(
        IncrementalDecoder(
            decode,
            stream_slots=stream_slot_count,
            max_window_frames=max(strides),
            device=device,
        ),
        num_codebooks=num_codebooks,
        frame_length=int(decode.codec.frame_length),
        device=device,
        batch_sizes=tuple(batch_sizes),
        window_lengths=incremental_window_lengths(strides),
        use_graphs=use_graphs,
    )
