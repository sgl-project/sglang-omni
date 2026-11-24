# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Zonos3 pipeline: preprocessing -> tts_engine -> vocoder."""

from __future__ import annotations

import os

from tokenizers import Tokenizer

from sglang_omni.models.zonos3.components.codec import (
    Zonos3CodecDecoder,
    codec_registry_entry,
)
from sglang_omni.models.zonos3.components.codec_encoder import Zonos3CodecEncoder
from sglang_omni.models.zonos3.components.codec_stream import Zonos3DecoderSlots
from sglang_omni.models.zonos3.engine_builder import Zonos3EngineBuilder
from sglang_omni.models.zonos3.hf_config import (
    LANGUAGE_MODEL_DIRECTORY,
    load_params,
    zonos3_config_from_params,
)
from sglang_omni.models.zonos3.prompt import Zonos3SpeechPreprocessor
from sglang_omni.models.zonos3.request_builders import Zonos3RequestData
from sglang_omni.models.zonos3.streaming_vocoder import Zonos3StreamingVocoderScheduler
from sglang_omni.platforms import current_platform
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

if current_platform.is_cuda():
    from sglang.kernels.ops.attention.flash_attention_v3 import (  # noqa: leading-underscore  # upstream spelling
        _is_fa3_supported,
    )
else:
    pass


def create_preprocessing_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int,
    max_reference_seconds: float,
    silence_prefill_frames: int,
    max_concurrency: int,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """Speech requests to prompt rows; reference audio is encoded with the codec on the GPU."""
    snapshot_directory = resolve_checkpoint(model_path)
    language_model_directory = os.path.join(
        snapshot_directory, LANGUAGE_MODEL_DIRECTORY
    )
    params = load_params(language_model_directory)
    model_params = params["model"]
    assert isinstance(model_params, dict)
    tokenizer_spec = model_params["text_tokenizer_spec"]
    assert isinstance(tokenizer_spec, dict)
    reference_encoder = Zonos3CodecEncoder.from_model_directory(
        snapshot_directory, resolve_concrete_device(device, gpu_id)
    )
    if reference_encoder.input_conv.weight.is_cuda:
        reference_encoder.capture_encode_graphs()
    else:
        pass
    preprocessor = Zonos3SpeechPreprocessor(
        tokenizer=Tokenizer.from_file(
            os.path.join(language_model_directory, tokenizer_spec["path"])
        ),
        text_token_offset=int(tokenizer_spec["offset"]),
        begin_of_text_id=int(tokenizer_spec["bos"]),
        end_of_text_id=int(tokenizer_spec["eos"]),
        config=zonos3_config_from_params(params),
        reference_encoder=reference_encoder,
        silence_frame=codec_registry_entry(snapshot_directory).silence_frame,
        silence_prefill_frames=silence_prefill_frames,
        max_new_tokens=max_new_tokens,
        max_reference_seconds=max_reference_seconds,
    )
    return SimpleScheduler(preprocessor, max_concurrency=max_concurrency)


def create_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    context_length: int,
    max_running_requests: int,
    max_new_tokens: int,
    cuda_graph: bool,
    async_decode: bool,
    vocoder_initial_chunk_frames: int,
    vocoder_steady_chunk_frames: int,
    server_args_overrides: dict[str, object] | None = None,
) -> OmniScheduler[Zonos3RequestData]:
    """The AR engine; max_new_tokens is consumed by preprocessing as the request default."""
    if max_new_tokens >= context_length:
        raise ValueError(
            f"Zonos3 max_new_tokens={max_new_tokens} leaves no prompt room in "
            f"context_length={context_length}"
        )
    else:
        pass
    return Zonos3EngineBuilder(
        context_length=context_length,
        max_running_requests=max_running_requests,
        mem_fraction_static=total_gpu_memory_fraction,
        cuda_graph=cuda_graph,
        async_decode=async_decode,
        vocoder_initial_chunk_frames=vocoder_initial_chunk_frames,
        vocoder_steady_chunk_frames=vocoder_steady_chunk_frames,
    ).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=total_gpu_memory_fraction,
    )


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    initial_chunk_frames: int,
    steady_chunk_frames: int,
    max_step_streams: int,
    pass_frame_buckets: tuple[int, ...],
    max_stream_frames: int,
    max_running_requests: int,
    stream_slots: int | None = None,
) -> Zonos3StreamingVocoderScheduler:
    """One codec handles whole requests and streaming prefixes at 44.1 kHz, at the registry's headroom.

    Streams that hold a decoder slot, one per running request unless stream_slots sets the count, decode only their new frames, in captured passes of up to max_step_streams rows over pass_frame_buckets frames.
    """
    snapshot_directory = resolve_checkpoint(model_path)
    decoder = Zonos3CodecDecoder.from_model_directory(
        snapshot_directory, resolve_concrete_device(device, gpu_id)
    )
    # note (zhaochenyang20): A fresh server otherwise captures or eagerly decodes about 40 single-request trunk lengths and builds a synthesis plan per new length while serving SeedTTS EN non-streaming; capturing them first served 3.5% more requests per second on one H100 (4 ABBA pairs, streaming unchanged) for about 2.5 s more startup and 0.25 GB.
    if decoder.graph_backend is not None:
        decoder.capture_decode_graphs()
    else:
        pass
    stream_slot_count = max_running_requests if stream_slots is None else stream_slots
    # Slots need FA3's paged appending window attention and were measured only on the sm90 GPUs that run the fused trunk.
    if stream_slot_count > 0 and decoder.uses_fused_trunk and _is_fa3_supported():
        slots = Zonos3DecoderSlots(
            decoder,
            stream_slots=stream_slot_count,
            row_buckets=tuple(
                sorted(
                    {
                        min(2**exponent, max_step_streams)
                        for exponent in range(max_step_streams.bit_length() + 1)
                    }
                )
            ),
            pass_frame_buckets=pass_frame_buckets,
            max_stream_frames=max_stream_frames,
        )
        slots.capture_graphs()
    else:
        slots = None
    return Zonos3StreamingVocoderScheduler(
        decoder,
        slots=slots,
        headroom_gain=10.0
        ** (-codec_registry_entry(snapshot_directory).headroom_db / 20.0),
        initial_chunk_frames=initial_chunk_frames,
        steady_chunk_frames=steady_chunk_frames,
        max_step_streams=max_step_streams,
    )
