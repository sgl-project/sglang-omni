# SPDX-License-Identifier: Apache-2.0
"""Factories for text preprocessing, native AR generation and offline audio."""

from collections.abc import Mapping
from functools import partial

from transformers import AutoTokenizer

from sglang_omni.models.moss_speech.codec import MossSpeechCodec
from sglang_omni.models.moss_speech.engine_builder import MossSpeechEngineBuilder
from sglang_omni.models.moss_speech.hf_config import (
    ensure_moss_speech_config_registered,
)
from sglang_omni.models.moss_speech.request_builders import (
    MossSpeechSGLangRequestData,
    preprocess_request,
)
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device


def create_preprocessing_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    ensure_moss_speech_config_registered()
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=False)
    return SimpleScheduler(partial(preprocess_request, tokenizer=tokenizer))


def create_ar_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    total_gpu_memory_fraction: float | None = None,
    process_total_gpu_memory_fraction: float | None = None,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[MossSpeechSGLangRequestData]:
    return MossSpeechEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        dtype="bfloat16",
        server_args_overrides=server_args_overrides,
        total_gpu_memory_fraction=(
            process_total_gpu_memory_fraction
            if process_total_gpu_memory_fraction is not None
            else total_gpu_memory_fraction
        ),
    )


def create_audio_vocoder_executor(
    *,
    codec_path: str | None,
    device: str | None = None,
    gpu_id: int | None = None,
) -> SimpleScheduler[StagePayload, StagePayload]:
    if codec_path is None:
        raise ValueError("Set audio_vocoder.factory.codec_path")
    else:
        pass
    codec = MossSpeechCodec(codec_path, resolve_concrete_device(device, gpu_id))
    return SimpleScheduler(codec.decode, shutdown_callback=codec.close)
