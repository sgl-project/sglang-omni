# SPDX-License-Identifier: Apache-2.0
"""Breeze stage factories. Depth decoding stays inside the AR feedback loop."""

from pathlib import Path

from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint
from sglang_omni.utils.device import resolve_device_spec


def load_audio_tokenizer(checkpoint: str, device: str):
    from sglang_omni.models.qwen3_tts.compat import (
        apply_qwen_tts_transformers_compatibility_patches,
    )

    apply_qwen_tts_transformers_compatibility_patches()
    try:
        from qwen_tts import Qwen3TTSTokenizer
    except ImportError as exc:
        raise ImportError(
            "Breeze-TTS-2 requires qwen-tts==0.1.1 (install with uv pip --no-deps; "
            "keep Omni's pinned Transformers). See docs/cookbook/breeze_tts.md."
        ) from exc
    # Not speech_tokenizer/ (Qwen3-TTS checkpoints) nor the legacy Mimi model
    # described in the outer Breeze config. This directory is bundled by Breeze.
    return Qwen3TTSTokenizer.from_pretrained(
        str(Path(checkpoint) / "audio_tokenizer"), device_map=device
    )


def create_preprocessing_executor(model_path: str, *, gpu_id=None, device=None):
    from .frontend import BreezeFrontend, load_text_tokenizer

    device = resolve_device_spec(device, gpu_id)
    checkpoint = resolve_checkpoint(model_path)
    frontend = BreezeFrontend.from_checkpoint(checkpoint, device)
    tokenizer = load_text_tokenizer(checkpoint)
    audio_tokenizer = load_audio_tokenizer(checkpoint, device)
    return SimpleScheduler(
        lambda payload: frontend.prepare(payload, tokenizer, audio_tokenizer)
    )


def create_tts_engine_executor(
    model_path: str,
    *,
    gpu_id=None,
    device=None,
    dtype="bfloat16",
    server_args_overrides=None,
):
    from .engine_builder import BreezeEngineBuilder

    return BreezeEngineBuilder().build(
        model_path,
        gpu_id=gpu_id,
        device=device,
        dtype=dtype,
        server_args_overrides=server_args_overrides,
    )


def create_vocoder_executor(model_path: str, *, gpu_id=None, device=None):
    from sglang_omni.models.qwen3_tts.streaming_vocoder import (
        Qwen3TTSStreamingVocoderScheduler,
    )

    device = resolve_device_spec(device, gpu_id)
    tokenizer = load_audio_tokenizer(resolve_checkpoint(model_path), device)
    # Reuse the codec's stream lifecycle and incremental decoder, not the
    # Qwen3-TTS speech model's request formatting or sampling policy.
    return Qwen3TTSStreamingVocoderScheduler(
        tokenizer,
        device=device,
        stream_stride=2,
        stream_followup_stride=2,
        initial_chunk_frames=2,
        max_batch_size=1,
        initial_max_batch_size=1,
        followup_max_batch_size=1,
        initial_cuda_graph=False,
        followup_cuda_graph=False,
        enable_stateful_codec_decoder=True,
        suppress_bootstrap_silence=False,
    )
