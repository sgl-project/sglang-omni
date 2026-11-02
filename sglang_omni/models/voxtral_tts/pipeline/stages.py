"""Stage executor factories for the Voxtral TTS pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from collections.abc import Mapping
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline.state_io import store_state
from sglang_omni.models.voxtral_tts.vocoder import (
    VoxtralStreamingVocoder,
    VoxtralTTSVocoder,
)
from sglang_omni.platforms import current_platform
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint
from sglang_omni.utils.device import resolve_concrete_device

if TYPE_CHECKING:
    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
    from sglang_omni.models.voxtral_tts.request_builders import VoxtralSGLangRequestData
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
else:
    pass

logger = logging.getLogger(__name__)
_VOXTRAL_MISTRAL_COMMON_HINT = "Voxtral TTS requires the `mistral_common` package (speech / Tekken tokenizer). Please install it in your active environment, for example:\n  pip install 'mistral_common[audio]>=1.11.0'\n  uv pip install 'mistral_common[audio]>=1.11.0'"


def import_mistral_common_for_voxtral():
    """Lazy import so the rest of sglang-omni does not depend on mistral-common."""
    try:
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(_VOXTRAL_MISTRAL_COMMON_HINT) from exc
    return (SpeechRequest, MistralTokenizer)


def validate_voxtral_speech_params(
    *,
    inputs: object,
    params: Mapping[str, object],
    tts_params: Mapping[str, object],
) -> None:
    unsupported: set[str] = set()
    for field in ("temperature", "top_p", "top_k", "repetition_penalty", "seed"):
        if params.get(field) is not None:
            unsupported.add(field)
        else:
            pass
    if params.get("stage_sampling"):
        unsupported.add("stage_sampling")
    else:
        pass
    if params.get("stage_params"):
        unsupported.add("stage_params")
    else:
        pass
    for field in ("task_type", "language", "instructions", "ref_audio", "ref_text"):
        if tts_params.get(field) not in (None, ""):
            unsupported.add(field)
        else:
            pass
    if isinstance(inputs, dict) and inputs.get("references"):
        unsupported.add("references")
    else:
        pass
    if unsupported:
        fields = ", ".join(sorted(unsupported))
        raise ValueError(
            f"Voxtral TTS does not support these /v1/audio/speech fields: {fields}. Supported model-specific fields are voice and max_new_tokens."
        )
    else:
        pass


def create_preprocessing_executor(
    model_path: str,
) -> SimpleScheduler[StagePayload, StagePayload]:
    """Factory for the preprocessing stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)
    SpeechRequest, MistralTokenizer = import_mistral_common_for_voxtral()
    tekken_path = os.path.join(checkpoint_dir, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        tts_params = metadata.get("tts_params", {})
        if not isinstance(tts_params, dict):
            tts_params = {}
        else:
            pass
        validate_voxtral_speech_params(
            inputs=inputs, params=params, tts_params=tts_params
        )
        if isinstance(inputs, str):
            text = inputs
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
        else:
            text = str(inputs) if inputs else ""
        voice = tts_params.get("voice") or params.get("voice")
        if voice in (None, "", "default"):
            voice = "cheerful_female"
        else:
            pass
        encoded = tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice=voice)
        )
        max_new_tokens = params.get("max_new_tokens", 4096)
        if isinstance(max_new_tokens, dict):
            max_new_tokens = max_new_tokens.get("max_new_tokens", 4096)
        else:
            pass
        input_ids = list(encoded.tokens)
        state = VoxtralTTSState(
            input_ids=input_ids, voice=voice, max_new_tokens=max_new_tokens
        )
        return store_state(payload, state)

    return SimpleScheduler(_preprocess)


def enable_inductor_gemm_autotune() -> None:
    try:
        from torch._inductor import config as inductor_config
    except Exception:
        return
    if hasattr(inductor_config, "max_autotune_gemm"):
        inductor_config.max_autotune_gemm = True
    else:
        pass
    if hasattr(inductor_config, "max_autotune_gemm_backends"):
        inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
    else:
        pass
    logger.info(
        "Voxtral: enabled inductor per-shape GEMM autotuning (TRITON,ATEN); adds one-time startup autotune cost."
    )


def create_generation_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int = 4096,
    server_args_overrides: Mapping[str, object] | None = None,
) -> OmniScheduler[VoxtralSGLangRequestData]:
    """Factory for the SGLang-backed AR generation stage."""
    del max_new_tokens
    from sglang_omni.models.voxtral_tts.pipeline.engine_builder import (
        VoxtralTtsEngineBuilder,
    )

    return VoxtralTtsEngineBuilder().build(
        model_path,
        device=device,
        gpu_id=gpu_id,
        server_args_overrides=server_args_overrides,
    )


def write_voxtral_sglang_config(checkpoint_dir: str) -> str:
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    cfg = VoxtralModelConfig.from_model_path(checkpoint_dir).text_config
    path = os.path.join(
        tempfile.gettempdir(), f"voxtral_sglang_config_{abs(hash(checkpoint_dir))}.json"
    )
    data = {
        "model_type": "llama",
        "architectures": ["VoxtralSGLangTTSModel"],
        "hidden_size": cfg.dim,
        "intermediate_size": cfg.hidden_dim,
        "num_hidden_layers": cfg.n_layers,
        "num_attention_heads": cfg.n_heads,
        "num_key_value_heads": cfg.n_kv_heads,
        "head_dim": cfg.head_dim,
        "vocab_size": cfg.vocab_size,
        "max_position_embeddings": cfg.max_seq_len,
        "rope_theta": cfg.rope_theta,
        "rms_norm_eps": cfg.norm_eps,
        "tie_word_embeddings": cfg.tied_embeddings,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return path


def load_voxtral_voice_embeddings(
    checkpoint_dir: str, device: str
) -> dict[str, torch.Tensor]:
    voice_embeddings: dict[str, torch.Tensor] = {}
    voice_dir = os.path.join(checkpoint_dir, "voice_embedding")
    if not os.path.isdir(voice_dir):
        return voice_embeddings
    else:
        pass
    map_location = "cpu" if current_platform.is_musa() else device
    for fname in sorted(os.listdir(voice_dir)):
        if not fname.endswith(".pt"):
            continue
        else:
            pass
        name = fname.removesuffix(".pt")
        emb = torch.load(
            os.path.join(voice_dir, fname), map_location=map_location, weights_only=True
        )
        voice_embeddings[name] = emb.to(device=device, dtype=torch.bfloat16)
    return voice_embeddings


def load_audio_tokenizer(
    checkpoint_dir: str, audio_config: dict, device: str
) -> "VoxtralTTSAudioTokenizer":
    """Load the VoxtralTTSAudioTokenizer (decoder) from checkpoint."""
    import glob

    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    config = VoxtralModelConfig.from_model_path(checkpoint_dir)
    tokenizer = VoxtralTTSAudioTokenizer(
        audio_tokenizer_args=config.audio_tokenizer_args,
        audio_config={
            "audio_model_args": config.audio_model_args.acoustic_transformer_args
        },
    )
    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not safetensors_files:
        raise RuntimeError(f"No .safetensors files found in {checkpoint_dir}")
    else:
        pass
    logger.info("Loading audio tokenizer weights...")
    t0 = time.perf_counter()
    remapping_rules = [
        ("^audio_tokenizer\\.(.*)$", "\\1"),
        (
            "^mm_audio_embeddings\\.audio_codebook_embeddings\\.embeddings\\.(weight|bias)",
            "audio_token_embedding.embeddings.\\1",
        ),
    ]
    for name, tensor in safetensors_weights_iterator(safetensors_files):
        is_audio_tokenizer = name.startswith(
            "mm_audio_embeddings.audio_codebook_embeddings"
        ) or name.startswith("audio_tokenizer.")
        if not is_audio_tokenizer:
            continue
        else:
            pass
        remapped = name
        for pattern, repl in remapping_rules:
            if re.fullmatch(pattern, remapped):
                remapped = re.sub(pattern, repl, remapped)
            else:
                pass
        tokenizer.load_weight((remapped, tensor))
    tokenizer = tokenizer.to(dtype=torch.bfloat16, device=device).eval()
    logger.info(f"Audio tokenizer loaded in {time.perf_counter() - t0:.2f}s")
    return tokenizer


def create_vocoder_executor(
    model_path: str, *, device: str | None = None, gpu_id: int | None = None
) -> VoxtralStreamingVocoder:
    device = str(resolve_concrete_device(device, gpu_id))
    checkpoint_dir = _resolve_checkpoint(model_path)
    logger.info("Loading Voxtral audio tokenizer for vocoding...")
    audio_tokenizer = load_audio_tokenizer(checkpoint_dir, {}, device)
    return VoxtralStreamingVocoder(VoxtralTTSVocoder(audio_tokenizer), device=device)
