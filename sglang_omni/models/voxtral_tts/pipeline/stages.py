# SPDX-License-Identifier: Apache-2.0
"""Stage executor factories for the Voxtral TTS pipeline."""

from __future__ import annotations

import json
import logging
import os
import re
import tempfile
import time
from typing import Any

import torch

from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.models.voxtral_tts.pipeline.state_io import load_state, store_state
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.checkpoint import resolve_checkpoint as _resolve_checkpoint

logger = logging.getLogger(__name__)

_VOXTRAL_MISTRAL_COMMON_HINT = (
    "Voxtral TTS requires the `mistral_common` package (speech / Tekken tokenizer). "
    "Please install it in your active environment, for example:\n"
    "  pip install 'mistral_common[audio]>=1.11.0'\n"
    "  uv pip install 'mistral_common[audio]>=1.11.0'"
)


def _import_mistral_common_for_voxtral():
    """Lazy import so the rest of sglang-omni does not depend on mistral-common."""
    try:
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(_VOXTRAL_MISTRAL_COMMON_HINT) from exc
    return SpeechRequest, MistralTokenizer


def _validate_voxtral_speech_params(
    *,
    inputs: Any,
    params: dict[str, Any],
    tts_params: dict[str, Any],
) -> None:
    explicit_generation_params = tts_params.get("explicit_generation_params")
    if isinstance(explicit_generation_params, (list, tuple, set)):
        explicit_fields = {str(field) for field in explicit_generation_params}
    else:
        explicit_fields = set()

    unsupported: set[str] = set()
    for field in explicit_fields:
        if field != "max_new_tokens":
            unsupported.add(field)

    if params.get("seed") is not None:
        unsupported.add("seed")
    if params.get("stage_sampling"):
        unsupported.add("stage_sampling")
    if params.get("stage_params"):
        unsupported.add("stage_params")

    for field in ("task_type", "language", "instructions", "ref_audio", "ref_text"):
        if tts_params.get(field) not in (None, ""):
            unsupported.add(field)

    if isinstance(inputs, dict) and inputs.get("references"):
        unsupported.add("references")

    if unsupported:
        fields = ", ".join(sorted(unsupported))
        raise ValueError(
            "Voxtral TTS does not support these /v1/audio/speech fields: "
            f"{fields}. Supported model-specific fields are voice and max_new_tokens."
        )


def _ensure_non_empty_audio_codes(audio_codes: Any) -> None:
    if audio_codes is None:
        raise ValueError("Voxtral TTS generated no audio codes")
    if isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0:
        raise ValueError("Voxtral TTS generated no audio codes")


# ---- Preprocessing ----


def create_preprocessing_executor(model_path: str) -> SimpleScheduler:
    """Factory for the preprocessing stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)

    SpeechRequest, MistralTokenizer = _import_mistral_common_for_voxtral()

    tekken_path = os.path.join(checkpoint_dir, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}
        tts_params = metadata.get("tts_params", {})
        if not isinstance(tts_params, dict):
            tts_params = {}
        _validate_voxtral_speech_params(
            inputs=inputs,
            params=params,
            tts_params=tts_params,
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

        encoded = tokenizer.encode_speech_request(
            SpeechRequest(input=text, voice=voice)
        )

        max_new_tokens = params.get("max_new_tokens", 4096)
        if isinstance(max_new_tokens, dict):
            max_new_tokens = max_new_tokens.get("max_new_tokens", 4096)

        input_ids = list(encoded.tokens)

        state = VoxtralTTSState(
            input_ids=input_ids,
            voice=voice,
            max_new_tokens=max_new_tokens,
        )

        return store_state(payload, state)

    return SimpleScheduler(_preprocess)


# ---- Generation ----


def _enable_inductor_gemm_autotune() -> None:
    # Note:(Chenchen Hong) on torch 2.11/cu13 inductor routes the compiled
    # matmuls to slow split-K cuBLAS (~8% RTF); per-shape GEMM autotuning makes
    # it benchmark triton vs aten and keep the fastest. One-time startup cost.
    try:
        from torch._inductor import config as inductor_config
    except Exception:
        return
    if hasattr(inductor_config, "max_autotune_gemm"):
        inductor_config.max_autotune_gemm = True
    if hasattr(inductor_config, "max_autotune_gemm_backends"):
        inductor_config.max_autotune_gemm_backends = "TRITON,ATEN"
    logger.info(
        "Voxtral: enabled inductor per-shape GEMM autotuning (TRITON,ATEN); "
        "adds one-time startup autotune cost."
    )


def create_generation_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
    max_new_tokens: int = 4096,
    server_args_overrides: dict[str, Any] | None = None,
) -> Any:
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


def _write_voxtral_sglang_config(checkpoint_dir: str) -> str:
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    cfg = VoxtralModelConfig.from_model_path(checkpoint_dir).text_config
    path = os.path.join(
        tempfile.gettempdir(),
        f"voxtral_sglang_config_{abs(hash(checkpoint_dir))}.json",
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


def _load_voxtral_voice_embeddings(
    checkpoint_dir: str,
    device: str,
) -> dict[str, torch.Tensor]:
    voice_embeddings: dict[str, torch.Tensor] = {}
    voice_dir = os.path.join(checkpoint_dir, "voice_embedding")
    if not os.path.isdir(voice_dir):
        return voice_embeddings
    for fname in sorted(os.listdir(voice_dir)):
        if not fname.endswith(".pt"):
            continue
        name = fname.removesuffix(".pt")
        emb = torch.load(
            os.path.join(voice_dir, fname),
            map_location=device,
            weights_only=True,
        )
        voice_embeddings[name] = emb.to(dtype=torch.bfloat16)
    return voice_embeddings


# ---- Vocoder ----


def _load_audio_tokenizer(checkpoint_dir: str, audio_config: dict, device: str):
    """Load the VoxtralTTSAudioTokenizer (decoder) from checkpoint."""
    import glob

    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

    from sglang_omni.models.voxtral_tts.audio_tokenizer import VoxtralTTSAudioTokenizer
    from sglang_omni.models.voxtral_tts.model_config import VoxtralModelConfig

    config = VoxtralModelConfig.from_model_path(checkpoint_dir)

    tokenizer = VoxtralTTSAudioTokenizer(
        audio_tokenizer_args=config.audio_tokenizer_args,
        audio_config={
            "audio_model_args": config.audio_model_args.acoustic_transformer_args,
        },
    )

    safetensors_files = sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors")))
    if not safetensors_files:
        raise RuntimeError(f"No .safetensors files found in {checkpoint_dir}")

    logger.info("Loading audio tokenizer weights...")
    t0 = time.perf_counter()

    remapping_rules = [
        (r"^audio_tokenizer\.(.*)$", r"\1"),
        (
            r"^mm_audio_embeddings\.audio_codebook_embeddings\.embeddings\.(weight|bias)",
            r"audio_token_embedding.embeddings.\1",
        ),
    ]

    for name, tensor in safetensors_weights_iterator(safetensors_files):
        is_audio_tokenizer = name.startswith(
            "mm_audio_embeddings.audio_codebook_embeddings"
        ) or name.startswith("audio_tokenizer.")

        if not is_audio_tokenizer:
            continue

        remapped = name
        for pattern, repl in remapping_rules:
            if re.fullmatch(pattern, remapped):
                remapped = re.sub(pattern, repl, remapped)
        tokenizer.load_weight((remapped, tensor))

    tokenizer = tokenizer.to(dtype=torch.bfloat16, device=device).eval()
    logger.info(f"Audio tokenizer loaded in {time.perf_counter() - t0:.2f}s")
    return tokenizer


class _VoxtralTTSVocoder(BatchVocoderBase):
    """Decode audio codes with repeated initial frames as warmup context."""

    _N_WARMUP = 2
    _FADE_IN_MS = 10

    def __init__(self, audio_tokenizer: Any) -> None:
        self._audio_tokenizer = audio_tokenizer

    def prepare_item(
        self, payload: StagePayload
    ) -> tuple[VoxtralTTSState, torch.Tensor]:
        state = load_state(payload)
        audio_codes = state.audio_codes

        _ensure_non_empty_audio_codes(audio_codes)

        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.tensor(audio_codes)
        # Note:(AkazaAkane) Keep the original note from #248 before refactoring.
        # Prepend warmup context frames so the causal decoder has initial
        # context (mitigates boundary artifacts / noise at the start of the
        # waveform).  After decoding, the samples corresponding to the
        # warmup frames are trimmed away.
        if audio_codes.shape[0] > 0:
            first_frame = audio_codes[0:1]
            warmup = first_frame.repeat(self._N_WARMUP, 1)
            codes_with_warmup = torch.cat([warmup, audio_codes], dim=0)
        else:
            codes_with_warmup = audio_codes

        return state, codes_with_warmup

    async def decode_batch(
        self, items: list[tuple[VoxtralTTSState, torch.Tensor]]
    ) -> list[tuple[torch.Tensor, int]]:
        codes_list = [codes for _, codes in items]
        results = self._audio_tokenizer.decode_helper_batch_async(codes_list)
        sample_rate = self._audio_tokenizer.sampling_rate
        return [(audio_np, sample_rate) for audio_np in results]

    def store_result(
        self,
        payload: StagePayload,
        state: VoxtralTTSState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        audio_np = wav

        original_codes = state.audio_codes
        original_len = (
            original_codes.shape[0]
            if isinstance(original_codes, torch.Tensor)
            else len(original_codes)
        )
        warmup_samples = (
            self._N_WARMUP * self._audio_tokenizer.downsample_factor
            if original_len > 0
            else 0
        )

        # Trim warmup samples from the beginning
        if warmup_samples > 0 and len(audio_np) > warmup_samples:
            audio_np = audio_np[warmup_samples:]

        # Apply a short fade-in to smooth any residual onset artifacts
        fade_samples = min(
            int(self._FADE_IN_MS * sample_rate / 1000),
            len(audio_np),
        )
        if fade_samples > 0:
            fade_in = torch.linspace(
                0,
                1,
                fade_samples,
                device=audio_np.device,
                dtype=audio_np.dtype,
            )
            audio_np[:fade_samples] = audio_np[:fade_samples] * fade_in

        audio_payload = audio_waveform_payload(audio_np, source_hint="Voxtral TTS")
        state.audio_samples = None
        state.sample_rate = sample_rate
        payload = store_state(payload, state)

        payload.data.update(audio_payload)
        payload.data["sample_rate"] = sample_rate
        payload.data["modality"] = "audio"

        if state.prompt_tokens or state.completion_tokens:
            payload.data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }

        return payload


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    gpu_id: int | None = None,
) -> SimpleScheduler:
    checkpoint_dir = _resolve_checkpoint(model_path)
    if gpu_id is not None:
        device = f"cuda:{gpu_id}"

    logger.info("Loading Voxtral audio tokenizer for vocoding...")
    audio_tokenizer = _load_audio_tokenizer(checkpoint_dir, {}, device)

    return _VoxtralTTSVocoder(audio_tokenizer).build_scheduler(
        max_batch_size=1, max_batch_wait_ms=0
    )
