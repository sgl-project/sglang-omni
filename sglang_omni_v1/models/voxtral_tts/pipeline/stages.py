"""Stage executor factories for the Voxtral TTS pipeline."""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any

import torch

from sglang_omni_v1.executors import PreprocessingExecutor
from sglang_omni_v1.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni_v1.models.voxtral_tts.pipeline.state_io import load_state, store_state
from sglang_omni_v1.proto import StagePayload

logger = logging.getLogger(__name__)

_VOXTRAL_MISTRAL_COMMON_HINT = (
    "Voxtral TTS requires the `mistral-common` package (speech / Tekken tokenizer). "
    "Please install it in your active environment, for example:\n"
    "  pip install 'mistral-common[audio]>=1.8.0'\n"
    "  uv pip install 'mistral-common[audio]>=1.8.0'"
)


def _import_mistral_common_for_voxtral():
    """Lazy import so the rest of sglang-omni does not depend on mistral-common."""
    try:
        from mistral_common.protocol.speech.request import SpeechRequest
        from mistral_common.tokens.tokenizers.mistral import MistralTokenizer
    except ImportError as exc:
        raise RuntimeError(_VOXTRAL_MISTRAL_COMMON_HINT) from exc
    return SpeechRequest, MistralTokenizer


def _resolve_checkpoint(checkpoint: str) -> str:
    if os.path.isdir(checkpoint):
        return checkpoint
    from huggingface_hub import snapshot_download

    return snapshot_download(checkpoint)


# ---- Preprocessing ----


def create_preprocessing_executor(model_path: str) -> PreprocessingExecutor:
    """Factory for the preprocessing stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)

    SpeechRequest, MistralTokenizer = _import_mistral_common_for_voxtral()

    tekken_path = os.path.join(checkpoint_dir, "tekken.json")
    tokenizer = MistralTokenizer.from_file(tekken_path)

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs
        params = payload.request.params or {}
        metadata = payload.request.metadata or {}

        if isinstance(inputs, str):
            text = inputs
        elif isinstance(inputs, dict):
            text = inputs.get("text", "")
        else:
            text = str(inputs) if inputs else ""

        tts_params = metadata.get("tts_params", {})
        voice = tts_params.get("voice") or params.get("voice")
        if voice is None:
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

    return PreprocessingExecutor(_preprocess)


# ---- Generation ----


@torch.no_grad()
def _run_ar_generation(
    model: Any,
    voice_embeddings: dict[str, torch.Tensor],
    config: Any,
    input_ids: list[int],
    voice: str,
    max_new_tokens: int,
    device: str,
) -> tuple[torch.Tensor, int, int]:
    """Run AR generation loop using VoxtralTTSAudioGeneration model."""
    from sglang_omni_v1.models.voxtral_tts.acoustic_transformer import (
        AudioSpecialTokens,
    )

    audio_token_id = config.audio_model_args.audio_token_id

    input_ids_t = torch.tensor(input_ids, dtype=torch.long, device=device).unsqueeze(0)
    prompt_len = input_ids_t.shape[1]

    input_embeds = model.language_model.embed_tokens(input_ids_t)

    voice_emb = voice_embeddings.get(voice)
    if voice_emb is not None:
        audio_mask = input_ids_t[0] == audio_token_id
        audio_positions = audio_mask.nonzero(as_tuple=True)[0]
        n_voice_frames = min(len(audio_positions), voice_emb.shape[0])
        if n_voice_frames > 0:
            input_embeds[0, audio_positions[:n_voice_frames]] = voice_emb[
                :n_voice_frames
            ].to(input_embeds.dtype)

    position_ids = torch.arange(prompt_len, device=device).unsqueeze(0)

    # Prefill with per-layer debug logging
    hidden, past_kv = model.forward_llm(
        inputs_embeds=input_embeds,
        position_ids=position_ids,
        past_key_values=None,
        use_cache=True,
        do_layer_debug=True,
    )

    last_hidden = hidden[:, -1:, :]

    audio_codes_list = []
    semantic_codes_debug = []

    for step in range(max_new_tokens):
        codes = model.acoustic_transformer(last_hidden.squeeze(1))
        audio_codes_list.append(codes)

        semantic_code = codes[:, 0].item()
        semantic_codes_debug.append(semantic_code)

        if semantic_code == AudioSpecialTokens.id(AudioSpecialTokens.end_audio):
            break

        codes_for_embed = codes.unsqueeze(2)
        multi_cb_emb = model.audio_token_embedding(codes_for_embed)
        next_embeds = multi_cb_emb.sum(dim=1)

        cur_pos = prompt_len + step
        next_pos = torch.tensor([[cur_pos]], dtype=torch.long, device=device)

        hidden, past_kv = model.forward_llm(
            inputs_embeds=next_embeds,
            position_ids=next_pos,
            past_key_values=past_kv,
            use_cache=True,
            do_layer_debug=False,
        )
        last_hidden = hidden[:, -1:, :]

    total_steps = len(audio_codes_list)

    if audio_codes_list:
        all_codes = torch.stack([c.squeeze(0) for c in audio_codes_list], dim=0)
    else:
        all_codes = torch.empty(0, 37, dtype=torch.long)

    return all_codes, prompt_len, len(audio_codes_list)


def create_generation_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
    max_new_tokens: int = 4096,
) -> PreprocessingExecutor:
    """Factory for the AR generation stage."""
    from sglang_omni_v1.models.voxtral_tts.voxtral_tts_audio_generation import (
        VoxtralTTSAudioGeneration,
    )

    checkpoint_dir = _resolve_checkpoint(model_path)

    logger.info("Loading Voxtral TTS model for generation...")
    model, voice_embeddings, config = VoxtralTTSAudioGeneration.from_checkpoint(
        checkpoint_dir, device
    )

    def _generate(payload: StagePayload) -> StagePayload:
        state = load_state(payload)

        effective_max = state.max_new_tokens or max_new_tokens
        all_codes, prompt_tokens, completion_tokens = _run_ar_generation(
            model=model,
            voice_embeddings=voice_embeddings,
            config=config,
            input_ids=state.input_ids,
            voice=state.voice or "cheerful_female",
            max_new_tokens=effective_max,
            device=device,
        )

        state.audio_codes = all_codes
        state.prompt_tokens = prompt_tokens
        state.completion_tokens = completion_tokens
        return store_state(payload, state)

    return PreprocessingExecutor(_generate)


# ---- Vocoder ----


def _load_audio_tokenizer(checkpoint_dir: str, audio_config: dict, device: str):
    """Load the VoxtralTTSAudioTokenizer (decoder) from checkpoint."""
    import glob

    from sglang.srt.model_loader.weight_utils import safetensors_weights_iterator

    from sglang_omni_v1.models.voxtral_tts.audio_tokenizer import (
        VoxtralTTSAudioTokenizer,
    )
    from sglang_omni_v1.models.voxtral_tts.model_config import VoxtralModelConfig

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
    logger.info("Audio tokenizer loaded in %.2fs", time.perf_counter() - t0)
    return tokenizer


def create_vocoder_executor(
    model_path: str,
    *,
    device: str = "cuda:0",
) -> PreprocessingExecutor:
    """Factory for the vocoder (audio tokenizer decode) stage."""
    checkpoint_dir = _resolve_checkpoint(model_path)

    logger.info("Loading Voxtral audio tokenizer for vocoding...")
    audio_tokenizer = _load_audio_tokenizer(checkpoint_dir, {}, device)

    def _vocode(payload: StagePayload) -> StagePayload:
        state = load_state(payload)
        audio_codes = state.audio_codes

        if audio_codes is None or (
            isinstance(audio_codes, torch.Tensor) and audio_codes.numel() == 0
        ):
            state.audio_samples = []
            payload = store_state(payload, state)
            payload.data["audio_data"] = []
            payload.data["sample_rate"] = 24000
            payload.data["modality"] = "audio"
            return payload

        if not isinstance(audio_codes, torch.Tensor):
            audio_codes = torch.tensor(audio_codes)

        # Prepend warmup context frames so the causal decoder has initial
        # context (mitigates boundary artifacts / noise at the start of the
        # waveform).  After decoding, the samples corresponding to the
        # warmup frames are trimmed away.
        n_warmup = 2
        warmup_samples = 0
        if audio_codes.shape[0] > 0:
            first_frame = audio_codes[0:1]
            warmup = first_frame.repeat(n_warmup, 1)
            codes_with_warmup = torch.cat([warmup, audio_codes], dim=0)
            warmup_samples = n_warmup * audio_tokenizer.downsample_factor
        else:
            codes_with_warmup = audio_codes

        results = audio_tokenizer.decode_helper_batch_async([codes_with_warmup])
        audio_np = results[0]

        # Trim warmup samples from the beginning
        if warmup_samples > 0 and len(audio_np) > warmup_samples:
            audio_np = audio_np[warmup_samples:]

        # Apply a short fade-in to smooth any residual onset artifacts
        fade_in_ms = 10  # milliseconds
        fade_samples = min(
            int(fade_in_ms * audio_tokenizer.sampling_rate / 1000),
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

        state.audio_samples = audio_np
        state.sample_rate = audio_tokenizer.sampling_rate
        payload = store_state(payload, state)

        payload.data["audio_data"] = audio_np.tolist()
        payload.data["sample_rate"] = audio_tokenizer.sampling_rate
        payload.data["modality"] = "audio"

        if state.prompt_tokens or state.completion_tokens:
            payload.data["usage"] = {
                "prompt_tokens": state.prompt_tokens,
                "completion_tokens": state.completion_tokens,
                "total_tokens": state.prompt_tokens + state.completion_tokens,
            }

        return payload

    return PreprocessingExecutor(_vocode)
