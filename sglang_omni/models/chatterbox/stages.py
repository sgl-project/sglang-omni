# SPDX-License-Identifier: Apache-2.0
"""Stage factories for the Chatterbox-Turbo TTS pipeline."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, TypeVar

import torch

from sglang_omni.models.chatterbox.payload_types import ChatterboxState
from sglang_omni.preprocessing.cache_key import (
    hash_bytes,
    reference_path_cache_key,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.reference_encoder import (
    KeyedReferenceEncodeHook,
    ReferenceEncodeService,
)
from sglang_omni.scheduling.vocoder_base import BatchVocoderBase
from sglang_omni.utils.checkpoint import resolve_checkpoint

# Reference audio for the T3 conditioning path is consumed at 16 kHz; the
# voice encoder and S3 tokenizer both accept 16 kHz input.
S3_SR = 16000
ENC_COND_LEN = 15 * S3_SR
SPEECH_COND_PROMPT_LEN = 375
# S3Gen consumes references at 24 kHz and produces 24 kHz output.
S3GEN_SR = 24000

CHATTERBOX_INSTALL_HINT = (
    "Chatterbox-Turbo support requires the `chatterbox-tts` package:\n"
    "    uv pip install --no-deps chatterbox-tts==0.1.7\n"
    "    uv pip install s3tokenizer resemble-perth pyloudnorm\n"
    "`--no-deps` is required: chatterbox-tts pins torch==2.6.0, transformers==5.2.0,\n"
    "numpy<2.0.0, which would downgrade the pinned sglang-omni stack."
)


def _punc_norm(text: str) -> str:
    """Normalize text punctuation before tokenization, matching Chatterbox."""
    if not text:
        return "You need to add some text for me to talk."
    if text[0].islower():
        text = text[0].upper() + text[1:]
    text = " ".join(text.split())
    for old, new in (
        ("…", ", "),
        (":", ","),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", '"'),
        ("”", '"'),
        ("‘", "'"),
        ("’", "'"),
    ):
        text = text.replace(old, new)
    text = text.rstrip(" ")
    if not any(text.endswith(p) for p in (".", "!", "?", "-", ",")):
        text += "."
    return text


@dataclass(frozen=True)
class _ChatterboxReferenceInput:
    source_kind: str
    source: Any
    media_type: str | None = None


@dataclass(frozen=True)
class ChatterboxT3Reference:
    """T3-side conditioning derived from a reference clip."""

    speaker_embedding: torch.Tensor
    cond_prompt_speech_tokens: list[int]


def _reference_payload_is_supported(ref_data: dict[str, Any]) -> bool:
    return (
        ref_data.get("audio_path") is not None
        or ref_data.get("bytes") is not None
        or ref_data.get("base64") is not None
        or ref_data.get("data") is not None
    )


_ArtifactT = TypeVar("_ArtifactT")
_StoredT = TypeVar("_StoredT")


class _ChatterboxReferenceHookBase(
    KeyedReferenceEncodeHook[_ChatterboxReferenceInput, _ArtifactT, _StoredT]
):
    """Shared reference normalization and caching for T3 and S3Gen hooks."""

    def __init__(self, *, checkpoint_id: str, target_sr: int) -> None:
        self.model_revision = str(checkpoint_id)
        self._target_sr = int(target_sr)

    def normalize_input(self, raw_input: Any) -> _ChatterboxReferenceInput:
        if not isinstance(raw_input, dict):
            raise TypeError("Chatterbox reference input must be a dict")
        if raw_input.get("audio_path") is not None:
            return _ChatterboxReferenceInput("path", str(raw_input["audio_path"]))
        if raw_input.get("bytes") is not None:
            return _ChatterboxReferenceInput("bytes", bytes(raw_input["bytes"]))
        data = raw_input.get("base64") or raw_input.get("data")
        if data is not None:
            return _ChatterboxReferenceInput(
                "base64",
                data,
                str(raw_input.get("media_type") or "audio/wav"),
            )
        raise ValueError("Chatterbox reference input has no audio payload")

    def input_key(self, item: _ChatterboxReferenceInput) -> str | None:
        if item.source_kind == "path":
            return reference_path_cache_key(str(item.source), trust_stat=False)
        if item.source_kind == "bytes":
            return f"bytes:{hash_bytes(item.source)}"
        if item.source_kind == "base64":
            payload = str(item.source).encode("utf-8")
            return f"base64:{item.media_type or 'audio/wav'}:{hash_bytes(payload)}"
        return None

    def _load_reference_wav(self, item: _ChatterboxReferenceInput) -> Any:
        if item.source_kind == "path":
            from sglang_omni.utils.audio import load_audio

            return load_audio(
                str(item.source), target_sample_rate=self._target_sr, mono=True
            )
        from sglang_omni.preprocessing.audio import AudioMediaIO

        audio_io = AudioMediaIO(target_sr=self._target_sr)
        if item.source_kind == "bytes":
            audio, _ = audio_io.load_bytes(item.source)
            return audio
        audio, _ = audio_io.load_base64(item.media_type or "audio/wav", item.source)
        return audio


class ChatterboxT3ReferenceEncodeHook(
    _ChatterboxReferenceHookBase[ChatterboxT3Reference, ChatterboxT3Reference]
):
    """Encodes a reference clip into T3 conditioning (VE + S3 tokens)."""

    model_id = "chatterbox_turbo"
    encoder_id = "chatterbox_t3_conditioning"
    artifact_kind = "chatterbox_t3_reference"

    def __init__(self, *, ve: Any, s3_tokenizer: Any, checkpoint_id: str) -> None:
        super().__init__(checkpoint_id=checkpoint_id, target_sr=S3_SR)
        self._ve = ve
        self._s3_tokenizer = s3_tokenizer
        config = (
            f"s3_sr:{S3_SR};enc_cond_len:{ENC_COND_LEN};"
            f"cond_prompt_len:{SPEECH_COND_PROMPT_LEN}"
        )
        self.encoder_config_hash = hash_bytes(config.encode("utf-8"))

    def encode_one(self, item: _ChatterboxReferenceInput) -> ChatterboxT3Reference:
        wav = self._load_reference_wav(item)
        ve_embed = torch.from_numpy(
            self._ve.embeds_from_wavs([wav], sample_rate=S3_SR)
        )
        ve_embed = ve_embed.mean(axis=0, keepdim=True)
        tokens, _ = self._s3_tokenizer.forward(
            [wav[:ENC_COND_LEN]], max_len=SPEECH_COND_PROMPT_LEN
        )
        return ChatterboxT3Reference(
            speaker_embedding=ve_embed,
            cond_prompt_speech_tokens=tokens.flatten().tolist(),
        )

    def store_artifact(self, artifact: ChatterboxT3Reference) -> ChatterboxT3Reference:
        return ChatterboxT3Reference(
            speaker_embedding=artifact.speaker_embedding.detach().cpu(),
            cond_prompt_speech_tokens=list(artifact.cond_prompt_speech_tokens),
        )

    def load_artifact(self, stored: ChatterboxT3Reference) -> ChatterboxT3Reference:
        return stored


class ChatterboxS3GenReferenceEncodeHook(
    _ChatterboxReferenceHookBase[dict[str, Any], dict[str, Any]]
):
    """Encodes a reference clip into S3Gen conditioning via embed_ref."""

    model_id = "chatterbox_turbo"
    encoder_id = "chatterbox_s3gen_conditioning"
    artifact_kind = "chatterbox_s3gen_reference"

    def __init__(self, *, s3gen: Any, checkpoint_id: str) -> None:
        super().__init__(checkpoint_id=checkpoint_id, target_sr=S3GEN_SR)
        self._s3gen = s3gen
        self.encoder_config_hash = hash_bytes(
            f"s3gen_sr:{S3GEN_SR}".encode("utf-8")
        )

    def encode_one(self, item: _ChatterboxReferenceInput) -> dict[str, Any]:
        wav = self._load_reference_wav(item)
        return self._s3gen.embed_ref(wav, S3GEN_SR)

    def store_artifact(self, artifact: dict[str, Any]) -> dict[str, Any]:
        return {
            key: value.detach().cpu() if torch.is_tensor(value) else value
            for key, value in artifact.items()
        }

    def load_artifact(self, stored: dict[str, Any]) -> dict[str, Any]:
        return stored


class ChatterboxVocoder(BatchVocoderBase):
    """Decodes speech tokens into waveform via S3Gen (flow + HiFi-GAN)."""

    def __init__(
        self,
        *,
        s3gen: Any,
        reference_encode_service: ReferenceEncodeService,
        builtin_gen: dict[str, Any] | None,
        device: str,
    ) -> None:
        self._s3gen = s3gen
        self._reference_encode_service = reference_encode_service
        self._builtin_gen = builtin_gen
        self._device = device

    def prepare_item(self, payload: StagePayload) -> tuple[ChatterboxState, Any]:
        state = ChatterboxState.from_dict(payload.data)
        inputs = payload.request.inputs or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}
        ref_data = None
        for ref in inputs.get("references") or []:
            if _reference_payload_is_supported(ref):
                ref_data = ref
                break
        return state, (list(state.speech_tokens), ref_data)

    async def decode_batch(
        self, items: list[tuple[ChatterboxState, Any]]
    ) -> list[tuple[torch.Tensor, int]]:
        results = []
        for _state, (speech_tokens, ref_data) in items:
            results.append((self._decode(speech_tokens, ref_data), S3GEN_SR))
        return results

    def _decode(self, speech_tokens: list[int], ref_data: dict[str, Any] | None) -> torch.Tensor:
        tokens = torch.tensor(speech_tokens, dtype=torch.long).unsqueeze(0).to(self._device)
        if ref_data is not None:
            ref_dict = self._reference_encode_service.get_or_encode(
                ref_data, desc="Chatterbox-Turbo S3Gen reference"
            )
        else:
            ref_dict = self._builtin_gen
        with torch.inference_mode():
            wav = self._s3gen.forward(
                tokens,
                ref_wav=None,
                ref_sr=None,
                ref_dict=ref_dict,
                finalize=True,
                n_cfm_timesteps=2,
            )
        return wav.detach().cpu()

    def store_result(
        self,
        payload: StagePayload,
        state: ChatterboxState,
        wav: torch.Tensor,
        sample_rate: int,
    ) -> StagePayload:
        del state
        from sglang_omni.utils.audio_payload import audio_waveform_payload

        payload.data = audio_waveform_payload(
            wav,
            sample_rate=sample_rate,
            modality="audio",
            source_hint="chatterbox",
        )
        return payload


def create_preprocessing_executor(model_path: str, *, max_concurrency: int = 8):
    """Returns a threaded scheduler for text tokenization and T3-side reference encoding."""
    from sglang_omni.scheduling.threaded_simple_scheduler import ThreadedSimpleScheduler

    worker_count = max(int(max_concurrency), 1)
    checkpoint_dir = resolve_checkpoint(model_path)

    from safetensors.torch import load_file
    from transformers import AutoTokenizer

    try:
        from chatterbox.models.s3tokenizer import S3Tokenizer
        from chatterbox.models.voice_encoder import VoiceEncoder
    except ImportError as exc:
        raise RuntimeError(CHATTERBOX_INSTALL_HINT) from exc

    ve = VoiceEncoder()
    ve.load_state_dict(load_file(os.path.join(checkpoint_dir, "ve.safetensors")))
    ve.eval()

    s3_tokenizer = S3Tokenizer("speech_tokenizer_v2_25hz")
    s3_tokenizer.eval()

    gpt_tokenizer = AutoTokenizer.from_pretrained(checkpoint_dir)
    if gpt_tokenizer.pad_token is None:
        gpt_tokenizer.pad_token = gpt_tokenizer.eos_token

    reference_encode_service = ReferenceEncodeService(
        ChatterboxT3ReferenceEncodeHook(
            ve=ve, s3_tokenizer=s3_tokenizer, checkpoint_id=checkpoint_dir
        ),
        max_items=256,
        max_bytes=64 * 1024 * 1024,
        timeout_s=130.0,
        log_prefix="Chatterbox-Turbo",
    )

    def _preprocess(payload: StagePayload) -> StagePayload:
        inputs = payload.request.inputs or {}
        params = payload.request.params or {}
        if isinstance(inputs, str):
            inputs = {"text": inputs}

        text = _punc_norm(inputs.get("text", ""))
        text_tokens = gpt_tokenizer(
            text, return_tensors="pt", padding=True, truncation=True
        ).input_ids[0].tolist()

        speaker_embedding = None
        cond_prompt_speech_tokens: list[int] = []
        raw_refs = inputs.get("references")
        if raw_refs:
            for ref_data in raw_refs:
                if _reference_payload_is_supported(ref_data):
                    reference = reference_encode_service.get_or_encode(
                        ref_data, desc="Chatterbox-Turbo reference"
                    )
                    speaker_embedding = reference.speaker_embedding
                    cond_prompt_speech_tokens = reference.cond_prompt_speech_tokens
                    break

        state = ChatterboxState(
            text_tokens=text_tokens,
            speaker_embedding=speaker_embedding,
            cond_prompt_speech_tokens=cond_prompt_speech_tokens,
            max_new_tokens=params.get("max_new_tokens", ChatterboxState.max_new_tokens),
            temperature=params.get("temperature", ChatterboxState.temperature),
            top_k=params.get("top_k", ChatterboxState.top_k),
            top_p=params.get("top_p", ChatterboxState.top_p),
            repetition_penalty=params.get(
                "repetition_penalty", ChatterboxState.repetition_penalty
            ),
            seed=params.get("seed"),
        )
        payload.data = state.to_dict()
        return payload

    return ThreadedSimpleScheduler(_preprocess, max_concurrency=worker_count)


def create_vocoder_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_batch_size: int = 8,
    max_batch_wait_ms: int = 2,
):
    """Returns a batch scheduler for S3Gen waveform decoding."""
    from sglang_omni.utils.device import resolve_concrete_device

    try:
        from chatterbox.models.s3gen import S3Gen
    except ImportError as exc:
        raise RuntimeError(CHATTERBOX_INSTALL_HINT) from exc

    device = str(resolve_concrete_device(device, gpu_id))
    checkpoint_dir = resolve_checkpoint(model_path)

    s3gen = S3Gen(meanflow=True)
    from safetensors.torch import load_file

    s3gen.load_state_dict(
        load_file(os.path.join(checkpoint_dir, "s3gen_meanflow.safetensors")),
        strict=True,
    )
    s3gen.to(device).eval()

    builtin_gen = None
    conds_path = os.path.join(checkpoint_dir, "conds.pt")
    if os.path.exists(conds_path):
        builtin = torch.load(conds_path, map_location="cpu", weights_only=True)
        builtin_gen = builtin.get("gen")

    reference_encode_service = ReferenceEncodeService(
        ChatterboxS3GenReferenceEncodeHook(
            s3gen=s3gen, checkpoint_id=checkpoint_dir
        ),
        max_items=256,
        max_bytes=64 * 1024 * 1024,
        timeout_s=130.0,
        log_prefix="Chatterbox-Turbo S3Gen",
    )

    vocoder = ChatterboxVocoder(
        s3gen=s3gen,
        reference_encode_service=reference_encode_service,
        builtin_gen=builtin_gen,
        device=device,
    )
    return vocoder.build_scheduler(
        max_batch_size=max_batch_size, max_batch_wait_ms=max_batch_wait_ms
    )


def create_sglang_tts_engine_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    max_new_tokens: int = 1024,
):
    """Returns OmniScheduler for the T3 autoregressive engine."""
    from sglang_omni.models.chatterbox.engine_builder import (
        ChatterboxT3EngineBuilder,
    )

    return ChatterboxT3EngineBuilder(max_new_tokens=max_new_tokens).build(
        model_path,
        device=device,
        gpu_id=gpu_id,
    )
