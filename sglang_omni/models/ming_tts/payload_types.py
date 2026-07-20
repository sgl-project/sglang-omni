# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni-TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.pipeline_state import PipelineStateBase
from sglang_omni.scheduling.pipeline_state import load_state as _load_pipeline_state
from sglang_omni.scheduling.pipeline_state import store_state as _store_pipeline_state

MING_TTS_SAMPLE_RATE = 44100
MING_TTS_DEFAULT_MAX_DECODE_STEPS = 200
MING_TTS_LATENT_TRANSPORT_DTYPE = "float32"
MING_TTS_LATENT_TRANSPORT_VERSION = 1


def encode_generated_latents(latents: Any) -> dict[str, Any]:
    """Encode generated latent chunks as the cross-stage wire format."""

    if not isinstance(latents, torch.Tensor):
        latents = torch.as_tensor(latents)
    latents = latents.detach().to(device="cpu", dtype=torch.float32).contiguous()
    return {
        "generated_latents_bytes": latents.numpy().tobytes(),
        "generated_latents_shape": list(latents.shape),
        "generated_latents_dtype": MING_TTS_LATENT_TRANSPORT_DTYPE,
        "generated_latents_transport_version": MING_TTS_LATENT_TRANSPORT_VERSION,
    }


def decode_generated_latents(
    data: "MingTTSState | dict[str, Any]",
    *,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any | None:
    """Restore generated latent chunks from the cross-stage wire format."""

    if isinstance(data, MingTTSState):
        raw = data.generated_latents_bytes
        shape = data.generated_latents_shape
    else:
        raw = data.get("generated_latents_bytes")
        shape = data.get("generated_latents_shape")

    return _decode_float_tensor(
        raw,
        shape,
        device=device,
        dtype=dtype,
    )


def encode_speaker_embedding(spk_emb: Any) -> dict[str, Any]:
    """Encode raw CampPlus speaker embeddings for cross-stage transport."""

    tensor = _encode_float_tensor(spk_emb)
    return {
        "spk_emb_bytes": tensor.numpy().tobytes(),
        "spk_emb_shape": list(tensor.shape),
        "spk_emb_dtype": MING_TTS_LATENT_TRANSPORT_DTYPE,
    }


def decode_speaker_embedding(
    data: "MingTTSState | dict[str, Any]",
    *,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any | None:
    """Restore raw CampPlus speaker embeddings from the state payload."""

    if isinstance(data, MingTTSState):
        raw = data.spk_emb_bytes
        shape = data.spk_emb_shape
    else:
        raw = data.get("spk_emb_bytes")
        shape = data.get("spk_emb_shape")
    return _decode_float_tensor(
        raw,
        shape,
        device=device,
        dtype=dtype,
    )


def encode_prompt_latent(prompt_latent: Any) -> dict[str, Any]:
    """Encode raw AudioVAE prompt latents for cross-stage transport."""

    tensor = _encode_float_tensor(prompt_latent)
    return {
        "prompt_latent_bytes": tensor.numpy().tobytes(),
        "prompt_latent_shape": list(tensor.shape),
        "prompt_latent_dtype": MING_TTS_LATENT_TRANSPORT_DTYPE,
    }


def decode_prompt_latent(
    data: "MingTTSState | dict[str, Any]",
    *,
    device: Any | None = None,
    dtype: Any | None = None,
) -> Any | None:
    """Restore raw AudioVAE prompt latents from the state payload."""

    if isinstance(data, MingTTSState):
        raw = data.prompt_latent_bytes
        shape = data.prompt_latent_shape
    else:
        raw = data.get("prompt_latent_bytes")
        shape = data.get("prompt_latent_shape")
    return _decode_float_tensor(
        raw,
        shape,
        device=device,
        dtype=dtype,
    )


def _encode_float_tensor(tensor: Any) -> Any:
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.as_tensor(tensor)
    return tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()


def _decode_float_tensor(
    raw: Any,
    shape: Any,
    *,
    device: Any | None,
    dtype: Any | None,
) -> Any | None:
    if raw is None or shape is None:
        return None
    raw = bytes(raw)
    dims = [int(dim) for dim in shape]
    if len(raw) == 0:
        # torch.frombuffer rejects empty buffers on torch >= 2.12; an empty
        # latent stream (finish-before-first-chunk) is a supported payload.
        tensor = torch.empty(dims, dtype=torch.float32)
    else:
        tensor = torch.frombuffer(raw, dtype=torch.float32).clone().reshape(dims)
    if device is not None or dtype is not None:
        tensor = tensor.to(device=device, dtype=dtype)
    return tensor


@dataclass
class MingTTSState(PipelineStateBase):
    """Per-request state for Ming-Omni-TTS generation."""

    text: str = ""
    prompt: str | None = None
    instructions: str | None = None
    language: str | None = None
    voice: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None

    input_ids: list[int] | None = None
    prompt_text: str | None = None
    spk_token_positions: list[int] | None = None
    spk_injection_positions: list[int] | None = None
    audio_token_position: int | None = None
    prompt_latent_start_position: int | None = None
    prompt_latent_token_count: int = 0

    spk_emb_bytes: bytes | None = None
    spk_emb_shape: list[int] | None = None
    spk_emb_dtype: str | None = None
    prompt_latent_bytes: bytes | None = None
    prompt_latent_shape: list[int] | None = None
    prompt_latent_dtype: str | None = None

    max_decode_steps: int = MING_TTS_DEFAULT_MAX_DECODE_STEPS
    cfg: float = 2.0
    sigma: float = 0.25
    temperature: float = 0.0

    generated_latents_bytes: bytes | None = None
    generated_latents_shape: list[int] | None = None
    generated_latents_dtype: str | None = None
    generated_latents_transport_version: int = MING_TTS_LATENT_TRANSPORT_VERSION
    generated_last_chunk: list[bool] | None = None
    stop_step: int | None = None
    finish_reason: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    sample_rate: int = MING_TTS_SAMPLE_RATE
    duration_s: float | None = None
    audio_decode_time_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "text": self.text,
            "max_decode_steps": int(self.max_decode_steps),
            "cfg": float(self.cfg),
            "sigma": float(self.sigma),
            "temperature": float(self.temperature),
            "sample_rate": int(self.sample_rate),
        }
        if self.prompt is not None:
            data["prompt"] = self.prompt
        if self.instructions is not None:
            data["instructions"] = self.instructions
        if self.language is not None:
            data["language"] = self.language
        if self.voice is not None:
            data["voice"] = self.voice
        if self.ref_audio is not None:
            data["ref_audio"] = self.ref_audio
        if self.ref_text is not None:
            data["ref_text"] = self.ref_text
        if self.input_ids is not None:
            data["input_ids"] = list(self.input_ids)
        if self.prompt_text is not None:
            data["prompt_text"] = self.prompt_text
        if self.spk_token_positions is not None:
            data["spk_token_positions"] = list(self.spk_token_positions)
        if self.spk_injection_positions is not None:
            data["spk_injection_positions"] = list(self.spk_injection_positions)
        if self.audio_token_position is not None:
            data["audio_token_position"] = int(self.audio_token_position)
        if self.prompt_latent_start_position is not None:
            data["prompt_latent_start_position"] = int(
                self.prompt_latent_start_position
            )
        if self.prompt_latent_token_count:
            data["prompt_latent_token_count"] = int(self.prompt_latent_token_count)
        if self.spk_emb_bytes is not None:
            data["spk_emb_bytes"] = self.spk_emb_bytes
            data["spk_emb_shape"] = list(self.spk_emb_shape or [])
            data["spk_emb_dtype"] = self.spk_emb_dtype
        if self.prompt_latent_bytes is not None:
            data["prompt_latent_bytes"] = self.prompt_latent_bytes
            data["prompt_latent_shape"] = list(self.prompt_latent_shape or [])
            data["prompt_latent_dtype"] = self.prompt_latent_dtype
        if self.generated_latents_bytes is not None:
            data["generated_latents_bytes"] = self.generated_latents_bytes
            data["generated_latents_shape"] = list(self.generated_latents_shape or [])
            data["generated_latents_dtype"] = (
                self.generated_latents_dtype or MING_TTS_LATENT_TRANSPORT_DTYPE
            )
            data["generated_latents_transport_version"] = int(
                self.generated_latents_transport_version
            )
        if self.generated_last_chunk is not None:
            data["generated_last_chunk"] = [
                bool(item) for item in self.generated_last_chunk
            ]
        if self.stop_step is not None:
            data["stop_step"] = int(self.stop_step)
        if self.finish_reason is not None:
            data["finish_reason"] = self.finish_reason
        self.append_usage_fields(data)
        if self.duration_s is not None:
            data["duration_s"] = float(self.duration_s)
        if self.audio_decode_time_s:
            data["audio_decode_time_s"] = float(self.audio_decode_time_s)
        return data

    @classmethod
    def from_dict(cls, data: Any) -> "MingTTSState":
        def bytes_or_none(value: Any) -> bytes | None:
            if value is None:
                return None
            return bytes(value)

        def int_list_or_none(value: Any) -> list[int] | None:
            if value is None:
                return None
            return [int(item) for item in value]

        def int_or_default(value: Any, default: int) -> int:
            return int(default if value is None else value)

        def float_or_default(value: Any, default: float) -> float:
            return float(default if value is None else value)

        if not isinstance(data, dict):
            data = {}
        return cls(
            text=str(data.get("text", "")),
            prompt=data.get("prompt"),
            instructions=data.get("instructions"),
            language=data.get("language"),
            voice=data.get("voice"),
            ref_audio=data.get("ref_audio"),
            ref_text=data.get("ref_text"),
            input_ids=int_list_or_none(data.get("input_ids")),
            prompt_text=data.get("prompt_text"),
            spk_token_positions=int_list_or_none(data.get("spk_token_positions")),
            spk_injection_positions=int_list_or_none(
                data.get("spk_injection_positions")
            ),
            audio_token_position=(
                int(data["audio_token_position"])
                if data.get("audio_token_position") is not None
                else None
            ),
            prompt_latent_start_position=(
                int(data["prompt_latent_start_position"])
                if data.get("prompt_latent_start_position") is not None
                else None
            ),
            prompt_latent_token_count=int(
                data.get("prompt_latent_token_count", 0) or 0
            ),
            spk_emb_bytes=bytes_or_none(data.get("spk_emb_bytes")),
            spk_emb_shape=int_list_or_none(data.get("spk_emb_shape")),
            spk_emb_dtype=data.get("spk_emb_dtype"),
            prompt_latent_bytes=bytes_or_none(data.get("prompt_latent_bytes")),
            prompt_latent_shape=int_list_or_none(data.get("prompt_latent_shape")),
            prompt_latent_dtype=data.get("prompt_latent_dtype"),
            max_decode_steps=int_or_default(
                data.get("max_decode_steps"),
                MING_TTS_DEFAULT_MAX_DECODE_STEPS,
            ),
            cfg=float_or_default(data.get("cfg"), 2.0),
            sigma=float_or_default(data.get("sigma"), 0.25),
            temperature=float_or_default(data.get("temperature"), 0.0),
            generated_latents_bytes=bytes_or_none(data.get("generated_latents_bytes")),
            generated_latents_shape=int_list_or_none(
                data.get("generated_latents_shape")
            ),
            generated_latents_dtype=data.get("generated_latents_dtype"),
            generated_latents_transport_version=int(
                data.get(
                    "generated_latents_transport_version",
                    MING_TTS_LATENT_TRANSPORT_VERSION,
                )
                or MING_TTS_LATENT_TRANSPORT_VERSION
            ),
            generated_last_chunk=(
                [bool(item) for item in data["generated_last_chunk"]]
                if data.get("generated_last_chunk") is not None
                else None
            ),
            stop_step=(
                int(data["stop_step"]) if data.get("stop_step") is not None else None
            ),
            finish_reason=(
                str(data["finish_reason"])
                if data.get("finish_reason") is not None
                else None
            ),
            prompt_tokens=int_or_default(data.get("prompt_tokens"), 0),
            completion_tokens=int_or_default(data.get("completion_tokens"), 0),
            engine_time_s=float_or_default(data.get("engine_time_s"), 0.0),
            sample_rate=int_or_default(
                data.get("sample_rate"),
                MING_TTS_SAMPLE_RATE,
            ),
            duration_s=(
                float(data["duration_s"])
                if data.get("duration_s") is not None
                else None
            ),
            audio_decode_time_s=float_or_default(
                data.get("audio_decode_time_s"),
                0.0,
            ),
        )


def load_ming_tts_state(payload: StagePayload) -> MingTTSState:
    return _load_pipeline_state(payload, MingTTSState)


def store_ming_tts_state(payload: StagePayload, state: MingTTSState) -> StagePayload:
    return _store_pipeline_state(payload, state)


__all__ = [
    "MING_TTS_DEFAULT_MAX_DECODE_STEPS",
    "MING_TTS_LATENT_TRANSPORT_DTYPE",
    "MING_TTS_LATENT_TRANSPORT_VERSION",
    "MING_TTS_SAMPLE_RATE",
    "MingTTSState",
    "decode_prompt_latent",
    "decode_generated_latents",
    "decode_speaker_embedding",
    "encode_prompt_latent",
    "encode_generated_latents",
    "encode_speaker_embedding",
    "load_ming_tts_state",
    "store_ming_tts_state",
]
