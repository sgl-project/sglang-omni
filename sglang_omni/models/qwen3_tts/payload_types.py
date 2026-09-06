# SPDX-License-Identifier: Apache-2.0
"""Qwen3-TTS pipeline state."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class Qwen3TTSState(DeclarativeStateBase):
    """Per-request state for Qwen3-TTS generation."""

    sample_rate: int = wire(24000, codec="int")

    text: str = wire("", codec="str")
    task_type: str = wire("Base", codec="str_or")
    task_type_explicit: bool = wire(False, codec="bool")
    language: str = wire("auto", codec="str_or")
    voice: str | None = None
    instructions: str | None = None
    ref_audio: Any | None = None
    ref_text: str | None = None
    uploaded_voice_name: str | None = None
    uploaded_voice_created_at: int | None = None
    x_vector_only_mode: bool = wire(False, codec="bool")
    non_streaming_mode: bool = wire(False, codec="bool")
    stream_codec_output: bool = wire(True, codec="bool")
    suppress_bootstrap_silence: bool = wire(False, codec="bool")
    generation_kwargs: dict[str, Any] = wire(default_factory=dict, codec="dict")
    seed: int | None = None
    audio_codes: Any | None = wire(None, codec="tensor_list")
    finish_reason: str | None = None
    ref_code_len: int = wire(0, emit="truthy", codec="int")
    audio_samples: Any | None = wire(None, codec="tensor_list")
    # Note (Jiaxin Deng): set only by a preprocessing stage that runs outside the
    # engine process; the AR request builder consumes and clears them. tensor_cpu
    # keeps them on the relay in their own dtype instead of widening bf16 to fp32
    # and base64 into the control-plane message.
    prepared_prompt_embeds: Any | None = wire(None, codec="tensor_cpu")
    prepared_text_tail: Any | None = wire(None, codec="tensor_cpu")
    prepared_ref_code: Any | None = wire(None, codec="tensor_cpu")
    prepared_pad_embed: Any | None = wire(None, codec="tensor_cpu")
    prepared_input_ids: list[int] | None = None
    prepared_gen_kwargs: dict[str, Any] | None = None
