# SPDX-License-Identifier: Apache-2.0
"""Request/result helpers for Voxtral-TTS SGLang AR stage."""

from __future__ import annotations

import collections
import hashlib
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch

from sglang_omni.models.voxtral_tts.acoustic_transformer import AudioSpecialTokens
from sglang_omni.models.voxtral_tts.io import VoxtralTTSState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

if TYPE_CHECKING:
    from sglang_omni.models.voxtral_tts.sglang_model import VoxtralSGLangTTSModel
else:
    pass


@dataclass
class VoxtralSGLangRequestData(SGLangARRequestData):
    enforce_request_limits: bool = True
    voice_embedding: torch.Tensor | None = None
    audio_token_id: int = 24
    output_codes: list[torch.Tensor] = field(default_factory=list)
    pending_feedback_queue: collections.deque[torch.Tensor] | list[torch.Tensor] = (
        field(default_factory=collections.deque)
    )


def voice_cache_key(voice: str, voice_embedding: torch.Tensor | None) -> str | None:
    if voice_embedding is None:
        return None
    else:
        pass
    digest = hashlib.blake2b(voice.encode("utf-8"), digest_size=16).hexdigest()
    return f"voxtral_voice:{digest}"


def build_sglang_voxtral_request(
    payload: StagePayload,
    *,
    model: "VoxtralSGLangTTSModel",
    voice_embeddings: dict[str, torch.Tensor],
) -> VoxtralSGLangRequestData:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    state = VoxtralTTSState.from_dict(payload.data)
    input_ids_list = [int(token_id) for token_id in (state.input_ids or [])]
    input_ids = torch.tensor(input_ids_list, dtype=torch.long)
    voice = state.voice or "cheerful_female"
    voice_embedding = voice_embeddings.get(voice)

    eos_id = AudioSpecialTokens.id(AudioSpecialTokens.end_audio)
    sampling_params = SamplingParams(
        max_new_tokens=int(state.max_new_tokens or 4096),
        temperature=0.0,
        stop_token_ids=[eos_id],
    )
    sampling_params.normalize(None)
    sampling_params.verify(model.voxtral_config.text_config.vocab_size)

    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids_list,
        sampling_params=sampling_params,
        eos_token_ids={eos_id},
        vocab_size=model.voxtral_config.text_config.vocab_size,
        extra_key=voice_cache_key(voice, voice_embedding),
    )
    req.tokenizer = None
    req._codec_suppress_tokens = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    data = VoxtralSGLangRequestData(
        input_ids=input_ids,
        max_new_tokens=int(state.max_new_tokens or 4096),
        output_ids=req.output_ids,
        req=req,
        voice_embedding=voice_embedding,
        audio_token_id=int(model.audio_token_id),
    )
    data.stage_payload = payload
    return data


def apply_sglang_voxtral_result(
    payload: StagePayload,
    data: VoxtralSGLangRequestData,
) -> StagePayload:
    state = VoxtralTTSState.from_dict(payload.data)
    if data.output_codes:
        codes = torch.stack(data.output_codes, dim=0).to(dtype=torch.long)
    else:
        codes = torch.empty((0, 0), dtype=torch.long)
    state.audio_codes = codes
    state.prompt_tokens = len(data.input_ids) if data.input_ids is not None else 0
    state.completion_tokens = len(data.output_codes)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=state.to_dict(),
    )


def make_voxtral_scheduler_adapters(
    *,
    model: "VoxtralSGLangTTSModel | None",
    voice_embeddings: dict[str, torch.Tensor],
) -> tuple[
    Callable[[StagePayload], VoxtralSGLangRequestData],
    Callable[[VoxtralSGLangRequestData], StagePayload],
]:
    def request_builder(payload: StagePayload) -> VoxtralSGLangRequestData:
        return build_sglang_voxtral_request(
            payload,
            model=model,
            voice_embeddings=voice_embeddings,
        )

    def result_adapter(data: VoxtralSGLangRequestData) -> StagePayload:
        return apply_sglang_voxtral_result(data.stage_payload, data)

    return request_builder, result_adapter
