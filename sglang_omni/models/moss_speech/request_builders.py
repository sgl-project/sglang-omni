# SPDX-License-Identifier: Apache-2.0
"""Build spoken-response prompts and carry reference audio to the codec."""

from __future__ import annotations

import hashlib
from array import array
from dataclasses import dataclass, field
from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams as EngineSamplingParams
from transformers import PreTrainedTokenizerBase

from sglang_omni.models.moss_speech.fsm import (
    AUDIO_PAD,
    EOSP,
    IM_END,
    MODALITY_PAD,
    TEXT_ENDOFTEXT,
    TEXT_VOCAB,
    Mode,
    MossSamplingParams,
    initial_mode,
)
from sglang_omni.models.moss_speech.hf_config import MAX_CONTEXT_TOKENS
from sglang_omni.proto.request import StagePayload
from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

TTS_SYSTEM_PROMPT = "You are a helpful voice assistant. Answer the user's questions with spoken responses."


class RequestValidationError(ValueError):
    """Use the existing public bad-request classification for rejected inputs."""

    def __init__(self, message: str) -> None:
        super().__init__(f"Native chat request rejected: MOSS-Speech: {message}")


class SpeechParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    temperature: float = Field(default=0.6, ge=0)
    top_p: float = Field(default=0.95, gt=0, le=1)
    top_k: int = Field(default=20, ge=-1, strict=True)
    repetition_penalty: float = Field(default=1.1, gt=0)
    seed: int | None = Field(default=None, ge=-(2**63), lt=2**64, strict=True)
    max_new_tokens: int | None = Field(
        default=None, ge=1, lt=MAX_CONTEXT_TOKENS, strict=True
    )
    stream: Literal[False] = False
    min_p: Literal[0.0] = 0.0
    stop: list[str] = Field(default_factory=list, max_length=0)
    stop_token_ids: list[int] = Field(default_factory=list, max_length=0)


class VoiceOptions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    voice: Literal["default"] = "default"
    speed: Literal[1.0] = 1.0
    response_format: Literal["wav", "pcm"] = "wav"
    seed: int | None = Field(default=None, ge=-(2**63), lt=2**64, strict=True)
    ref_audio: str | None = Field(default=None, min_length=1)


class AudioReference(BaseModel):
    model_config = ConfigDict(extra="forbid")

    audio_path: str | None = Field(default=None, min_length=1)
    data: str | None = Field(default=None, min_length=1)
    media_type: str = "audio/wav"


class SpeechPrompt(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    references: list[AudioReference] = Field(min_length=1, max_length=1)


@dataclass(kw_only=True)
class MossSpeechState(DeclarativeStateBase):
    text_token_ids: list[int] = wire(default_factory=list, codec="list")
    prompt_token_count: int = wire(0, codec="int")
    completion_token_count: int = wire(0, codec="int")
    temperature: float = wire(0.0, codec="float")
    top_p: float = wire(0.0, codec="float")
    top_k: int = wire(0, codec="int")
    repetition_penalty: float = wire(0.0, codec="float")
    effective_seed: int = wire(0, codec="int")
    max_new_tokens: int = wire(0, codec="int")
    audio_codes: list[int] = wire(default_factory=list, codec="list")
    reference_audio: str = wire("", codec="str")


@dataclass
class MossSpeechSGLangRequestData(SGLangARRequestData):
    prompt_rows: torch.Tensor | None = None
    output_rows: list[tuple[int, int]] = field(default_factory=list)
    mode: Mode = 0
    params: MossSamplingParams = field(default_factory=MossSamplingParams)
    effective_seed: int = 0
    finished: bool = False
    stop_reason: Literal["stop_token", "length"] | None = None
    rng_generator: torch.Generator | None = None


def preprocess_request(
    payload: StagePayload, tokenizer: PreTrainedTokenizerBase
) -> StagePayload:
    request = payload.request
    if request.metadata.get("task") not in (None, "tts"):
        raise RequestValidationError("only text-to-speech is supported")
    elif request.metadata.get("output_modalities") not in (None, ["audio"]):
        raise RequestValidationError("only audio output is supported")
    elif any(
        request.metadata.get(name)
        for name in ("audios", "images", "videos", "audio_config")
    ):
        raise RequestValidationError("media inputs are unsupported")
    else:
        pass
    try:
        parameters = SpeechParameters.model_validate(
            {name: value for name, value in request.params.items() if value is not None}
        )
        voice_options = VoiceOptions.model_validate(
            request.metadata.get("tts_params", {})
        )
        reference_audio = voice_options.ref_audio
        if isinstance(request.inputs, str):
            text = request.inputs
        else:
            prompt = SpeechPrompt.model_validate(request.inputs)
            text = prompt.text
            reference = prompt.references[0]
            if reference.audio_path is not None and reference.data is None:
                prompt_reference_audio = reference.audio_path
            elif reference.data is not None and reference.audio_path is None:
                prompt_reference_audio = (
                    f"data:{reference.media_type};base64,{reference.data}"
                )
            else:
                raise RequestValidationError("Provide one reference audio source")
            if (
                reference_audio is not None
                and reference_audio != prompt_reference_audio
            ):
                raise RequestValidationError("Conflicting reference audio sources")
            else:
                reference_audio = prompt_reference_audio
    except ValidationError as error:
        raise RequestValidationError(str(error)) from error
    if not text.strip():
        raise RequestValidationError("TTS requires a non-empty text prompt")
    elif reference_audio is None or not reference_audio.strip():
        raise RequestValidationError("Provide ref_audio with each TTS request")
    else:
        pass
    if (
        parameters.seed is not None
        and voice_options.seed is not None
        and parameters.seed != voice_options.seed
    ):
        raise RequestValidationError("conflicting speech seeds")
    else:
        seed = parameters.seed if parameters.seed is not None else voice_options.seed
    if seed is None:
        seed = int.from_bytes(
            hashlib.blake2b(payload.request_id.encode(), digest_size=8).digest(), "big"
        )
    else:
        pass
    # Separate segments preserve the checkpoint processor's token boundaries.
    segments = (
        "<|im_start|>system\n",
        TTS_SYSTEM_PROMPT,
        "<|im_end|>\n",
        "<|im_start|>user\n",
        text,
        "<|im_end|>\n",
        "<|im_start|>assistant\n<|object_ref_start|>",
    )
    text_token_ids = [
        token_id
        for segment in segments
        for token_id in tokenizer.encode(segment, add_special_tokens=False)
    ]
    remaining_tokens = MAX_CONTEXT_TOKENS - len(text_token_ids)
    max_new_tokens = (
        remaining_tokens
        if parameters.max_new_tokens is None
        else parameters.max_new_tokens
    )
    if not 1 <= max_new_tokens <= remaining_tokens:
        raise RequestValidationError(
            f"Prompt plus generation budget exceeds {MAX_CONTEXT_TOKENS} tokens"
        )
    else:
        pass
    state = MossSpeechState(
        text_token_ids=text_token_ids,
        prompt_token_count=len(text_token_ids),
        temperature=parameters.temperature,
        top_p=parameters.top_p,
        top_k=parameters.top_k,
        repetition_penalty=parameters.repetition_penalty,
        effective_seed=seed,
        max_new_tokens=max_new_tokens,
        reference_audio=reference_audio,
    )
    return StagePayload(payload.request_id, request, state.to_dict())


def build_sglang_moss_request(payload: StagePayload) -> MossSpeechSGLangRequestData:
    state = MossSpeechState.from_dict(payload.data)
    text_tokens = torch.tensor(state.text_token_ids, dtype=torch.long)
    prompt_rows = torch.stack(
        (text_tokens, torch.full_like(text_tokens, AUDIO_PAD)), dim=1
    )
    parameters = MossSamplingParams(
        do_sample=state.temperature > 0 and state.top_k != 1,
        temperature=state.temperature,
        top_p=state.top_p,
        top_k=state.top_k,
        repetition_penalty=state.repetition_penalty,
    )
    engine_sampling = EngineSamplingParams(
        max_new_tokens=state.max_new_tokens, temperature=0.0
    )
    engine_sampling.normalize(None)
    engine_sampling.verify(TEXT_VOCAB)
    engine_request = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=array("q", state.text_token_ids),
        sampling_params=engine_sampling,
        eos_token_ids={TEXT_ENDOFTEXT, IM_END},
        vocab_size=TEXT_VOCAB,
    )
    engine_request.tokenizer = None
    return MossSpeechSGLangRequestData(
        req=engine_request,
        input_ids=text_tokens,
        prompt_rows=prompt_rows,
        mode=initial_mode(prompt_rows),
        params=parameters,
        effective_seed=state.effective_seed,
        max_new_tokens=state.max_new_tokens,
        enforce_request_limits=True,
        stage_payload=payload,
    )


def extract_output_codes(rows: list[tuple[int, int]]) -> list[int]:
    codes: list[int] = []
    for text_token, audio_token in rows:
        if text_token != MODALITY_PAD:
            continue
        elif audio_token == EOSP:
            break
        else:
            assert 0 <= audio_token < EOSP
            codes.append(audio_token)
    return codes


def build_result(request_data: MossSpeechSGLangRequestData) -> StagePayload:
    payload = request_data.stage_payload
    assert payload is not None
    state = MossSpeechState.from_dict(payload.data)
    state.audio_codes = extract_output_codes(request_data.output_rows)
    state.completion_token_count = len(request_data.output_rows)
    state.finish_reason = request_data.finish_reason or ""
    state.text_token_ids = []
    return StagePayload(payload.request_id, payload.request, state.to_dict())
