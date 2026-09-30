# SPDX-License-Identifier: Apache-2.0
"""Request lowering for EasyMagpie TTS."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.easymagpie_tts.payload_types import (
    DEFAULT_CONTEXT_TEXT,
    DEFAULT_MAX_NEW_FRAMES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_VOICE,
    EasyMagpieTTSState,
)
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.seed import resolve_row_seed
from sglang_omni.scheduling.sglang_backend import SGLangARRequestData

CONTINUE_TOKEN_ID = 0
STOP_TOKEN_ID = 1
BACKBONE_VOCAB_SIZE = 2


def build_easymagpie_state(payload: StagePayload) -> EasyMagpieTTSState:
    inputs = payload.request.inputs
    params = payload.request.params or {}
    metadata = payload.request.metadata or {}
    tts_params = metadata.get("tts_params") or {}

    if isinstance(inputs, dict):
        text = inputs.get("input", inputs.get("text", ""))
    elif inputs is None:
        text = ""
    else:
        text = inputs
    text = str(text).strip()
    if not text:
        raise ValueError("EasyMagpie TTS requires non-empty input text")
    else:
        pass

    # Endpoint defaults for params must not override model defaults; only
    # values the client set explicitly are honoured.
    explicit = set(tts_params.get("explicit_generation_params") or ())

    def pick(name: str, default: float | int | None) -> float | int | None:
        if tts_params.get(name) is not None:
            return tts_params[name]
        elif name in explicit and params.get(name) is not None:
            return params[name]
        else:
            return default

    temperature = float(pick("temperature", DEFAULT_TEMPERATURE))
    top_k = int(pick("top_k", DEFAULT_TOP_K))
    max_new_frames = int(tts_params.get("max_new_frames") or DEFAULT_MAX_NEW_FRAMES)
    if "max_new_tokens" in explicit and params.get("max_new_tokens") is not None:
        max_new_frames = int(params["max_new_tokens"])
    else:
        pass
    seed = pick("seed", None)
    if isinstance(seed, bool):
        raise ValueError("EasyMagpie TTS seed must be an integer")
    else:
        pass
    if temperature <= 0 or top_k <= 0 or max_new_frames <= 0:
        raise ValueError(
            "EasyMagpie TTS temperature, top_k, and max_new_frames must be positive"
        )
    else:
        pass

    return EasyMagpieTTSState(
        text=text,
        voice=str(tts_params.get("voice") or params.get("voice") or DEFAULT_VOICE),
        context_text=str(tts_params.get("context_text") or DEFAULT_CONTEXT_TEXT),
        temperature=temperature,
        top_k=top_k,
        max_new_frames=max_new_frames,
        seed=None if seed is None else int(seed),
    )


@dataclass
class EasyMagpieSGLangRequestData(SGLangARRequestData):
    state: EasyMagpieTTSState = field(default_factory=EasyMagpieTTSState)
    output_codes: list[torch.Tensor] = field(default_factory=list)
    decode_offset: int = 0
    last_audio_codes: torch.Tensor | None = None
    last_phoneme_tokens: torch.Tensor | None = None
    last_phoneme_is_eos: bool = False
    phoneme_ended: bool = False
    sampling_seed: int = 0
    engine_start_s: float = 0.0


def prompt_cache_key(state: EasyMagpieTTSState) -> str:
    """Radix key for the spliced prompt; the placeholder ids carry no content."""
    digest = hashlib.sha256()
    for part in (state.voice, state.context_text, state.text):
        digest.update(part.encode())
        digest.update(b"\0")
    return f"easymagpie:{digest.hexdigest()}"


def max_decode_tokens(state: EasyMagpieTTSState) -> int:
    """Output-token budget that yields at most max_new_frames acoustic frames.

    The prefill emits one token, and the decode steps before the speech delay
    produce no audio.
    """
    return state.max_new_frames + state.speech_delay - state.text_prefill_num + 1


def build_sglang_easymagpie_request(
    payload: StagePayload,
) -> EasyMagpieSGLangRequestData:
    state = EasyMagpieTTSState.from_dict(payload.data)
    speaker_frames = (
        0 if state.speaker_embedding is None else int(state.speaker_embedding.shape[0])
    )
    prompt_len = speaker_frames + len(state.context_token_ids) + state.text_prefill_num
    max_new_tokens = max_decode_tokens(state)
    sampling = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        stop_token_ids=[STOP_TOKEN_ID],
    )
    sampling.normalize(None)
    sampling.verify(BACKBONE_VOCAB_SIZE)
    prompt_ids = [CONTINUE_TOKEN_ID] * prompt_len
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=prompt_ids,
        sampling_params=sampling,
        eos_token_ids={STOP_TOKEN_ID},
        vocab_size=BACKBONE_VOCAB_SIZE,
        extra_key=prompt_cache_key(state),
    )
    req.tokenizer = None
    return EasyMagpieSGLangRequestData(
        state=state,
        decode_offset=state.text_prefill_num,
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        max_new_tokens=max_new_tokens,
        input_embeds_are_projected=True,
        sampling_seed=resolve_row_seed(state.seed),
        engine_start_s=time.perf_counter(),
    )


def apply_easymagpie_result(data: EasyMagpieSGLangRequestData) -> StagePayload:
    state = data.state
    if data.output_codes:
        state.audio_codes = torch.stack(data.output_codes, dim=0).to(torch.long)
    else:
        state.audio_codes = None
    # The speaker rows are prompt-only; the codec stage does not need them.
    state.speaker_embedding = None
    state.prompt_tokens = int(data.input_ids.numel())
    state.completion_tokens = len(data.output_codes)
    state.engine_time_s = time.perf_counter() - data.engine_start_s
    payload = data.stage_payload
    return StagePayload(
        request_id=payload.request_id, request=payload.request, data=state.to_dict()
    )


__all__ = [
    "EasyMagpieSGLangRequestData",
    "apply_easymagpie_result",
    "build_easymagpie_state",
    "build_sglang_easymagpie_request",
    "max_decode_tokens",
    "prompt_cache_key",
]
