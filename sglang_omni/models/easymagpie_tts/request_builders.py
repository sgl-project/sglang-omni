# SPDX-License-Identifier: Apache-2.0
"""Request lowering for EasyMagpie TTS."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass, field
from typing import Any

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.easymagpie_tts.payload_types import (
    DEFAULT_CONTEXT_TEXT,
    DEFAULT_MAX_NEW_FRAMES,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_K,
    DEFAULT_VOICE,
    MAX_TEXT_TOKENS,
    MAX_TOP_K,
    EasyMagpieTTSState,
)
from sglang_omni.proto import StagePayload
from sglang_omni.sampling.seed import resolve_row_seed
from sglang_omni.scheduling.message import OutgoingMessage
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
    if top_k > MAX_TOP_K:
        raise ValueError(f"EasyMagpie TTS top_k must be at most {MAX_TOP_K}")
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
    stream_code_count: int = 0
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

    The prefill emits one token, and the decode steps before the emit delay
    produce no audio.
    """
    return state.max_new_frames + state.audio_emit_delay - state.text_prefill_num + 1


def build_sglang_easymagpie_request(
    payload: StagePayload,
) -> EasyMagpieSGLangRequestData:
    state = EasyMagpieTTSState.from_dict(payload.data)
    if len(state.text_token_ids) > MAX_TEXT_TOKENS:
        raise ValueError(
            f"EasyMagpie TTS text has {len(state.text_token_ids)} tokens; "
            f"at most {MAX_TEXT_TOKENS} are supported"
        )
    else:
        pass
    prompt_len = (
        state.speaker_frames + len(state.context_token_ids) + state.text_prefill_num
    )
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
        stage_payload=payload,
        req=req,
        output_ids=req.output_ids,
        input_ids=torch.tensor(prompt_ids, dtype=torch.long),
        max_new_tokens=max_new_tokens,
        input_embeds_are_projected=True,
        sampling_seed=resolve_row_seed(state.seed),
        engine_start_s=time.perf_counter(),
    )


def is_streaming_request(data: EasyMagpieSGLangRequestData) -> bool:
    params = data.stage_payload.request.params
    return isinstance(params, dict) and bool(params.get("stream", False))


class EasyMagpieStreamOutputBuilder:
    """Forward the acoustic frames produced since the last step to the vocoder."""

    def __call__(
        self, request_id: str, data: EasyMagpieSGLangRequestData, req_output: Any
    ) -> list[OutgoingMessage]:
        del req_output
        if not is_streaming_request(data) or data.stream_code_count == len(
            data.output_codes
        ):
            return []
        else:
            pass
        rows = data.output_codes[data.stream_code_count :]
        data.stream_code_count = len(data.output_codes)
        return [
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target="vocoder",
                data=torch.stack(rows, dim=0).to(torch.long),
                metadata=dict(STREAM_METADATA),
            )
        ]

    def build_batch(
        self, entries: list[tuple[str, EasyMagpieSGLangRequestData, Any]]
    ) -> list[OutgoingMessage] | None:
        """One ``[B, 1, codebooks]`` message for a step that streamed one frame
        per request; None when some request has a backlog of several."""
        streaming = []
        for request_id, data, _ in entries:
            if not is_streaming_request(data):
                continue
            else:
                pass
            pending = len(data.output_codes) - data.stream_code_count
            if pending > 1:
                return None
            elif pending == 1:
                streaming.append((request_id, data))
            else:
                pass
        if not streaming:
            return []
        else:
            pass
        rows = []
        for _, data in streaming:
            rows.append(data.output_codes[data.stream_code_count])
            data.stream_code_count += 1
        request_ids = tuple(request_id for request_id, _ in streaming)
        return [
            OutgoingMessage(
                request_id=request_ids[0],
                type="stream",
                target="vocoder",
                data=torch.stack(rows, dim=0).to(torch.long).unsqueeze(1),
                metadata=dict(STREAM_METADATA),
                request_ids=request_ids,
            )
        ]


STREAM_METADATA = {"modality": "audio_codes", "stream": True}
easymagpie_stream_output_builder = EasyMagpieStreamOutputBuilder()


def apply_easymagpie_result(data: EasyMagpieSGLangRequestData) -> StagePayload:
    state = data.state
    # Streamed frames already reached the vocoder; it only needs usage here.
    if data.output_codes and not is_streaming_request(data):
        state.audio_codes = torch.stack(data.output_codes, dim=0).to(torch.long)
    else:
        state.audio_codes = None
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
    "easymagpie_stream_output_builder",
    "is_streaming_request",
    "max_decode_tokens",
    "prompt_cache_key",
]
