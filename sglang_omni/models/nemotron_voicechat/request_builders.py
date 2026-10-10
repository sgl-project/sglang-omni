from __future__ import annotations

import logging
from dataclasses import dataclass, field

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import RequestOutput

logger = logging.getLogger(__name__)

BOS_TOKEN_ID = 1
TALKER_PLACEHOLDER_ID = 0
# What the checkpoint's stt config names, resolved through its tokenizer, is
# authoritative for these; see NemotronVoiceChatEngineBuilder._prompt_tokens.
SYSTEM_PROMPT = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Answer in a spoken, conversational style rather than a written one. "
    "Do not repeat the same sentence over and over again. "
    "Start the conversation by greeting the user."
)


@dataclass
class NemotronVoiceChatRequestData(SGLangARRequestData):
    acoustic_frames: torch.Tensor | None = None
    function_ids: list[int] = field(default_factory=list)
    pending_stream_tokens: list[int] = field(default_factory=list)


def build_autoregressive_request(
    payload: StagePayload, *, input_ids: list[int], max_new_tokens: int, vocab_size: int
) -> Req:
    # Greedy sampling keeps Thinker's sampled tokens equal to those sent to Talker.
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    sampling_params.normalize(tokenizer=None)
    return Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )


def build_thinker_request(
    payload: StagePayload,
    *,
    vocab_size: int,
    prompt_token_ids: list[int],
    pad_token_id: int,
) -> NemotronVoiceChatRequestData:
    """One request per utterance: the system prompt plus a position for the
    first acoustic frame, then one decode step per frame."""
    params = payload.request.params
    thinker_sampling = (params.get("stage_sampling") or {}).get("thinker") or {}
    temperature = thinker_sampling.get("temperature", params.get("temperature"))
    if temperature is not None and float(temperature) != 0.0:
        logger.warning(
            f"Ignoring text temperature={temperature}; using temperature=0."
        )
    else:
        pass
    state = NemotronVoiceChatState.from_dict(payload.data)
    acoustic_frames = state.acoustic_frames
    if acoustic_frames is None:
        raise ValueError("Nemotron VoiceChat Thinker request has no acoustic frames")
    else:
        pass
    if acoustic_frames.ndim != 2 or acoustic_frames.shape[0] != state.num_frames + 1:
        raise ValueError(
            "Nemotron VoiceChat acoustic frame shape does not match its frame count: "
            f"shape={tuple(acoustic_frames.shape)}, num_frames={state.num_frames}"
        )
    else:
        pass

    opening = [*prompt_token_ids, pad_token_id]
    request = build_autoregressive_request(
        payload,
        input_ids=opening,
        max_new_tokens=state.num_frames,
        vocab_size=vocab_size,
    )
    # The Thinker result only needs the frame count and generated text. Keeping
    # the wire bytes here would duplicate the request-owned CPU and device tensors.
    state.waveform = None
    state.acoustic_frames = None
    payload.data = state.to_dict()
    return NemotronVoiceChatRequestData(
        req=request,
        input_ids=torch.tensor(opening, dtype=torch.long),
        stage_payload=payload,
        max_new_tokens=state.num_frames,
        temperature=0.0,
        acoustic_frames=acoustic_frames.contiguous(),
    )


def apply_thinker_result(data: NemotronVoiceChatRequestData) -> StagePayload:
    payload = data.stage_payload
    if payload is None:
        raise RuntimeError("Nemotron VoiceChat Thinker result has no stage payload")
    else:
        pass
    # Release the request-owned acoustic tensor before forwarding the result.
    data.acoustic_frames = None
    state = NemotronVoiceChatState.from_dict(payload.data)
    state.text_ids = list(data.output_ids)
    payload.data = state.to_dict()
    return payload


def thinker_stream_output_builder(
    request_id: str,
    data: NemotronVoiceChatRequestData,
    request_output: RequestOutput,
) -> list[OutgoingMessage]:
    del request_output
    tokens = data.pending_stream_tokens
    data.pending_stream_tokens = []
    # The stages run in separate processes, and the relay between them moves
    # tensors; a bare int never reaches the other side.
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=torch.tensor([int(token)], dtype=torch.long),
            target="talker",
            metadata={"modality": "text_token"},
        )
        for token in tokens
    ]


def build_talker_request(
    payload: StagePayload, *, vocab_size: int, prompt_frames: int
) -> SGLangARRequestData:
    """Whole-utterance request used by the offline pipeline."""
    num_frames = NemotronVoiceChatState.from_dict(payload.data).num_frames
    input_ids = [TALKER_PLACEHOLDER_ID] * prompt_frames
    request = build_autoregressive_request(
        payload,
        input_ids=input_ids,
        # One more than the frames: the prefill's own step does not emit codes.
        max_new_tokens=num_frames + 1,
        vocab_size=vocab_size,
    )
    return SGLangARRequestData(
        req=request,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        stage_payload=payload,
        max_new_tokens=num_frames + 1,
        temperature=0.0,
    )


def apply_talker_result(data: SGLangARRequestData) -> StagePayload:
    payload = data.stage_payload
    state = NemotronVoiceChatState.from_dict(payload.data)
    state.codes = torch.stack(data.talker_model_inputs["codes_rows"]).cpu()
    payload.data = state.to_dict()
    return payload


def talker_stream_output_builder(
    request_id: str,
    data: SGLangARRequestData,
    request_output: RequestOutput,
) -> list[OutgoingMessage]:
    del request_output
    codes = data.talker_model_inputs.pop("stream_chunk", None)
    if codes is None:
        return []
    else:
        pass
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=codes,
            target="code2wav",
            metadata={"modality": "audio_codes"},
        )
    ]


def merge_for_talker(payloads: dict[str, StagePayload]) -> StagePayload:
    payload = payloads["perception"]
    state = NemotronVoiceChatState.from_dict(payload.data)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=NemotronVoiceChatState(num_frames=state.num_frames).to_dict(),
    )
