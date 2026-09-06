from __future__ import annotations

import logging

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.nemotron_voicechat.payload_types import NemotronVoiceChatState
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

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


def _ar_request(
    payload: StagePayload, *, input_ids: list[int], max_new_tokens: int, vocab_size: int
) -> SGLangARRequestData:
    # Greedy sampling keeps Thinker's sampled tokens equal to those sent to Talker.
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        ignore_eos=True,
    )
    sampling_params.normalize(tokenizer=None)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    return SGLangARRequestData(
        req=req,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        stage_payload=payload,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
    )


def build_thinker_request(
    payload: StagePayload,
    *,
    vocab_size: int,
    prompt_token_ids: list[int],
    pad_token_id: int,
) -> SGLangARRequestData:
    """One request per utterance: the system prompt plus a position for the
    first acoustic frame, then one decode step per frame."""
    params = payload.request.params
    thinker_sampling = (params.get("stage_sampling") or {}).get("thinker") or {}
    temperature = thinker_sampling.get("temperature", params.get("temperature"))
    if temperature is not None and float(temperature) != 0.0:
        logger.warning(
            "Ignoring text temperature=%s; using temperature=0.", temperature
        )
    num_frames = NemotronVoiceChatState.from_dict(payload.data).num_frames
    opening = [*prompt_token_ids, pad_token_id]
    data = _ar_request(
        payload,
        input_ids=opening,
        max_new_tokens=num_frames,
        vocab_size=vocab_size,
    )
    data.pending_stream_tokens = []
    return data


def apply_thinker_result(data: SGLangARRequestData) -> StagePayload:
    payload = data.stage_payload
    state = NemotronVoiceChatState.from_dict(payload.data)
    state.text_ids = list(data.output_ids)
    payload.data = state.to_dict()
    return payload


def thinker_stream_output_builder(
    request_id: str, data: SGLangARRequestData, req_output
) -> list[OutgoingMessage]:
    del req_output
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
    return _ar_request(
        payload,
        input_ids=[TALKER_PLACEHOLDER_ID] * prompt_frames,
        # One more than the frames: the prefill's own step does not emit codes.
        max_new_tokens=num_frames + 1,
        vocab_size=vocab_size,
    )


def apply_talker_result(data: SGLangARRequestData) -> StagePayload:
    payload = data.stage_payload
    state = NemotronVoiceChatState.from_dict(payload.data)
    state.codes = torch.stack(data.talker_model_inputs["codes_rows"]).cpu()
    payload.data = state.to_dict()
    return payload


def talker_stream_output_builder(
    request_id: str, data: SGLangARRequestData, req_output
) -> list[OutgoingMessage]:
    del req_output
    codes = data.talker_model_inputs.pop("stream_chunk", None)
    if codes is None:
        return []
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=codes,
            target="code2wav",
            metadata={"modality": "audio_codes"},
        )
    ]


def merge_for_talker(payloads: dict) -> StagePayload:
    payload = payloads["perception"]
    state = NemotronVoiceChatState.from_dict(payload.data)
    return StagePayload(
        request_id=payload.request_id,
        request=payload.request,
        data=NemotronVoiceChatState(num_frames=state.num_frames).to_dict(),
    )
