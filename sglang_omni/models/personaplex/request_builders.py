# SPDX-License-Identifier: Apache-2.0
"""Turns a stage payload into an SGLang request, and the result back into one."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.personaplex.architecture import (
    DEFAULT_AUDIO_TEMPERATURE,
    DEFAULT_AUDIO_TOP_K,
    DEFAULT_TEXT_TEMPERATURE,
    DEFAULT_TEXT_TOP_K,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
    TEXT_CARD,
    TEXT_PAD_ID,
)
from sglang_omni.models.personaplex.config import CODE2WAV_STAGE, LM_STAGE
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.models.personaplex.prompts import VoicePrompt, load_packaged_voice
from sglang_omni.models.personaplex.sampling import AudioSampling
from sglang_omni.models.personaplex.timeline import (
    PromptFrames,
    Timeline,
    build_prompt_frames,
    build_timeline,
)
from sglang_omni.proto import EXPLICIT_GENERATION_PARAMS_KEY, StagePayload
from sglang_omni.sampling.seed import derive_sampling_seed
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

SEED_NAMESPACE = "personaplex"


@dataclass(frozen=True)
class RequestSampling:
    text_temperature: float
    text_top_k: int
    audio: AudioSampling
    seed: int | None

    @property
    def text_seed(self) -> int | None:
        return (
            None
            if self.seed is None
            else derive_sampling_seed(SEED_NAMESPACE, self.seed, "text")
        )

    @property
    def audio_seed(self) -> int | None:
        return (
            None
            if self.seed is None
            else derive_sampling_seed(SEED_NAMESPACE, self.seed, "audio")
        )


def stage_overrides(params: dict, stage: str) -> dict:
    stage_params = params.get("stage_params")
    overrides = stage_params.get(stage) if isinstance(stage_params, dict) else None
    return overrides if isinstance(overrides, dict) else {}


def stage_request_params(params: dict, stage: str) -> dict:
    """Request params with stage_params[stage] layered on top.

    The in-process client can set PersonaPlex options at the top level; an HTTP
    request reaches them only through stage_params.
    """
    return {**params, **stage_overrides(params, stage)}


def request_param(params: dict, key: str, default, cast):
    value = params.get(key)
    return default if value is None else cast(value)


def resolve_sampling(params: dict, explicit_fields=()) -> RequestSampling:
    """temperature/top_k steer the text, audio_temperature /
    audio_top_k the codes; seed makes both draws reproducible.

    Text values come from stage_sampling or stage_params for the lm stage, or
    from the top level when explicit_fields names them; the client fills the
    top level for every request, so anything else keeps the reference default.
    """
    stage_sampling = (params.get("stage_sampling") or {}).get(LM_STAGE) or {}
    lm_overrides = stage_overrides(params, LM_STAGE)
    lm_params = {**params, **lm_overrides}
    explicit = set(explicit_fields)
    seed = lm_params.get("seed")
    if isinstance(seed, bool):
        raise ValueError("PersonaPlex seed must be an integer")
    else:
        pass

    def text(key: str, default, cast):
        top_level = params if key in explicit else {}
        for source in (stage_sampling, lm_overrides, top_level):
            if source.get(key) is not None:
                return cast(source[key])
            else:
                pass
        return default

    return RequestSampling(
        text_temperature=text("temperature", DEFAULT_TEXT_TEMPERATURE, float),
        text_top_k=text("top_k", DEFAULT_TEXT_TOP_K, int),
        audio=AudioSampling(
            temperature=request_param(
                lm_params, "audio_temperature", DEFAULT_AUDIO_TEMPERATURE, float
            ),
            top_k=request_param(lm_params, "audio_top_k", DEFAULT_AUDIO_TOP_K, int),
        ),
        seed=None if seed is None else int(seed),
    )


def packaged_voice_for(
    state: PersonaPlexState, voice_cache: dict[str, VoicePrompt]
) -> VoicePrompt | None:
    """The packaged voice state names, loaded once per path."""
    if state.voice_path is None:
        return None
    else:
        pass
    packaged_voice = voice_cache.get(state.voice_path)
    if packaged_voice is None:
        packaged_voice = load_packaged_voice(Path(state.voice_path))
        voice_cache[state.voice_path] = packaged_voice
    else:
        pass
    return packaged_voice


def prompt_from_state(
    state: PersonaPlexState, packaged_voice: VoicePrompt | None = None
) -> tuple[PromptFrames, VoicePrompt]:
    voice = packaged_voice or VoicePrompt(frames=int(state.voice_frames))
    voice_codes = state.voice_codes
    prompt = build_prompt_frames(
        voice_frames=voice.frames,
        text_prompt_ids=[int(i) for i in state.text_prompt_ids],
        voice_codes=None if voice_codes is None else voice_codes.to(torch.long),
    )
    return prompt, voice


def timeline_from_state(
    state: PersonaPlexState, packaged_voice: VoicePrompt | None = None
) -> Timeline:
    if state.user_codes is None:
        raise ValueError("PersonaPlex LM request has no encoded caller audio")
    else:
        pass
    prompt, voice = prompt_from_state(state, packaged_voice)
    return build_timeline(
        prompt,
        state.user_codes.to(torch.long),
        voice_embeddings=voice.embeddings,
        voice_tail_codes=voice.tail_codes,
    )


def prompt_input_ids(timeline: Timeline) -> list[int]:
    """Placeholder ids for SGLang's bookkeeping; the model runner embeds the real rows."""
    # Note (wilsonzheng0327): The text stream's initial token is outside the
    # vocabulary, so it is masked.
    text_ids = timeline.prefill_tokens[:, 0].clone()
    text_ids[text_ids >= TEXT_CARD] = TEXT_PAD_ID
    return [int(i) for i in text_ids.tolist()]


def lm_sampling_params(
    sampling: RequestSampling, max_new_tokens: int
) -> SamplingParams:
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=sampling.text_temperature,
        top_k=sampling.text_top_k,
        ignore_eos=True,
    )
    sampling_params.normalize(tokenizer=None)
    if sampling.text_seed is not None:
        sampling_params.sampling_seed = sampling.text_seed
    else:
        pass
    return sampling_params


def new_model_inputs(timeline: Timeline, sampling: RequestSampling) -> dict:
    """The runner's per-request state, shared by every unit of a session."""
    return {
        "timeline": timeline,
        "sampling": sampling,
        "agent_row": timeline.agent_row_before_start,
        "agent_rows": [],
        "frames": [],
        "pending_frames": [],
    }


def build_lm_request(
    payload: StagePayload,
    *,
    vocab_size: int,
    voice_cache: dict[str, VoicePrompt],
    context_length: int | None = None,
) -> SGLangARRequestData:
    """One request per recording: the whole prompt as prefill, then one
    decode step per 80 ms frame of the caller's audio.

    voice_cache holds the packaged voices this stage has loaded, by path.
    """
    state = PersonaPlexState.from_dict(payload.data)
    packaged_voice = packaged_voice_for(state, voice_cache)
    timeline = timeline_from_state(state, packaged_voice)
    metadata = payload.request.metadata or {}
    sampling = resolve_sampling(
        payload.request.params, metadata.get(EXPLICIT_GENERATION_PARAMS_KEY) or ()
    )
    if timeline.num_frames < 1:
        raise ValueError("PersonaPlex needs at least one 80 ms frame of caller audio")
    else:
        pass
    positions = timeline.num_prompt_positions + timeline.num_frames
    if context_length is not None and positions > context_length - 1:
        raise ValueError(
            f"PersonaPlex request needs {positions} positions "
            f"({timeline.num_prompt_positions} prompt + {timeline.num_frames} caller "
            f"frames, {timeline.num_frames * SAMPLES_PER_FRAME / SAMPLE_RATE:.1f} s) "
            f"but the LM context holds {context_length - 1}; shorten the recording "
            "or raise the lm stage's context_length"
        )
    else:
        pass

    sampling_params = lm_sampling_params(sampling, timeline.num_frames)
    input_ids = prompt_input_ids(timeline)
    req = Req(
        rid=payload.request_id,
        origin_input_text="",
        origin_input_ids=input_ids,
        sampling_params=sampling_params,
        vocab_size=vocab_size,
    )
    data = SGLangARRequestData(
        req=req,
        input_ids=torch.tensor(input_ids, dtype=torch.long),
        stage_payload=payload,
        max_new_tokens=timeline.num_frames,
        temperature=sampling.text_temperature,
    )
    data.talker_model_inputs = new_model_inputs(timeline, sampling)
    data.talker_model_inputs["num_samples"] = int(state.num_samples)
    return data


def apply_lm_result(data: SGLangARRequestData) -> StagePayload:
    payload = data.stage_payload
    state = PersonaPlexState.from_dict(payload.data)
    frames = data.talker_model_inputs["frames"]
    state.text_ids = [int(i) for i in data.output_ids]
    state.codes = (
        torch.stack(frames).cpu() if frames else torch.zeros(0, 8, dtype=torch.long)
    )
    for name in (
        "waveform",
        "voice_waveform",
        "voice_path",
        "user_codes",
        "voice_codes",
    ):
        setattr(state, name, None)
    state.text_prompt_ids = []
    payload.data = state.to_dict()
    return payload


def lm_stream_output_builder(
    request_id: str, data: SGLangARRequestData, req_output
) -> list[OutgoingMessage]:
    pending = data.talker_model_inputs.get("pending_frames")
    if not pending:
        return []
    else:
        pass
    frames = torch.stack(pending).cpu()
    pending.clear()
    return [
        OutgoingMessage(
            request_id=request_id,
            type="stream",
            data=frames,
            target=CODE2WAV_STAGE,
            metadata={
                "modality": "audio_codes",
                "num_samples": data.talker_model_inputs["num_samples"],
            },
        )
    ]


__all__ = [
    "RequestSampling",
    "apply_lm_result",
    "build_lm_request",
    "lm_sampling_params",
    "lm_stream_output_builder",
    "new_model_inputs",
    "packaged_voice_for",
    "prompt_from_state",
    "prompt_input_ids",
    "resolve_sampling",
    "stage_request_params",
    "timeline_from_state",
]
