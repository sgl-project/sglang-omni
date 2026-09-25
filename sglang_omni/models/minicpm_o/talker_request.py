# SPDX-License-Identifier: Apache-2.0
"""Speech-condition slicing and scheduler adapters for MiniCPM-o."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import resolve_sampling_seed
from sglang_omni.models.minicpm_o.routing import TALKER_STAGE, payload_with_state
from sglang_omni.proto.request import StagePayload

if TYPE_CHECKING:
    from sglang_omni.models.minicpm_o.components.sglang_talker import (
        MiniCPMOTalkerForCausalLM,
    )
    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
else:
    pass


def build_talker_request(
    state: MiniCPMOPipelineState,
    *,
    tts_bos_token_id: int,
    tts_eos_token_id: int,
) -> dict[str, torch.Tensor]:
    """Pair speech-span token ids with captured hidden states at the same positions."""
    prompt = state.prompt or {}
    input_ids = prompt.get("input_ids")
    prompt_ids = (
        input_ids.reshape(-1).tolist()
        if isinstance(input_ids, torch.Tensor)
        else list(input_ids or [])
    )
    thinker_out = state.thinker_out if isinstance(state.thinker_out, dict) else {}
    output_ids = [int(t) for t in (thinker_out.get("output_ids") or [])]
    extra = thinker_out.get("extra_model_outputs") or {}
    hidden_seq = extra.get("hidden_states_seq") or []

    full_sequence = [int(t) for t in prompt_ids] + output_ids
    prompt_len = len(prompt_ids)

    tts_bos_indices = [i for i, t in enumerate(full_sequence) if t == tts_bos_token_id]
    tts_eos_indices = [i for i, t in enumerate(full_sequence) if t == tts_eos_token_id]
    if not tts_bos_indices:
        empty = torch.empty(0, dtype=torch.long)
        return {"tts_token_ids": empty, "tts_hidden": empty}
    else:
        pass
    start = tts_bos_indices[-1] + 1
    # Only an end marker inside the current segment may close the span: a
    # history turn's <|tts_eos|> sits before the last <|tts_bos|>, and slicing
    # at it would drop the truncated current speech (end < start → empty).
    segment_eos = [i for i in tts_eos_indices if i >= start]
    end = segment_eos[0] if segment_eos else len(full_sequence)

    # note (MayDomine): the first captured hidden state is the last prompt position.
    hidden_base = prompt_len - 1
    if start < hidden_base:
        raise ValueError(
            f"tts span start {start} precedes first captured hidden position "
            f"{hidden_base}; prompt-side spans are not supported"
        )
    else:
        pass
    end = min(end, hidden_base + len(hidden_seq))
    if end <= start:
        empty = torch.empty(0, dtype=torch.long)
        return {"tts_token_ids": empty, "tts_hidden": empty}
    else:
        pass

    tokens = torch.tensor(full_sequence[start:end], dtype=torch.long)
    hidden = torch.stack([hidden_seq[i - hidden_base] for i in range(start, end)])
    return {"tts_token_ids": tokens, "tts_hidden": hidden}


@dataclass(kw_only=True)
class CodecTokenizer:
    """Supply the codec EOS id required by SGLang's minimum-length penalizer."""

    eos_token_id: int
    bos_token_id: int | None = None
    additional_stop_token_ids: set[int] | None = None


def build_sglang_talker_request(
    state: MiniCPMOPipelineState,
    *,
    model: MiniCPMOTalkerForCausalLM,
    codec_vocab_size: int,
    codec_eos_id: int,
    tts_bos_token_id: int,
    tts_eos_token_id: int,
    params: dict[str, Any],
    request_id: str | None = None,
) -> SGLangARRequestData:
    """Build a codec request carrying speech-condition embeddings."""
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams

    from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

    span = build_talker_request(
        state,
        tts_bos_token_id=tts_bos_token_id,
        tts_eos_token_id=tts_eos_token_id,
    )
    tts_token_ids = span["tts_token_ids"]
    empty_span = tts_token_ids.numel() == 0
    condition = model.build_condition_embeddings(tts_token_ids, span["tts_hidden"])
    if empty_span:
        # note (MayDomine): built requests cannot bypass the engine; discard this step.
        sampling_params = SamplingParams(max_new_tokens=1, temperature=0.0)
        rep_penalty = 1.0
    else:
        # note (MayDomine): the runner applies a windowed penalty, not SGLang's penalty.
        sampling_params = SamplingParams(
            max_new_tokens=int(params.get("talker_max_new_tokens", 2048)),
            min_new_tokens=int(params.get("talker_min_new_tokens", 50)),
            temperature=float(params.get("talker_temperature", 0.8)),
            top_p=float(params.get("talker_top_p", 0.85)),
            top_k=int(params.get("talker_top_k", 25)),
            repetition_penalty=1.0,
            stop_token_ids=[int(codec_eos_id)],
            sampling_seed=resolve_sampling_seed(params),
        )
        rep_penalty = float(params.get("talker_repetition_penalty", 1.05))
    shim = CodecTokenizer(eos_token_id=int(codec_eos_id))
    sampling_params.normalize(shim)
    sampling_params.verify(codec_vocab_size)

    prompt_len = int(condition.shape[0])
    req = Req(
        rid=request_id or "talker-req-0",
        origin_input_text="",
        # note (MayDomine): prefill embeds replace these position-tracking ids.
        origin_input_ids=[int(codec_eos_id)] * prompt_len,
        sampling_params=sampling_params,
        eos_token_ids={int(codec_eos_id)},
        vocab_size=codec_vocab_size,
    )
    req.tokenizer = shim
    req._input_embeds_are_projected = True  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
    req._codec_suppress_tokens = None  # noqa: leading-underscore  # upstream spelling, or the public name is already taken

    data = SGLangARRequestData(
        prefill_input_embeds=condition,
        input_embeds_are_projected=True,
        talker_model_inputs={"rep_penalty": rep_penalty},
        max_new_tokens=int(sampling_params.max_new_tokens),
        output_ids=req.output_ids,
        req=req,
    )
    data.talker_model_inputs["empty_span"] = empty_span
    return data


def make_talker_scheduler_adapters(
    *,
    model: MiniCPMOTalkerForCausalLM,
    codec_vocab_size: int,
    codec_eos_id: int,
    tts_bos_token_id: int,
    tts_eos_token_id: int,
) -> tuple[
    Callable[[StagePayload], SGLangARRequestData],
    Callable[[SGLangARRequestData], StagePayload],
]:
    """Build StagePayload <-> scheduler adapters for the sglang talker."""

    def request_builder(payload: StagePayload) -> SGLangARRequestData:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        req_data = build_sglang_talker_request(
            state,
            model=model,
            codec_vocab_size=codec_vocab_size,
            codec_eos_id=codec_eos_id,
            tts_bos_token_id=tts_bos_token_id,
            tts_eos_token_id=tts_eos_token_id,
            params=payload.request.params or {},
            request_id=payload.request_id,
        )
        req_data.stage_payload = payload
        return req_data

    def result_adapter(data: SGLangARRequestData) -> StagePayload:
        payload = data.stage_payload
        state = MiniCPMOPipelineState.from_dict(payload.data)
        if data.talker_model_inputs.get("empty_span"):
            codec = torch.empty(0, dtype=torch.long)
        else:
            output_ids = [int(t) for t in data.output_ids]
            while output_ids and output_ids[-1] == int(codec_eos_id):
                output_ids.pop()
            codec = torch.tensor(output_ids, dtype=torch.long)
        state.engine_outputs[TALKER_STAGE] = {"codec_tokens": codec}
        return payload_with_state(payload, state)

    return request_builder, result_adapter
