# SPDX-License-Identifier: Apache-2.0
"""Map MiniCPM-o duplex units to append-only SGLang session turns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from sglang.srt.managers.io_struct import SessionParams, TokenizedGenerateReqInput
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.models.minicpmo_4_5.state import (
    EmbeddingSpan,
    MiniCPMOSessionState,
    MiniCPMOSpecialTokens,
    MiniCPMOUnitRequestData,
)


@dataclass(frozen=True)
class MiniCPMOUnitBuild:
    internal_request_id: str
    state: MiniCPMOSessionState
    prepared_unit: Any
    forced_listen: bool
    close_speaking_turn: bool
    sampling: dict[str, Any]


def prepare_session_prefix(
    state: MiniCPMOSessionState,
    *,
    tokenizer: Any,
    special_tokens: MiniCPMOSpecialTokens,
    reference_embedding: torch.Tensor | None,
) -> None:
    """Prepare the checkpoint's system framing once per logical session."""

    prefix = tokenizer.encode(
        f"<|im_start|>system\n{state.system_prompt}\n<|audio_start|>",
        add_special_tokens=False,
    )
    suffix = tokenizer.encode(
        "<|audio_end|><|im_end|>",
        add_special_tokens=False,
    )
    input_ids = [int(token_id) for token_id in prefix]
    spans: list[EmbeddingSpan] = []
    if reference_embedding is not None:
        if reference_embedding.ndim != 2:
            raise ValueError("reference embedding must have shape [tokens, hidden]")
        start = len(input_ids)
        input_ids.extend(
            [special_tokens.media_placeholder] * int(reference_embedding.shape[0])
        )
        spans.append(
            EmbeddingSpan(
                start=start,
                end=len(input_ids),
                embedding=reference_embedding,
                modality="audio",
            )
        )
    input_ids.extend(int(token_id) for token_id in suffix)
    state.prefix_input_ids = input_ids
    state.prefix_embedding_spans = spans
    state.prefix_pending = True


def build_unit_request_data(
    build: MiniCPMOUnitBuild,
    *,
    tokenizer: Any,
    vocab_size: int,
    special_tokens: MiniCPMOSpecialTokens,
) -> MiniCPMOUnitRequestData:
    """Build a request that Session.create_req will append to the saved KV."""

    state = build.state
    prepared = build.prepared_unit
    unit_ids = [int(token_id) for token_id in prepared.input_ids]
    unit_spans = [_coerce_span(span) for span in prepared.embedding_spans]

    local_ids: list[int] = []
    local_spans: list[EmbeddingSpan] = []
    if state.prefix_pending:
        local_ids.extend(state.prefix_input_ids)
        local_spans.extend(state.prefix_embedding_spans)
    else:
        # The prior sampled terminator is carried by StreamingSession.  Feeding
        # only </unit> here realizes the Demo's deferred finalize without a
        # separate forward or Python KV cache.
        local_ids.append(special_tokens.unit_end)

    unit_offset = len(local_ids)
    local_ids.extend(unit_ids)
    local_spans.extend(_shift_span(span, unit_offset) for span in unit_spans)
    if build.close_speaking_turn:
        # Interrupt semantics in the unified Demo: ingest the new unit first,
        # then feed turn_eos and force the following decision to listen.
        local_ids.append(special_tokens.turn_eos)

    max_new_tokens = int(build.sampling.get("max_new_speak_tokens_per_chunk", 20))
    sampling_params = SamplingParams(
        max_new_tokens=max_new_tokens,
        temperature=1.0,
        top_p=1.0,
        top_k=-1,
        stop_token_ids=set(special_tokens.chunk_terminators),
        no_stop_trim=True,
        skip_special_tokens=False,
    )
    sampling_params.normalize(tokenizer)
    sampling_params.verify(int(vocab_size))

    tokenized = TokenizedGenerateReqInput(
        rid=build.internal_request_id,
        input_text=None,
        # Keep appends as ordinary lists. StreamingSession owns the shared
        # append-only token storage; feeding an ``array`` here makes later
        # list appends version-dependent in SGLang's session materializer.
        input_ids=list(local_ids),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=sampling_params,
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=True,
        return_hidden_states=True,
        session_params=SessionParams(id=state.session_id),
    )

    data = MiniCPMOUnitRequestData(
        input_ids=torch.tensor(local_ids, dtype=torch.long),
        max_new_tokens=max_new_tokens,
        temperature=float(build.sampling.get("temperature", 0.7)),
        tokenized_session_req=tokenized,
        session_tokenizer=tokenizer,
        session_state=state,
        outer_request_id=state.request_id,
        input_seq=(
            state.inflight_input_seq if state.inflight_input_seq is not None else 0
        ),
        response_epoch=(
            state.inflight_response_epoch
            if state.inflight_response_epoch is not None
            else state.response_epoch
        ),
        local_input_ids=local_ids,
        local_embedding_spans=local_spans,
        forced_listen=build.forced_listen,
        input_mode=str(prepared.mode).lower(),
    )
    data.extra_model_outputs["duplex_sampling"] = dict(build.sampling)

    def setup(req: Any) -> Any:
        old_span_count = len(state.embedding_spans)
        old_prefix_pending = state.prefix_pending
        old_turn_ended = state.current_turn_ended
        absolute_offset = len(req.origin_input_ids) - len(local_ids)
        if absolute_offset < 0:
            raise RuntimeError(
                "SGLang session produced a shorter origin than its append"
            )
        absolute = [_shift_span(span, absolute_offset) for span in local_spans]
        state.embedding_spans.extend(absolute)
        data.absolute_embedding_spans = absolute
        state.prefix_pending = False
        if build.close_speaking_turn:
            state.current_turn_ended = True
        req._codec_suppress_tokens = None

        def rollback() -> None:
            del state.embedding_spans[old_span_count:]
            state.prefix_pending = old_prefix_pending
            state.current_turn_ended = old_turn_ended
            data.absolute_embedding_spans = []

        return rollback

    data.session_req_setup = setup
    return data


def _coerce_span(value: Any) -> EmbeddingSpan:
    if isinstance(value, EmbeddingSpan):
        return value
    embedding = getattr(value, "embedding", None)
    if embedding is None:
        embedding = getattr(value, "embeddings", None)
    return EmbeddingSpan(
        start=int(value.start),
        end=int(value.end),
        embedding=embedding,
        modality=str(value.modality),
    )


def _shift_span(span: EmbeddingSpan, amount: int) -> EmbeddingSpan:
    return EmbeddingSpan(
        start=span.start + amount,
        end=span.end + amount,
        embedding=span.embedding,
        modality=span.modality,
    )


__all__ = [
    "MiniCPMOUnitBuild",
    "build_unit_request_data",
    "prepare_session_prefix",
]
