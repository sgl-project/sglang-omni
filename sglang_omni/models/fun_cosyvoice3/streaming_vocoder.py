# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Fun-CosyVoice3."""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping, NamedTuple

import torch

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.stages import FlowBatchInput
from sglang_omni.models.fun_cosyvoice3.streaming import (
    LEFTOVER_FLOW_STREAMING,
    PRE_LOOKAHEAD_LEN,
    SAMPLE_RATE,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    TOKEN_MEL_RATIO,
    as_flow_embedding,
    as_flow_prompt_feat,
    as_flow_prompt_token,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

FIRST_HOP = "first"
FOLLOW_UP_HOP = "follow_up"
FINAL = "final"


@dataclass
class _CosyVoice3StreamState:
    tokens: list[int] = field(default_factory=list)
    token_offset: int = 0
    hop_len: int = TOKEN_HOP_LEN
    prompt_token: torch.Tensor | None = None
    prompt_feat: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    hift_mel: torch.Tensor | None = None
    speech_offset: int = 0
    done: bool = False
    ready_since: float | None = None
    first_emit_at: float | None = None


class _Candidate(NamedTuple):
    request_id: str
    state: _CosyVoice3StreamState
    kind: str


def _window_end(state: _CosyVoice3StreamState) -> int:
    return state.token_offset + state.hop_len + PRE_LOOKAHEAD_LEN


def _is_hop_ready(state: _CosyVoice3StreamState) -> bool:
    return state.prompt_token is not None and len(state.tokens) >= _window_end(state)


def _step_kind(state: _CosyVoice3StreamState) -> str | None:
    if _is_hop_ready(state):
        return FIRST_HOP if state.token_offset == 0 else FOLLOW_UP_HOP
    if state.done:
        return FINAL
    return None


def _step_key(state: _CosyVoice3StreamState) -> tuple[int, int]:
    return state.token_offset, state.hop_len


def _slack_s(state: _CosyVoice3StreamState, *, now: float, sample_rate: int) -> float:
    # note(ratish): a stream that has emitted nothing has nothing to play, so
    # it is as urgent as a stream whose buffer just ran out.
    if state.first_emit_at is None:
        return 0.0
    return state.speech_offset / sample_rate - (now - state.first_emit_at)


class FunCosyVoice3StreamingVocoderScheduler(
    StreamingVocoderBase[_CosyVoice3StreamState, str]
):
    """Decode CosyVoice3 speech tokens incrementally through Flow + HiFT.

    Messages only change per-request state. The serving loop runs one step
    once the inbox is drained: the runnable stream with the least playback
    slack goes first and every runnable stream with the same token window
    joins it, so equal-shape hops share one causal Flow call.
    """

    _can_batch_stream_chunks = True

    def __init__(
        self,
        vocoder: Any,
        *,
        max_batch_size: int = 8,
        max_batch_wait_ms: int = 2,
        sample_rate: int = SAMPLE_RATE,
        request_cost_fn: Callable[[Any], int] | None = None,
        max_batch_cost: int | None = None,
        token_hop_len: int = TOKEN_HOP_LEN,
        token_max_hop_len: int = TOKEN_MAX_HOP_LEN,
        disable_hop_growth: bool = False,
    ) -> None:
        hop = int(token_hop_len)
        max_hop = int(token_max_hop_len)
        if hop <= 0:
            raise ValueError(f"token_hop_len must be positive, got {token_hop_len}")
        if max_hop < hop:
            raise ValueError(
                f"token_max_hop_len ({token_max_hop_len}) must be >= "
                f"token_hop_len ({token_hop_len})"
            )
        self._token_hop_len = hop
        self._token_max_hop_len = max_hop
        self._disable_hop_growth = bool(disable_hop_growth)
        self._vocoder = vocoder
        self._clock: Callable[[], float] = time.monotonic
        super().__init__(
            self._vocode_payload,
            batch_compute_fn=self._vocode_payloads,
            sample_rate=int(sample_rate),
            stream_source_hint="Fun-CosyVoice3",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
        )

    async def _vocode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self._vocoder.decode_payloads([payload])
        return results[0]

    async def _vocode_payloads(
        self, payloads: list[StagePayload]
    ) -> list[StagePayload]:
        return await self._vocoder.decode_payloads(payloads)

    def create_stream_state(self, request_id: str) -> _CosyVoice3StreamState:
        del request_id
        return _CosyVoice3StreamState(hop_len=self._token_hop_len)

    def _advance_hop_len(self, state: _CosyVoice3StreamState) -> None:
        state.hop_len = next_stream_hop_len(
            state.hop_len,
            max_hop_len=self._token_max_hop_len,
            disable_growth=self._disable_hop_growth,
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        source: StagePayload | Mapping[str, Any],
        *,
        origin: str,
    ) -> None:
        if origin == "payload":
            payload = source
            if not isinstance(payload, StagePayload):
                raise TypeError(
                    f"Fun-CosyVoice3 streaming payload for {request_id!r} must "
                    f"be a StagePayload, got {type(payload).__name__}"
                )
            pipeline_state = FunCosyVoice3State.from_dict(payload.data)
            self._latch_prompts(
                request_id,
                state,
                prompt_token=pipeline_state.flow_prompt_speech_token,
                prompt_feat=pipeline_state.flow_prompt_speech_feat,
                embedding=pipeline_state.flow_embedding,
            )
            return
        metadata: Mapping[str, Any] = source
        if any(
            key in metadata
            for key in (
                "flow_prompt_speech_token",
                "flow_prompt_speech_feat",
                "flow_embedding",
            )
        ):
            self._latch_prompts(
                request_id,
                state,
                prompt_token=metadata.get("flow_prompt_speech_token"),
                prompt_feat=metadata.get("flow_prompt_speech_feat"),
                embedding=metadata.get("flow_embedding"),
            )

    def _latch_prompts(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        *,
        prompt_token: Any,
        prompt_feat: Any,
        embedding: Any,
    ) -> None:
        token = as_flow_prompt_token(prompt_token)
        feat = as_flow_prompt_feat(prompt_feat)
        spk = as_flow_embedding(embedding)
        # note (guozhihao-224): pad prompt to a hop multiple here so the
        # first generated hop stays hop+lookahead instead of waiting for
        # prompt_pad extra AR tokens.
        token, feat = pad_flow_prompt_to_hop(token, feat, hop_len=self._token_hop_len)
        if state.prompt_token is not None:
            # note (guozhihao-224): latch is shape-stable; payload and first
            # chunk metadata must carry the same prompt tensors.
            if (
                tuple(token.shape) != tuple(state.prompt_token.shape)
                or tuple(feat.shape) != tuple(state.prompt_feat.shape)
                or tuple(spk.shape) != tuple(state.embedding.shape)
            ):
                raise ValueError(
                    f"Fun-CosyVoice3 stream prompt tensors changed for {request_id!r}"
                )
            return
        state.prompt_token = token
        state.prompt_feat = feat
        state.embedding = spk

    def validate_chunk(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        del request_id, state
        chunk = codes.to(dtype=torch.long)
        if chunk.ndim == 2 and chunk.shape[-1] == 1:
            chunk = chunk.reshape(-1)
        if chunk.ndim != 1:
            raise ValueError(
                f"Fun-CosyVoice3 stream chunk must be 1-D speech tokens, "
                f"got {tuple(chunk.shape)}"
            )
        return chunk.contiguous()

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        super().on_streaming_new_request(request_id, payload)
        state = self._stream_states.get(request_id)
        if state is not None:
            self._mark_ready(state)

    def on_stream_chunk_batch(self, items: list[tuple[str, StreamItem]]) -> None:
        failed: list[str] = []
        with self._state_lock:
            for request_id, item in items:
                if self._is_aborted(request_id):
                    continue
                try:
                    self._ingest_stream_item(request_id, item)
                except Exception as exc:
                    self._emit_error(request_id, exc)
                    self._abort_state(request_id)
                    failed.append(request_id)
        for request_id in failed:
            self._cleanup_aborted_request(request_id)

    def ingest(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.tokens.extend(int(token) for token in codes.tolist())
        self._mark_ready(state)

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        state = self._get_or_create_stream_state(request_id)
        if state is None:
            return []
        state.done = True
        self._mark_ready(state)
        return None

    def _mark_ready(self, state: _CosyVoice3StreamState) -> None:
        if state.ready_since is None and _step_kind(state) is not None:
            state.ready_since = self._clock()

    def _ranked_candidates(self) -> list[_Candidate]:
        now = self._clock()
        candidates: list[_Candidate] = []
        for request_id, state in self._stream_state_items():
            kind = _step_kind(state)
            if kind is not None and not self._is_aborted(request_id):
                candidates.append(_Candidate(request_id, state, kind))
        candidates.sort(
            key=lambda candidate: (
                _slack_s(candidate.state, now=now, sample_rate=self._sample_rate),
                candidate.state.ready_since,
                candidate.request_id,
            )
        )
        return candidates

    def _has_ready_work(self) -> bool:
        with self._state_lock:
            return bool(self._ranked_candidates())

    def select_step_participants(
        self,
    ) -> list[tuple[str, _CosyVoice3StreamState]]:
        ranked = self._ranked_candidates()
        if not ranked:
            return []
        head = ranked[0]
        if head.kind == FINAL:
            return [(head.request_id, head.state)]
        key = _step_key(head.state)
        peers = [
            (candidate.request_id, candidate.state)
            for candidate in ranked
            if candidate.kind != FINAL and _step_key(candidate.state) == key
        ]
        return peers[: self._max_batch_size]

    def build_step_plan(
        self, participants: list[tuple[str, _CosyVoice3StreamState]]
    ) -> str:
        return _step_kind(participants[0][1])

    def run_step(
        self,
        participants: list[tuple[str, _CosyVoice3StreamState]],
        plan: str,
    ) -> dict[str, torch.Tensor]:
        if plan == FINAL:
            request_id, _ = participants[0]
            self._complete_stream_request(request_id, self._finish_stream(request_id))
            return {}
        # note (guozhihao-224): B>1 uses packed inference_causal; B=1 keeps
        # native CosyVoice Flow.inference. Packed singleton-vs-row tests
        # cover the batch adapter; native hops stay on the official signature.
        if len(participants) > 1:
            decoded = self._run_causal_hop_batch(participants)
        else:
            request_id, state = participants[0]
            delta = self._run_one_causal_hop(state)
            decoded = {request_id: delta} if delta.numel() > 0 else {}
        now = self._clock()
        for request_id, state in participants:
            if request_id in decoded and state.first_emit_at is None:
                state.first_emit_at = now
            state.ready_since = None
            self._mark_ready(state)
        return decoded

    def _run_causal_hop_batch(
        self, participants: list[tuple[str, _CosyVoice3StreamState]]
    ) -> dict[str, torch.Tensor]:
        head = participants[0][1]
        hop = head.hop_len
        token_offset = head.token_offset
        token_end = _window_end(head)
        items = [
            FlowBatchInput(
                token=torch.tensor(
                    state.tokens[:token_end], dtype=torch.int32
                ).unsqueeze(0),
                prompt_token=state.prompt_token,
                prompt_feat=state.prompt_feat,
                embedding=state.embedding,
            )
            for _, state in participants
        ]
        logger.info(
            f"Fun-CosyVoice3 causal Flow batch size={len(items)} hop={hop} "
            f"token_offset={token_offset}"
        )
        mels = self._vocoder.first_hop_batch(items)
        offset_frames = token_offset * TOKEN_MEL_RATIO
        decoded: dict[str, torch.Tensor] = {}
        for (request_id, state), mel in zip(participants, mels, strict=True):
            delta, hift_mel, speech_offset = self._vocoder.hift_delta(
                mel[:, :, offset_frames:],
                hift_mel=state.hift_mel,
                speech_offset=state.speech_offset,
                finalize=False,
            )
            state.token_offset += hop
            self._advance_hop_len(state)
            state.hift_mel = hift_mel
            state.speech_offset = speech_offset
            if delta.numel() > 0:
                decoded[request_id] = delta
        return decoded

    def _run_one_causal_hop(self, state: _CosyVoice3StreamState) -> torch.Tensor:
        delta = self._run_flow_hift(
            state, token_end=_window_end(state), streaming=True, finalize=False
        )
        state.token_offset += state.hop_len
        self._advance_hop_len(state)
        return delta

    def decode_delta(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        del request_id
        if not is_final:
            if not _is_hop_ready(state):
                return None
            delta = self._run_one_causal_hop(state)
            return delta if delta.numel() > 0 else None
        pieces: list[torch.Tensor] = []
        while _is_hop_ready(state):
            pieces.append(self._run_one_causal_hop(state))
        if state.tokens:
            # note (guozhihao-224): leftover keeps finalize=True so HiFT
            # flushes and pre_lookahead consumes the tail. DiT stays
            # bidirectional; leftover streaming=True did not win the A/B.
            pieces.append(
                self._run_flow_hift(
                    state,
                    token_end=len(state.tokens),
                    streaming=LEFTOVER_FLOW_STREAMING,
                    finalize=True,
                )
            )
        pieces = [piece for piece in pieces if piece.numel() > 0]
        if not pieces:
            return None
        return torch.cat(pieces, dim=-1)

    def _run_flow_hift(
        self,
        state: _CosyVoice3StreamState,
        *,
        token_end: int,
        streaming: bool,
        finalize: bool,
    ) -> torch.Tensor:
        token = torch.tensor(state.tokens[:token_end], dtype=torch.int32).unsqueeze(0)
        wav, hift_mel, speech_offset = self._vocoder.token2wav_chunk(
            token=token,
            prompt_token=state.prompt_token,
            prompt_feat=state.prompt_feat,
            embedding=state.embedding,
            token_offset=state.token_offset,
            streaming=streaming,
            finalize=finalize,
            hift_mel=state.hift_mel,
            speech_offset=state.speech_offset,
        )
        state.hift_mel = hift_mel
        state.speech_offset = speech_offset
        return wav

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: _CosyVoice3StreamState,
    ) -> torch.Tensor | None:
        del request_id, state
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        if pipeline_state.audio_codes is None:
            codes = torch.zeros(0, dtype=torch.long)
        else:
            codes = torch.as_tensor(
                pipeline_state.audio_codes, dtype=torch.long
            ).reshape(-1)
        if codes.numel() == 0:
            raise RuntimeError(
                "Fun-CosyVoice3 generation produced no usable speech tokens"
            )
        prompt_token = as_flow_prompt_token(pipeline_state.flow_prompt_speech_token)
        prompt_feat = as_flow_prompt_feat(pipeline_state.flow_prompt_speech_feat)
        embedding = as_flow_embedding(pipeline_state.flow_embedding)
        return self._vocoder.token2wav(
            token=codes.unsqueeze(0),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: _CosyVoice3StreamState,
    ) -> dict[str, Any]:
        del request_id, state
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self._sample_rate,
        }
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        usage = build_usage(pipeline_state)
        if usage is not None:
            final_data["usage"] = usage
        return final_data

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        del request_id
        return audio_waveform_payload(
            waveform,
            sample_rate=self._sample_rate,
            modality="audio",
            source_hint="Fun-CosyVoice3",
        )

    def release_stream_resources(
        self, request_id: str, state: _CosyVoice3StreamState
    ) -> None:
        del request_id
        state.tokens.clear()
        state.hift_mel = None
        state.prompt_token = None
        state.prompt_feat = None
        state.embedding = None
