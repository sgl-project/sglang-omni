# SPDX-License-Identifier: Apache-2.0
"""Streaming vocoder scheduler for Fun-CosyVoice3."""

from __future__ import annotations

import logging
import queue as _queue_mod
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Mapping

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
    stream_hop_len,
    tokens_needed_for_causal_chunk,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import IncomingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)


@dataclass
class _CosyVoice3StreamState:
    tokens: list[int] = field(default_factory=list)
    token_offset: int = 0
    hop_len: int = TOKEN_HOP_LEN
    prompt_pad: int = 0
    prompt_token: torch.Tensor | None = None
    prompt_feat: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    prompts_latched: bool = False
    hift_mel: torch.Tensor | None = None
    speech_offset: int = 0


@dataclass(frozen=True)
class _CosyVoice3FirstHopPlan:
    hop: int
    token_end: int
    token_offset: int
    batched: bool


class FunCosyVoice3StreamingVocoderScheduler(
    StreamingVocoderBase[_CosyVoice3StreamState, _CosyVoice3FirstHopPlan]
):
    """Decode CosyVoice3 speech tokens incrementally through Flow + HiFT."""

    _can_batch_stream_chunks = True
    _stream_chunk_batch_distinct_requests = True
    # note (guozhihao-224): follow-up hops are a separate knife from
    # first-hop coalescing. Keep this on once equal-shape causal batch
    # is the default; A/B turns it off to isolate ITL/C50.
    _can_batch_follow_up_hops = True
    # note (guozhihao-224): c=1 has no joinable peer so this is a no-op.
    # c=16 holds a singleton first hop or follow-up hop up to this
    # window while equal-shape peers arrive, otherwise the vocoder
    # decodes B=1 forever.
    _first_hop_peer_wait_ms = 30

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

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        super().on_streaming_new_request(request_id, payload)
        # note (guozhihao-224): payload can arrive after buffered chunks.
        # Serial path pumps immediately. Coalescing waits until
        # _handle_new_request_batch has latched every queued streaming
        # payload so equal-shape first hops share one causal Flow call.
        if request_id in self._pending_done:
            return
        if self._can_batch_stream_chunks:
            return
        with self._state_lock:
            failed = self._pump_streams()
        for failed_id in failed:
            self._cleanup_aborted_request(failed_id)

    def _collect_new_request_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        if not self._can_batch_stream_chunks:
            return super()._collect_new_request_batch(first_msg)
        try:
            first_is_streaming = self.is_streaming_payload(first_msg.data)
        except Exception:
            return super()._collect_new_request_batch(first_msg)
        if not first_is_streaming:
            return super()._collect_new_request_batch(first_msg)
        batch = [first_msg]
        seen = {first_msg.request_id}
        cap = max(int(self._max_batch_size), 1)
        while len(batch) < cap:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break
            if self._is_aborted(msg.request_id):
                continue
            if msg.type != "new_request":
                self._pending_messages.appendleft(msg)
                break
            try:
                is_streaming = self.is_streaming_payload(msg.data)
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                continue
            if not is_streaming or msg.request_id in seen:
                self._pending_messages.appendleft(msg)
                break
            seen.add(msg.request_id)
            batch.append(msg)
        return batch

    def _handle_new_request_batch(
        self,
        batch: list[IncomingMessage],
        loop: Any | None = None,
    ) -> None:
        super()._handle_new_request_batch(batch, loop)
        if not self._can_batch_stream_chunks:
            return
        has_streaming = False
        for msg in batch:
            if msg.type != "new_request" or self._is_aborted(msg.request_id):
                continue
            try:
                has_streaming = self.is_streaming_payload(msg.data)
            except Exception:
                continue
            if has_streaming:
                break
        if not has_streaming:
            return
        with self._state_lock:
            failed = self._pump_streams()
        for failed_id in failed:
            self._cleanup_aborted_request(failed_id)

    def _collect_stream_chunk_batch(
        self, first_msg: IncomingMessage
    ) -> list[IncomingMessage]:
        if not self._can_batch_stream_chunks:
            return super()._collect_stream_chunk_batch(first_msg)
        batch = [first_msg]
        seen = {first_msg.request_id}
        deferred: list[IncomingMessage] = []
        leftover: IncomingMessage | None = None
        cap = self._stream_chunk_batch_max or max(self._max_batch_size, 1)
        while len(batch) < cap:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                break
            if msg.type != "stream_chunk":
                leftover = msg
                break
            if self._is_aborted(msg.request_id):
                continue
            if msg.request_id in seen:
                # note (guozhihao-224): keep scanning for other requests'
                # first hops instead of stopping behind this request's
                # follow-up chunk.
                deferred.append(msg)
                continue
            batch.append(msg)
            seen.add(msg.request_id)
        if leftover is not None:
            self._pending_messages.appendleft(leftover)
        for msg in reversed(deferred):
            self._pending_messages.appendleft(msg)
        return batch

    def _first_hop_group_size(self) -> int:
        groups: dict[int, int] = {}
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or not self._ready_for_causal_chunk(state):
                continue
            if state.token_offset != 0:
                continue
            key = self._first_hop_key(state)
            groups[key] = groups.get(key, 0) + 1
        return max(groups.values(), default=0)

    def _has_joinable_first_hop_peer(self) -> bool:
        ready_keys: set[int] = set()
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or not self._ready_for_causal_chunk(state):
                continue
            if state.token_offset == 0:
                ready_keys.add(self._first_hop_key(state))
        if len(ready_keys) != 1:
            return False
        ready_key = next(iter(ready_keys))
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or state.token_offset != 0:
                continue
            if self._ready_for_causal_chunk(state):
                continue
            if state.prompts_latched and self._first_hop_key(state) != ready_key:
                continue
            return True
        return False

    def _ingest_peer_message(self, msg: IncomingMessage) -> bool:
        if self._is_aborted(msg.request_id):
            return True
        if msg.type == "stream_chunk":
            try:
                item = self._validate_stream_chunk_item(msg.request_id, msg.data)
                self._ingest_stream_item(msg.request_id, item)
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
            return True
        if msg.type == "new_request":
            try:
                if self.is_streaming_payload(msg.data):
                    self._handle_streaming_new_request(msg.request_id, msg.data)
                    return True
            except Exception as exc:
                self._emit_error(msg.request_id, exc)
                self.abort(msg.request_id)
                return True
            self._pending_messages.appendleft(msg)
            return False
        self._pending_messages.appendleft(msg)
        return False

    def _wait_for_first_hop_peers(self) -> None:
        if self._first_hop_group_size() >= 2:
            return
        if not self._has_joinable_first_hop_peer():
            return
        wait_s = max(float(self._first_hop_peer_wait_ms), 0.0) / 1000.0
        deadline = time.monotonic() + wait_s
        while True:
            if self._first_hop_group_size() >= 2:
                return
            if not self._has_joinable_first_hop_peer():
                return
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    msg = self.inbox.get_nowait()
                else:
                    msg = self.inbox.get(timeout=remaining)
            except _queue_mod.Empty:
                if remaining <= 0:
                    return
                continue
            if not self._ingest_peer_message(msg):
                return

    def _follow_up_key(self, state: _CosyVoice3StreamState) -> tuple[int, int]:
        # note (guozhihao-224): drop prompt_len so SeedTTS mixed prompts
        # with the same hop/offset share one causal Flow call.
        return int(state.hop_len), int(state.token_offset)

    def _follow_up_group_size(self) -> int:
        groups: dict[tuple[int, int], int] = {}
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or not self._ready_for_causal_chunk(state):
                continue
            if state.token_offset == 0:
                continue
            key = self._follow_up_key(state)
            groups[key] = groups.get(key, 0) + 1
        return max(groups.values(), default=0)

    def _has_joinable_follow_up_peer(self) -> bool:
        ready_keys: set[tuple[int, int]] = set()
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or not self._ready_for_causal_chunk(state):
                continue
            if state.token_offset == 0:
                continue
            ready_keys.add(self._follow_up_key(state))
        if len(ready_keys) != 1:
            return False
        ready_key = next(iter(ready_keys))
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or state.token_offset == 0:
                continue
            if self._ready_for_causal_chunk(state):
                continue
            if not state.prompts_latched:
                continue
            if self._follow_up_key(state) != ready_key:
                continue
            return True
        return False

    def _wait_for_follow_up_peers(self) -> None:
        # note (guozhihao-224): same 30ms window as first hops; without it
        # equal follow-ups arrive staggered and stay B=1 native.
        if self._follow_up_group_size() >= 2:
            return
        if not self._has_joinable_follow_up_peer():
            return
        wait_s = max(float(self._first_hop_peer_wait_ms), 0.0) / 1000.0
        deadline = time.monotonic() + wait_s
        while True:
            if self._follow_up_group_size() >= 2:
                return
            if not self._has_joinable_follow_up_peer():
                return
            remaining = deadline - time.monotonic()
            try:
                if remaining <= 0:
                    msg = self.inbox.get_nowait()
                else:
                    msg = self.inbox.get(timeout=remaining)
            except _queue_mod.Empty:
                if remaining <= 0:
                    return
                continue
            if not self._ingest_peer_message(msg):
                return

    def _ingest_ready_inbox(self) -> None:
        """Pull already-queued peers between hops without blocking.

        The base pump drains every ready hop before returning to the
        serving loop. CosyVoice first hops must be able to join after a
        follow-up step, otherwise a backlogged request monopolizes the GPU.
        """
        while True:
            try:
                msg = self.inbox.get_nowait()
            except _queue_mod.Empty:
                return
            if not self._ingest_peer_message(msg):
                return

    def _pump_one_step(self) -> list[str] | None:
        participants = self.select_step_participants()
        if not participants:
            return []
        plan = self.build_step_plan(participants)
        try:
            decoded = self.run_step(participants, plan)
        except Exception as exc:
            return list(self.on_step_failure(participants, exc))
        for request_id, _ in participants:
            waveform = decoded.get(request_id)
            if waveform is not None and not self._is_aborted(request_id):
                self._mark_stream_emitted(request_id)
                self.outbox.put(self._stream_chunk_message(request_id, waveform))
        return None

    def _pump_streams(self) -> list[str]:
        # note (guozhihao-224): one hop per step. Keep looping while work
        # remains so a lone request is not stalled until the next inbox
        # message, but ingest between steps so a new first hop can preempt
        # a follow-up backlog. 30ms peer wait stays at pump start and when
        # a singleton first hop appears after ingest.
        first = True
        while True:
            if self._can_batch_stream_chunks:
                if first:
                    self._wait_for_first_hop_peers()
                    if (
                        self._first_hop_group_size() == 0
                        and self._can_batch_follow_up_hops
                    ):
                        self._wait_for_follow_up_peers()
                else:
                    self._ingest_ready_inbox()
                    if self._first_hop_group_size() == 1:
                        self._wait_for_first_hop_peers()
            first = False
            failed = self._pump_one_step()
            if failed is not None:
                return failed

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
        if state.prompts_latched:
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
        state.prompt_pad = 0
        state.prompts_latched = True

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

    def ingest(
        self,
        request_id: str,
        state: _CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> None:
        del request_id
        state.tokens.extend(int(token) for token in codes.tolist())

    def should_decode(self, state: _CosyVoice3StreamState, *, is_final: bool) -> bool:
        del is_final
        return self._ready_for_causal_chunk(state)

    def _ready_for_causal_chunk(self, state: _CosyVoice3StreamState) -> bool:
        if not state.prompts_latched:
            return False
        needed = tokens_needed_for_causal_chunk(
            state.token_offset,
            hop_len=state.hop_len,
            prompt_pad=state.prompt_pad,
        )
        return len(state.tokens) >= needed

    def _first_hop_key(self, state: _CosyVoice3StreamState) -> int:
        hop = stream_hop_len(0, hop_len=state.hop_len, prompt_pad=state.prompt_pad)
        return hop + PRE_LOOKAHEAD_LEN

    def select_step_participants(
        self,
    ) -> list[tuple[str, _CosyVoice3StreamState]]:
        first_hops: list[tuple[str, _CosyVoice3StreamState]] = []
        follow_ups: list[tuple[str, _CosyVoice3StreamState]] = []
        for request_id, state in self._stream_state_items():
            if self._is_aborted(request_id) or not self._ready_for_causal_chunk(state):
                continue
            if state.token_offset == 0:
                first_hops.append((request_id, state))
            else:
                follow_ups.append((request_id, state))
        if first_hops:
            if not self._can_batch_stream_chunks:
                return first_hops[:1]
            groups: dict[int, list[tuple[str, _CosyVoice3StreamState]]] = {}
            for entry in first_hops:
                groups.setdefault(self._first_hop_key(entry[1]), []).append(entry)
            best = max(groups.values(), key=len)
            return best[: self._max_batch_size]
        if follow_ups:
            if not self._can_batch_stream_chunks or not self._can_batch_follow_up_hops:
                return follow_ups[:1]
            follow_groups: dict[
                tuple[int, int], list[tuple[str, _CosyVoice3StreamState]]
            ] = {}
            for entry in follow_ups:
                follow_groups.setdefault(self._follow_up_key(entry[1]), []).append(
                    entry
                )
            best_follow = max(follow_groups.values(), key=len)
            return best_follow[: self._max_batch_size]
        return []

    def build_step_plan(
        self, participants: list[tuple[str, _CosyVoice3StreamState]]
    ) -> _CosyVoice3FirstHopPlan:
        state = participants[0][1]
        hop = stream_hop_len(
            state.token_offset,
            hop_len=state.hop_len,
            prompt_pad=state.prompt_pad,
        )
        return _CosyVoice3FirstHopPlan(
            hop=hop,
            token_end=state.token_offset + hop + PRE_LOOKAHEAD_LEN,
            token_offset=state.token_offset,
            batched=len(participants) > 1,
        )

    def run_step(
        self,
        participants: list[tuple[str, _CosyVoice3StreamState]],
        plan: _CosyVoice3FirstHopPlan,
    ) -> dict[str, torch.Tensor]:
        # note (guozhihao-224): B>1 uses packed inference_causal; B=1 keeps
        # native CosyVoice Flow.inference. Packed singleton-vs-row tests
        # cover the batch adapter; native hops stay on the official signature.
        if plan.batched:
            return self._run_causal_hop_batch(participants, plan)
        request_id, state = participants[0]
        waveform = self.decode_delta(request_id, state, is_final=False)
        if waveform is None:
            return {}
        return {request_id: waveform}

    def _run_causal_hop_batch(
        self,
        participants: list[tuple[str, _CosyVoice3StreamState]],
        plan: _CosyVoice3FirstHopPlan,
    ) -> dict[str, torch.Tensor]:
        items: list[FlowBatchInput] = []
        for _, state in participants:
            if state.prompt_token is None or state.prompt_feat is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 streaming vocoder decoded before prompt "
                    "conditioning was latched"
                )
            if state.embedding is None:
                raise RuntimeError(
                    "Fun-CosyVoice3 streaming vocoder decoded before speaker "
                    "embedding was latched"
                )
            generated = state.tokens[: plan.token_end]
            items.append(
                FlowBatchInput(
                    token=torch.tensor(generated, dtype=torch.int32).unsqueeze(0),
                    prompt_token=state.prompt_token,
                    prompt_feat=state.prompt_feat,
                    embedding=state.embedding,
                )
            )
        if plan.token_offset == 0:
            logger.info(
                "Fun-CosyVoice3 first-hop Flow batch size=%d hop=%d",
                len(items),
                plan.hop,
            )
        else:
            logger.info(
                "Fun-CosyVoice3 follow-up Flow batch size=%d hop=%d token_offset=%d",
                len(items),
                plan.hop,
                plan.token_offset,
            )
        mels = self._vocoder.first_hop_batch(items)
        offset_frames = int(plan.token_offset) * TOKEN_MEL_RATIO
        decoded: dict[str, torch.Tensor] = {}
        for (request_id, state), mel in zip(participants, mels, strict=True):
            if mel.shape[-1] < offset_frames:
                raise RuntimeError(
                    "Fun-CosyVoice3 causal Flow batch returned "
                    f"{mel.shape[-1]} frames, need offset {offset_frames}"
                )
            delta, hift_mel, speech_offset = self._vocoder._hift_delta(
                mel[:, :, offset_frames:],
                hift_mel=state.hift_mel,
                speech_offset=state.speech_offset,
                finalize=False,
            )
            state.token_offset += plan.hop
            self._advance_hop_len(state)
            state.hift_mel = hift_mel
            state.speech_offset = speech_offset
            if delta is not None and delta.numel() > 0:
                decoded[request_id] = delta
        return decoded

    def _run_one_causal_hop(self, state: _CosyVoice3StreamState) -> torch.Tensor | None:
        hop = stream_hop_len(
            state.token_offset,
            hop_len=state.hop_len,
            prompt_pad=state.prompt_pad,
        )
        token_end = state.token_offset + hop + PRE_LOOKAHEAD_LEN
        delta = self._run_flow_hift(
            state,
            token_end=token_end,
            token_offset=state.token_offset,
            streaming=True,
            finalize=False,
        )
        state.token_offset += hop
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
        pieces: list[torch.Tensor] = []
        if not is_final:
            # note (guozhihao-224): one hop per non-final step so a
            # backlogged request cannot drain every ready hop inside one
            # run_step. stream_done still catches up below.
            if not self._ready_for_causal_chunk(state):
                return None
            delta = self._run_one_causal_hop(state)
            if delta is None or delta.numel() == 0:
                return None
            return delta
        while self._ready_for_causal_chunk(state):
            delta = self._run_one_causal_hop(state)
            if delta is not None and delta.numel() > 0:
                pieces.append(delta)
        if not state.tokens:
            return None if not pieces else torch.cat(pieces, dim=-1)
        # note (guozhihao-224): leftover keeps finalize=True so HiFT
        # flushes and pre_lookahead consumes the tail. DiT stays
        # bidirectional; leftover streaming=True did not win the A/B.
        delta = self._run_flow_hift(
            state,
            token_end=len(state.tokens),
            token_offset=state.token_offset,
            streaming=LEFTOVER_FLOW_STREAMING,
            finalize=True,
        )
        if delta is not None and delta.numel() > 0:
            pieces.append(delta)
        if not pieces:
            return None
        return torch.cat(pieces, dim=-1)

    def _run_flow_hift(
        self,
        state: _CosyVoice3StreamState,
        *,
        token_end: int,
        token_offset: int,
        streaming: bool,
        finalize: bool,
    ) -> torch.Tensor | None:
        if state.prompt_token is None or state.prompt_feat is None:
            raise RuntimeError(
                "Fun-CosyVoice3 streaming vocoder decoded before prompt "
                "conditioning was latched"
            )
        if state.embedding is None:
            raise RuntimeError(
                "Fun-CosyVoice3 streaming vocoder decoded before speaker "
                "embedding was latched"
            )
        generated = state.tokens[:token_end]
        if not generated:
            raise RuntimeError(
                "Fun-CosyVoice3 streaming vocoder has no speech tokens to decode"
            )
        token = torch.tensor(generated, dtype=torch.int32).unsqueeze(0)
        wav, hift_mel, speech_offset = self._vocoder.token2wav_chunk(
            token=token,
            prompt_token=state.prompt_token,
            prompt_feat=state.prompt_feat,
            embedding=state.embedding,
            token_offset=token_offset,
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
