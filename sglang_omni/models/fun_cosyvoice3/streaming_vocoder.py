# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 streaming vocoder.

Note (chenyang):

Flow is not autoregressive: it cannot emit one token of mel from one new
token. Each causal step decodes the whole prefix plus several lookahead tokens
(PRE_LOOKAHEAD_LEN). Those last several tokens are context only and are not played;
the next step (or the stream-done flush) is when they become audio.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, Mapping

import torch

from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    PreparedFlowRequest,
    adaptive_flow_requests_grouping,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    TOKEN_MEL_RATIO,
    as_flow_embedding,
    as_flow_prompt_feat,
    as_flow_prompt_token,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
)
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

NextDecode = Literal["causal_window", "leftover", "fallback", "wait"]


@dataclass
class CosyVoice3StreamState:
    tokens: list[int] = field(default_factory=list)
    token_offset: int = 0
    hop_len: int = TOKEN_HOP_LEN
    prompt_token: torch.Tensor | None = None
    prompt_feat: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    hift_mel: torch.Tensor | None = None
    speech_offset: int = 0
    leftover: torch.Tensor | None = None
    done: bool = False
    ready_since: float | None = None
    first_emit_at: float | None = None

    def next_decode(self) -> NextDecode:
        causal_token_end = self.token_offset + self.hop_len + PRE_LOOKAHEAD_LEN
        if self.prompt_token is not None and len(self.tokens) >= causal_token_end:
            return "causal_window"
        elif self.done and self.tokens:
            return "leftover"
        elif self.done:
            return "fallback"
        else:
            return "wait"


@dataclass
class CosyVoice3BufferedState:
    payload: StagePayload
    pipeline_state: FunCosyVoice3State
    prepared: PreparedFlowRequest
    admission_cost: int


class FunCosyVoice3StreamingVocoderScheduler(
    StreamingVocoderBase[CosyVoice3StreamState, NextDecode]
):
    """Decode CosyVoice3 speech tokens incrementally through Flow + HiFT."""

    can_batch_stream_chunks = True
    pump_on_chunk_batch = False

    def __init__(
        self,
        vocoder: CosyVoice3Vocoder,
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
        elif max_hop < hop:
            raise ValueError(
                f"token_max_hop_len ({token_max_hop_len}) must be >= "
                f"token_hop_len ({token_hop_len})"
            )
        else:
            self.token_hop_len = hop
            self.token_max_hop_len = max_hop
            self.disable_hop_growth = bool(disable_hop_growth)
            self.vocoder = vocoder
            self.clock: Callable[[], float] = time.monotonic
        super().__init__(
            self.vocode_payload,
            batch_compute_fn=self.vocode_payloads,
            sample_rate=int(sample_rate),
            stream_source_hint="Fun-CosyVoice3",
            max_batch_size=max_batch_size,
            max_batch_wait_ms=max_batch_wait_ms,
            request_cost_fn=request_cost_fn,
            max_batch_cost=max_batch_cost,
        )
        self.buffered_pending: dict[str, CosyVoice3BufferedState] = {}
        self.next_buffered_request_index = 0
        self.buffered_plan: list[tuple[str, ...]] = []
        self.buffered_plan_dirty = True
        self.buffered_active_group_ids: set[str] = set()

    def pump_one_step(self) -> list[str] | None:
        with self.vocoder.stream_context:
            return super().pump_one_step()

    async def vocode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.vocoder.decode_payloads([payload])
        return results[0]

    async def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        return await self.vocoder.decode_payloads(payloads)

    def new_request_batch_wait_s(self, first_msg: IncomingMessage) -> float:
        del first_msg
        with self.state_lock:
            if self.buffered_pending:
                return 0.0
        return self.max_batch_wait_s

    def run_non_streaming_batch(
        self,
        batch: list[IncomingMessage],
        loop: asyncio.AbstractEventLoop,
    ) -> None:
        """Prepare and register non-streaming requests for a later decode step."""
        active = [msg for msg in batch if not self.is_aborted(msg.request_id)]
        if not active:
            return

        with self.state_lock:
            for msg in active:
                self.pending_done.discard(msg.request_id)

        for msg in active:
            try:
                self.validate_non_streaming_payload(msg.data)
                pipeline_state, codes = self.vocoder.prepare_item(msg.data)
                with self.state_lock:
                    request_index = self.next_buffered_request_index
                    self.next_buffered_request_index += 1
                prepared = self.vocoder.prepare_flow_request(
                    request_index, pipeline_state, codes
                )
                admission_cost = prepared.total_mel_frames
            except Exception as exc:
                if not self.is_aborted(msg.request_id):
                    self.emit_error(msg.request_id, exc)
                    self.record_completed_non_streaming_request_id(msg.request_id)
                continue
            if self.is_aborted(msg.request_id):
                continue
            with self.state_lock:
                self.buffered_pending[msg.request_id] = CosyVoice3BufferedState(
                    payload=msg.data,
                    pipeline_state=pipeline_state,
                    prepared=prepared,
                    admission_cost=admission_cost,
                )
                self.buffered_plan_dirty = True

    def clear_request_state(
        self, request_id: str, *, keep_aborted: bool = False
    ) -> None:
        with self.state_lock:
            if self.buffered_pending.pop(request_id, None) is not None:
                if request_id not in self.buffered_active_group_ids:
                    self.buffered_plan_dirty = True
            super().clear_request_state(request_id, keep_aborted=keep_aborted)

    def create_stream_state(self, request_id: str) -> CosyVoice3StreamState:
        return CosyVoice3StreamState(hop_len=self.token_hop_len)

    def warmup_now(self) -> None:
        # note(ratish): one hop and one final through Flow and HiFT before the
        # stage publishes readiness, so the first request pays neither the
        # attention kernel load nor the f0 cast.
        flow = self.vocoder.flow
        item = FlowBatchInput(
            token=torch.zeros(
                1, self.token_hop_len + PRE_LOOKAHEAD_LEN, dtype=torch.int32
            ),
            prompt_token=torch.zeros(1, self.token_hop_len, dtype=torch.int32),
            prompt_feat=torch.zeros(
                1, self.token_hop_len * TOKEN_MEL_RATIO, flow.output_size
            ),
            embedding=torch.zeros(1, flow.spk_embed_affine_layer.in_features),
        )
        # note(ratish): under the vocoder's stream, so the warmup and not the
        # first request builds that stream's memory pool and cuBLAS workspaces,
        # which PyTorch keeps per stream.
        started = time.monotonic()
        with self.vocoder.stream_context:
            mel = self.vocoder.hop_batch([item])[0]
            self.vocoder.hift_delta(mel, hift_mel=None, speech_offset=0, finalize=False)
            hop_s = time.monotonic() - started
            mel = self.vocoder.leftover_batch([item])[0]
            self.vocoder.hift_delta(mel, hift_mel=None, speech_offset=0, finalize=True)
        final_s = time.monotonic() - started - hop_s
        logger.info(
            f"Fun-CosyVoice3 vocoder warmup: hop {hop_s:.1f} s, final {final_s:.1f} s"
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: CosyVoice3StreamState,
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
            else:
                pipeline_state = FunCosyVoice3State.from_dict(payload.data)
                self.latch_prompts(
                    request_id,
                    state,
                    prompt_token=pipeline_state.flow_prompt_speech_token,
                    prompt_feat=pipeline_state.flow_prompt_speech_feat,
                    embedding=pipeline_state.flow_embedding,
                )
        else:
            metadata: Mapping[str, Any] = source
            if any(
                key in metadata
                for key in (
                    "flow_prompt_speech_token",
                    "flow_prompt_speech_feat",
                    "flow_embedding",
                )
            ):
                self.latch_prompts(
                    request_id,
                    state,
                    prompt_token=metadata.get("flow_prompt_speech_token"),
                    prompt_feat=metadata.get("flow_prompt_speech_feat"),
                    embedding=metadata.get("flow_embedding"),
                )
            else:
                return

    def latch_prompts(
        self,
        request_id: str,
        state: CosyVoice3StreamState,
        *,
        prompt_token: Any,
        prompt_feat: Any,
        embedding: Any,
    ) -> None:
        token = as_flow_prompt_token(prompt_token)
        feat = as_flow_prompt_feat(prompt_feat)
        speaker_embedding = as_flow_embedding(embedding)
        # note (guozhihao-224): pad prompt to a hop multiple here so the
        # first generated hop stays hop+lookahead instead of waiting for
        # prompt_pad extra AR tokens.
        token, feat = pad_flow_prompt_to_hop(token, feat, hop_len=self.token_hop_len)
        if state.prompt_token is None:
            state.prompt_token = token
            state.prompt_feat = feat
            state.embedding = speaker_embedding
        elif (
            tuple(token.shape) != tuple(state.prompt_token.shape)
            or tuple(feat.shape) != tuple(state.prompt_feat.shape)
            or tuple(speaker_embedding.shape) != tuple(state.embedding.shape)
        ):
            # note (guozhihao-224): latch is shape-stable; payload and first
            # chunk metadata must carry the same prompt tensors.
            raise ValueError(
                f"Fun-CosyVoice3 stream prompt tensors changed for {request_id!r}"
            )
        else:
            return

    def validate_chunk(
        self,
        request_id: str,
        state: CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> torch.Tensor:
        chunk = codes.to(dtype=torch.long)
        if chunk.ndim == 2 and chunk.shape[-1] == 1:
            return chunk.reshape(-1).contiguous()
        elif chunk.ndim != 1:
            raise ValueError(
                f"Fun-CosyVoice3 stream chunk must be 1-D speech tokens, "
                f"got {tuple(chunk.shape)}"
            )
        else:
            return chunk.contiguous()

    def on_streaming_new_request(self, request_id: str, payload: StagePayload) -> None:
        super().on_streaming_new_request(request_id, payload)
        state = self.stream_states.get(request_id)
        if state is None:
            return
        elif state.ready_since is None and state.next_decode() != "wait":
            ready_since = self.clock()
        else:
            ready_since = state.ready_since
        state.ready_since = ready_since

    def ingest(
        self,
        request_id: str,
        state: CosyVoice3StreamState,
        codes: torch.Tensor,
    ) -> None:
        state.tokens.extend(int(token) for token in codes.tolist())
        if state.ready_since is None and state.next_decode() != "wait":
            ready_since = self.clock()
        else:
            ready_since = state.ready_since
        state.ready_since = ready_since

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage] | None:
        state = self.get_or_create_stream_state(request_id)
        if state is None:
            return []
        else:
            state.done = True
            if state.ready_since is None and state.next_decode() != "wait":
                ready_since = self.clock()
            else:
                ready_since = state.ready_since
            state.ready_since = ready_since
            return None

    def has_ready_work(self) -> bool:
        with self.state_lock:
            ready = any(
                state.next_decode() != "wait" and not self.is_aborted(request_id)
                for request_id, state in self.stream_state_items()
            )
            return ready or bool(self.buffered_pending)

    def remove_buffered_group_locked(
        self, group_ids: tuple[str, ...]
    ) -> dict[str, CosyVoice3BufferedState]:
        if self.buffered_plan and self.buffered_plan[0] == group_ids:
            self.buffered_plan.pop(0)
        group: dict[str, CosyVoice3BufferedState] = {}
        for request_id in group_ids:
            buffered = self.buffered_pending.pop(request_id, None)
            if buffered is not None:
                group[request_id] = buffered
        for request_id in group_ids:
            self.pending_done.discard(request_id)
        self.buffered_active_group_ids.difference_update(group_ids)
        return group

    def run_ready_step(self) -> None:
        flow_group: list[PreparedFlowRequest] | None = None
        group_ids: tuple[str, ...] = ()
        requests_by_index: dict[int, tuple[str, CosyVoice3BufferedState]] = {}
        with self.state_lock:
            streaming_ready = any(
                state.next_decode() != "wait" and not self.is_aborted(request_id)
                for request_id, state in self.stream_state_items()
            )
            if streaming_ready:
                failed = self._pump_one_step() or []
            else:
                failed = []
                if self.buffered_plan_dirty or not self.buffered_plan:
                    candidates: list[tuple[str, CosyVoice3BufferedState]] = []
                    candidate_cost = 0
                    for request_id, buffered in list(self.buffered_pending.items()):
                        if self.is_aborted(request_id):
                            self.buffered_pending.pop(request_id, None)
                            self.buffered_plan_dirty = True
                            continue
                        if len(candidates) >= self.max_batch_size:
                            break
                        request_cost = buffered.admission_cost
                        if (
                            self.max_batch_cost is not None
                            and candidates
                            and candidate_cost + request_cost > self.max_batch_cost
                        ):
                            break
                        candidates.append((request_id, buffered))
                        candidate_cost += request_cost

                    if candidates:
                        request_ids_by_index = {
                            buffered.prepared.index: request_id
                            for request_id, buffered in candidates
                        }
                        groups = adaptive_flow_requests_grouping(
                            [buffered.prepared for _, buffered in candidates],
                            flow_merge_max_gap_frames=self.vocoder.flow_merge_max_gap_frames,
                            flow_merge_pad_budget_percent=self.vocoder.flow_merge_pad_budget_percent,
                        )
                        groups.sort(key=len, reverse=True)
                        self.buffered_plan = [
                            tuple(
                                request_ids_by_index[request.index] for request in group
                            )
                            for group in groups
                        ]
                    else:
                        self.buffered_plan = []
                    self.buffered_plan_dirty = False

                if self.buffered_plan:
                    group_ids = self.buffered_plan[0]
                    group = [
                        (request_id, self.buffered_pending.get(request_id))
                        for request_id in group_ids
                    ]
                    if all(
                        buffered is not None and not self.is_aborted(request_id)
                        for request_id, buffered in group
                    ):
                        requests_by_index = {
                            buffered.prepared.index: (request_id, buffered)
                            for request_id, buffered in group
                            if buffered is not None
                        }
                        flow_group = [
                            buffered.prepared
                            for _, buffered in group
                            if buffered is not None
                        ]
                        self.buffered_active_group_ids = set(group_ids)
                    else:
                        self.buffered_plan_dirty = True

        if streaming_ready:
            for request_id in failed:
                self.cleanup_aborted_request(request_id)
            return
        if flow_group is None:
            return

        try:
            completed = self.vocoder.decode_flow_group(flow_group)
            completed_by_index = {
                request.index: (request, wav, sample_rate)
                for request, wav, sample_rate in completed
            }
            if set(completed_by_index) != set(requests_by_index):
                raise RuntimeError(
                    "Fun-CosyVoice3 vocoder did not decode every request in the "
                    "Flow group"
                )
        except Exception as exc:
            with self.state_lock:
                self.remove_buffered_group_locked(group_ids)
            for request_id in group_ids:
                if not self.is_aborted(request_id):
                    self.emit_error(request_id, exc)
                    self.record_completed_non_streaming_request_id(request_id)
            return

        with self.state_lock:
            buffered_group = self.remove_buffered_group_locked(group_ids)
        for request_index, (request_id, buffered) in requests_by_index.items():
            if request_id not in buffered_group or self.is_aborted(request_id):
                continue
            _, wav, sample_rate = completed_by_index[request_index]
            try:
                result = self.vocoder.store_result(
                    buffered.payload,
                    buffered.pipeline_state,
                    wav,
                    sample_rate,
                )
            except Exception as exc:
                if not self.is_aborted(request_id):
                    self.emit_error(request_id, exc)
                    self.record_completed_non_streaming_request_id(request_id)
            else:
                if not self.is_aborted(request_id):
                    self.emit_result(request_id, result)
                    self.record_completed_non_streaming_request_id(request_id)

    def select_step_participants(
        self,
    ) -> list[tuple[str, CosyVoice3StreamState]]:
        now = self.clock()
        started: list[tuple[float, float, str, CosyVoice3StreamState]] = []
        unstarted: list[tuple[float, str, CosyVoice3StreamState]] = []
        for request_id, state in self.stream_state_items():
            if state.next_decode() == "wait" or self.is_aborted(request_id):
                continue
            elif state.first_emit_at is None:
                assert state.ready_since is not None
                unstarted.append((state.ready_since, request_id, state))
            else:
                assert state.ready_since is not None
                playback_slack = state.speech_offset / self.sample_rate - (
                    now - state.first_emit_at
                )
                started.append((playback_slack, state.ready_since, request_id, state))
        ranked = [(request_id, state) for _, _, request_id, state in sorted(started)]
        ranked += [(request_id, state) for _, request_id, state in sorted(unstarted)]
        if not ranked:
            return []
        else:
            head_decode = ranked[0][1].next_decode()
            if head_decode == "fallback":
                return ranked[:1]
            else:
                peers = [
                    (request_id, state)
                    for request_id, state in ranked
                    if state.next_decode() == head_decode
                ]
                return peers[: self.max_batch_size]

    def build_step_plan(
        self, participants: list[tuple[str, CosyVoice3StreamState]]
    ) -> NextDecode:
        decode = participants[0][1].next_decode()
        assert decode != "wait"
        return decode

    def run_step(
        self,
        participants: list[tuple[str, CosyVoice3StreamState]],
        plan: NextDecode,
    ) -> dict[str, torch.Tensor]:
        assert plan != "wait"
        if plan == "fallback":
            request_id, _ = participants[0]
            self.complete_stream_request(request_id, self.finish_stream(request_id))
            return {}
        elif plan == "leftover":
            items = [
                FlowBatchInput(
                    token=torch.tensor(state.tokens, dtype=torch.int32).unsqueeze(0),
                    prompt_token=state.prompt_token,
                    prompt_feat=state.prompt_feat,
                    embedding=state.embedding,
                )
                for _, state in participants
            ]
            logger.info(f"Fun-CosyVoice3 leftover Flow batch size={len(items)}")
            mels = self.vocoder.leftover_batch(items)
            for (_, state), mel in zip(participants, mels, strict=True):
                state.leftover, state.hift_mel, state.speech_offset = (
                    self.vocoder.hift_delta(
                        mel[:, :, state.token_offset * TOKEN_MEL_RATIO :],
                        hift_mel=state.hift_mel,
                        speech_offset=state.speech_offset,
                        finalize=True,
                    )
                )
            for request_id, _ in participants:
                self.complete_stream_request(request_id, self.finish_stream(request_id))
            return {}
        else:
            items = [
                FlowBatchInput(
                    token=torch.tensor(
                        state.tokens[
                            : state.token_offset + state.hop_len + PRE_LOOKAHEAD_LEN
                        ],
                        dtype=torch.int32,
                    ).unsqueeze(0),
                    prompt_token=state.prompt_token,
                    prompt_feat=state.prompt_feat,
                    embedding=state.embedding,
                )
                for _, state in participants
            ]
            logger.info(f"Fun-CosyVoice3 causal Flow batch size={len(items)}")
            mels = self.vocoder.hop_batch(items)
            decoded: dict[str, torch.Tensor] = {}
            for (request_id, state), mel in zip(participants, mels, strict=True):
                delta, state.hift_mel, state.speech_offset = self.vocoder.hift_delta(
                    mel[:, :, state.token_offset * TOKEN_MEL_RATIO :],
                    hift_mel=state.hift_mel,
                    speech_offset=state.speech_offset,
                    finalize=False,
                )
                if delta.numel() > 0:
                    decoded[request_id] = delta
            now = self.clock()
            for request_id, state in participants:
                state.token_offset += state.hop_len
                state.hop_len = next_stream_hop_len(
                    state.hop_len,
                    max_hop_len=self.token_max_hop_len,
                    disable_growth=self.disable_hop_growth,
                )
                if request_id in decoded and state.first_emit_at is None:
                    state.first_emit_at = now
                if state.next_decode() != "wait":
                    state.ready_since = now
                else:
                    state.ready_since = None
            return decoded

    def decode_delta(
        self,
        request_id: str,
        state: CosyVoice3StreamState,
        *,
        is_final: bool,
    ) -> torch.Tensor | None:
        # note(ratish): hops and leftovers run in steps, so the per-chunk
        # decode never emits and the final flush returns the batched leftover.
        if is_final:
            return state.leftover
        else:
            return None

    def fallback_full_decode(
        self,
        request_id: str,
        payload: StagePayload,
        state: CosyVoice3StreamState,
    ) -> torch.Tensor | None:
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
        else:
            prompt_token = as_flow_prompt_token(pipeline_state.flow_prompt_speech_token)
            prompt_feat = as_flow_prompt_feat(pipeline_state.flow_prompt_speech_feat)
            embedding = as_flow_embedding(pipeline_state.flow_embedding)
        return self.vocoder.token2wav(
            token=codes.unsqueeze(0),
            prompt_token=prompt_token,
            prompt_feat=prompt_feat,
            embedding=embedding,
        )

    def final_result_data(
        self,
        request_id: str,
        payload: StagePayload,
        state: CosyVoice3StreamState,
    ) -> dict[str, Any]:
        final_data: dict[str, Any] = {
            "modality": "audio",
            "sample_rate": self.sample_rate,
        }
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        usage = build_usage(pipeline_state)
        if usage is None:
            return final_data
        else:
            final_data["usage"] = usage
            return final_data

    def stream_payload(self, request_id: str, waveform: torch.Tensor) -> dict[str, Any]:
        return audio_waveform_payload(
            waveform,
            sample_rate=self.sample_rate,
            modality="audio",
            source_hint="Fun-CosyVoice3",
        )

    def release_stream_resources(
        self, request_id: str, state: CosyVoice3StreamState
    ) -> None:
        state.tokens.clear()
        state.hift_mel = None
        state.prompt_token = None
        state.prompt_feat = None
        state.embedding = None
