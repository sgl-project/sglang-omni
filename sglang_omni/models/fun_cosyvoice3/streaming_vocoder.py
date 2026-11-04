# SPDX-License-Identifier: Apache-2.0
"""Fun-CosyVoice3 streaming vocoder.

Note (chenyang):

Flow is not autoregressive: it cannot emit one token of mel from one new
token. Each causal step decodes the whole prefix plus several lookahead tokens
(PRE_LOOKAHEAD_LEN). Those last several tokens are context only and are not played;
the next step (or the stream-done flush) is when they become audio.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal, Mapping

import torch
from sglang.srt.utils.common import is_pin_memory_available

from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.prefix_cache import BLOCK_FRAMES, PrefixCacheRow
from sglang_omni.models.fun_cosyvoice3.stages import (
    CosyVoice3Vocoder,
    FlowBatchInput,
    HiftStepRow,
    HiftStepSamples,
)
from sglang_omni.models.fun_cosyvoice3.streaming import (
    PRE_LOOKAHEAD_LEN,
    TOKEN_HOP_LEN,
    TOKEN_MAX_HOP_LEN,
    as_flow_embedding,
    as_flow_prompt_feat,
    as_flow_prompt_token,
    next_stream_hop_len,
    pad_flow_prompt_to_hop,
    prompt_token_pad,
)
from sglang_omni.models.fun_cosyvoice3.utils import MAX_REFERENCE_SECONDS
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import build_usage
from sglang_omni.scheduling.streaming_vocoder import StreamingVocoderBase
from sglang_omni.utils.audio_payload import audio_waveform_payload

logger = logging.getLogger(__name__)

SAMPLE_RATE = 24000

NextDecode = Literal["causal_window", "leftover", "fallback", "wait"]


def upload_prompt(
    tensor: torch.Tensor, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """A host prompt tensor as dtype on device, staged in pinned memory where the
    platform has it, so the copy does not wait for the work queued on the stream."""
    host = torch.empty(
        tensor.shape, dtype=dtype, pin_memory=is_pin_memory_available(device)
    )
    host.copy_(tensor)
    return host.to(device, non_blocking=True)


def causal_hop_frames(item: FlowBatchInput, *, token_mel_ratio: int) -> int:
    """Mel frames a causal hop over item solves, prompt included."""
    return (
        int(item.prompt_token.shape[1]) + int(item.token.shape[1]) - PRE_LOOKAHEAD_LEN
    ) * token_mel_ratio


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
    flow_cache: tuple[PrefixCacheRow, PrefixCacheRow] | None = None
    leftover: torch.Tensor | None = None
    done: bool = False
    ready_since: float | None = None
    first_emit_at: float | None = None
    is_final_launched: bool = False

    def next_decode(self) -> NextDecode:
        causal_token_end = self.token_offset + self.hop_len + PRE_LOOKAHEAD_LEN
        if self.is_final_launched:
            return "wait"
        elif self.prompt_token is not None and len(self.tokens) >= causal_token_end:
            return "causal_window"
        elif self.done and self.tokens:
            return "leftover"
        elif self.done:
            return "fallback"
        else:
            return "wait"


@dataclass(kw_only=True, frozen=True)
class LaunchedStep:
    """A step whose Flow and HiFT are queued on the vocoder's stream and whose streams
    have advanced; its audio is read at its completion."""

    participants: list[tuple[str, CosyVoice3StreamState]]
    plan: Literal["causal_window", "leftover"]
    samples: HiftStepSamples


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
        request_cost_fn: Callable[[StagePayload], int] | None = None,
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
            self.launched_step: LaunchedStep | None = None
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

    def pump_one_step(self) -> list[str] | None:
        """Launch the next step, then complete the step launched before it, so the device
        runs one while the host reads the other. None when no stream is ready and no step
        is in flight; otherwise the request ids a failed step aborted."""
        previous = self.launched_step
        participants = self.select_step_participants()
        if not participants and previous is None:
            return None
        else:
            pass
        failed: list[str] = []
        launched: LaunchedStep | None = None
        if participants:
            try:
                with self.vocoder.stream_context:
                    launched = self.launch_step(participants)
            except Exception as exc:
                failed.extend(self.on_step_failure(participants, exc))
        else:
            pass
        if previous is not None:
            failed.extend(self.complete_step(previous))
        else:
            pass
        # note (ratish): samples already on the host need no wait, so such a step
        # completes at its launch, as before.
        if launched is not None and launched.samples.copy_done is None:
            failed.extend(self.complete_step(launched))
            launched = None
        else:
            pass
        self.launched_step = launched
        return failed

    def on_serving_stop(self) -> None:
        """Drop the step in flight: its device work finishes and nothing is sent."""
        self.launched_step = None

    async def vocode_payload(self, payload: StagePayload) -> StagePayload:
        results = await self.vocoder.decode_payloads([payload])
        return results[0]

    async def vocode_payloads(self, payloads: list[StagePayload]) -> list[StagePayload]:
        return await self.vocoder.decode_payloads(payloads)

    def create_stream_state(self, request_id: str) -> CosyVoice3StreamState:
        return CosyVoice3StreamState(hop_len=self.token_hop_len)

    def warmup_now(self) -> None:
        # note(ratish): one hop and one final through Flow and HiFT before the
        # stage publishes readiness, so the first request pays neither the
        # attention kernel load nor the f0 cast.
        item = self.make_warmup_flow_input(self.token_hop_len + PRE_LOOKAHEAD_LEN)
        # note(ratish): under the vocoder's stream, so the warmup and not the
        # first request builds that stream's memory pool and cuBLAS workspaces,
        # which PyTorch keeps per stream.
        started = time.monotonic()
        with self.vocoder.stream_context:
            mel = self.vocoder.hop_batch([item])[0]
            self.vocoder.hift_step(
                [HiftStepRow(history=mel, emitted_samples=0, is_final=False)]
            ).wait()
            self.warmup_prefix_hops()
            hop_s = time.monotonic() - started
            mel = self.vocoder.leftover_batch([item])[0]
            self.vocoder.hift_step(
                [HiftStepRow(history=mel, emitted_samples=0, is_final=True)]
            ).wait()
        final_s = time.monotonic() - started - hop_s
        logger.info(
            f"Fun-CosyVoice3 vocoder warmup: hop {hop_s:.1f} s, final {final_s:.1f} s"
        )

    def warmup_packed_dit_compile(self) -> None:
        """Materialize PackedDiT contracts before the process becomes ready."""
        packed_estimator = self.vocoder.flow.packed_estimator
        if not isinstance(packed_estimator, PackedDiT):
            raise RuntimeError(
                "Fun-CosyVoice3 PackedDiT compile warmup requires a PackedDiT estimator"
            )
        else:
            pass
        if not packed_estimator.compile(self.vocoder.autocast_dtype):
            return
        else:
            pass
        hop_tokens = self.token_hop_len + PRE_LOOKAHEAD_LEN
        first = [self.make_warmup_flow_input(hop_tokens)]
        # note(ratish): a second row count and length,
        # so the sizes serving varies turn symbolic at startup, not on a request.
        second = [
            self.make_warmup_flow_input(hop_tokens),
            self.make_warmup_flow_input(hop_tokens + self.token_hop_len),
        ]
        started = time.monotonic()
        with self.vocoder.stream_context:
            for items in (first, second):
                self.vocoder.hop_batch(items)
                self.vocoder.leftover_batch(items)
        logger.info(
            f"Fun-CosyVoice3 PackedDiT compile warmup, hop and final, with "
            f"torch_num_threads={torch.get_num_threads()} "
            f"({time.monotonic() - started:.1f} s)"
        )

    def measure_peak_step_bytes(
        self, device: torch.device, *, row_count: int, generated_tokens_per_row: int
    ) -> int:
        """Device memory the largest step this scheduler admits maps beyond what stays
        resident, over rows of generated_tokens_per_row tokens after the longest
        reference prompt: the larger of row_count streams as uncached hops and finals,
        and the largest non-streaming batch."""
        # note (ratish): without ref_text the engine's prompt omits the reference tokens,
        # so Flow gets the whole reference on top of a context of generated tokens.
        prompt_token_count = MAX_REFERENCE_SECONDS * self.vocoder.flow.input_frame_rate
        prompt_token_count += prompt_token_pad(
            prompt_token_count, hop_len=self.token_hop_len
        )
        items = [
            self.make_warmup_flow_input(
                generated_tokens_per_row, prompt_token_count=prompt_token_count
            )
            for _ in range(row_count)
        ]
        started = time.monotonic()
        # note (ratish): each pass starts from an emptied cache, since the allocator
        # frees cached blocks before it fails, so the passes need the larger peak.
        peak_bytes: list[int] = []
        for run in (self.run_largest_streaming_step, self.run_largest_batch_decode):
            torch.accelerator.synchronize(device)
            torch.accelerator.empty_cache()
            resident_bytes = torch.accelerator.memory_reserved(device)
            torch.accelerator.reset_peak_memory_stats(device)
            with self.vocoder.stream_context:
                run(items)
            torch.accelerator.synchronize(device)
            peak_bytes.append(
                torch.accelerator.max_memory_reserved(device) - resident_bytes
            )
        streaming_bytes, batch_decode_bytes = peak_bytes
        logger.info(
            f"Fun-CosyVoice3 largest admitted step, {row_count} rows of "
            f"{generated_tokens_per_row} tokens after a {prompt_token_count} token "
            f"prompt: streaming {streaming_bytes / 2**30:.2f} GB, non-streaming "
            f"{batch_decode_bytes / 2**30:.2f} GB mapped "
            f"({time.monotonic() - started:.1f} s)"
        )
        return max(peak_bytes)

    def run_largest_streaming_step(self, items: list[FlowBatchInput]) -> None:
        """The items as uncached hops and as finals, each followed by a HiFT step."""
        hop_frames = self.token_max_hop_len * self.vocoder.flow.token_mel_ratio
        samples_per_frame = self.vocoder.hift_samples_per_mel_frame
        for decode, is_final in (
            (self.vocoder.hop_batch, False),
            (self.vocoder.leftover_batch, True),
        ):
            mels = decode(items)
            self.vocoder.hift_step(
                [
                    HiftStepRow(
                        history=mel,
                        emitted_samples=max(int(mel.shape[2]) - hop_frames, 0)
                        * samples_per_frame,
                        is_final=is_final,
                    )
                    for mel in mels
                ]
            ).wait()

    def run_largest_batch_decode(self, items: list[FlowBatchInput]) -> None:
        """The non-streaming decode of one item alone, then HiFT over the largest group
        a batch at the admission budget forms."""
        # note (ratish): a non-streaming request runs the final's Flow, then HiFT over
        # its whole mel. Batched requests fit the admission budget, and a HiFT group
        # pads to its longest row within the Flow merge budget and HiFT's padding cap.
        mels = self.vocoder.leftover_batch(items[:1])
        self.vocoder.mel2wav_batch(mels)
        if self.max_batch_size > 1:
            row_frames = (
                int(items[0].prompt_token.shape[1]) + int(items[0].token.shape[1])
            ) * self.vocoder.flow.token_mel_ratio
            if self.max_batch_cost is None:
                batch_frames = self.max_batch_size * row_frames
            else:
                batch_frames = min(
                    self.max_batch_cost, self.max_batch_size * row_frames
                )
            padding_ratio = min(
                self.vocoder.hift_max_padding_waste,
                1 + self.vocoder.flow_merge_pad_budget_percent / 100,
            )
            padded_frames = int(batch_frames * padding_ratio) // self.max_batch_size
            self.vocoder.mel2wav_batch(
                [
                    mels[0].new_zeros(1, int(mels[0].shape[1]), padded_frames)
                    for _ in range(self.max_batch_size)
                ]
            )
        else:
            pass

    def warmup_prefix_compile(self) -> None:
        """The pool's compiled prefix forward through its hops, so its graphs capture
        compiled code."""
        started = time.monotonic()
        with self.vocoder.stream_context:
            self.warmup_prefix_hops()
        logger.info(
            f"Fun-CosyVoice3 prefix compile warmup ({time.monotonic() - started:.1f} s)"
        )

    def prefix_warmup_hop_inputs(self) -> tuple[FlowBatchInput, FlowBatchInput]:
        """The prefix warmup's first hop and its follow-up over twice the tokens."""
        first_hop = self.make_warmup_flow_input(self.token_hop_len + PRE_LOOKAHEAD_LEN)
        follow_up_hop = FlowBatchInput(
            token=torch.cat((first_hop.token, first_hop.token), dim=1),
            prompt_token=first_hop.prompt_token,
            prompt_feat=first_hop.prompt_feat,
            embedding=first_hop.embedding,
        )
        return first_hop, follow_up_hop

    def prefix_warmup_block_count(self) -> int:
        """Pool blocks the prefix warmup holds at its follow-up hop, for both CFG
        twins."""
        _, follow_up_hop = self.prefix_warmup_hop_inputs()
        frames = causal_hop_frames(
            follow_up_hop, token_mel_ratio=self.vocoder.flow.token_mel_ratio
        )
        return 2 * -(-frames // BLOCK_FRAMES)

    def warmup_prefix_hops(self) -> None:
        """A first hop and a follow-up hop through the prefix cache, so both
        an empty and a filled prefix are materialized before serving."""
        first_hop, follow_up_hop = self.prefix_warmup_hop_inputs()
        token_mel_ratio = self.vocoder.flow.token_mel_ratio
        cache = self.vocoder.prefix_cache_rows(
            causal_hop_frames(first_hop, token_mel_ratio=token_mel_ratio)
        )
        if cache is None:
            return
        else:
            pass
        try:
            self.vocoder.hop_batch_prefix([first_hop], [cache])
            if self.vocoder.grow_prefix_cache(
                cache, causal_hop_frames(follow_up_hop, token_mel_ratio=token_mel_ratio)
            ):
                self.vocoder.hop_batch_prefix([follow_up_hop], [cache])
            else:
                pass
        finally:
            self.vocoder.release_prefix_cache(cache)

    def make_warmup_flow_input(
        self, token_count: int, *, prompt_token_count: int | None = None
    ) -> FlowBatchInput:
        flow = self.vocoder.flow
        if prompt_token_count is None:
            prompt_token_count = self.token_hop_len
        else:
            pass
        return FlowBatchInput(
            token=torch.zeros(1, token_count, dtype=torch.int32),
            prompt_token=torch.zeros(1, prompt_token_count, dtype=torch.int32),
            prompt_feat=torch.zeros(
                1, prompt_token_count * flow.token_mel_ratio, flow.output_size
            ),
            embedding=torch.zeros(1, flow.spk_embed_affine_layer.in_features),
        )

    def latch_stream_contract(
        self,
        request_id: str,
        state: CosyVoice3StreamState,
        source: StagePayload | Mapping[str, object],
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
            metadata: Mapping[str, object] = source
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
        prompt_token: object,
        prompt_feat: object,
        embedding: object,
    ) -> None:
        token = as_flow_prompt_token(prompt_token)
        feat = as_flow_prompt_feat(prompt_feat)
        speaker_embedding = as_flow_embedding(embedding)
        # note (guozhihao-224): pad prompt to a hop multiple here so the
        # first generated hop stays hop+lookahead instead of waiting for
        # prompt_pad extra AR tokens.
        token, feat = pad_flow_prompt_to_hop(
            token,
            feat,
            hop_len=self.token_hop_len,
            token_mel_ratio=self.vocoder.flow.token_mel_ratio,
        )
        if state.prompt_token is None:
            # note (ratish): on the vocoder's stream, which every launch reading the
            # prompt follows; each step's pack then copies it on the device.
            parameter = next(self.vocoder.flow.parameters())
            device, dtype = parameter.device, parameter.dtype
            with self.vocoder.stream_context:
                state.prompt_token = upload_prompt(token, device, torch.int32)
                state.prompt_feat = upload_prompt(feat, device, dtype)
                state.embedding = upload_prompt(speaker_embedding, device, dtype)
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
            ready = self.launched_step is not None or any(
                state.next_decode() != "wait" and not self.is_aborted(request_id)
                for request_id, state in self.stream_state_items()
            )
            if ready:
                return True
            else:
                return False

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

    def launch_step(
        self, participants: list[tuple[str, CosyVoice3StreamState]]
    ) -> LaunchedStep | None:
        """Queue one step's Flow and HiFT and advance its streams; their audio is read by
        complete_step. A fallback decodes and completes here."""
        plan = participants[0][1].next_decode()
        assert plan != "wait"
        pin_memory = is_pin_memory_available(
            next(self.vocoder.flow.parameters()).device
        )
        if plan == "fallback":
            request_id, _ = participants[0]
            self.complete_stream_request(request_id, self.finish_stream(request_id))
            return None
        elif plan == "leftover":
            items = [
                FlowBatchInput(
                    token=torch.tensor(
                        state.tokens, dtype=torch.int32, pin_memory=pin_memory
                    ).unsqueeze(0),
                    prompt_token=state.prompt_token,
                    prompt_feat=state.prompt_feat,
                    embedding=state.embedding,
                )
                for _, state in participants
            ]
            logger.info(f"Fun-CosyVoice3 leftover Flow batch size={len(items)}")
            mels = self.vocoder.leftover_batch(items)
            samples = self.hift_step(participants, mels, is_final=True)
            for (_, state), offset in zip(
                participants, samples.emitted_samples, strict=True
            ):
                state.speech_offset = offset
                state.is_final_launched = True
            return LaunchedStep(participants=participants, plan=plan, samples=samples)
        else:
            items = [
                FlowBatchInput(
                    token=torch.tensor(
                        state.tokens[
                            : state.token_offset + state.hop_len + PRE_LOOKAHEAD_LEN
                        ],
                        dtype=torch.int32,
                        pin_memory=pin_memory,
                    ).unsqueeze(0),
                    prompt_token=state.prompt_token,
                    prompt_feat=state.prompt_feat,
                    embedding=state.embedding,
                )
                for _, state in participants
            ]
            logger.info(f"Fun-CosyVoice3 causal Flow batch size={len(items)}")
            mels = self.hop_batch_with_prefix(participants, items)
            samples = self.hift_step(participants, mels, is_final=False)
            now = self.clock()
            for (_, state), new_samples, offset in zip(
                participants, samples.new_samples, samples.emitted_samples, strict=True
            ):
                # note (ratish): playback starts with the launch of the first samples, so
                # the next selection, made before their completion, ranks the stream started.
                if new_samples.numel() > 0 and state.first_emit_at is None:
                    state.first_emit_at = now
                else:
                    pass
                state.speech_offset = offset
                state.token_offset += state.hop_len
                state.hop_len = next_stream_hop_len(
                    state.hop_len,
                    max_hop_len=self.token_max_hop_len,
                    disable_growth=self.disable_hop_growth,
                )
                if state.next_decode() != "wait":
                    state.ready_since = now
                else:
                    state.ready_since = None
            return LaunchedStep(participants=participants, plan=plan, samples=samples)

    def complete_step(self, launched: LaunchedStep) -> list[str]:
        """Read a launched step's audio and send it: a hop's chunk, or a final's last
        chunk and then its result. Streams cleared since the launch get nothing."""
        try:
            deltas = launched.samples.wait()
            for (request_id, state), delta in zip(
                launched.participants, deltas, strict=True
            ):
                is_cleared = self.stream_states.get(request_id) is not state
                if is_cleared or self.is_aborted(request_id):
                    continue
                elif launched.plan == "leftover":
                    state.leftover = delta
                    self.complete_stream_request(
                        request_id, self.finish_stream(request_id)
                    )
                elif delta.numel() > 0:
                    self.mark_stream_emitted(request_id)
                    self.outbox.put(self.stream_chunk_message(request_id, delta))
                else:
                    pass
        except Exception as exc:
            return self.on_step_failure(
                [
                    (request_id, state)
                    for request_id, state in launched.participants
                    if self.stream_states.get(request_id) is state
                ],
                exc,
            )
        return []

    def hop_batch_with_prefix(
        self,
        participants: list[tuple[str, CosyVoice3StreamState]],
        items: list[FlowBatchInput],
    ) -> list[torch.Tensor]:
        """Flow for the step: rows with room in the prefix pool run over their
        new frames against their cached prefix, the rest over their whole
        history as one plain call."""
        if self.vocoder.flow.prefix_pool is None:
            return self.vocoder.hop_batch(items)
        else:
            pass
        cached: list[int] = []
        caches: list[tuple[PrefixCacheRow, PrefixCacheRow]] = []
        plain: list[int] = []
        token_mel_ratio = self.vocoder.flow.token_mel_ratio
        for index, ((_, state), item) in enumerate(
            zip(participants, items, strict=True)
        ):
            total_frames = causal_hop_frames(item, token_mel_ratio=token_mel_ratio)
            cache = state.flow_cache
            if cache is None:
                cache = self.vocoder.prefix_cache_rows(total_frames)
            elif not self.vocoder.grow_prefix_cache(cache, total_frames):
                # Note (Jiaxin Deng): a row the pool cannot hold runs over its
                # whole history this hop and may re-enter the pool on a later one.
                self.vocoder.release_prefix_cache(cache)
                cache = None
            else:
                pass
            state.flow_cache = cache
            if cache is None:
                plain.append(index)
            else:
                cached.append(index)
                caches.append(cache)
        mels: list[torch.Tensor | None] = [None] * len(items)
        if cached:
            outputs = self.vocoder.hop_batch_prefix(
                [items[index] for index in cached], caches
            )
            for index, mel in zip(cached, outputs, strict=True):
                mels[index] = mel
        else:
            pass
        if plain:
            outputs = self.vocoder.hop_batch([items[index] for index in plain])
            for index, mel in zip(plain, outputs, strict=True):
                mels[index] = mel
        else:
            pass
        return [mel for mel in mels if mel is not None]

    def hift_step(
        self,
        participants: list[tuple[str, CosyVoice3StreamState]],
        mels: list[torch.Tensor],
        *,
        is_final: bool,
    ) -> HiftStepSamples:
        """Append each participant's new mel frames to its history and run one
        HiFT call over the step."""
        rows: list[HiftStepRow] = []
        token_mel_ratio = self.vocoder.flow.token_mel_ratio
        for (_, state), mel in zip(participants, mels, strict=True):
            new_frames = mel[:, :, state.token_offset * token_mel_ratio :].detach()
            if state.hift_mel is None:
                state.hift_mel = new_frames
            else:
                state.hift_mel = torch.cat(
                    [state.hift_mel.to(new_frames.device), new_frames], dim=2
                )
            rows.append(
                HiftStepRow(
                    history=state.hift_mel,
                    emitted_samples=state.speech_offset,
                    is_final=is_final,
                )
            )
        return self.vocoder.hift_step(rows)

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
    ) -> dict[str, str | int | dict[str, int | float]]:
        final_data: dict[str, str | int | dict[str, int | float]] = {
            "modality": "audio",
            "sample_rate": self.sample_rate,
        }
        pipeline_state = FunCosyVoice3State.from_dict(payload.data)
        final_data["finish_reason"] = pipeline_state.finish_reason
        usage = build_usage(pipeline_state)
        if usage is None:
            return final_data
        else:
            final_data["usage"] = usage
            return final_data

    def stream_payload(
        self, request_id: str, waveform: torch.Tensor
    ) -> dict[str, bytes | list[int] | str | int]:
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
        self.vocoder.release_prefix_cache(state.flow_cache)
        state.flow_cache = None
        state.prompt_token = None
        state.prompt_feat = None
        state.embedding = None
