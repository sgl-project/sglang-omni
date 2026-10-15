# SPDX-License-Identifier: Apache-2.0
"""Own session-local flow and vocoder state."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field

import numpy as np
import torch

from sglang_omni.models.minicpm_o.components.code2wav import (
    OUTPUT_SAMPLE_RATE,
    MiniCPMOCode2Wav,
)
from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    SILENCE_TOKEN_ID,
    SpeakerPrompt,
    StreamCaches,
    StreamChunk,
    Token2Wav,
)
from sglang_omni.proto.session import ResourceUsage
from sglang_omni.scheduling.speaker_cache import estimate_cache_bytes

SILENCE_PREFIX_LENGTH = 3
CODEC_CHUNK_SIZE = 25
# note (Junnan Li): A forward holds its streams' flow caches once more in its batch window and once more in the copies taken out of it.
FORWARD_CACHE_COPIES = 2
# note (Junnan Li): A graph replays a chunk forward's kernels in one launch, but wider forwards gain little and each graphed stream holds a static attention window.
CHUNK_GRAPH_MAX_STREAMS = 8


def clone_caches(caches: StreamCaches) -> StreamCaches:
    flow_cache, hift_cache = caches
    return (
        {name: tensor.clone() for name, tensor in flow_cache.items()},
        {name: tensor.clone() for name, tensor in hift_cache.items()},
    )


@dataclass(kw_only=True, frozen=True)
class SynthesisRequest:
    session_id: str
    codec_token_ids: list[int]
    is_turn_start: bool
    end_of_turn: bool


@dataclass(kw_only=True)
class SharedSpeaker:
    """Stream caches prefilled from one reference voice, shared by its open sessions."""

    reference_key: str
    prompt: SpeakerPrompt
    # note (Junnan Li): Every turn decodes on a copy, so these caches are never written after prefill.
    base_caches: StreamCaches
    session_ids: list[str] = field(default_factory=list)


@dataclass(kw_only=True)
class MiniCPMOVocoderSessionState:
    speaker: SharedSpeaker
    caches: StreamCaches
    pre_lookahead_tokens: int
    pending_codec_token_ids: list[int] = field(
        default_factory=lambda: [SILENCE_TOKEN_ID] * SILENCE_PREFIX_LENGTH
    )
    has_pending_turn: bool = False

    def held(self) -> ResourceUsage:
        size = estimate_cache_bytes(
            (self.caches, self.pending_codec_token_ids)
        ) + estimate_cache_bytes(self.speaker.base_caches)
        return ResourceUsage(slots={"tts": 1}, bytes=size)


class MiniCPMOVocoderRuntime:
    """Own streaming vocoder state independently per session."""

    def __init__(
        self,
        code2wav: MiniCPMOCode2Wav,
        *,
        max_state_bytes_per_session: int,
        max_open_sessions: int,
    ) -> None:
        self.code2wav = code2wav
        self.max_state_bytes_per_session = max_state_bytes_per_session
        self.max_open_sessions = max_open_sessions
        self.token2wav: Token2Wav = code2wav.token2wav
        self.sessions: dict[str, MiniCPMOVocoderSessionState] = {}
        self.speakers: dict[str, SharedSpeaker] = {}

    def open_session(
        self, session_id: str, *, reference_audio: bytes
    ) -> MiniCPMOVocoderSessionState:
        if session_id in self.sessions:
            raise ValueError(f"TTS session {session_id!r} is already open")
        else:
            pass
        reference_key, _ = self.code2wav.resolve_reference_key(reference_audio)
        speaker = self.speakers.get(reference_key)
        if speaker is None:
            (prompt,) = self.code2wav.prepare_references([reference_audio])
            # note (Dayuxiaoshui): references are prepared on the codec's decode stream.
            device_module = torch.get_device_module(self.token2wav.device)
            device_module.current_stream().wait_stream(self.code2wav.decode_stream)
            speaker = SharedSpeaker(
                reference_key=reference_key,
                prompt=prompt,
                base_caches=self.token2wav.open_stream(prompt),
            )
            self.speakers[reference_key] = speaker
        else:
            pass
        speaker.session_ids.append(session_id)
        state = MiniCPMOVocoderSessionState(
            speaker=speaker,
            caches=clone_caches(speaker.base_caches),
            pre_lookahead_tokens=self.token2wav.flow.pre_lookahead_len,
        )
        self.sessions[session_id] = state
        return state

    def synthesize_batch(
        self, requests: list[SynthesisRequest]
    ) -> Iterator[tuple[int, np.ndarray | None]]:
        """Synthesize one unit of each of several distinct sessions, round by round.

        Yields each request's index and waveform as soon as its last chunk is decoded.
        """
        states = [self.sessions[request.session_id] for request in requests]
        plans: list[list[tuple[list[int], bool]]] = []
        for request, state in zip(requests, states, strict=True):
            if request.codec_token_ids:
                state.has_pending_turn = True
            else:
                pass
            if state.has_pending_turn:
                plans.append(
                    self.plan_chunks(
                        state,
                        request.codec_token_ids,
                        force_flush=request.is_turn_start,
                        is_last_chunk=request.end_of_turn,
                    )
                )
            else:
                plans.append([])
        pcm_chunks: list[list[bytes]] = [[] for _ in requests]

        def finish(index: int) -> np.ndarray | None:
            pcm = b"".join(pcm_chunks[index])
            if not pcm:
                waveform = None
            else:
                waveform = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
                if (
                    not requests[index].end_of_turn
                    and waveform.size < OUTPUT_SAMPLE_RATE
                ):
                    waveform = np.pad(waveform, (OUTPUT_SAMPLE_RATE - waveform.size, 0))
                else:
                    pass
            if requests[index].end_of_turn:
                self.reset_turn_state(states[index])
            else:
                pass
            return waveform

        for index, plan in enumerate(plans):
            if not plan:
                yield index, finish(index)
            else:
                pass
        speaking_states = [state for state, plan in zip(states, plans) if plan]
        if speaking_states:
            width = self.streams_per_forward(speaking_states)
        else:
            width = 1
        for round_index in range(max(map(len, plans), default=0)):
            round_indices = [
                index for index, plan in enumerate(plans) if round_index < len(plan)
            ]
            for start in range(0, len(round_indices), width):
                forward_indices = round_indices[start : start + width]
                decoded = self.token2wav.stream_batch(
                    [
                        StreamChunk(
                            token_ids=plans[index][round_index][0],
                            prompt=states[index].speaker.prompt,
                            caches=states[index].caches,
                            is_last_chunk=plans[index][round_index][1],
                        )
                        for index in forward_indices
                    ]
                )
                for index, (pcm, caches) in zip(forward_indices, decoded, strict=True):
                    pcm_chunks[index].append(pcm)
                    states[index].caches = caches
                    if round_index == len(plans[index]) - 1:
                        yield index, finish(index)
                    else:
                        pass

    def streams_per_forward(self, states: list[MiniCPMOVocoderSessionState]) -> int:
        unheld_bytes = sum(
            self.max_state_bytes_per_session - self.held(session_id).bytes
            for session_id in self.sessions
        )
        forward_bytes_per_stream = FORWARD_CACHE_COPIES * max(
            estimate_cache_bytes(state.caches) for state in states
        )
        if forward_bytes_per_stream == 0:
            return len(states)
        else:
            return max(1, unheld_bytes // forward_bytes_per_stream)

    def warm_up(self, reference_audio: bytes) -> None:
        """Capture the flow's chunk graphs for this voice, then decode a stream prefill and an equal-length and a ragged chunk forward of every width a forward can reach."""
        # note (Junnan Li): A forward wider than every graph runs eagerly, and a compiled estimator compiles its equal and ragged forms once each.
        if self.max_open_sessions > CHUNK_GRAPH_MAX_STREAMS:
            forward_stream_counts = (2, CHUNK_GRAPH_MAX_STREAMS + 1)
        else:
            forward_stream_counts = (2,)
        # note (Junnan Li): Every stream decodes a steady chunk in the first round; in the second the last stream decodes only its turn's lookahead tail, so that forward is ragged.
        batches = [
            [
                SynthesisRequest(
                    session_id=f"warm-up-{stream_count}-continuing-{index}",
                    codec_token_ids=[SILENCE_TOKEN_ID] * (2 * CODEC_CHUNK_SIZE),
                    is_turn_start=False,
                    end_of_turn=False,
                )
                for index in range(stream_count - 1)
            ]
            + [
                SynthesisRequest(
                    session_id=f"warm-up-{stream_count}-ending",
                    codec_token_ids=[SILENCE_TOKEN_ID] * CODEC_CHUNK_SIZE,
                    is_turn_start=False,
                    end_of_turn=True,
                )
            ]
            for stream_count in forward_stream_counts
        ]
        requests = [request for batch in batches for request in batch]
        for request in requests:
            self.open_session(request.session_id, reference_audio=reference_audio)
        speaker = self.sessions[requests[0].session_id].speaker
        up_rate = self.token2wav.flow.up_rate
        widest_graph = min(CHUNK_GRAPH_MAX_STREAMS, self.max_open_sessions)
        # note (Junnan Li): A steady chunk decodes CODEC_CHUNK_SIZE tokens; a turn's last chunk also decodes its lookahead, at most one token short of a full window.
        self.token2wav.capture_chunk_graphs(
            speaker.prompt,
            speaker.base_caches,
            stream_counts=tuple(
                2**power
                for power in range(widest_graph.bit_length())
                if 2**power < widest_graph
            )
            + (widest_graph,),
            frame_counts=(
                CODEC_CHUNK_SIZE * up_rate,
                (CODEC_CHUNK_SIZE + self.token2wav.flow.pre_lookahead_len - 1)
                * up_rate,
            ),
        )
        for batch in batches:
            list(self.synthesize_batch(batch))
        for request in requests:
            self.close_session(request.session_id)

    def warm_up_vocoder(self) -> None:
        # note (Junnan Li): A forward takes at most every open session and a chunk holds at most one window of tokens, and cuDNN keeps its plans per thread.
        flow = self.token2wav.flow
        self.token2wav.warm_up_vocoder(
            self.max_open_sessions,
            flow.up_rate * (CODEC_CHUNK_SIZE + flow.pre_lookahead_len),
        )

    def close_session(self, session_id: str) -> None:
        speaker = self.sessions.pop(session_id).speaker
        speaker.session_ids.remove(session_id)
        if not speaker.session_ids:
            self.speakers.pop(speaker.reference_key)
        else:
            pass

    def held(self, session_id: str) -> ResourceUsage:
        return self.sessions[session_id].held()

    def plan_chunks(
        self,
        state: MiniCPMOVocoderSessionState,
        token_ids: list[int],
        *,
        force_flush: bool,
        is_last_chunk: bool,
    ) -> list[tuple[list[int], bool]]:
        """Consume pending codec tokens into (token ids, is last chunk) decode steps."""
        state.pending_codec_token_ids.extend(token_ids)
        plan: list[tuple[list[int], bool]] = []
        minimum_flush_tokens = state.pre_lookahead_tokens + 5
        window_tokens = CODEC_CHUNK_SIZE + state.pre_lookahead_tokens

        if force_flush:
            while len(state.pending_codec_token_ids) >= minimum_flush_tokens:
                window_length = min(window_tokens, len(state.pending_codec_token_ids))
                plan.append((state.pending_codec_token_ids[:window_length], False))
                consumed_tokens = min(
                    CODEC_CHUNK_SIZE, window_length - state.pre_lookahead_tokens
                )
                del state.pending_codec_token_ids[:consumed_tokens]
        else:
            while len(state.pending_codec_token_ids) >= window_tokens:
                plan.append((state.pending_codec_token_ids[:window_tokens], False))
                del state.pending_codec_token_ids[:CODEC_CHUNK_SIZE]

        if is_last_chunk and state.pending_codec_token_ids:
            plan.append((list(state.pending_codec_token_ids), True))
            state.pending_codec_token_ids.clear()
        else:
            pass
        return plan

    def reset_turn_state(self, state: MiniCPMOVocoderSessionState) -> None:
        state.has_pending_turn = False
        state.pending_codec_token_ids = [SILENCE_TOKEN_ID] * SILENCE_PREFIX_LENGTH
        state.caches = clone_caches(state.speaker.base_caches)


__all__ = [
    "CODEC_CHUNK_SIZE",
    "SILENCE_PREFIX_LENGTH",
    "MiniCPMOVocoderRuntime",
    "MiniCPMOVocoderSessionState",
    "SharedSpeaker",
    "SynthesisRequest",
]
