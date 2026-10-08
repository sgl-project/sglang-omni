"""Code2Wav scheduler: streaming vocoder with inbox/outbox interface.

Receives codec code chunks via inbox (stream_chunk), takes every queued chunk before
it decodes, then decodes the ready windows of one length in one replay.
"""

from __future__ import annotations

import itertools
import json
import logging
import queue
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TypedDict

import numpy as np
import torch
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeCode2Wav,
)

from sglang_omni.models.qwen3_omni.components.code2wav import Qwen3OmniCode2Wav
from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
    Code2WavCudaGraphRunner,
    Code2WavRunResult,
    GraphKey,
)
from sglang_omni.platforms import current_platform
from sglang_omni.profiler.event_recorder import emit as _emit_event
from sglang_omni.profiler.event_recorder import get_recorder as _get_event_recorder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.message import IncomingMessage, OutgoingMessage
from sglang_omni.scheduling.streaming_vocoder import (
    StreamingVocoderBase,
    vocoder_decode_stream_priority,
)
from sglang_omni.utils.audio_payload import audio_waveform_payload
from sglang_omni.utils.cuda_staging import PinnedTransferSlot
from sglang_omni.utils.snake_beta import fuse_vocoder_decoder

logger = logging.getLogger(__name__)


class IngestProfile(TypedDict):
    run_id: str | None
    messages: int
    accepted_frames: int
    ingest_host_ns: int
    eos_check_host_ns: int
    eos_checks: int
    started_with_frames: int
    ready_emitted: bool


class ExecutionMetadata(TypedDict):
    execution_mode: str
    graph_key: dict[str, int] | None
    fallback_reason: str | None


def serial_window_frames(
    stream_chunk_size: int, left_context_size: int, initial_chunk_frames: int = 0
) -> tuple[int, ...]:
    """Window lengths the serial walk actually visits, in visit order.

    Each decode advances by one chunk while its context grows to the
    configured cap. A configured ``initial_chunk_frames`` makes the first
    decode advance by less than a chunk, which offsets every window until the
    context saturates, so the walk is replayed here rather than assumed: keys
    derived from the wrong offset miss on exactly the windows that carry
    time-to-first-audio, and a missed key runs eager.
    """
    initial = min(max(int(initial_chunk_frames), 0), stream_chunk_size)
    steady = left_context_size + stream_chunk_size
    frames: list[int] = []
    seen: set[int] = set()
    emitted = 0
    for _ in range(left_context_size + 2):
        step = initial if emitted == 0 and initial else stream_chunk_size
        window = min(left_context_size, emitted) + step
        if window not in seen:
            seen.add(window)
            frames.append(window)
        else:
            pass
        emitted += step
        if window == steady:
            break
        else:
            pass
    return tuple(frames)


def final_window_frames(
    stream_chunk_size: int, left_context_size: int, initial_chunk_frames: int = 0
) -> tuple[int, ...]:
    """Window lengths a finished stream's last decode can take, in walk order.

    A stream ends 1 to step - 1 frames past its last threshold window, read with
    the context it holds by then.
    """
    initial = min(max(int(initial_chunk_frames), 0), stream_chunk_size)
    frames: list[int] = []
    emitted = 0
    while True:
        step = initial if emitted == 0 and initial else stream_chunk_size
        context = min(left_context_size, emitted)
        for new_frames in range(1, step):
            if context + new_frames not in frames:
                frames.append(context + new_frames)
            else:
                pass
        if context + step == left_context_size + stream_chunk_size:
            break
        else:
            pass
        emitted += step
    return tuple(frames)


def window_graph_keys(
    stream_chunk_size: int,
    left_context_size: int,
    max_replay_rows: int,
    initial_chunk_frames: int = 0,
) -> tuple[GraphKey, ...]:
    """Every window length the walk visits, at every row count up to max_replay_rows."""
    return tuple(
        GraphKey(batch_size=rows, frames=frames)
        for rows in range(1, max_replay_rows + 1)
        for frames in serial_window_frames(
            stream_chunk_size, left_context_size, initial_chunk_frames
        )
    )


def load_code2wav_model(
    model_path: str, *, device: str = "cuda", dtype: str | None = None
) -> Qwen3OmniCode2Wav:
    """Load Code2Wav model from HF checkpoint."""
    from transformers import AutoConfig

    from sglang_omni.models.weight_loader import load_module, resolve_dtype

    torch_dtype = resolve_dtype(dtype)
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    code2wav_config = config.code2wav_config
    model = Qwen3OmniCode2Wav._from_config(code2wav_config)  # noqa: leading-underscore
    model = load_module(
        model,
        model_path,
        prefix="code2wav.",
        dtype=torch_dtype,
        device=device,
        strict=False,
    )
    if current_platform.is_cuda() and torch.device(device).type == "cuda":
        model.use_channels_last()
        if model.dtype in (torch.bfloat16, torch.float16):
            model.use_fused_transformer(
                current_platform.get_joint_rope_inplace_kernel()
            )
        else:
            pass
    else:
        pass
    return model.eval()


@dataclass
class PendingWindow:
    """Depth-2 pipeline slot reference held by a stream state.

    The sample count is recorded at launch so the flush never has to read a
    shape back from the device.
    """

    slot: PinnedTransferSlot
    samples: int
    launch_index: int


@dataclass
class RetiredChunks:
    """A dropped request's codec chunks, held until the event recorded after
    their last queued window read completes; freeing them earlier would let the
    producer reuse memory a window is still reading."""

    event: torch.cuda.Event
    chunks: list[torch.Tensor]


@dataclass
class Code2WavStreamState:
    chunks: list[torch.Tensor] = field(default_factory=list)
    emitted: int = 0
    audio_parts: list[np.ndarray] = field(default_factory=list)
    stream_enabled: bool | None = None
    ready_since: float | None = None
    checked: int = 0
    pending: PendingWindow | None = None
    codes_ready_event: torch.Event | None = None
    _critical_ingest_profile: IngestProfile | None = None


class Code2WavScheduler(StreamingVocoderBase[Code2WavStreamState, "list[int]"]):
    """Streaming vocoder scheduler. Same inbox/outbox interface as OmniScheduler."""

    MAX_PINNED_SLOTS = 32

    def __init__(
        self,
        model: Qwen3OmniMoeCode2Wav,
        device: str,
        stream_chunk_size: int = 10,
        left_context_size: int = 25,
        sample_rate: int = 24000,
        codec_eos_token_id: int = 2150,
        initial_codec_chunk_frames: int = 0,
        max_replay_rows: int = 8,
        enable_output_overlap: bool = True,
        enable_cuda_graph: bool = False,
        cuda_graph_runner: Code2WavCudaGraphRunner | None = None,
        decode_stream: torch.Stream | None = None,
    ) -> None:
        self.model = model
        self.device = torch.device(device)
        self.decode_stream = decode_stream
        self.stream_chunk_size = max(int(stream_chunk_size), 1)
        self.left_context_size = max(int(left_context_size), 0)
        self.codec_eos_token_id = codec_eos_token_id
        self.total_upsample = int(model.total_upsample)
        self.cuda_graph_runner = cuda_graph_runner if bool(enable_cuda_graph) else None
        super().__init__(
            None, sample_rate=sample_rate, stream_source_hint="Qwen3-Omni code2wav"
        )
        self.initial_codec_chunk_frames = min(
            max(int(initial_codec_chunk_frames), 0), int(stream_chunk_size)
        )
        self.max_replay_rows = max(int(max_replay_rows), 1)
        self.enable_output_overlap = bool(enable_output_overlap)
        self.eos_lazy_scan = self.enable_output_overlap
        self.pipeline_active = self.eos_lazy_scan and self.device.type == "cuda"
        self.default_slot_samples = self.stream_chunk_size * self.total_upsample
        self.pinned_free: list[PinnedTransferSlot] = []
        self.pinned_created = 0
        self.window_launch_count = 0
        self.pinned_retired: list[PinnedTransferSlot] = []
        self.pinned_quarantined: list[PinnedTransferSlot] = []
        self.retired_chunks: list[RetiredChunks] = []
        self.max_pinned_slots = self.MAX_PINNED_SLOTS + self.max_replay_rows

    def on_serving_start(self) -> None:
        """Every device op of the serving thread runs on the decode stream."""
        if self.decode_stream is not None:
            torch.get_device_module(self.device).set_stream(self.decode_stream)
        else:
            pass

    def wait_codes_ready(self, state: Code2WavStreamState) -> None:
        """Order the decode stream after the producer of the request's newest chunk."""
        if self.decode_stream is None:
            pass
        elif state.codes_ready_event is None:
            # note (ratish): a chunk from another process is made ready on the
            # receiving thread's default stream.
            self.decode_stream.wait_stream(
                torch.get_device_module(self.device).default_stream(self.device)
            )
        else:
            self.decode_stream.wait_event(state.codes_ready_event)

    def is_streaming_payload(self, payload: StagePayload) -> bool:
        del payload
        return True

    def create_stream_state(self, request_id: str) -> Code2WavStreamState:
        del request_id
        return Code2WavStreamState()

    def latch_stream_contract(
        self,
        request_id: str,
        state: Code2WavStreamState,
        source: StagePayload | Mapping[str, object],
        *,
        origin: str,
    ) -> None:
        del request_id
        if origin != "stream metadata":
            return
        else:
            pass
        state.codes_ready_event = source.get("codes_ready_event")
        if state.stream_enabled is None:
            state.stream_enabled = bool(source["stream"])
        else:
            pass

    def validate_chunk(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> torch.Tensor:
        del request_id, state
        return codes.to(device=self.device, dtype=torch.long)

    def ingest(
        self, request_id: str, state: Code2WavStreamState, codes: torch.Tensor
    ) -> None:
        profile = None
        if _get_event_recorder().is_active():
            profile = self.start_ingest_profile(state)
        else:
            pass
        if codes.ndim == 2:
            state.chunks.extend(codes.unbind(0))
            state.checked = len(state.chunks)
        elif self.eos_lazy_scan:
            state.chunks.append(codes)
        elif codes.ndim >= 1:
            self.wait_codes_ready(state)
            if profile is None:
                is_eos = codes[0].item() == self.codec_eos_token_id
            else:
                eos_start_ns = time.perf_counter_ns()
                is_eos = codes[0].item() == self.codec_eos_token_id
                profile[0]["eos_check_host_ns"] += time.perf_counter_ns() - eos_start_ns
                profile[0]["eos_checks"] += 1
            if not is_eos:
                state.chunks.append(codes)
            else:
                pass
        else:
            state.chunks.append(codes)
        if profile is not None:
            self.finish_ingest_profile(request_id, state, profile)
        else:
            pass

    def start_ingest_profile(
        self, state: Code2WavStreamState
    ) -> tuple[IngestProfile, int, int, int] | None:
        """Called only with event recording active; stop timing after first readiness."""
        if state.emitted > 0:
            return None
        else:
            pass
        run_id = _get_event_recorder().active_run_id()
        profile: IngestProfile | None = (
            state._critical_ingest_profile
        )  # noqa: leading-underscore
        if profile is None or profile["run_id"] != run_id:
            profile = {
                "run_id": run_id,
                "messages": 0,
                "accepted_frames": 0,
                "ingest_host_ns": 0,
                "eos_check_host_ns": 0,
                "eos_checks": 0,
                "started_with_frames": len(state.chunks),
                "ready_emitted": False,
            }
            state._critical_ingest_profile = profile  # noqa: leading-underscore
        else:
            pass
        if profile["ready_emitted"]:
            return None
        else:
            pass
        return (profile, time.time_ns(), time.perf_counter_ns(), len(state.chunks))

    def finish_ingest_profile(
        self,
        request_id: str,
        state: Code2WavStreamState,
        context: tuple[IngestProfile, int, int, int],
    ) -> None:
        profile, start_wall_ns, start_ns, before_frames = context
        profile["messages"] += 1
        profile["accepted_frames"] += len(state.chunks) - before_frames
        profile["ingest_host_ns"] += time.perf_counter_ns() - start_ns
        threshold = self.window_new_frames(state)
        first_ingest = profile["messages"] == 1
        ready = self.ready(state) >= threshold
        if not (first_ingest or ready):
            return
        else:
            pass
        metadata = {
            key: profile[key]
            for key in (
                "messages",
                "accepted_frames",
                "ingest_host_ns",
                "eos_check_host_ns",
                "eos_checks",
                "started_with_frames",
            )
        }
        metadata.update(
            ready_frames=self.ready(state),
            threshold_frames=threshold,
            eos_scan_deferred=self.eos_lazy_scan,
            inbox_depth=self.inbox.qsize(),
            pending_message_depth=len(self.pending_messages),
            active_request_count=len(self.stream_states),
        )
        if first_ingest:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_ingest",
                timestamp_ns=start_wall_ns,
                metadata=metadata,
            )
        else:
            pass
        if ready:
            profile["ready_emitted"] = True
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_window_ready",
                metadata=metadata,
            )
        else:
            pass

    def window_new_frames(self, state: Code2WavStreamState) -> int:
        """Frames the stream's next threshold window decodes past its left context."""
        if state.emitted == 0 and self.initial_codec_chunk_frames:
            return self.initial_codec_chunk_frames
        else:
            return self.stream_chunk_size

    def window_ready(self, state: Code2WavStreamState) -> bool:
        new_frames = self.window_new_frames(state)
        if (
            self.eos_lazy_scan
            and state.checked < len(state.chunks)
            and self.ready(state) >= new_frames
        ):
            self.scan_unchecked(state)
        else:
            pass
        return self.ready(state) >= new_frames

    def scan_unchecked(self, state: Code2WavStreamState) -> None:
        """Batched EOS scan over frames staged by the lazy-ingest path.

        Replays the eager per-frame check exactly: every staged frame whose
        leading code equals the codec EOS id is dropped and every other frame
        keeps its arrival order (0-dim frames bypass the check, as in the
        eager path). One host sync per scan instead of one per frame.

        The producer emits 1-D ``[num_quantizers]`` frames; any other rank
        falls back to the per-frame path rather than stacking into a nested
        truthy list that would silently drop malformed chunks.
        """
        chunks = state.chunks
        start = state.checked
        if start >= len(chunks):
            return
        else:
            pass
        unchecked = chunks[start:]
        self.wait_codes_ready(state)
        if all((codes.ndim == 1 for codes in unchecked)):
            heads = torch.stack([codes[0] for codes in unchecked])
            is_eos = (heads == self.codec_eos_token_id).tolist()
        else:
            is_eos = [
                codes.ndim >= 1 and codes[0].item() == self.codec_eos_token_id
                for codes in unchecked
            ]
        if any(is_eos):
            chunks[start:] = [codes for codes, eos in zip(unchecked, is_eos) if not eos]
        else:
            pass
        state.checked = len(chunks)

    def decode_delta(
        self, request_id: str, state: Code2WavStreamState, *, is_final: bool
    ) -> torch.Tensor | None:
        """The frames a finished stream has left, as one window decoded alone."""
        del is_final
        if self.eos_lazy_scan:
            self.scan_unchecked(state)
        else:
            pass
        start, end = (state.emitted, len(state.chunks))
        if start >= end:
            return None
        else:
            pass
        context = min(self.left_context_size, start)
        profile_metadata = self.window_profile(
            state, "stream_done", end - start, rows=1
        )
        if profile_metadata is not None:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_start",
                metadata=profile_metadata,
            )
        else:
            pass
        self.wait_codes_ready(state)
        window = torch.stack(state.chunks[start - context : end], dim=0)
        codes = window.transpose(0, 1).unsqueeze(0)
        wav, execution_metadata = self.forward_codes(codes, graph_eligible=True)
        wav = wav[..., -(end - start) * self.total_upsample :]
        audio = wav.reshape(-1).detach().cpu().float().numpy().copy()
        if profile_metadata is not None:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_decode_end",
                metadata={
                    **profile_metadata,
                    "audio_samples": int(audio.shape[0]),
                    **execution_metadata,
                    **self.overlap_profile(False, 0),
                },
            )
        else:
            pass
        state.emitted = end
        state.ready_since = None
        return self.keep_audio(request_id, state, audio)

    def window_profile(
        self, state: Code2WavStreamState, trigger: str, new_frames: int, *, rows: int
    ) -> dict[str, str | int] | None:
        """Request profiler metadata of one window, None while no profile runs."""
        if not _get_event_recorder().is_active():
            return None
        else:
            pass
        context = min(self.left_context_size, state.emitted)
        return {
            "trigger": trigger,
            "start_frame": state.emitted,
            "end_frame": state.emitted + new_frames,
            "new_frames": new_frames,
            "context_frames": context,
            "window_frames": new_frames + context,
            "rows": rows,
            "active_request_count": len(self.stream_states),
            "threshold_ready_request_count": sum(
                self.ready(other) >= self.window_new_frames(other)
                for _, other in self.stream_state_items()
            ),
            "inbox_depth": self.inbox.qsize(),
            "pending_message_depth": len(self.pending_messages),
        }

    def overlap_profile(self, pipelined: bool, wait_ns: int) -> dict[str, bool | int]:
        """Output overlap's end-event keys: whether the window copies asynchronously, and the
        wait for the request's previous window; none while overlap is off."""
        if self.pipeline_active:
            return {"pipelined": pipelined, "d2h_wait_ns": wait_ns}
        else:
            return {}

    def decode_windows(
        self, participants: list[tuple[str, Code2WavStreamState]]
    ) -> list[OutgoingMessage]:
        """Decode one threshold window of every participant, all of one length, in one replay.

        First windows reach the host at once; later ones copy into a pinned slot each and leave when
        their copy completes, after the request's previous window.
        """
        new_frames = self.window_new_frames(participants[0][1])
        profiles = [
            self.window_profile(state, "threshold", new_frames, rows=len(participants))
            for _, state in participants
        ]
        rows = []
        for (request_id, state), profile_metadata in zip(participants, profiles):
            if profile_metadata is not None:
                _emit_event(
                    request_id=request_id,
                    stage=None,
                    event_name="code2wav_decode_start",
                    metadata=profile_metadata,
                )
            else:
                pass
            start = state.emitted
            context = min(self.left_context_size, start)
            self.wait_codes_ready(state)
            rows.append(
                torch.stack(state.chunks[start - context : start + new_frames], dim=1)
            )
        wav, execution_metadata = self.forward_codes(
            torch.stack(rows, dim=0), graph_eligible=True
        )
        wav = wav[..., -new_frames * self.total_upsample :].reshape(
            len(participants), -1
        )
        pipelined = self.pipeline_active and participants[0][1].emitted > 0
        if pipelined:
            host_rows = None
            device_rows = wav.to(torch.float32)
        else:
            host_rows = wav.detach().cpu().float().numpy()
            device_rows = None
        messages: list[OutgoingMessage] = []
        for row, ((request_id, state), profile_metadata) in enumerate(
            zip(participants, profiles)
        ):
            state.emitted += new_frames
            state.ready_since = None
            wait_ns = 0
            if host_rows is None:
                waveforms, wait_ns = self.stage_window(
                    request_id, state, device_rows[row]
                )
                if profile_metadata is not None and state.pending is not None:
                    _emit_event(
                        request_id=request_id,
                        stage=None,
                        event_name="code2wav_decode_launched",
                        metadata={
                            **execution_metadata,
                            "window_frames": profile_metadata["window_frames"],
                            "new_frames": new_frames,
                            "rows": len(participants),
                        },
                    )
                else:
                    pass
            else:
                waveforms = [self.keep_audio(request_id, state, host_rows[row].copy())]
            for waveform in waveforms:
                if waveform is not None:
                    self.mark_stream_emitted(request_id)
                    messages.append(self.stream_chunk_message(request_id, waveform))
                else:
                    pass
            if profile_metadata is not None:
                _emit_event(
                    request_id=request_id,
                    stage=None,
                    event_name="code2wav_decode_end",
                    metadata={
                        **profile_metadata,
                        "audio_samples": int(wav.shape[1]),
                        **execution_metadata,
                        **self.overlap_profile(
                            pipelined and state.pending is not None, wait_ns
                        ),
                    },
                )
            else:
                pass
        return messages

    def stage_window(
        self, request_id: str, state: Code2WavStreamState, waveform: torch.Tensor
    ) -> tuple[list[torch.Tensor | None], int]:
        """Copy one decoded window toward the host through a pinned slot.

        Returns the waveforms to send now, in order (the request's previous window, then this one
        when no slot is free and it copied at once), and the wait for the previous window's copy.
        """
        samples = int(waveform.numel())
        ready: list[torch.Tensor | None] = []
        wait_ns = 0
        slot = self.acquire_slot(samples)
        if slot is None and state.pending is not None:
            wait_ns, previous = self.flush_pending(request_id, state)
            ready.append(previous)
            slot = self.acquire_slot(samples)
        else:
            pass
        if slot is None:
            audio = waveform.detach().cpu().numpy().copy()
            ready.append(self.keep_audio(request_id, state, audio))
            return (ready, wait_ns)
        else:
            pass
        event_recorded = False
        try:
            slot.view(samples).copy_(waveform, non_blocking=True)
            slot.record(torch.cuda.current_stream(self.device))
            event_recorded = True
            if state.pending is not None:
                wait_ns, previous = self.flush_pending(request_id, state)
                ready.append(previous)
            else:
                pass
        except Exception:
            if event_recorded:
                self.retire_slot(slot)
            else:
                self.quarantine_slot(slot)
            raise
        self.window_launch_count += 1
        state.pending = PendingWindow(
            slot=slot, samples=samples, launch_index=self.window_launch_count
        )
        return (ready, wait_ns)

    def keep_audio(
        self, request_id: str, state: Code2WavStreamState, audio: np.ndarray
    ) -> torch.Tensor | None:
        """Add a window's audio to its request; returns the waveform to stream, if it streams."""
        if audio.size == 0:
            return None
        else:
            pass
        if not state.audio_parts:
            _emit_event(
                request_id=request_id,
                stage=None,
                event_name="code2wav_first_audio",
                metadata={"samples": int(audio.shape[0])},
            )
        else:
            pass
        state.audio_parts.append(audio)
        if not state.stream_enabled:
            return None
        else:
            pass
        return torch.from_numpy(audio)

    def decode_and_emit(
        self, request_id: str, state: Code2WavStreamState
    ) -> list[OutgoingMessage]:
        """A chunk only sends its request's finished window; windows decode in the ready steps."""
        pending = state.pending
        if pending is not None and pending.slot.query():
            _, waveform = self.flush_pending(request_id, state)
            if waveform is not None:
                self.mark_stream_emitted(request_id)
                return [self.stream_chunk_message(request_id, waveform)]
            else:
                pass
        else:
            pass
        return []

    def flush_pending(
        self, request_id: str, state: Code2WavStreamState
    ) -> tuple[int, torch.Tensor | None]:
        """Materialize the pending window: wait for its D2H copy, append the
        audio, and return (wait_ns, waveform-or-None). The slot returns to the
        pool only after the audio is copied into owned memory."""
        pending = state.pending
        if pending is None:
            return (0, None)
        else:
            pass
        slot = pending.slot
        wait_start = time.monotonic_ns()
        # note (ratish): synchronize() drops the GIL even for a finished copy, and in
        # the talker's process the talker thread can then hold it for a switch interval.
        if not slot.query():
            slot.synchronize()
        else:
            pass
        wait_ns = time.monotonic_ns() - wait_start
        audio = slot.view(pending.samples).numpy().copy()
        state.pending = None
        self.release_slot(slot)
        return (wait_ns, self.keep_audio(request_id, state, audio))

    def acquire_slot(self, samples: int) -> PinnedTransferSlot | None:
        self.reap_retired()
        if self.pinned_free:
            slot = self.pinned_free.pop()
            try:
                slot.ensure_capacity(samples)
            except Exception:
                self.release_slot(slot)
                raise
            return slot
        else:
            pass
        if self.pinned_created < self.max_pinned_slots:
            slot = PinnedTransferSlot(
                self.device,
                torch.float32,
                initial_capacity=max(samples, self.default_slot_samples),
                blocking=True,
            )
            self.pinned_created += 1
            return slot
        else:
            pass
        return None

    def release_slot(self, slot: PinnedTransferSlot) -> None:
        self.pinned_free.append(slot)

    def retire_slot(self, slot: PinnedTransferSlot) -> None:
        self.pinned_retired.append(slot)

    def quarantine_slot(self, slot: PinnedTransferSlot) -> None:
        self.pinned_quarantined.append(slot)
        self.pipeline_active = False

    def reap_retired(self) -> None:
        """Non-blockingly return completed retired slots to the free pool and
        drop retired chunks the device has finished reading.

        Callers hold state_lock. An event-query error leaves the buffer owned
        by the scheduler but permanently unavailable for reuse.
        """
        self.retired_chunks = [
            held for held in self.retired_chunks if not held.event.query()
        ]
        if not self.pinned_retired:
            return
        else:
            pass
        still_retired: list[PinnedTransferSlot] = []
        for slot in self.pinned_retired:
            try:
                complete = slot.query()
            except Exception:
                logger.exception("code2wav failed to query a retired D2H copy")
                self.quarantine_slot(slot)
                continue
            if complete:
                self.release_slot(slot)
            else:
                still_retired.append(slot)
        self.pinned_retired = still_retired

    def final_result_data(
        self, request_id: str, payload: StagePayload, state: Code2WavStreamState
    ) -> dict[str, bytes | list[int] | str | int]:
        del payload
        if not state.audio_parts:
            raise RuntimeError(f"code2wav produced no audio for {request_id!r}")
        else:
            pass
        if state.stream_enabled:
            return {"modality": "audio", "sample_rate": self.sample_rate}
        else:
            pass
        full = np.concatenate(state.audio_parts).astype(np.float32, copy=False)
        return audio_waveform_payload(
            full,
            sample_rate=self.sample_rate,
            modality="audio",
            source_hint="Qwen3-Omni code2wav",
        )

    def forward_codes(
        self, codes: torch.Tensor, *, graph_eligible: bool = False
    ) -> tuple[torch.Tensor, ExecutionMetadata]:
        with torch.no_grad():
            if self.device.type != "cpu":
                torch.get_device_module(self.device).set_device(self.device)
            else:
                pass
            if self.cuda_graph_runner is None:
                result = Code2WavRunResult(
                    output=self.model(codes),
                    execution_mode="eager",
                    key=None,
                    fallback_reason=None,
                )
            else:
                result = self.cuda_graph_runner.run(codes, eligible=graph_eligible)
        graph_key = None
        if result.key is not None:
            graph_key = {
                "batch_size": int(result.key.batch_size),
                "frames": int(result.key.frames),
            }
        else:
            pass
        return (
            result.output,
            {
                "execution_mode": str(result.execution_mode),
                "graph_key": graph_key,
                "fallback_reason": (
                    None
                    if result.fallback_reason is None
                    else str(result.fallback_reason)
                ),
            },
        )

    def next_message(self) -> IncomingMessage | None:
        with self.state_lock:
            self.reap_retired()
            failed = self.emit_completed_windows()
            pending_windows = self.streaming_pending_windows()
            earliest_window = pending_windows[0] if pending_windows else None
        for request_id in failed:
            self.cleanup_aborted_request(request_id)
        if self.pending_messages:
            return self.pending_messages.popleft()
        else:
            pass
        if earliest_window is not None and self.inbox.empty():
            # note (ratish): with nothing queued, sleep on the launched window rather
            # than hold its audio until the stream's next codes; the next pass sends it.
            request_id, pending = earliest_window
            try:
                pending.slot.synchronize()
            except Exception as exc:
                logger.exception(f"Qwen3-Omni code2wav failed waiting on {request_id}")
                self.emit_error(request_id, exc)
                self.abort(request_id)
            return None
        else:
            pass
        try:
            return self.inbox.get(timeout=0.1)
        except queue.Empty:
            return None

    def ready(self, state: Code2WavStreamState) -> int:
        return len(state.chunks) - state.emitted

    def window_length(self, state: Code2WavStreamState) -> int:
        return min(self.left_context_size, state.emitted) + self.window_new_frames(
            state
        )

    def ready_streams(self) -> list[tuple[str, Code2WavStreamState]]:
        """Streams with a threshold window ready, each stamped with when it became ready."""
        now = time.monotonic()
        streams = []
        for request_id, state in self.stream_state_items():
            if not self.is_aborted(request_id) and self.window_ready(state):
                if state.ready_since is None:
                    state.ready_since = now
                else:
                    pass
                streams.append((request_id, state))
            else:
                pass
        return streams

    def has_ready_work(self) -> bool:
        with self.state_lock:
            return bool(self.ready_streams())

    def ready_window_group(self) -> list[tuple[str, Code2WavStreamState]]:
        """The ready windows the next replay decodes: the longest waiting window's length, oldest first."""
        streams = self.ready_streams()
        if not streams:
            return []
        else:
            pass
        # note (ratish): a first window carries its request's time to first audio, so it goes
        # ahead of every later window whatever their wait.
        anchor = min(
            streams, key=lambda stream: (stream[1].emitted > 0, stream[1].ready_since)
        )
        window_frames = self.window_length(anchor[1])
        group = sorted(
            (
                stream
                for stream in streams
                if self.window_length(stream[1]) == window_frames
            ),
            key=lambda stream: stream[1].ready_since,
        )
        if self.cuda_graph_runner is None:
            rows = min(len(group), self.max_replay_rows)
        else:
            rows = max(
                (
                    size
                    for size in self.cuda_graph_runner.available_batch_sizes(
                        window_frames
                    )
                    if size <= len(group)
                ),
                default=1,
            )
        return group[:rows]

    def run_ready_step(self) -> None:
        """One replay over the ready windows of one length, once the inbox holds nothing more."""
        failed: list[str] = []
        with self.state_lock:
            self.reap_retired()
            failed.extend(self.emit_completed_windows())
            participants = self.ready_window_group()
            if participants:
                try:
                    messages = self.decode_windows(participants)
                except Exception as exc:
                    failed.extend(self.on_step_failure(participants, exc))
                else:
                    for message in messages:
                        if not self.is_aborted(message.request_id):
                            self.outbox.put(message)
                        else:
                            pass
            else:
                pass
        for request_id in failed:
            self.cleanup_aborted_request(request_id)

    def streaming_pending_windows(self) -> list[tuple[str, PendingWindow]]:
        """Launched windows of streaming requests that no abort has claimed, in
        launch order. A non-streaming request returns its audio in the final result,
        so sending a window early gains it nothing. Callers hold state_lock."""
        # note (ratish): every window copies on the serving thread's one stream, so
        # launch order is completion order.
        return sorted(
            (
                (request_id, state.pending)
                for request_id, state in self.stream_state_items()
                if state.pending is not None
                and state.stream_enabled
                and not self.is_aborted(request_id)
            ),
            key=lambda request_window: request_window[1].launch_index,
        )

    def emit_completed_windows(self) -> list[str]:
        """Send every streaming window whose host copy has finished. Callers hold
        state_lock and run abort cleanup for the returned failed request ids once
        it is released, as pump_due_streams does."""
        failed: list[str] = []
        for request_id, pending in self.streaming_pending_windows():
            try:
                messages = (
                    self.drain_pending_window(request_id)
                    if pending.slot.query()
                    else []
                )
            except Exception as exc:
                logger.exception(f"Qwen3-Omni code2wav failed to send {request_id}")
                self.emit_error(request_id, exc)
                self.abort_state(request_id)
                failed.append(request_id)
                continue
            for message in messages:
                self.outbox.put(message)
        return failed

    def drain_pending_window(self, request_id: str) -> list[OutgoingMessage]:
        state = self.stream_states.get(request_id)
        if state is None or state.pending is None:
            return []
        else:
            pass
        _, waveform = self.flush_pending(request_id, state)
        if waveform is None:
            return []
        else:
            pass
        self.mark_stream_emitted(request_id)
        return [self.stream_chunk_message(request_id, waveform)]

    def finish_windows(self, request_id: str) -> list[OutgoingMessage]:
        """A finished stream's ready threshold windows decoded alone, then its pending window, in order."""
        messages: list[OutgoingMessage] = []
        state = self.stream_states.get(request_id)
        while state is not None and self.window_ready(state):
            messages.extend(self.decode_windows([(request_id, state)]))
        messages.extend(self.drain_pending_window(request_id))
        return messages

    def on_stream_done(self, request_id: str) -> list[OutgoingMessage]:
        messages = self.finish_windows(request_id)
        messages.extend(super().on_stream_done(request_id))
        return messages

    def on_stream_done_before_payload(self, request_id: str) -> list[OutgoingMessage]:
        messages = self.finish_windows(request_id)
        state = self.stream_states.get(request_id)
        if state is not None:
            waveform = self.decode_delta(request_id, state, is_final=True)
            if waveform is not None:
                self.mark_stream_emitted(request_id)
                messages.append(self.stream_chunk_message(request_id, waveform))
            else:
                pass
        else:
            pass
        return messages

    def release_stream_resources(
        self, request_id: str, state: Code2WavStreamState
    ) -> None:
        del request_id
        if state.pending is not None:
            self.retire_slot(state.pending.slot)
            state.pending = None
        else:
            pass
        if state.chunks and self.device.type == "cuda":
            # note (ratish): windows read the chunks on the decode stream, or on
            # the default stream when the serving thread has none of its own.
            read_stream = self.decode_stream
            if read_stream is None:
                read_stream = torch.cuda.default_stream(self.device)
            else:
                pass
            event = torch.cuda.Event()
            event.record(read_stream)
            self.retired_chunks.append(RetiredChunks(event=event, chunks=state.chunks))
            state.chunks = []
        else:
            pass

    def on_serving_stop(self) -> None:
        """Drain retired slots and chunks at shutdown, when blocking costs no
        latency."""
        retired = self.pinned_retired
        self.pinned_retired = []
        for slot in retired:
            try:
                slot.synchronize()
            except Exception:
                logger.exception(
                    "code2wav failed to synchronize a retired D2H copy on shutdown"
                )
                self.pinned_quarantined.append(slot)
            else:
                self.release_slot(slot)
        for held in self.retired_chunks:
            held.event.synchronize()
        self.retired_chunks = []


def create_code2wav_scheduler(
    model_path: str,
    *,
    device: str | None = None,
    dtype: str | None = None,
    gpu_id: int | None = None,
    stream_chunk_size: int = 10,
    left_context_size: int = 25,
    initial_codec_chunk_frames: int = 0,
    max_replay_rows: int = 8,
    enable_output_overlap: bool = True,
    enable_cuda_graph: bool = False,
    total_gpu_memory_fraction: float | None = None,
    fused_snake_activation: bool = True,
    talker_in_process: bool = False,
) -> Code2WavScheduler:
    """Factory: returns Code2WavScheduler."""
    from sglang_omni.utils.device import resolve_concrete_device

    if enable_cuda_graph and total_gpu_memory_fraction is None:
        raise ValueError(
            "Code2Wav device graph requires gpu_memory_fraction on the code2wav stage"
        )
    else:
        pass
    concrete_device = resolve_concrete_device(device, gpu_id)
    device = str(concrete_device)
    stream_chunk_size = max(int(stream_chunk_size), 1)
    left_context_size = max(int(left_context_size), 0)
    model = load_code2wav_model(model_path, device=device, dtype=dtype)
    if fused_snake_activation:
        replaced = fuse_vocoder_decoder(model.decoder)
        logger.info(f"Code2Wav fused SnakeBeta modules: {replaced}")
    else:
        pass
    decode_stream: torch.Stream | None = None
    # note (ratish): the priority stream only orders code2wav ahead of the
    # talker's stream in the same context; alone in its process code2wav keeps
    # the default stream.
    if talker_in_process and concrete_device.type == "cuda":
        device_module = torch.get_device_module(concrete_device)
        decode_stream = device_module.Stream(
            device=concrete_device,
            priority=vocoder_decode_stream_priority(device_module),
        )
    else:
        pass
    cuda_graph_runner = None
    if enable_cuda_graph:
        graph_keys = window_graph_keys(
            stream_chunk_size,
            left_context_size,
            max(int(max_replay_rows), 1),
            initial_codec_chunk_frames,
        )
        final_window_keys = tuple(
            key
            for key in (
                GraphKey(batch_size=1, frames=frames)
                for frames in final_window_frames(
                    stream_chunk_size, left_context_size, initial_codec_chunk_frames
                )
            )
            if key not in graph_keys
        )
        cuda_graph_runner = Code2WavCudaGraphRunner.build(
            model,
            device=concrete_device,
            num_quantizers=int(model.config.num_quantizers),
            total_gpu_memory_fraction=total_gpu_memory_fraction,
            graph_keys=graph_keys,
            best_effort_keys=final_window_keys,
            model_footprint_bytes=sum(
                tensor.nbytes
                for tensor in itertools.chain(model.parameters(), model.buffers())
            ),
            decode_stream=decode_stream,
        )
        logger.info(
            "Code2Wav device graph startup stats=%s",
            json.dumps(
                cuda_graph_runner.stats(), sort_keys=True, separators=(",", ":")
            ),
        )
    else:
        pass
    return Code2WavScheduler(
        model,
        device=device,
        stream_chunk_size=stream_chunk_size,
        left_context_size=left_context_size,
        initial_codec_chunk_frames=initial_codec_chunk_frames,
        max_replay_rows=max_replay_rows,
        enable_output_overlap=enable_output_overlap,
        enable_cuda_graph=enable_cuda_graph,
        cuda_graph_runner=cuda_graph_runner,
        decode_stream=decode_stream,
    )
