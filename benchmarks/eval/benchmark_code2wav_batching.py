# SPDX-License-Identifier: Apache-2.0
"""Code2Wav-bound component benchmark for the #1126 batching sweep.

The serving-level benchmark cannot saturate the vocoder: #1126's E2E A/B at
c1-8 showed no change because Talker windows rarely align, so batching seldom
fires. This harness drives ``Code2WavScheduler`` directly through its
inbox/outbox contract with synthetic codec streams and sweeps
``max_batch_wait_ms`` x ``batch_floor`` (issue #1026, roadmap #1018 item 4.2)
across four arms:

- ``serial-eager``: batching off, CUDA graph off (#1126's control)
- ``serial-graph``: batching off, CUDA graph on (production default, #1101)
- ``batched``: batching on, CUDA graph off (#1126's bounded-wait/floor policy)
- ``quantized``: batching and CUDA graph both on — chunk-aligned dispatch
  (the arm keeps its historical label for continuity with recorded JSONs)
  (perf/code2wav-adaptive-dispatch) with batched graph keys captured;
  ``--quantized-wait-ms`` x ``--quantized-floor`` sweeps the wait-vs-fire
  policy frontier (default 0/1 = fire every due bucket immediately)

``--drain-inbox on,off`` adds an inbox-drain dimension on the batching arms
(the #1862 2x2): ``off`` suppresses only b90d859's drain-and-coalesce block
in ``_next_message`` (its sole reader there), while deliveries stay on the
coalesced ``_collect_stream_chunk_batch`` -> ``on_stream_chunk_batch`` path
with the wait/floor pump policy intact — the exact pre-b90d859 loop — so the
drain mechanism and the wait/floor policy vary independently on one head.

Arrival modes:

- ``aligned``: every stream receives one full decode window per tick, back to
  back, the coalescing upper bound, matching #1126's component A/B
- ``staggered``: per-stream Poisson frame arrivals at ``--frame-interval-ms``
  with random phase, so windows rarely align: the default-on question

Run from the repo root::

    python -m benchmarks.eval.benchmark_code2wav_batching \\
        --model-path <hf-checkpoint> --arms serial-graph,batched \\
        --streams 8,16 --modes staggered --wait-ms 2,4,8 --floor 2,4 \\
        --repeats 3 --output-json results/code2wav_sweep.json

    # CPU smoke without a checkpoint:
    python -m benchmarks.eval.benchmark_code2wav_batching --fake \\
        --arms serial-eager,batched --streams 4 --windows 4 --modes aligned

Measured repeats run in serpentine (ABBA) order over the full config list so
slow clock/thermal drift cancels across cells; warmup rounds cover every
config once first.

Per run: TTFA p50/p95 (first frame enqueued -> first stream message), audio
seconds emitted per wall second (xRT), per-request RTF, chunks/request,
forward batch-size histogram, single-request forward share (the item 4.2
gate), fire-reason counts, and per-forward execution mode; plus steady-state
xRT over the all-streams-active window, per-execution-mode call/busy split,
EOS final-flush cost (calls/frames/busy), inter-forward idle gaps, observed
oldest-wait stats, and the drained-messages-per-wake histogram. Failed
requests invalidate the run rather than being dropped.

``--compare-waveforms`` additionally runs a lockstep batched-vs-serial
equivalence pass on identical codes and reports SNR / peak-diff /
exceed-fraction through ``benchmarks.metrics.waveform_tolerance`` (windows in
the free-running sweep are timing-dependent, so waveforms are only comparable
under lockstep feeding).
"""

from __future__ import annotations

import argparse
import itertools
import json
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import torch

from benchmarks.metrics.waveform_tolerance import compare_waveforms, tolerance_failures
from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
    Code2WavScheduler,
    load_code2wav_model,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage

ARM_CHOICES = ("serial-eager", "serial-graph", "batched", "quantized")
MODE_CHOICES = ("aligned", "staggered")


class FakeVocoder:
    """Deterministic, batch-invariant stand-in for CPU smoke runs."""

    def __init__(self, *, total_upsample: int = 2) -> None:
        self.total_upsample = total_upsample

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        samples = int(codes.shape[-1]) * self.total_upsample
        base = codes.to(dtype=torch.float32).flatten(1).sum(dim=1).view(-1, 1, 1)
        ramp = torch.arange(samples, dtype=torch.float32).view(1, 1, samples)
        return ramp / samples + base * 1e-4


class InstrumentedCode2Wav(Code2WavScheduler):
    """Code2WavScheduler with per-forward, per-step, and per-request capture."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.forward_log: list[dict[str, Any]] = []
        self.step_log: list[dict[str, Any]] = []
        self.captured_audio: dict[str, np.ndarray] = {}
        self.wake_drain_counts: list[int] = []
        self._in_final_decode = False
        self._bench_inbox_drain = True

    def _forward_codes(self, codes, **kwargs):
        start = time.perf_counter()
        wav, metadata = super()._forward_codes(codes, **kwargs)
        if self._device.type == "cuda":
            torch.cuda.synchronize(self._device)
        self.forward_log.append(
            {
                "batch": int(codes.shape[0]),
                "frames": int(codes.shape[-1]),
                "t_start": start,
                "seconds": time.perf_counter() - start,
                "execution_mode": metadata["execution_mode"],
                "fallback_reason": metadata["fallback_reason"],
                "final": self._in_final_decode,
            }
        )
        return wav, metadata

    def run_step(self, participants, plan):
        decoded = super().run_step(participants, plan)
        self.step_log.append(
            {
                "participants": len(participants),
                "fire_reason": self._last_fire_reason,
                "due_buckets": self._last_due_bucket_count,
                "oldest_wait_ms": self._last_oldest_wait_ms,
            }
        )
        return decoded

    def _next_message(self):
        if self._bench_inbox_drain:
            return super()._next_message()
        # Note (ruoyu): pre-b90d859 wake — the drain-and-coalesce block is
        # the only reader of _can_batch_stream_chunks inside _next_message,
        # so clearing the flag just for this call skips it; once restored,
        # _handle_message still routes chunks through
        # _collect_stream_chunk_batch -> on_stream_chunk_batch, keeping the
        # wait/floor pump policy live.
        restore = self._can_batch_stream_chunks
        self._can_batch_stream_chunks = False
        try:
            return super()._next_message()
        finally:
            self._can_batch_stream_chunks = restore

    def _drain_inbox(self):
        drained = 0
        for message in super()._drain_inbox():
            drained += 1
            yield message
        self.wake_drain_counts.append(drained)

    def decode_delta(self, request_id, state, *, is_final):
        self._in_final_decode = is_final
        try:
            return super().decode_delta(request_id, state, is_final=is_final)
        finally:
            self._in_final_decode = False

    def final_result_data(self, request_id, payload, state):
        if state.audio_parts:
            self.captured_audio[request_id] = np.concatenate(state.audio_parts)
        return super().final_result_data(request_id, payload, state)


@dataclass
class ModelContext:
    model: Any
    device: str
    num_quantizers: int
    total_upsample: int
    sample_rate: int
    graph_runner: Any = None


@dataclass
class StreamPlan:
    request_id: str
    codes: torch.Tensor  # (frames, num_quantizers)
    start_offset_s: float
    intervals_s: np.ndarray  # per-frame inter-arrival delays


@dataclass
class RunClock:
    first_frame: dict[str, float] = field(default_factory=dict)
    first_stream: dict[str, float] = field(default_factory=dict)
    result: dict[str, float] = field(default_factory=dict)
    stream_chunks: dict[str, int] = field(default_factory=dict)
    stream_events: list[tuple[float, str]] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)


def build_model_context(args: argparse.Namespace) -> ModelContext:
    if args.fake:
        model = FakeVocoder(total_upsample=args.fake_upsample)
        return ModelContext(
            model=model,
            device=args.device,
            num_quantizers=args.num_quantizers,
            total_upsample=model.total_upsample,
            sample_rate=args.sample_rate,
        )
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    args.device = str(device)
    model = load_code2wav_model(args.model_path, device=args.device, dtype=args.dtype)
    graph_runner = None
    if any(arm in args.arms for arm in ("serial-graph", "quantized")):
        from sglang_omni.models.qwen3_omni.components.code2wav_cuda_graph import (
            Code2WavCudaGraphRunner,
        )
        from sglang_omni.models.qwen3_omni.components.code2wav_scheduler import (
            _batched_graph_keys,
            _serial_threshold_graph_keys,
        )

        if "quantized" in args.arms:
            # Serial keys are included by _batched_graph_keys since the 1-pool
            # merge; adding them again would capture every serial graph twice.
            graph_keys = _batched_graph_keys(
                args.stream_chunk_size, args.left_context_size, args.ceiling
            )
        else:
            graph_keys = _serial_threshold_graph_keys(
                args.stream_chunk_size, args.left_context_size
            )
        graph_runner = Code2WavCudaGraphRunner.build(
            model,
            device=torch.device(args.device),
            num_quantizers=int(model.config.num_quantizers),
            total_gpu_memory_fraction=args.graph_memory_fraction,
            graph_keys=graph_keys,
        )
        startup_stats = graph_runner.stats()
        if not startup_stats["enabled"]:
            # A degraded arm would silently measure the wrong mechanism.
            raise RuntimeError(
                "code2wav graph build failed: " f"{startup_stats['disable_reason']}"
            )
        published = startup_stats["build"]["published_graph_count"]
        if published < len(graph_keys):
            # Shrink is legitimate under a tight budget (that IS the prod
            # behavior at 0.02) — record it loudly instead of failing.
            print(
                f"[graph] published {published}/{len(graph_keys)} keys at "
                f"fraction {args.graph_memory_fraction}",
                flush=True,
            )
    return ModelContext(
        model=model,
        device=args.device,
        num_quantizers=int(model.config.num_quantizers),
        total_upsample=int(model.total_upsample),
        sample_rate=args.sample_rate,
        graph_runner=graph_runner,
    )


def build_scheduler(
    ctx: ModelContext,
    arm: str,
    args: argparse.Namespace,
    wait_ms: int,
    floor: int,
    *,
    output_overlap: bool = True,
) -> InstrumentedCode2Wav:
    graph_arm = arm in ("serial-graph", "quantized")
    return InstrumentedCode2Wav(
        ctx.model,
        device=ctx.device,
        stream_chunk_size=args.stream_chunk_size,
        left_context_size=args.left_context_size,
        sample_rate=ctx.sample_rate,
        enable_batching=arm in ("batched", "quantized"),
        max_batch_wait_ms=wait_ms,
        batch_floor=floor,
        batch_ceiling=args.ceiling,
        enable_output_overlap=output_overlap,
        enable_cuda_graph=graph_arm,
        _cuda_graph_runner=ctx.graph_runner if graph_arm else None,
    )


def build_stream_plans(
    ctx: ModelContext,
    args: argparse.Namespace,
    *,
    streams: int,
    mode: str,
    repeat_seed: int,
    frame_interval_s: float,
) -> list[StreamPlan]:
    """Codes and arrival timings depend on (seed, repeat, stream) only, so the
    same plans pair runs across arms and wait/floor settings."""
    frames = args.windows * args.stream_chunk_size + args.tail_frames
    window_interval_s = frame_interval_s * args.stream_chunk_size
    plans = []
    for i in range(streams):
        rng = np.random.default_rng([args.seed, repeat_seed, i])
        codes = torch.from_numpy(
            rng.integers(
                0, args.code_vocab, size=(frames, ctx.num_quantizers), dtype=np.int64
            )
        )
        if mode == "aligned":
            offset, intervals = 0.0, np.zeros(frames)
        else:
            offset = float(rng.uniform(0.0, window_interval_s))
            intervals = rng.exponential(frame_interval_s, size=frames)
        plans.append(
            StreamPlan(
                request_id=f"bench-{i}",
                codes=codes,
                start_offset_s=offset,
                intervals_s=intervals,
            )
        )
    return plans


def _chunk_message(request_id: str, chunk_id: int, frame: torch.Tensor):
    return IncomingMessage(
        request_id=request_id,
        type="stream_chunk",
        data=StreamItem(chunk_id, frame, "bench", metadata={"stream": True}),
    )


def _feed_staggered(
    scheduler: Code2WavScheduler, plan: StreamPlan, clock: RunClock
) -> None:
    time.sleep(plan.start_offset_s)
    for k in range(plan.codes.shape[0]):
        delay = float(plan.intervals_s[k])
        if delay > 0:
            time.sleep(delay)
        if k == 0:
            clock.first_frame[plan.request_id] = time.perf_counter()
        scheduler.inbox.put(_chunk_message(plan.request_id, k, plan.codes[k]))
    scheduler.inbox.put(IncomingMessage(request_id=plan.request_id, type="stream_done"))


def _feed_aligned(
    scheduler: Code2WavScheduler,
    plans: list[StreamPlan],
    clock: RunClock,
    chunk_size: int,
) -> None:
    frames = plans[0].codes.shape[0]
    for window_start in range(0, frames, chunk_size):
        for plan in plans:
            if window_start == 0:
                clock.first_frame.setdefault(plan.request_id, time.perf_counter())
            # The last window may be a sub-chunk --tail-frames remainder.
            for k in range(window_start, min(window_start + chunk_size, frames)):
                scheduler.inbox.put(_chunk_message(plan.request_id, k, plan.codes[k]))
    for plan in plans:
        scheduler.inbox.put(
            IncomingMessage(request_id=plan.request_id, type="stream_done")
        )


def _collect(
    scheduler: Code2WavScheduler,
    plans: list[StreamPlan],
    clock: RunClock,
    timeout_s: float,
) -> str | None:
    """Drain the outbox until every request has a terminal message."""
    outstanding = {plan.request_id for plan in plans}
    deadline = time.perf_counter() + timeout_s
    while outstanding:
        remaining = deadline - time.perf_counter()
        if remaining <= 0:
            return f"timed out with {len(outstanding)} unfinished request(s)"
        try:
            message = scheduler.outbox.get(timeout=min(remaining, 0.5))
        except queue.Empty:
            continue
        now = time.perf_counter()
        if message.type == "stream":
            clock.first_stream.setdefault(message.request_id, now)
            clock.stream_chunks[message.request_id] = (
                clock.stream_chunks.get(message.request_id, 0) + 1
            )
            clock.stream_events.append((now, message.request_id))
        elif message.type == "result":
            clock.result[message.request_id] = now
            outstanding.discard(message.request_id)
        elif message.type == "error":
            clock.errors[message.request_id] = repr(message.data)
            outstanding.discard(message.request_id)
    return None


def _pct(values: list[float], p: float) -> float | None:
    return round(float(np.percentile(values, p)), 4) if values else None


def run_single(
    ctx: ModelContext,
    args: argparse.Namespace,
    *,
    arm: str,
    streams: int,
    mode: str,
    wait_ms: int | None,
    floor: int | None,
    drain: str | None,
    repeat: int,
) -> dict[str, Any]:
    frame_interval_s = args.frame_interval_ms / 1000.0
    plans = build_stream_plans(
        ctx,
        args,
        streams=streams,
        mode=mode,
        repeat_seed=repeat,
        frame_interval_s=frame_interval_s,
    )
    scheduler = build_scheduler(ctx, arm, args, wait_ms or 0, floor or 1)
    if drain == "off":
        scheduler._bench_inbox_drain = False
    clock = RunClock()
    for plan in plans:
        scheduler.inbox.put(
            IncomingMessage(
                request_id=plan.request_id,
                type="new_request",
                data=StagePayload(
                    request_id=plan.request_id,
                    request=OmniRequest(inputs="", params={"stream": True}),
                    data=None,
                ),
            )
        )
    serve_thread = threading.Thread(target=scheduler.start, daemon=True)
    serve_thread.start()
    started = time.perf_counter()
    if mode == "aligned":
        feeders = [
            threading.Thread(
                target=_feed_aligned,
                args=(scheduler, plans, clock, args.stream_chunk_size),
                daemon=True,
            )
        ]
    else:
        feeders = [
            threading.Thread(
                target=_feed_staggered, args=(scheduler, plan, clock), daemon=True
            )
            for plan in plans
        ]
    for feeder in feeders:
        feeder.start()
    invalid = _collect(scheduler, plans, clock, args.timeout_s)
    for feeder in feeders:
        feeder.join(timeout=5)
    scheduler.stop()
    serve_thread.join(timeout=10)
    if serve_thread.is_alive():
        invalid = invalid or "scheduler thread failed to stop"
    if clock.errors:
        invalid = invalid or f"{len(clock.errors)} failed request(s)"

    record: dict[str, Any] = {
        "arm": arm,
        "streams": streams,
        "mode": mode,
        "max_batch_wait_ms": wait_ms,
        "batch_floor": floor,
        "batch_ceiling": (args.ceiling if arm in ("batched", "quantized") else None),
        "inbox_drain": drain,
        "repeat": repeat,
        "errors": clock.errors,
    }
    if invalid:
        record["invalid"] = invalid
        return record

    wall = max(clock.result.values()) - started
    ttfas = sorted(
        clock.first_stream[rid] - clock.first_frame[rid]
        for rid in clock.first_stream
        if rid in clock.first_frame
    )
    audio_s = {
        rid: audio.shape[0] / ctx.sample_rate
        for rid, audio in scheduler.captured_audio.items()
    }
    rtfs = sorted(
        (clock.result[rid] - clock.first_frame[rid]) / audio_s[rid]
        for rid in audio_s
        if rid in clock.first_frame and audio_s[rid] > 0
    )
    batches = [entry["batch"] for entry in scheduler.forward_log]
    histogram: dict[int, int] = {}
    for batch in batches:
        histogram[batch] = histogram.get(batch, 0) + 1
    fire_reasons: dict[str, int] = {}
    for entry in scheduler.step_log:
        reason = entry["fire_reason"] or "unknown"
        fire_reasons[reason] = fire_reasons.get(reason, 0) + 1
    forward_busy_s = sum(entry["seconds"] for entry in scheduler.forward_log)
    exec_split: dict[str, dict[str, Any]] = {}
    for entry in scheduler.forward_log:
        bucket = exec_split.setdefault(
            entry["execution_mode"], {"calls": 0, "busy_s": 0.0, "frames": 0}
        )
        bucket["calls"] += 1
        bucket["busy_s"] += entry["seconds"]
        bucket["frames"] += entry["frames"] * entry["batch"]
    for bucket in exec_split.values():
        bucket["busy_s"] = round(bucket["busy_s"], 4)
    final_entries = [e for e in scheduler.forward_log if e["final"]]
    ordered_fwd = sorted(scheduler.forward_log, key=lambda e: e["t_start"])
    idle_gaps = [
        max(b["t_start"] - (a["t_start"] + a["seconds"]), 0.0)
        for a, b in zip(ordered_fwd, ordered_fwd[1:])
    ]
    waits_ms = [e["oldest_wait_ms"] for e in scheduler.step_log]
    drain_hist: dict[int, int] = {}
    for count in scheduler.wake_drain_counts:
        drain_hist[count] = drain_hist.get(count, 0) + 1
    steady = None
    if clock.first_stream and clock.result:
        # Note (ruoyu): all-streams-active window — last first-emission ->
        # first completion. Interior stream chunks are exactly one window of
        # audio each; the trimmed first chunk sits at/before t0 and flush
        # tails at/after t1.
        t0 = max(clock.first_stream.values())
        t1 = min(clock.result.values())
        if t1 > t0:
            chunk_audio_s = (
                args.stream_chunk_size * ctx.total_upsample / ctx.sample_rate
            )
            interior = sum(1 for t, _ in clock.stream_events if t0 <= t <= t1)
            busy = sum(
                e["seconds"] for e in scheduler.forward_log if t0 <= e["t_start"] <= t1
            )
            steady = {
                "window_s": round(t1 - t0, 4),
                "stream_chunks": interior,
                "xrt": round(interior * chunk_audio_s / (t1 - t0), 3),
                "forward_busy_fraction": round(busy / (t1 - t0), 4),
            }
    execution_modes = sorted(
        {entry["execution_mode"] for entry in scheduler.forward_log}
    )
    record.update(
        {
            "wall_s": round(wall, 4),
            "audio_s_total": round(sum(audio_s.values()), 3),
            "xrt": round(sum(audio_s.values()) / wall, 3) if wall > 0 else None,
            "ttfa_p50_s": _pct(ttfas, 50),
            "ttfa_p95_s": _pct(ttfas, 95),
            "ttfa_sorted_s": [round(v, 4) for v in ttfas],
            "rtf_p50": _pct(rtfs, 50),
            "rtf_p95": _pct(rtfs, 95),
            "chunks_per_request_mean": (
                round(sum(clock.stream_chunks.values()) / len(clock.stream_chunks), 2)
                if clock.stream_chunks
                else 0
            ),
            "forward_calls": len(batches),
            "forward_busy_s": round(forward_busy_s, 4),
            "forward_busy_fraction": (
                round(forward_busy_s / wall, 4) if wall > 0 else None
            ),
            "single_request_forward_share": (
                round(histogram.get(1, 0) / len(batches), 4) if batches else None
            ),
            "batched_window_share": (
                round(sum(b for b in batches if b > 1) / sum(batches), 4)
                if batches
                else None
            ),
            "batch_histogram": {str(k): v for k, v in sorted(histogram.items())},
            "fire_reasons": fire_reasons,
            "execution_modes": execution_modes,
            "execution_split": exec_split,
            "final_flush": {
                "calls": len(final_entries),
                "frames": sum(e["frames"] for e in final_entries),
                "busy_s": round(sum(e["seconds"] for e in final_entries), 4),
                "modes": sorted({e["execution_mode"] for e in final_entries}),
            },
            "idle_gap_total_s": round(sum(idle_gaps), 4),
            "idle_gap_p95_s": _pct(idle_gaps, 95),
            "oldest_wait_ms_mean": (
                round(float(np.mean(waits_ms)), 3) if waits_ms else None
            ),
            "oldest_wait_ms_p95": _pct(waits_ms, 95),
            "wake_drain_histogram": {str(k): v for k, v in sorted(drain_hist.items())},
            "steady_state": steady,
        }
    )
    return record


def run_equivalence(
    ctx: ModelContext,
    args: argparse.Namespace,
    *,
    streams: int,
    candidate_arm: str = "batched",
) -> dict[str, Any]:
    """Lockstep candidate-vs-serial decode on identical codes.

    Free-running window boundaries are timing-dependent, so waveforms are only
    well-defined under lockstep feeding: every stream advances one aligned
    window per round, mirroring test_code2wav_batching's equivalence test but
    on the benchmark model with tolerance metrics instead of bitwise equality.
    """
    plans = build_stream_plans(
        ctx,
        args,
        streams=streams,
        mode="aligned",
        repeat_seed=0,
        frame_interval_s=0.0,
    )
    # #1567's output-overlap path (serial-only, CUDA-only) merges the last
    # pending window into the tail and shifts total length by one sample vs
    # the non-overlap paths; the gate compares dispatch numerics, so overlap
    # is disabled on both sides to keep lengths comparable.
    control = build_scheduler(ctx, "serial-eager", args, 0, 1, output_overlap=False)
    batched = build_scheduler(ctx, candidate_arm, args, 0, 2, output_overlap=False)
    chunk = args.stream_chunk_size
    frames = plans[0].codes.shape[0]
    for plan in plans:
        for k in range(frames):
            control._handle_stream_chunk(
                plan.request_id,
                StreamItem(k, plan.codes[k], "bench", metadata={"stream": True}),
            )
    for window_start in range(0, frames, chunk):
        items = [
            (
                plan.request_id,
                StreamItem(k, plan.codes[k], "bench", metadata={"stream": True}),
            )
            for plan in plans
            for k in range(window_start, window_start + chunk)
        ]
        batched.on_stream_chunk_batch(items)

    batched_forwards = [entry["batch"] for entry in batched.forward_log]
    report: dict[str, Any] = {
        "candidate_arm": candidate_arm,
        "streams": streams,
        "windows": args.windows,
        "max_attained_batch": max(batched_forwards, default=0),
        "per_request": {},
        "failures": {},
    }
    for plan in plans:
        rid = plan.request_id
        reference = np.concatenate(control._stream_states[rid].audio_parts)
        candidate = np.concatenate(batched._stream_states[rid].audio_parts)
        comparison = compare_waveforms(
            reference, candidate, diff_threshold=args.diff_threshold
        )
        failures = tolerance_failures(
            comparison,
            min_snr_db=args.min_snr_db,
            max_peak_diff=args.max_peak_diff,
            max_exceed_fraction=args.max_exceed_fraction,
        )
        report["per_request"][rid] = comparison.to_dict()
        if failures:
            report["failures"][rid] = failures
    return report


def _parse_csv(raw: str, cast, choices=None) -> list:
    values = [cast(part.strip()) for part in raw.split(",") if part.strip()]
    if not values:
        raise argparse.ArgumentTypeError(f"empty list: {raw!r}")
    if choices is not None:
        for value in values:
            if value not in choices:
                raise argparse.ArgumentTypeError(f"{value!r} not in {sorted(choices)}")
    return values


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--model-path", help="HF checkpoint with code2wav weights")
    parser.add_argument("--dtype", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--fake", action="store_true", help="deterministic CPU vocoder, no checkpoint"
    )
    parser.add_argument("--fake-upsample", type=int, default=2)
    parser.add_argument(
        "--arms",
        type=lambda raw: _parse_csv(raw, str, ARM_CHOICES),
        default=list(ARM_CHOICES),
    )
    parser.add_argument(
        "--streams", type=lambda raw: _parse_csv(raw, int), default=[4, 8, 16, 32]
    )
    parser.add_argument("--windows", type=int, default=20)
    parser.add_argument(
        "--tail-frames",
        type=int,
        default=0,
        help="extra non-chunk-aligned frames per stream, so the EOS flush "
        "always carries a sub-chunk remainder (real Talker streams are "
        "never window-aligned)",
    )
    parser.add_argument(
        "--modes",
        type=lambda raw: _parse_csv(raw, str, MODE_CHOICES),
        default=["staggered"],
    )
    parser.add_argument(
        "--frame-interval-ms",
        type=float,
        default=None,
        help="staggered inter-frame arrival mean; default = real-time rate",
    )
    parser.add_argument(
        "--wait-ms", type=lambda raw: _parse_csv(raw, int), default=[1, 2, 4, 8]
    )
    parser.add_argument(
        "--floor", type=lambda raw: _parse_csv(raw, int), default=[2, 4]
    )
    parser.add_argument("--ceiling", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260728)
    parser.add_argument("--stream-chunk-size", type=int, default=10)
    parser.add_argument("--left-context-size", type=int, default=25)
    parser.add_argument("--sample-rate", type=int, default=24000)
    parser.add_argument("--num-quantizers", type=int, default=16)
    parser.add_argument(
        "--code-vocab", type=int, default=2048, help="codes drawn from [0, vocab)"
    )
    parser.add_argument(
        "--quantized-wait-ms", type=lambda raw: _parse_csv(raw, int), default=[0]
    )
    parser.add_argument(
        "--quantized-floor", type=lambda raw: _parse_csv(raw, int), default=[1]
    )
    parser.add_argument("--graph-memory-fraction", type=float, default=0.02)
    parser.add_argument(
        "--drain-inbox",
        type=lambda raw: _parse_csv(raw, str, ("on", "off")),
        default=["on"],
        help="inbox-drain dimension for the batching arms; off restores the "
        "pre-b90d859 per-message inbox loop",
    )
    parser.add_argument("--timeout-s", type=float, default=300.0)
    parser.add_argument("--compare-waveforms", action="store_true")
    parser.add_argument("--diff-threshold", type=float, default=1e-2)
    parser.add_argument("--min-snr-db", type=float, default=40.0)
    parser.add_argument("--max-peak-diff", type=float, default=0.2)
    parser.add_argument("--max-exceed-fraction", type=float, default=0.01)
    parser.add_argument("--output-json", default=None)
    args = parser.parse_args(argv)

    if args.fake and any(arm in args.arms for arm in ("serial-graph", "quantized")):
        parser.error("--fake cannot run the serial-graph or quantized arms")
    if not args.fake and not args.model_path:
        parser.error("--model-path is required without --fake")
    if args.device is None:
        args.device = "cpu" if args.fake else "cuda"
    return args


def _resolve_frame_interval(ctx: ModelContext, args: argparse.Namespace) -> None:
    if args.frame_interval_ms is None:
        args.frame_interval_ms = ctx.total_upsample / ctx.sample_rate * 1000.0


def _combos(arm: str, args: argparse.Namespace) -> list[tuple[int | None, int | None]]:
    if arm == "batched":
        return list(itertools.product(args.wait_ms, args.floor))
    if arm == "quantized":
        # The sweep lists measure the wait-vs-fire policy frontier. Zero wait
        # and floor 1 each fire every due bucket immediately, so any combo
        # containing either collapses to the (0, 1) baseline.
        combos: list[tuple[int | None, int | None]] = []
        for wait_ms, floor in itertools.product(
            args.quantized_wait_ms, args.quantized_floor
        ):
            combo = (0, 1) if wait_ms == 0 or floor == 1 else (wait_ms, floor)
            if combo not in combos:
                combos.append(combo)
        return combos
    return [(None, None)]


def _describe(record: dict[str, Any]) -> str:
    tag = (
        f"[{record['mode']} c{record['streams']} {record['arm']}"
        f" w={record['max_batch_wait_ms']} f={record['batch_floor']}"
        f" d={record['inbox_drain']} r{record['repeat']}]"
    )
    if "invalid" in record:
        return f"{tag} INVALID: {record['invalid']}"
    return (
        f"{tag} ttfa_p50={record['ttfa_p50_s']} p95={record['ttfa_p95_s']}"
        f" xrt={record['xrt']} rtf_p50={record['rtf_p50']}"
        f" fwd={record['forward_calls']}"
        f" single_share={record['single_request_forward_share']}"
        f" busy={record['forward_busy_fraction']}"
        f" batches={record['batch_histogram']}"
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ctx = build_model_context(args)
    _resolve_frame_interval(ctx, args)
    print(
        f"code2wav benchmark: device={ctx.device} quantizers={ctx.num_quantizers}"
        f" upsample={ctx.total_upsample}"
        f" frame_interval_ms={args.frame_interval_ms:.3f}",
        flush=True,
    )

    configs: list[dict[str, Any]] = []
    for mode, arm, streams in itertools.product(args.modes, args.arms, args.streams):
        for wait_ms, floor in _combos(arm, args):
            drains = args.drain_inbox if arm in ("batched", "quantized") else [None]
            for drain in drains:
                configs.append(
                    {
                        "mode": mode,
                        "arm": arm,
                        "streams": streams,
                        "wait_ms": wait_ms,
                        "floor": floor,
                        "drain": drain,
                    }
                )

    records: list[dict[str, Any]] = []
    invalid_runs = 0
    for warmup in range(args.warmup_runs):
        for config in configs:
            run_single(
                ctx,
                args,
                arm=config["arm"],
                streams=config["streams"],
                mode=config["mode"],
                wait_ms=config["wait_ms"],
                floor=config["floor"],
                drain=config["drain"],
                repeat=args.repeats + warmup,
            )
    for repeat in range(args.repeats):
        # Serpentine (ABBA) config order per round.
        round_order = configs if repeat % 2 == 0 else list(reversed(configs))
        for config in round_order:
            record = run_single(
                ctx,
                args,
                arm=config["arm"],
                streams=config["streams"],
                mode=config["mode"],
                wait_ms=config["wait_ms"],
                floor=config["floor"],
                drain=config["drain"],
                repeat=repeat,
            )
            records.append(record)
            invalid_runs += 1 if "invalid" in record else 0
            print(_describe(record), flush=True)

    equivalence = None
    if args.compare_waveforms:
        candidate_arms = [
            arm for arm in args.arms if arm in ("batched", "quantized")
        ] or ["batched"]
        equivalence = []
        for candidate_arm in candidate_arms:
            report = run_equivalence(
                ctx, args, streams=max(args.streams), candidate_arm=candidate_arm
            )
            equivalence.append(report)
            worst = min(
                (entry["snr_db"] for entry in report["per_request"].values()),
                default=None,
            )
            print(
                f"equivalence[{candidate_arm}]: streams={report['streams']}"
                f" max_batch={report['max_attained_batch']}"
                f" worst_snr_db={worst}"
                f" failures={len(report['failures'])}",
                flush=True,
            )

    graph_runner_stats = None
    if ctx.graph_runner is not None:
        graph_runner_stats = ctx.graph_runner.stats()
        runtime = graph_runner_stats["runtime"]
        print(
            f"graph_runner: enabled={graph_runner_stats['enabled']}"
            f" graphs={graph_runner_stats['build']['published_graph_count']}"
            f" replays={runtime['graph_replays']}"
            f" fallbacks={runtime['fallback_counts']}",
            flush=True,
        )

    if args.output_json:
        payload = {
            "config": {
                key: value for key, value in vars(args).items() if key != "output_json"
            },
            "model": {
                "num_quantizers": ctx.num_quantizers,
                "total_upsample": ctx.total_upsample,
                "device": ctx.device,
            },
            "runs": records,
            "equivalence": equivalence,
            "graph_runner_stats": graph_runner_stats,
        }
        with open(args.output_json, "w") as handle:
            json.dump(payload, handle, indent=2)
        print(f"results written: {args.output_json}", flush=True)

    if invalid_runs:
        print(f"{invalid_runs} invalid run(s); not formal output", flush=True)
        return 1
    if equivalence and any(report["failures"] for report in equivalence):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
