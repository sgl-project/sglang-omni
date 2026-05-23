# Request-level profiler

This document describes the request-level profiler introduced for
[issue #501](https://github.com/sgl-project/sglang-omni/issues/501).

The goal is to reconstruct the full latency timeline of a request as it flows
through the multi-stage pipeline, aggregate stage-local time, and surface
stage-to-stage hop costs. The existing `TorchProfiler` keeps doing what it does
(kernel-level Chrome traces); the request-level profiler runs alongside it and
shares the same `run_id`.

## Event model

Every instrumentation point appends a single line of JSON to a per-process
JSONL file. One event has the shape:

```jsonc
{
  "request_id": "req-123",
  "stage": "thinker",
  "event_name": "scheduler_first_emit",
  "timestamp_ns": 1717000000123456789,
  "run_id": "demo-run",
  "pid": 42,
  "metadata": {"chunk_id": 0}
}
```

Files are written under
`<event_dir>/events_<stage>_<pid>.jsonl`. Stage files from different processes
can be merged transparently by passing the directory to the views layer.

### Standard event names

The event taxonomy maps the high-level milestones laid out in issue #501 to
concrete callsites. The recorder always attaches the active `stage` name to
every event, so the same `scheduler_prefill_start` becomes "thinker prefill
start" when emitted from the thinker process and "talker prefill start" when
emitted from the talker process.

| #501 milestone | Concrete event | Where |
|---|---|---|
| request admission | `request_admission` | `Coordinator._submit_request` |
| preprocessing start / end | `preprocess_start` / `preprocess_end` | `Qwen3OmniPreprocessor.__call__` |
| encoder start / end | `encoder_start` / `encoder_end` (metadata `modality`, `batch_size`) | `create_image_encoder_executor`, `create_audio_encoder_executor` |
| aggregate ready | `stage_aggregate_ready` | `Stage._on_data_ready` after `InputHandler.receive` returns a merged payload |
| thinker prefill start | `scheduler_prefill_start` (stage=thinker) | `OmniScheduler.process_input_requests` |
| thinker first token | `stage_first_stream_chunk_sent` (stage=thinker) | `Stage._send_stream_to_target` / `_send_stream_to_coordinator` |
| first stream chunk sent | `stage_first_stream_chunk_sent` (terminal stage → coordinator) | same |
| talker request build start / end | `scheduler_request_build_start` / `_end` (stage=talker) | `OmniScheduler.process_input_requests` |
| talker prefill start | `scheduler_prefill_start` (stage=talker) | same |
| first code chunk | `stage_first_stream_chunk_sent` (stage=talker) | `Stage._send_stream_to_target` |
| code2wav first audio | `code2wav_first_audio` | `Code2WavScheduler._decode_and_emit` |
| terminal response | `terminal_response` | `Coordinator._handle_completion` |

Additional supporting events used for finer-grained breakdown:

| Layer | Event | Notes |
|---|---|---|
| Coordinator | `coordinator_stream_received` | Each `StreamMessage` received on the coordinator |
| Stage | `stage_input_received` | Submit or relay payload accepted (metadata `from_stage`) |
| Stage | `stage_dispatch` | Scheduler inbox put |
| Stage | `stage_complete` | Scheduler result routed onward (metadata `terminal`, `next`) |
| Stage | `stage_hop_sent` | Payload `DataReadyMessage` sent to next stage |
| Stage | `stage_stream_chunk_sent` | Each stream chunk (metadata `to_stage`, `chunk_id`, `modality`) |
| Stage | `stage_stream_chunk_received` | Each stream chunk received |
| AR scheduler | `scheduler_first_emit` | First `stream_output_builder` emission per request |

Custom callsites can call `sglang_omni.profiler.event_recorder.emit(...)` to add
domain-specific events — for example, model-specific preprocess timings or
encoder warmups. Events from inactive recorders are no-ops; instrumentation
sites do not need to guard against the disabled case.

## Lifecycle

The recorder is process-local. It is started on every stage and on the
coordinator when `/start_profile` (or `/start_request_profile`) is hit:

1. Launcher receives the HTTP request.
2. Coordinator starts its local recorder pointed at `<event_dir>`.
3. Launcher broadcasts `ProfilerStartMessage` over ZMQ to every stage,
   carrying both the torch trace template and the `event_dir`.
4. Each stage starts its own recorder for `event_dir` + its `stage` name.
5. On `/stop_profile`, the recorder is closed everywhere; files remain on
   disk under `<event_dir>`.

The torch profiler and the event recorder share a `run_id`. Setting
`enable_torch=false` on the request lets you record cheap JSONL events without
paying for a kernel trace.

## Generating reports

Use the views module — directly in Python:

```python
from sglang_omni.profiler.views import build_report
report = build_report("/tmp/profiles/demo-run/events")
print(report["request_count"], len(report["stage_breakdown"]))
```

…or as a CLI:

```bash
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format table
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format json --out report.json
```

The CLI / `build_report` returns three views derived from the same event
stream:

1. **Timeline** — per-request event list with `t_rel_ms` anchored at admission.
2. **Stage breakdown** — pair `(open_event, close_event)` durations aggregated
   per stage (count, total, avg, p50, p95, max).
3. **Hop breakdown** — pair `stage_hop_sent`/`stage_input_received` and
   `stage_stream_chunk_sent`/`stage_stream_chunk_received` durations per
   (source, destination, kind).

Hop pairs match across processes by `(request_id, source_stage, dest_stage,
chunk_id?)`, so single requests can be reconstructed even when each stage is in
its own subprocess.

## Discipline

- Recorder failures must never break serving: the emitter swallows write errors
  and counts drops; the first failure is logged.
- Tensors and large blobs MUST stay out of event metadata — keep metadata to
  small scalars (ids, counts, durations, modality, error strings).
- New event names should mirror existing naming: lowercase snake_case, prefix
  with the layer that owns the event (`stage_*`, `scheduler_*`, etc.).
