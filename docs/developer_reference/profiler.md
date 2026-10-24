# Request-level profiler

`sglang-omni` ships two complementary profilers that share the same `run_id`
and are controlled by the same HTTP surface:

- a **request-level event recorder** that writes a JSONL stream of
  per-request milestones (admission, preprocess, encoder, prefill, first
  token / first code chunk, hops, terminal response) — used to reconstruct
  a single request's end-to-end timeline and to aggregate stage / hop costs
  across a batch;
- a **torch profiler** that produces a Chrome trace of kernel-level CPU /
  device activity — used to drill into a specific window once the event
  recorder has identified where the time is going. Activities are `CPU` plus
  whatever device activity this torch build reports via
  `torch.profiler.supported_activities()` — `CUDA` on NVIDIA / ROCm, `XPU` on
  Intel.

Most diagnostics use the event recorder. The torch profiler is opt-in for
deeper kernel investigation.

## Event model

Every instrumentation point appends a single line of JSON to a per-process
JSONL file. The shape:

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

Files are written under `<event_dir>/events_<stage>_<pid>.jsonl`. Multiple
co-located stages in the same OS process share **one** JSONL file — the
filename uses the first stage to start, and the per-event `stage` field
identifies the owner. The views layer merges files from every process by
`request_id`.

### Standard event names

The recorder always attaches the active `stage` name to every event, so the
same `scheduler_prefill_start` becomes "thinker prefill start" when emitted
from the thinker process and "talker prefill start" when emitted from the
talker process. `scheduler_queue_enter` marks a built request entering the
scheduler queue; `scheduler_prefill_start` is emitted later, when the request's
first executable prefill / extend batch is selected.

| Pipeline milestone | Concrete event | Source |
|---|---|---|
| Request admission | `request_admission` | `Coordinator._submit_request` |
| Preprocessing start / end | `preprocess_start` / `preprocess_end` | model preprocessor `__call__` |
| Encoder start / end | `encoder_start` / `encoder_end` (metadata `modality`, `batch_size`) | image / audio encoder executors |
| Aggregate ready | `stage_aggregate_ready` | `Stage._on_data_ready` after `InputHandler.receive` returns a merged payload |
| Thinker prefill start | `scheduler_prefill_start` (stage = thinker) | `OmniScheduler.run_batch` |
| Thinker first token | `stage_first_stream_chunk_sent` (stage = thinker) | `Stage._send_stream_to_target` / `_send_stream_to_coordinator` |
| First stream chunk to client | `stage_first_stream_chunk_sent` (terminal stage → coordinator) | same |
| Talker request build execution start / end | `scheduler_request_build_start` / `_end` (stage = talker) | `OmniScheduler._run_request_builder` |
| Talker prefill start | `scheduler_prefill_start` (stage = talker) | same |
| First code chunk | `stage_first_stream_chunk_sent` (stage = talker) | `Stage._send_stream_to_target` |
| Code2Wav first audio | `code2wav_first_audio` | `Code2WavScheduler.keep_audio` (every decoded or flushed window) |
| Terminal response | `terminal_response` | `Coordinator._handle_completion` |

Supporting events used for finer-grained breakdown:

| Layer | Event | Notes |
|---|---|---|
| Coordinator | `coordinator_stream_received` | Each `StreamMessage` received on the coordinator |
| Stage | `stage_input_received` | Submit or relay payload accepted (metadata `from_stage`) |
| Stage | `stage_dispatch` | Scheduler inbox put |
| Stage | `stage_complete` | Scheduler result routed onward (metadata `terminal`, `next`) |
| Stage | `stage_hop_sent` | Payload `DataReadyMessage` sent to next stage |
| Stage | `stage_stream_chunk_sent` | Each stream chunk (metadata `to_stage`, `chunk_id`, `modality`) |
| Stage | `stage_stream_chunk_received` | Each stream chunk materialized and ready for the receiver scheduler, including coordinator terminal chunks |
| AR scheduler | `scheduler_queue_enter` | Built request entered the scheduler queue |
| AR scheduler | `scheduler_first_emit` | First `stream_output_builder` emission per request |
| Code2Wav | `code2wav_decode_start` | Each window's decode start: trigger, start/end/new/context/window frames, `rows` (the windows its replay decodes together), active and threshold-ready requests, inbox depth |
| Code2Wav | `code2wav_decode_launched` | Pipelined window whose vocoder work and asynchronous D2H copy have been enqueued; includes execution mode, window/new-frame counts and `rows` |
| Code2Wav | `code2wav_decode_end` | Repeats start metadata and adds the current decode's `audio_samples` plus execution metadata; output-overlap runs also include `pipelined` and the previous window's post-EOS-scan `d2h_wait_ns` |

Read `d2h_wait_ns` narrowly. With output overlap on, the lazy codec-EOS scan
calls `.tolist()` on the staged frame heads, which is itself a host
synchronization point, and it runs before `code2wav_decode_start`. So
`d2h_wait_ns` measures only the residual wait left after that scan, not the
whole overlap interval. Use the process-wide CUDA API trace as the load-bearing
measurement of whether blocking synchronization was actually removed.

Custom callsites can call `sglang_omni.profiler.event_recorder.emit(...)` to
add domain-specific events. Events from inactive recorders are no-ops, so
instrumentation sites do not need to guard against the disabled case.

## Full Duplex session Unit metrics

Enable the existing request event recorder with `POST /start_request_profile`
and stop it with `POST /stop_request_profile`. The JSONL files then include a
small session vocabulary that can be converted with
`sglang_omni.profiler.duplex_metrics.build_duplex_session_metrics`.

Each session-specific event carries its `request_id`; the report uses session
metadata to name the Unit as `(session_id, session_open_index, input_seq)`.
Unit lifecycle and start events carry the input modality and media clock.
Output events carry the output sequence, modality, media clock, duration, and
kind. Payload bytes are never stored.

The event boundaries have these meanings:

- `session_unit_ready` is declared by the ingress owner when a logical Unit
  has all required input needed to run without waiting for more required
  input. It is modality-independent and is not inferred by the profiler from
  PCM, frames, or any other payload boundary.
- `session_unit_admitted` is emitted after the coordinator accepts an append;
  a rejected `QueueFullError` has no event.
- `stage_dispatch` is the existing stage ingress observation used as the queue
  boundary for a non-AR session Unit.
- `scheduler_queue_enter` and `scheduler_prefill_start` are the existing AR
  scheduler boundaries. For an AR Unit, they replace `stage_dispatch` and
  `session_stage_started` for queue and start timing.
- `session_stage_started` marks execution after per-session ordering for a
  `SessionScheduler` stage.
- `session_stage_bypassed` is the authoritative AR decision that an input
  Unit completed without generation. Its `reason` metadata identifies the
  bypass path, such as `generation_bypassed` or `finish_input`.
- `stage_complete` is the existing stage result observation used as the finish
  boundary for a session Unit.
- `session_output_emitted` marks a model content chunk entering the
  coordinator output buffer. `input_done` is a lifecycle completion and is
  recorded as `session_unit_finished`, not as model output.

The report joins generic stage events by `request_id`. For AR stages, `queue_ms`
is `scheduler_queue_enter -> scheduler_prefill_start`; for
`SessionScheduler` stages, it is `stage_dispatch -> session_stage_started`.
`service_ms` is the observed execution-start to `stage_complete` interval, not
pure GPU compute time. A stage is `observed` when its selected boundaries are
complete, `bypassed` when `session_stage_bypassed` is present, and `incomplete`
when a boundary is missing. Bypassed stages retain their timestamp and reason;
queue and service durations remain `null`. `handoff_from_previous_ms` ends at
the current stage's generic `stage_dispatch` boundary. The recorder carries
the input media clock for later benchmarking, but this slice does not claim
semantic input-to-output latency, playback underrun, or deadline misses.

Readiness is observed only while the request event recorder is active. The
internal observation carries that profiler run ID, and admission emits the
ready event only when the same run is still active; a run rotation leaves the
admitted-based values valid while ready-based values remain unavailable.

The report names the ready-based values explicitly:
`ready_to_admitted_ms`, `ready_to_first_output_ms`, and
`ready_to_finished_ms`. They measure ingress wait, ready-to-first-output, and
ready-to-finish directly from their event timestamps. The admitted-based
decomposition remains available as `admitted_to_first_output_ms` and
`admitted_to_finished_ms`; those intervals start when
`CoordinatorSessions.append_session` accepts the Unit. They exclude any
earlier `SessionRuntime.pending_pcm` buffering before submission and do not
claim producer-ready, client/network, semantic-response, playback, or
deadline latency. For a complete Unit, `ready_to_first_output_ms` is the
ready-to-admission wait plus the post-admission time to first output; each
value is derived per Unit before aggregation.

The `SessionScheduler` scalar hook emits one start event per Unit immediately
before append. `BatchedSessionHooks.append_batch` emits one event per row with
one shared directly observed batch-start timestamp.

### Active-stage attribution

`emit(...)` accepts an explicit `stage=...` parameter; when the caller can't
plumb the stage name down (preprocessor `__call__`, encoder callables,
`OmniScheduler` / `Code2WavScheduler` internals), it can pass `stage=None`
and the recorder fills it in from the **per-thread / per-task active
stage**.

`Stage._run_scheduler` binds `set_active_stage(self.name)` on the scheduler
thread before invoking the scheduler. The binding uses both a
`threading.local` slot (for plain `threading.Thread` workers) and a
`contextvars.ContextVar` (so it propagates through `asyncio.to_thread` /
`loop.run_in_executor`, which copy contextvars but not thread-local).
Explicit `stage=...` on emit always wins; the active-stage binding is only
consulted when the caller passes `stage=None`.

To bind / unbind manually from your own thread:

```python
from sglang_omni.profiler.event_recorder import set_active_stage, reset_active_stage

token = set_active_stage("my_stage")
try:
    ...
finally:
    reset_active_stage(token)
```

`reset_active_stage(None)` is the "scrub" form (used by test fixtures) and
clears both the thread-local slot and the contextvar.

## Lifecycle

The recorder is process-local. It is started on every stage and on the
coordinator when `POST /start_profile` (or `POST /start_request_profile`)
is hit:

1. Launcher receives the HTTP request.
2. Coordinator starts its local recorder pointed at `<event_dir>`.
3. Launcher broadcasts `ProfilerStartMessage` over ZMQ to every stage,
   carrying both the torch trace template and the `event_dir`.
4. Each stage joins the per-process recorder. In a shared-process topology
   the first stage to call `start()` wins the filename; every subsequent
   stage in the same process writes to the same file and the per-event
   `stage` field disambiguates.
5. On `POST /stop_profile`, the recorder is closed everywhere; files
   remain on disk under `<event_dir>`.

`POST /stop_profile` and `POST /stop_request_profile` accept an optional
`run_id` field. When **omitted**, the request is a wildcard: every stage
stops whatever profiler session is currently active. When **set**, only
stages whose active run matches stop. This makes the common case (caller
didn't specify a run_id on either start or stop) work without ceremony.

The torch profiler and the event recorder share a `run_id`. Setting
`enable_torch=false` on the start request records JSONL events without
paying for a kernel trace.

## Generating reports

Use the views module directly:

```python
from sglang_omni.profiler.views import build_report
report = build_report("/tmp/profiles/demo-run/events")
print(report["request_count"], len(report["stage_breakdown"]))
```

…or via the CLI:

```bash
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format table
python -m sglang_omni.profiler /tmp/profiles/demo-run/events --format json --out report.json
```

The CLI / `build_report` returns three views derived from the same event
stream:

1. **Timeline** — per-request event list with `t_rel_ms` anchored at
   admission.
2. **Stage breakdown** — `(open_event, close_event)` interval durations
   aggregated per stage (count, total, avg, p50, p95, max). The same opener
   can participate in multiple pairs (e.g. `scheduler_prefill_start` closes
   against both `scheduler_first_emit` AND `stage_first_stream_chunk_sent`);
   every pair gets its own pending stack so a close event for pair A does
   not consume the opener of pair B.
3. **Hop breakdown** — `stage_hop_sent` / `stage_input_received` and
   `stage_stream_chunk_sent` / `stage_stream_chunk_received` durations per
   (source, destination, kind). Terminal stage stream chunks are paired the
   same way with destination `coordinator`.

Hop pairs match across processes by `(request_id, source_stage, dest_stage,
chunk_id?)`, so a single request's path through subprocesses can be
reconstructed even when each stage runs in its own process.

## Torch profiler

The torch profiler runs alongside the event recorder when
`enable_torch=true` (the default for `/start_profile`). It records
continuously between `start()` and `stop()` — no `schedule(...)`, no
`step()` requirement — and exports a Chrome trace `*.trace.json.gz` on stop.

The expensive introspection flags are opt-in via env vars so the default
trace stays small enough to load in `chrome://tracing` or
[`ui.perfetto.dev`](https://ui.perfetto.dev):

| Env var | Effect |
|---|---|
| `SGLANG_TORCH_PROFILER_RECORD_SHAPES=1` | Record input tensor shapes per op |
| `SGLANG_TORCH_PROFILER_PROFILE_MEMORY=1` | Record tensor memory allocation / free events |
| `SGLANG_TORCH_PROFILER_WITH_STACK=1` | Record the Python (and C++) call stack per op |
| `SGLANG_TORCH_PROFILER_WITH_FLOPS=1` | Estimate FLOPs per op |

With all four off (the default), a typical 10-sample MMMU run produces a
trace in the tens of MB. With all four on, the same workload can produce a
multi-GB trace — only opt in when you need that specific information.

## HTTP surface

| Method | Path | Body | Notes |
|---|---|---|---|
| POST | `/start_profile` | `{"run_id": ?, "trace_path_template": ?, "event_dir": ?, "enable_torch": true \| false, "config": ?}` | Starts torch trace + event recorder. `run_id` auto-generated if omitted. |
| POST | `/stop_profile` | `{"run_id": ?}` | Stops torch trace + event recorder. Omitting `run_id` is a wildcard ("stop whatever's active"). |
| POST | `/start_request_profile` | `{"run_id": ?, "event_dir": ?}` | Event recorder only — no torch trace. Lower overhead; safer to leave on. |
| POST | `/stop_request_profile` | `{"run_id": ?}` | Same wildcard semantics as `/stop_profile`. |

Example: record cheap events on every request without a kernel trace:

```bash
curl -X POST http://localhost:8000/start_request_profile \
     -d '{"run_id":"demo","event_dir":"/tmp/profiles/demo/events"}'
# … run traffic …
curl -X POST http://localhost:8000/stop_request_profile -d '{}'
python -m sglang_omni.profiler /tmp/profiles/demo/events --format table
```

## Discipline

- **Profiling must never break serving.** The emitter swallows write
  errors and counts drops; the first failure is logged once.
- **Tensors and large blobs stay out of event metadata.** Keep metadata
  to small scalars (ids, counts, durations, modality, error strings). The
  recorder enforces this defensively: if a tensor / numpy array ends up
  in metadata, `_json_default` serializes a summary
  (`{"__tensor_summary__": true, "type": ..., "shape": [...], "dtype":
  "...", "device": "..."}`) instead of materializing the contents. 0-D
  tensors / numpy scalars still serialize as plain scalars.
- **Event naming.** Lowercase snake_case, prefix with the layer that
  owns the event (`stage_*`, `scheduler_*`, `encoder_*`, etc.). Use the
  stage name (not the event name) to distinguish "thinker prefill start"
  from "talker prefill start".
