# Realtime Session Lifecycle Stress

This tool opens, closes, drops, cancels, idles out, reconnects and interrupts `/v1/realtime` sessions in bulk for many cycles, and checks after every cycle that the server holds no session and admits a full set of new ones. It is item C11 of #2994.

## GPU run

This is the part that is still open: the tool has only run against a model-free mock. One script runs every cell against a turn-based Qwen3-Omni server, with the client on the server's host and no other job on the GPU.

```bash
MODEL=$MODEL MAX_SESSIONS=$MAX_SESSIONS SERVER_PID=$SERVER_PID SERVER_LOG=$SERVER_LOG PAIR=$FDB_WORK/dataset/v1.5/user_interruption/1 bash benchmarks/duplex/lifecycle_stress/gpu_matrix.sh runs/gpu
```

- `MAX_SESSIONS` is the server's `max_sessions`, `SERVER_PID` its main process, `SERVER_LOG` the file its log goes to.
- `PAIR` is a directory with `input.wav`, in which the user talks over the reply, and `clean_input.wav`, in which nobody does.
- `URL` (default `ws://127.0.0.1:8000/v1/realtime`) and `CYCLES` (default 10) can be set the same way.

Each cell first runs one cycle of two clients. A cell that fails there is skipped; the others run `CYCLES` cycles at `MAX_SESSIONS` clients. No failure stops the script, and a rerun with the same output directory skips what is already there. It ends with `runs/gpu/report.txt`, and `runs/gpu` is what to send back.

What `report.txt` should show:

| Cell | Expected |
|---|---|
| every cell | `failed_checks` is empty; `unmeasured_checks` is `["no_sessions_left"]`, because a model server has no debug route to read its session count from |
| `interrupt` | `interrupted_replies` is above zero |
| `interrupt_control` | `interrupted_replies` is zero |
| `churn`, `long` | every session `completed` |
| `churn_overflow` | `MAX_SESSIONS` sessions `completed` per cycle and the rest `denied` |
| `drop_*`, `reconnect_reply_reset` | every session `dropped`; in the `reply` cells `drops_mid_reply` counts them and `drops_after_reply` is zero |
| `cancel` | every session `completed`, `cancel_statuses` all `cancelled` |
| `cancel_reply_reset` | every session `dropped` |

The two interrupt cells run first, because they establish that the scenario does what it says before any of its numbers count. A zero in `interrupt` means the server's voice detection did not cancel a reply, or the tool did not count it; the traces under `traces/` tell which.

Also worth a look:

- `server_log` counts the tracebacks and `ERROR` lines the server logged during a cell. A line with `unknown session open index` is #3565; its count is worth recording either way.
- `device_megabytes`, `rss_megabytes`, `thread_count` and `open_files` were tested against stand-ins only. A reading that is empty or absurd is a bug in the tool, not a finding about the server.

Setup notes:

- The tool reads from `/v1/realtime/capabilities` whether the deployment is turn-based. If it is, every session of the run asks for `server_vad` turn detection, and the recorded sessions use the duplex recorder's turn-based profile.
- With `--audio-file` a short session sends the whole file, because a turn-based model replies only after the speech and the pause that ends the turn. So a cycle takes at least as long as the file.
- The tool cannot see how the server was started. State the voice, the encoder scheduling (`batch_sessions` and the like), `max_sessions` and the GPU and driver with the results; `--runtime` takes a free text that is stored in every run's record.
- The script was run end to end against the mock's turn-taking stand-in only. Nothing here has met a model.

## Mock results

- A run passes five checks: `no_sessions_left`, `slots_recovered`, `final_session_completed`, `no_errored_sessions` and `no_unexpected_denials`. A check the server gives no data for is reported as unmeasured, not as passed.
- The matrix of mock runs is not recorded with this version of the tool yet. Its commands are under Results; the table goes here once they have run.
- The mock is model-free and runs on one CPU host. It replaces only the model adapter; the route, the admission check, the session manager and the session runtime are the code a model server runs. So it says how the serving layer cleans up, and nothing about races between the stages of a model pipeline, which is where #3565 was found.

Try it against the mock:

```bash
python -m benchmarks.duplex.lifecycle_stress mock --port 8801 --max-connections 64
```

```bash
python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8801/v1/realtime --server-revision $(git rev-parse HEAD) --model lifecycle-stress-mock --scenario churn --concurrency 64 --cycles 10 --max-connections 64 --output runs/churn
```

`runs/churn/summary.json` holds the checks and the numbers, `samples.json` one record per session and per cycle.

## Related fixes

The tool changes no serving code. It is built on `main` and holds what `main` holds:

| Change | State | What it does | Where the tool exercises it |
|---|---|---|---|
| #3013 | merged | close codes for shared sessions | idle rows: the close is 1008 with the name of the limit |
| #3014 | merged | release when admission or input processing is cancelled | drop and cancel rows: no session left, every slot back |
| #3015 | merged | configurable idle timeouts that release the slot | idle rows: no close before the published limit |
| #3504 | open | release the legacy session when its client vanishes mid-send | `legacy_drop_frame` row |
| #3561 | open | one Silero session shared across VAD streams | not exercised: the mock replaces Silero |
| #3565 | open bug | `unknown session open index` in the log as sessions close | `--server-log`, on a model server only |

Until #3504 is merged, the `legacy_drop_frame` row leaks on `main` and fails `no_sessions_left`. That is the leak #3504 fixes and not a fault of the tool; the same row against the serving code of #3504 is the comparison. #3504 and #3561 are on the legacy path, which the GPU run above does not take.

## Scenarios

Every scenario runs `--concurrency` clients at once per cycle and repeats for `--cycles` cycles. A short session sends `--packets-per-session` packets of 80 ms: 5 of silence by default, 0.4 s of input, and the whole file with `--audio-file`.

- churn: connect, `session.update`, send the packets, then say goodbye the way the duplex recorder does: `sglang.input_audio.end`, wait for `sglang.input_audio.drained`, `session.close`, wait for `session.closed`. With more clients than the connection cap, the excess is expected to be denied.
- drop: leave early without reading what the server still has to say. `--leave-point update` leaves right after `session.update`, `append` after the last packet, `reply` while the first reply is in progress. `--leave-style reset` resets the TCP connection; `close_frame` sends a close frame.
- cancel: send `response.cancel` on the first `response.created`, wait for that reply's `response.done`, send the remaining packets and say goodbye. With `--leave-point reply` the client leaves right after the cancel instead. A deployment that publishes `supports_cancel` false, such as PersonaPlex, has to refuse the cancel with a `not_supported` error and keep the session going; the client then expects that refusal and the session still has to complete.
- idle: stop sending, before `session.update` (`--idle-point created`) or after the last packet (`open`), and read until the server closes the session.
- reconnect: each client opens `--reconnects` further sessions per cycle, each the moment its previous one ended. The server has no session resume, so a reconnect is a new session; a denied one retries every 20 ms.
- long: whole conversations of 15 s through `benchmarks.duplex.client.run_session`, the recorder the other duplex benchmarks use, with one trace file per session.
- interrupt: whole conversations through the same recorder with `--audio-file` speech in which the user talks over the reply, so that the server's voice detection cancels the reply by itself. Only a turn-based deployment has this path, and the tool refuses the scenario elsewhere; a native full-duplex model decides for itself when to speak. The same command with a file in which nobody talks over the reply is the control.

A reply cannot start before the server has one whole unit of input, and the unit belongs to the deployment: 80 ms for Nemotron VoiceChat and PersonaPlex, 1000 ms for MiniCPM-o, 200 ms for the turn-based pipeline. A turn-based pipeline also answers only once it has heard the user's turn end, which takes the packets after the speech. So the client keeps sending at its pace while it waits for the first `response.created`, and acts on it when it arrives. It reads `native_unit_ms` from `/v1/realtime/capabilities` and refuses cancel and a mid-reply drop when a session's packets do not fill one unit (13 packets for a 1000 ms unit).

Pending work is whatever the server still has in flight for a session when the client leaves or cancels: input not yet processed, a reply being generated, output not yet sent. The rows that leave at `reply` and the interrupt rows produce it. Late output is what arrives after the close or the cancel. The part a client can see is counted in the cancel and interrupt rows: events of the ended reply that arrive after its `response.done`. The part that stays inside the server, an operation reaching a stage that has already dropped the session, shows only in the server's log, which `--server-log` collects.

## Definitions

- Left: when a client counts as gone. After a goodbye that is when the server has closed the socket; after a reset or a close frame it is when the client has let go of it. The server has done part of its cleanup by the first moment and none by the second, so cleanup latencies of the two are not comparable.
- Cleanup latency: from the moment the last client of a cycle has left until the server's session count is zero, polled every 5 ms on the mock's debug route. The wait gives up after `--cleanup-wait-s` (35 s by default, above the server's default 30 s cleanup timeout) and the cycle counts as timed out.
- Slots recovered: after the cleanup wait, a probe opens `--max-connections` sessions together up to `session.updated` and closes them. Every one admitted means no slot is still taken. A server refuses a session it has no slot for either at the connect, with a 503, or at `session.update`, with `admission_rejected`, which is where a turn-based pipeline counts its `max_sessions`; the probe goes as far as the second so that it sees both. The server frees a slot by removing the session from its table, so with a session count to read the time of this would repeat the cleanup latency and is not recorded; without a debug route it is recorded, at a resolution of about 0.1 s.
- Sessions left: the server's session count once the last probe and the final session have had their time to leave.
- Resource growth: resident memory, asyncio tasks, threads and open file descriptors of the server are read once per cycle, after the cleanup wait and before the probe. Memory growth is the slope of a line fitted to the readings after five warm-up cycles; the growth of a count is the last such reading minus the first. The mock reports all four. For a model server on the client's host, `--server-pid` gives the resident memory of that process and all its descendants together, because a model server runs its stages in child processes, the number of those processes and, on Linux, their threads and open file descriptors; sockets are among the descriptors, so a connection that is never closed shows there. `--device-memory` adds the memory in use on the host's GPUs as `nvidia-smi` reports it, with the same slope.
- Denied: the server refused the session, with a 503 at connect when the connection cap is reached, or with an `admission_rejected` error in answer to `session.update` when the model has no room. Any other failure is errored and fails the run.
- Cancelled in: from sending `response.cancel` to the `response.done` of the reply that was in progress, and the status that event carries. Against a deployment that does not cancel it is the time to the refusal, and the status is `not_supported`.
- Interrupted replies: in the interrupt scenario, the replies whose `response.done` has status `cancelled` with reason `interrupted`, per session, and the events of such a reply that arrive after that `response.done`. The tool counts them and measures cleanup as in every other scenario; it does not score whether or how fast the model gave way, which is what the Full-Duplex-Bench evaluation does.
- What is told apart, and what is not: natural interruption (`interrupted_replies`), explicit cancel support (`cancel_statuses`, with `not_supported` for a deployment that does not cancel) and the end of server output (`late_outputs`) are separate figures. When playback actually stops at the listener is not measured: the tool plays no audio, so it sees when the server stopped sending and not when a client stopped being heard.
- Server log: with `--server-log`, what the server's log file gains during the run is copied to `server.log` in the output, and the summary counts its tracebacks, its `ERROR` lines and the most frequent kinds of those lines. These are counts and not a check, because what a healthy model server logs is not known yet.
- Expired: the server closed the idle session with code 1008 and the name of the limit it waited on. Closed after the limit: the time from the client's last send to the close, minus the limit the server publishes. A negative value is a close before the limit.
- Denied first and readmitted in: of the reconnects, how many were denied before they were admitted, and the time from the end of a client's previous session to the start of the connect that was admitted.
- Final session: one ordinary recorded session after the last cycle, which must complete. `samples.json` is written after every cycle without it, so a run that dies keeps the cycles it finished, and a file without a final session is an unfinished run.

A summary has five checks: `no_sessions_left`, `slots_recovered`, `final_session_completed`, `no_errored_sessions` and `no_unexpected_denials`. A denial is expected only for the clients above `--max-connections` in a cycle; any further one fails the last check, and without `--max-connections` every denial does, so a server that admits nobody fails the run although none of its sessions errors. A recorded session counts as completed only if the server answered its `session.update`; one that was refused there and then closed in good order is denied. `failed_checks` lists the ones that failed and `unmeasured_checks` the ones the server gave no data for, so a run against a real server states what it could not check instead of passing it.

## Commands

- `mock` serves the real `/v1/realtime` route over an adapter that answers every unit with a short text reply, and adds a `/debug/lifecycle` route with the session count, resident memory, task and thread counts. `--protocol native` (default) is the shared-session path with a connection cap; `--protocol legacy` is the per-connection path with its turn detector replaced by one that never hears speech. `--unit-ms` sets the unit the mock publishes and `--no-cancel` makes it refuse `response.cancel`. `--reply-chunks N --chunk-delay-ms D` spreads each reply over N text deltas D ms apart, which keeps a reply in progress. `--model-slots`, `--close-delay-ms`, `--idle-timeout-s` and `--session-update-timeout-s` imitate a model that is full, slow to close, or quick to time out. `--takes-turns` serves a stand-in for a turn-based pipeline: a unit with any nonzero sample is speech, every silent unit of a reply carries one delta, and speech during a reply ends it as interrupted. The reply starts on the first silent unit after the speech, as a pipeline that waits for the end of the turn does. It exists to test the interrupt scenario's counting and the wait for a reply, and proves nothing about whether a model interrupts.
- `run` drives one scenario against any realtime URL and writes `samples.json` (one record per session and per cycle) and `summary.json`. Both carry when the run started and how long it took, the server identity the operator passes with `--server-revision` and `--model`, the settings the mock was started with, the unit and the cancel support the server publishes, the client's commit, with `--device-memory` the model and total memory of each GPU, and the input: its length, and for `--audio-file` the file path, the SHA-256 of the file and how it was converted to mono 16 kHz. The conversion is `benchmarks.duplex.v15_audio.normalize_audio`, the one the Full-Duplex-Bench recorder applies. The final session and the long scenario also leave the recorder's traces under `traces/`.

## Results

Not recorded yet. The matrix has one row per cell; each cell runs three times against a mock of its own, and a figure that differs between the three is given as its range. These are the commands:

```bash
start_mock() { python -m benchmarks.duplex.lifecycle_stress mock --port 8801 "$@" > mock.log & mock_pid=$!; until grep -q listening mock.log; do sleep 0.1; done; }
stop_mock() { kill -INT $mock_pid; wait $mock_pid; }
run() { python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8801/v1/realtime --server-revision $(git rev-parse HEAD) --model lifecycle-stress-mock "$@"; }

start_mock --max-connections 64 --reply-chunks 5 --chunk-delay-ms 50
run --scenario churn --concurrency 0 --cycles 50 --max-connections 64 --output runs/probe_only
run --scenario churn --concurrency 64 --cycles 50 --max-connections 64 --output runs/churn_fill
run --scenario churn --concurrency 80 --cycles 50 --max-connections 64 --output runs/churn_overflow
run --scenario drop --leave-point reply --leave-style reset --concurrency 64 --cycles 50 --max-connections 64 --output runs/drop_reply_reset
run --scenario cancel --concurrency 64 --cycles 50 --max-connections 64 --output runs/cancel
run --scenario cancel --leave-point reply --leave-style reset --concurrency 64 --cycles 50 --max-connections 64 --output runs/cancel_reply_reset
run --scenario reconnect --reconnects 2 --leave-point reply --leave-style reset --concurrency 64 --cycles 50 --max-connections 64 --output runs/reconnect_reply_reset
stop_mock

start_mock --max-connections 64 --idle-timeout-s 2 --session-update-timeout-s 2
run --scenario idle --idle-point open --concurrency 64 --cycles 20 --max-connections 64 --output runs/idle_open
stop_mock

start_mock --max-connections 64
run --scenario long --concurrency 64 --cycles 5 --max-connections 64 --output runs/long
stop_mock

start_mock --max-connections 64 --reply-chunks 5 --chunk-delay-ms 50 --unit-ms 1000
run --scenario cancel --packets-per-session 15 --concurrency 64 --cycles 20 --max-connections 64 --output runs/cancel_unit_1000
stop_mock

start_mock --max-connections 64 --reply-chunks 5 --chunk-delay-ms 50 --no-cancel
run --scenario cancel --concurrency 64 --cycles 20 --max-connections 64 --output runs/cancel_not_supported
stop_mock

start_mock --max-connections 64 --takes-turns --reply-chunks 20
run --scenario interrupt --audio-file talks_over.wav --concurrency 64 --cycles 10 --max-connections 64 --output runs/interrupt
run --scenario interrupt --audio-file talks_once.wav --concurrency 64 --cycles 10 --max-connections 64 --output runs/interrupt_control
stop_mock

start_mock --protocol legacy
run --scenario drop --protocol legacy --leave-point update --leave-style close_frame --concurrency 64 --cycles 20 --cleanup-wait-s 2 --output runs/legacy_drop_frame
stop_mock
```

The other drop, idle, reconnect and legacy rows differ from these only in `--leave-point`, `--leave-style` and `--idle-point`. `talks_over.wav` is 0.4 s of a 440 Hz tone, 0.8 s of silence, the tone again and 2 s of silence; `talks_once.wav` has silence in place of the second tone. To the stand-in pipeline any nonzero sample is speech.

## Environment

Where the tool was developed and its tests ran:

- Host: Apple M4 Pro, 48 GB, macOS 27.2; client and mock on the same host.
- Python 3.12.15, websockets 17.2, uvicorn 0.54.0, starlette 1.7.0, fastapi 0.142.2.
- Input: silence, 5 packets of 80 ms per short session and 15 s per long one.

## Limitations

- A real server has no debug route. Sessions left, cleanup latency and task counts are not measurable there; denials, close codes, the capacity probe, the final session and, on the same host, resident and device memory are, and the process, thread and open file counts of the server's process tree, the last two on Linux only. The thread and open file counts were tested against a stand-in `/proc` only.
- Without the debug route a leak shows as a capacity probe that is denied, which needs a connection cap. The legacy path has none, so there a leak shows only as memory growth.
- The capacity probe opens `--max-connections` sessions of its own after every cycle, and each takes a slot of the model while it is held. With a debug route, resources are read before it, and the probe-only row shows what it leaves behind by itself. Without one, a probe that got every slot is the only sign that the cycle's sessions are gone, so resources are read after it; a probe that fails is retried after 0.1 s, then 0.2 s and so on up to 2 s, to let the sessions it did get in close first. The probe's own sessions have to be gone before the next cycle, or they deny it. With no session count, the tool probes again after 0.1 s, 0.2 s and so on until every slot is back, then waits as long again as that took, for the sessions of that last probe. So a cycle without a debug route costs at least two probes, and the wait assumes the server releases a probe's sessions about as fast each time. `capacity_recovery_ms` is taken before any of this and moves in steps of the retry interval, so it reads as an upper bound.
- How often a reconnect is denied first depends on how much work the server has in flight when the client resets, which on the mock is set by `--unit-ms` and the reply flags. The figure describes that configuration and does not carry over to a model.
- Resident memory on macOS is reclaimed and compressed by the system, and readings moved by tens of megabytes in either direction between cycles of a healthy run. A memory slope from such a host is not evidence that nothing grows; that needs a Linux host.
- Device memory is what all processes hold on all GPUs of the host, cached and reserved memory included. A leaked session may not move it, and another job on the same GPU does. The reading was tested against a stand-in `nvidia-smi` only.
- The server's output buffer drops every event of a reply after that reply's `response.done`, so `late_outputs` is zero by construction on the current server. The figure guards that guarantee; a nonzero value means the guarantee broke.
- Three times start or end a little off their names. The cleanup latency of the long and interrupt rows starts after the recorder has finished writing and the tool has read the trace, tens of milliseconds late. The time an idle session is closed after its limit includes the time the server takes to close it. `readmit_ms` includes reconnects that were still denied when the cleanup wait ran out.
- On macOS the mock's debug route runs `ps` for its resident memory on every poll of the session count, which takes about 11 ms, so the polls come about 18 ms apart instead of 5 ms. Cleanup latency on the mock carries 15 to 20 ms of that: a churn row near 20 ms means no more than one poll, and its true value is near zero.
- A drop at the reply tests a leave mid-reply only if the reply was still going when the client left. `drops_mid_reply` and `drops_after_reply` count the two; a reply that is short or quick is over first, as on the mock without `--chunk-delay-ms`, and such a row says nothing about a leave mid-reply.
- The debug route counts Python threads only, and gives no open file count on macOS.
- A failed reading after the first, of `nvidia-smi`, `ps` or the debug route, is left empty and the run goes on. The first reading has to succeed.
- The mock ends a cancelled reply at its next delta and answers silence. A model does neither by itself: it needs speech in the input to reply, and without a reply the cancel scenario has nothing to cancel and its sessions time out as errored.
- The client sends 16 kHz audio, so it cannot drive a deployment that takes another input rate, such as PersonaPlex at 24 kHz.
- The idle scenario reads its limit from `/v1/realtime/capabilities`, which only the shared-session path serves, so the tool refuses it on the legacy protocol. With the default limits of 60 s and 300 s a cycle takes that long. The server puts off the idle limit while a reply is in progress, so against a server that is still answering, the close comes later by that much; the idle rows use a mock that answers at once.
- One client process runs up to 80 sessions at once in the matrix; above about 128 the duplex client needs its own host, see #3288.
- A run records the operating system of the client's host and, with `--device-memory`, the GPU models. CPU model, host memory and driver versions are not collected; they belong in the description of a run, as in Environment above.
- The server revision is what the operator passes. The client's commit is the checkout's HEAD, so uncommitted edits to the tool do not show in a run's record.
