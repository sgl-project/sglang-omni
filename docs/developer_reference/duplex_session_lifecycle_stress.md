# Duplex Session Lifecycle Stress

This is the issue #2994 item C11 record: whether a `/v1/realtime` server cleans up after sessions that open, talk briefly and leave, in bulk and repeatedly, and whether it then admits new sessions as if nothing had happened. The tool drives the real realtime route of a model-free server on one CPU host and reports per cycle what the server still holds. No model or GPU was involved in the numbers below; the same client takes any realtime URL, so the model-backed run is the same command against a GPU server.

The shared-session path, which every deployment with a realtime factory uses, released every session in every scenario, including clients that reset the TCP connection mid-send. The legacy per-connection path, which serves pipelines without a realtime factory, keeps a session forever whenever the client sends its close frame before reading the reply to its last event, which an ordinary client does when the user leaves right after connecting. Each such session leaks about 27 kB of host memory with the model-free turn detector; with the Silero model that each legacy connection loads, #3561 measured about 5.6 MB per connection. #3504 fixes the leak; the before and after rows below are the same command against main and against that branch.

On the shared path the connection cap is exact, both idle limits close a silent client within 30 ms of the published value at the median, and `response.cancel` ends the reply in progress with nothing of it sent afterwards. A client that resets its connection and reconnects at once while the cap is full gets a 503 until the server has closed its old session: half of such reconnects were denied first against the plain mock, and nearly all of them, for the length of the close, once the adapter took 100 ms to close.

## Scenarios

- churn: N clients connect at once, send `session.update`, send a few 80 ms packets of silence, close with `session.close` (or a close frame on the legacy path) and wait for the server's close frame. This repeats for M cycles.
- drop: the same, but the client leaves before reading the server's replies. `--drop-point append` leaves after the last packet, `--drop-point update` right after `session.update`. `--drop-style abort` resets the TCP connection; `--drop-style close` sends a close frame and waits for the server to answer it.
- fill: the client opens `--max-connections` sessions and holds them open, then tries `--concurrency` more at once, which must all be denied. The holders then close and the cycle probes the whole cap again.
- idle: N clients connect and stop sending, before `session.update` (`--idle-point created`) or after the last packet (`--idle-point open`), and read until the server closes them. The run reads the limit the server enforces from `/v1/realtime/capabilities`.
- reconnect: N clients each open `--reconnects` further sessions per cycle, each the moment the previous one ended. The server has no session resume, so a reconnect is a new session; one that is denied retries every 20 ms.
- cancel: each client sends one packet, sends `response.cancel` on the first `response.created`, waits for that reply's `response.done`, then sends the remaining packets and leaves.

Reconnect and cancel sessions leave like churn by default and like drop when `--drop-point` is given. Against a mock started with `--reply-chunks` and `--chunk-delay-ms`, every session that leaves does so with a reply in progress and further units waiting, which is the pending work and late output case.

## Definitions

- Sessions left: the server's own session count after the last cycle, read from the mock's debug route. A real server does not expose it.
- Cleanup latency: from the moment the last client of a cycle has left until the server's session count reaches zero, polled every 20 ms. The wait gives up after `--cleanup-wait-s` (default 35 s, above the server's default 30 s adapter cleanup timeout) and the cycle is marked timed out.
- Capacity recovery: from the same moment until a probe that opens `--max-connections` sockets at once is admitted in full. This is the signal a real server gives, since a held slot shows up as a 503 denial.
- RSS growth per cycle: the server's resident set after the last cycle minus after the fifth, divided by the cycles in between. The first five cycles are warm-up: the mock grows up to 15 MB once while its allocator and thread pool fill, which a run of ten cycles would otherwise read as 1.5 MB per cycle. A run of five cycles or fewer has no such figure. The mock reports its own RSS; for a real server on the same host pass `--server-pid`.
- Denied: the server refused the session, either with a 503 at connect, when the connection cap is reached, or with an `admission_rejected` error in answer to `session.update`, when the model has no room. The client closes the socket and does not retry. Any other error event ends the session as errored, with the server's error recorded.
- Above the cap: of the connections the fill scenario tries while the cap is held, how many were denied and how many got in. Any admitted one fails the run.
- Expired: the server closed the idle session with code 1008 and the reason of the limit it waited on, `session_update_timeout` before `session.update` and `idle_timeout` after. Closed after the timeout: the time from the client's last send, or from the start of its connect when it never sent, to the close, minus the published limit. The server's clock restarts no earlier than that send, so a negative value is a close before the limit and fails the run.
- Denied first and readmitted in: of the reconnects, how many got a denial before they were admitted, and the time from the end of a client's last session to the start of the connect that was admitted, over all reconnects.
- Cancelled in: from sending `response.cancel` to the `response.done` of the reply that was in progress. Not cancelled: that `response.done` carried another status than `cancelled`. Late outputs: events of the cancelled reply that arrived after its `response.done`, counted when the session closes with `session.close`. This is explicit cancellation and the end of server output; it says nothing about natural interruption by speech or about when playback stops at the client.
- Final probe: one ordinary churn session run after the last cycle, which must complete.

A summary lists `unmet_targets` for the checks that failed and `targets_not_measurable` for those the server gave no data for, so a run against a real server states what it could not check instead of passing it.

## Results

### CPU-only mock, main branch

Every row is one client process of 64 concurrent sessions per cycle, 5 packets of 80 ms silence per session, against a mock with a 128-connection cap (deployment) or none (legacy). Cleanup and recovery are per cycle; RSS is the mock's resident set before the first and after the last cycle.

| Scenario | Path | Client leaves by | Cycles | Sessions | Sessions left | Cleanup p50 / p99 ms | Capacity recovery p99 ms | Server RSS MB | Final probe |
|---|---|---|---|---|---|---|---|---|---|
| churn | shared | `session.close`, waits for `session.closed` | 50 | 3,200 completed, close 1000 | 0 | 18 / 20 | 20 | 524 to 539 | completed |
| drop | shared | TCP reset after the last packet | 50 | 3,200 dropped | 0 | 65 / 123 | 123 | 539 to 544 | completed |
| drop | shared | close frame right after `session.update`, replies unread | 50 | 3,200 dropped | 0 | 17 / 18 | 18 | 544 to 545 | completed |
| churn | legacy | close frame after the last packet | 20 | 1,280 completed, close 1000 | 0 | 30 / 33 | not measurable | 521 to 530 | completed |
| drop | legacy | TCP reset after the last packet | 20 | 1,280 dropped | 0 | 32 / 35 | not measurable | 530 to 536 | completed |
| drop | legacy | close frame right after `session.update`, replies unread | 20 | 1,280 dropped | 1,280 | timed out in 20 of 20 cycles | not measurable | 536 to 570 | completed |

The shared path releases a reset connection in about 65 ms at the median, with a tail past 120 ms, and a cleanly closed one in under 20 ms. The legacy path releases both, except when the client's close frame arrives before the server has answered `session.update`: then `RealtimeSession.teardown` calls `websocket.close()` on a socket whose reply already failed, the call raises, and `RealtimeSessionManager.close` never removes the entry. Every one of the 1,280 such sessions stayed registered, 27 kB each with the model-free detector, and the server kept admitting new sessions because this path has no connection cap. The legacy churn row's 18 new threads are the default thread pool that `asyncio.to_thread` fills for the turn detector on the first cycles; they did not grow afterwards. Latencies move by a few milliseconds between runs on this host: an earlier run of the same commands measured 23 ms for the legacy churn median that reads 30 ms here, and one 145 ms cycle in the shared close-frame drop that did not recur. The session counts, the timeouts and the RSS growth of the leaking row were the same in both.

### CPU-only mock, legacy path before and after #3504

The same two legacy rows against #3504 at `074ab1d86`, which pops the manager entry even when teardown raises. That branch starts from `c6158da69`, 36 commits before the main commit measured above; `sglang_omni/serve/realtime` and `sglang_omni/audio` did not change between the two, and the changes to `openai_api.py` in that range are outside the realtime route.

| Scenario | Client leaves by | Branch | Sessions left | Cleanup p50 / p99 ms | Server RSS MB |
|---|---|---|---|---|---|
| churn | close frame after the last packet | main | 0 | 30 / 33 | 521 to 530 |
| churn | close frame after the last packet | #3504 | 0 | 31 / 34 | 523 to 532 |
| drop | close frame right after `session.update`, replies unread | main | 1,280 | timed out | 536 to 570 |
| drop | close frame right after `session.update`, replies unread | #3504 | 0 | 18 / 21 | 532 to 532 |

### CPU-only mock, fill, idle, reconnect and cancel

All rows are the shared path. Reconnects is the number of sessions opened right after another one ended; every cycle ends with the same cleanup wait and capacity probe as above.

| Scenario | Mock | Client | Cycles | Sessions | Sessions left | Cleanup p50 / p99 ms | Scenario figures | Server RSS MB | Final probe |
|---|---|---|---|---|---|---|---|---|---|
| fill | cap 128 | holds 128 sessions, tries 64 more, closes with `session.close` | 20 | 2,560 held and completed | 0 | 16 / 20 | above the cap: 1,280 denied, 0 admitted | 523 to 540 | completed |
| reconnect | cap 128 | 128 clients, 5 reconnects each after `session.close` | 10 | 7,680 completed | 0 | 18 / 20 | 0 of 6,400 reconnects denied first | 540 to 541 | completed |
| reconnect | cap 128 | 128 clients, 5 reconnects each after a TCP reset | 10 | 7,680 dropped | 0 | 22 / 29 | 3,236 of 6,400 denied first; readmitted in 22 / 166 ms, 403 ms at most | 541 to 551 | completed |
| reconnect | cap 32, adapter closes in 100 ms | 32 clients, 5 reconnects each after a TCP reset | 10 | 1,920 dropped | 0 | 146 / 155 | 1,590 of 1,600 denied first; readmitted in 116 / 239 ms, 274 ms at most | 521 to 545 | completed |
| idle | cap 128, both limits 2 s | 64 clients silent before `session.update` | 10 | 640 expired, close 1008 `session_update_timeout` | 0 | 22 / 22 | closed 24 / 29 ms after the limit | 519 to 531 | completed |
| idle | cap 128, both limits 2 s | 64 clients silent after the last packet | 10 | 640 expired, close 1008 `idle_timeout` | 0 | 22 / 23 | closed 14 / 115 ms after the limit | 531 to 535 | completed |
| cancel | cap 128, replies of 5 deltas 50 ms apart | cancels the first reply, closes with `session.close` | 20 | 1,280 completed | 0 | 20 / 21 | cancelled in 51 / 54 ms, 0 not cancelled, 0 late outputs | 521 to 538 | completed |
| cancel | cap 128, replies of 5 deltas 50 ms apart | cancels the first reply, TCP reset after the last packet | 20 | 1,280 dropped | 0 | 65 / 69 | cancelled in 51 / 54 ms, 0 not cancelled, late outputs unread | 538 to 539 | completed |
| drop | cap 128, replies of 5 deltas 50 ms apart | TCP reset after the last packet | 20 | 1,280 dropped | 0 | 25 / 61 | - | 539 to 540 | completed |

No run left a session, timed out a cleanup wait or errored a session, and every capacity probe got the whole cap back. The cap held exactly: none of the 1,280 connections tried above it got in, and the cap was free 16 ms after the holders closed. Each idle limit closed all 640 clients with the reason of that limit, none before it.

A reconnect after `session.close` was never denied: the slot was free by the time the client saw the socket close. After a reset the server still holds the old session while it closes it, so a client that reconnects at once at a full cap is denied until then. Against the plain mock that was half of the reconnects and 22 ms at the median; with an adapter that takes 100 ms to close it was 1,590 of 1,600 and 116 ms. The time a dropped client waits to get back in is the time the server takes to close its old session, which with a model is the time its adapter takes to close.

The mock ends a cancelled reply at its next delta, hence the 51 ms; the figure that carries over to a real server is that every `response.cancel` got a `cancelled` `response.done` and nothing of that reply followed it. The last three rows leave with a reply in progress and further units waiting; the server released those sessions within 70 ms.

### Model-backed

Not recorded yet. The intended command is `run` with the GPU server's URL, its `--max-connections` for the probe, and `--server-pid` when the client runs on the same host.

## Tool

`benchmarks/duplex/lifecycle_stress` has three subcommands.

- `mock` serves the real `/v1/realtime` route of `sglang_omni.serve.openai_api` over an adapter that answers each 20 ms unit with a short text response, and adds a `/debug/lifecycle` route with the session count, RSS, task, thread and file descriptor counts. `--mode deployment` (default) is the shared-session path with a connection cap; `--mode legacy` is the per-connection path with the Silero and Smart Turn loads replaced by a detector that never hears speech. `--model-slots N` makes the stand-in model hold at most N sessions and reject the rest on `session.update`, as a model server whose capacity is below its connection cap does. `--reply-chunks N --chunk-delay-ms D` spreads each reply over N text deltas D ms apart and lets `response.cancel` end it at the next delta; `--close-delay-ms` makes the adapter slow to close; `--idle-timeout-s` and `--session-update-timeout-s` shorten the two limits the idle scenario waits on.
- `run` drives one scenario against any realtime URL and writes `samples.json` and `summary.json`. Both carry the server identity the operator passes (`--server-revision`, `--model`, and optionally `--model-revision` and `--runtime`), the model IDs the server lists at `/v1/models`, and the client's commit, Python and package versions.
- `report` pools run directories into one summary per scenario, mode and server revision and writes `report.md`, so runs against two branches stay in separate rows.

```bash
server_revision=$(git rev-parse HEAD)
python -m benchmarks.duplex.lifecycle_stress mock --port 8765 --max-connections 128 > mock.log &
mock_pid=$!
until grep -q listening mock.log; do sleep 0.2; done
python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8765/v1/realtime --server-revision $server_revision --model lifecycle-stress-mock --scenario churn --concurrency 64 --cycles 50 --max-connections 128 --server-pid $mock_pid --output runs/churn
python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8765/v1/realtime --server-revision $server_revision --model lifecycle-stress-mock --scenario drop --drop-point append --drop-style abort --concurrency 64 --cycles 50 --max-connections 128 --server-pid $mock_pid --output runs/drop
kill -INT $mock_pid; wait $mock_pid
```

```bash
server_revision=$(git rev-parse HEAD)
python -m benchmarks.duplex.lifecycle_stress mock --port 8766 --mode legacy > legacy.log &
mock_pid=$!
until grep -q listening legacy.log; do sleep 0.2; done
python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8766/v1/realtime --server-revision $server_revision --model lifecycle-stress-mock --scenario drop --drop-point update --drop-style close --close-style frame --concurrency 64 --cycles 20 --cleanup-wait-s 2 --server-pid $mock_pid --output runs/legacy_drop
kill -INT $mock_pid; wait $mock_pid
```

The fill, idle, reconnect and cancel rows, each group against a mock started with the flags it needs:

```bash
start_mock() { python -m benchmarks.duplex.lifecycle_stress mock --port 8770 --max-connections 128 "$@" > mock.log & mock_pid=$!; until grep -q listening mock.log; do sleep 0.2; done; }
stop_mock() { kill -INT $mock_pid; wait $mock_pid; }
run() { python -m benchmarks.duplex.lifecycle_stress run --url ws://127.0.0.1:8770/v1/realtime --server-revision $(git rev-parse HEAD) --model lifecycle-stress-mock --max-connections 128 --server-pid $mock_pid "$@"; }
start_mock
run --scenario fill --concurrency 64 --cycles 20 --output runs/fill
run --scenario reconnect --concurrency 128 --reconnects 5 --cycles 10 --output runs/reconnect_close
run --scenario reconnect --drop-point append --concurrency 128 --reconnects 5 --cycles 10 --output runs/reconnect_reset
stop_mock
start_mock --close-delay-ms 100 --max-connections 32
run --scenario reconnect --drop-point append --concurrency 32 --max-connections 32 --reconnects 5 --cycles 10 --output runs/reconnect_reset_slow_close
stop_mock
start_mock --idle-timeout-s 2 --session-update-timeout-s 2
run --scenario idle --idle-point created --concurrency 64 --cycles 10 --output runs/idle_created
run --scenario idle --idle-point open --concurrency 64 --cycles 10 --output runs/idle_open
stop_mock
start_mock --reply-chunks 5 --chunk-delay-ms 50
run --scenario cancel --concurrency 64 --cycles 20 --output runs/cancel_close
run --scenario cancel --drop-point append --concurrency 64 --cycles 20 --output runs/cancel_reset
run --scenario drop --drop-point append --concurrency 64 --cycles 20 --output runs/drop_reset_slow_reply
stop_mock
```

```bash
python -m benchmarks.duplex.lifecycle_stress report runs/churn runs/drop runs/legacy_drop --output results
```

## Environment

- Host: Apple M4 Pro, 48 GB, macOS 27.2; client and mock on the same host.
- Python 3.12.15, websockets 17.2, uvicorn 0.54.0, starlette 1.7.0, fastapi 0.142.2.
- Commit: main `c0776499c`; the fixed rows use #3504 at `074ab1d86`. Each run was recorded with `--server-revision` set to the commit its mock served. The fill, idle, reconnect and cancel rows ran this branch on top of main `dace4cdc6`, which leaves `sglang_omni` as it is on main.
- Measured on 2026-11-11; the fill, idle, reconnect and cancel rows on 2026-11-12.

## Limitations

- A real server has no debug route, so sessions left, cleanup latency and task counts are not measurable there; capacity recovery, denials, close codes, the final probe and, on the same host, RSS are. None of this has been run against a model-backed server yet.
- Without the debug route a leak shows as a capacity probe that is denied, which needs a connection cap. The legacy path has none: there the leak of the last row shows only as RSS growth, 1.95 MB per cycle against at most 0.31 in the rows that release every session, and without `--server-pid` not at all.
- The capacity probe opens `--max-connections` sessions and expects all of them admitted, so against a model that holds fewer sessions than the server's connection cap, pass the number the model holds.
- The capacity probe opens and closes `--max-connections` sockets of its own, so it perturbs the server it measures; the tool waits for those sessions to clear (or 0.5 s without a debug route) before the next cycle.
- The mock answers every unit with text and never speaks; its RSS per leaked session is a floor, not the cost with a model.
- One client process ran up to 64 sessions at once here at up to 0.4 of one core; above about 128 the duplex client needs its own host, see #3288.
- The idle scenario reads the limit from `/v1/realtime/capabilities`, which only the shared-session path serves, so it does not run against the legacy path. A server with the default 300 s idle timeout takes five minutes per cycle.
- The cancel scenario sends silence. A model that does not answer silence never sends `response.created`, and the session ends as errored after its 30 s timeout; a model-backed run needs speech in the first packet. A server whose adapter does not cancel answers `response.cancel` with a `not_supported` error, and those sessions count as errored with that code.
- A reconnect that is denied retries every 20 ms, so readmission times have that resolution, and every retry is a connection of its own. On loopback a closed connection keeps its client port for 30 s and this host has 16,384 of them, so one client host sustains about 540 connections per second. A first attempt at the slow-close reconnect row with 128 clients and a 200 ms close opened about 1,070 per second and 712 of its 7,680 sessions failed to connect with `EADDRNOTAVAIL`; the row in Results uses 32 clients and a 100 ms close.
- The mock and the client disable WebSocket pings; a real uvicorn server pings every 20 s, which does not affect these short sessions.
- The server revision is what the operator passes, not something the tool reads from the server. The client's commit is the checkout's HEAD, so uncommitted edits to this tool do not show in a run's record.
