# Duplex Session Lifecycle Stress

This is the issue #2994 item C11 record: whether a `/v1/realtime` server cleans up after sessions that open, talk briefly and leave, in bulk and repeatedly, and whether it then admits new sessions as if nothing had happened. The tool drives the real realtime route of a model-free server on one CPU host and reports per cycle what the server still holds. No model or GPU was involved in the numbers below; the same client takes any realtime URL, so the model-backed run is the same command against a GPU server.

The shared-session path, which every deployment with a realtime factory uses, released every session in every scenario, including clients that reset the TCP connection mid-send. The legacy per-connection path, which serves pipelines without a realtime factory, keeps a session forever whenever the client sends its close frame before reading the reply to its last event, which an ordinary client does when the user leaves right after connecting. Each such session leaks about 27 kB of host memory with the model-free turn detector; with the Silero model that each legacy connection loads, #3561 measured about 5.6 MB per connection. #3504 fixes the leak; the before and after rows below are the same command against main and against that branch.

## Scenarios

- churn: N clients connect at once, send `session.update`, send a few 80 ms packets of silence, close with `session.close` (or a close frame on the legacy path) and wait for the server's close frame. This repeats for M cycles.
- drop: the same, but the client leaves before reading the server's replies. `--drop-point append` leaves after the last packet, `--drop-point update` right after `session.update`. `--drop-style abort` resets the TCP connection; `--drop-style close` sends a close frame and waits for the server to answer it.

Cancel mid-reply, idle timeout expiry, immediate reconnection and filling the connection cap are planned as further scenarios of the same tool.

## Definitions

- Sessions left: the server's own session count after the last cycle, read from the mock's debug route. A real server does not expose it.
- Cleanup latency: from the moment the last client of a cycle has left until the server's session count reaches zero, polled every 20 ms. The wait gives up after `--cleanup-wait-s` (default 35 s, above the server's default 30 s adapter cleanup timeout) and the cycle is marked timed out.
- Capacity recovery: from the same moment until a probe that opens `--max-connections` sockets at once is admitted in full. This is the signal a real server gives, since a held slot shows up as a 503 denial.
- RSS growth per cycle: the server's resident set after the last cycle minus before the first, divided by the number of cycles. The mock reports its own RSS; for a real server on the same host pass `--server-pid`.
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

### Model-backed

Not recorded yet. The intended command is `run` with the GPU server's URL, its `--max-connections` for the probe, and `--server-pid` when the client runs on the same host.

## Tool

`benchmarks/duplex/lifecycle_stress` has three subcommands.

- `mock` serves the real `/v1/realtime` route of `sglang_omni.serve.openai_api` over an adapter that answers each 20 ms unit with a short text response, and adds a `/debug/lifecycle` route with the session count, RSS, task, thread and file descriptor counts. `--mode deployment` (default) is the shared-session path with a connection cap; `--mode legacy` is the per-connection path with the Silero and Smart Turn loads replaced by a detector that never hears speech.
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

```bash
python -m benchmarks.duplex.lifecycle_stress report runs/churn runs/drop runs/legacy_drop --output results
```

## Environment

- Host: Apple M4 Pro, 48 GB, macOS 27.2; client and mock on the same host.
- Python 3.12.15, websockets 17.2, uvicorn 0.54.0, starlette 1.7.0, fastapi 0.142.2.
- Commit: main `c0776499c`; the fixed rows use #3504 at `074ab1d86`. Each run was recorded with `--server-revision` set to the commit its mock served.
- Measured on 2026-11-11.

## Limitations

- A real server has no debug route, so sessions left, cleanup latency and task counts are not measurable there; capacity recovery, denials, close codes, the final probe and, on the same host, RSS are.
- The capacity probe opens and closes `--max-connections` sockets of its own, so it perturbs the server it measures; the tool waits for those sessions to clear (or 0.5 s without a debug route) before the next cycle.
- The mock answers every unit with text and never speaks; its RSS per leaked session is a floor, not the cost with a model.
- One client process ran up to 64 sessions at once here at up to 0.4 of one core; above about 128 the duplex client needs its own host, see #3288.
- The mock and the client disable WebSocket pings; a real uvicorn server pings every 20 s, which does not affect these short sessions.
- The server revision is what the operator passes, not something the tool reads from the server. The client's commit is the checkout's HEAD, so uncommitted edits to this tool do not show in a run's record.
