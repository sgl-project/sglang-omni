# Duplex Load Generator Pacing

This is the issue #2994 item 4.3 record: whether the duplex benchmark client keeps its 80 ms input pacing when many sessions stream at once, so that a recording measures the server and not the load generator. It is a CPU-only study against a mock server on one host. No model, GPU or real server was involved.

The client did fall behind, and the cause was its own trace writes, not event-loop scheduling or WebSocket back-pressure. `run_session` wrote every sent and received event, base64 audio included, to the trace with a line-buffered write on the event loop. When the kernel wrote those pages back to disk, an append could wait for the page for hundreds of milliseconds, and every session on that loop missed its packet slots. One writer thread per client process now does these writes. With it, 30 s and 90 s sessions stay within the 80 ms tolerance up to 128 sessions in one process and up to 4 processes of 32 sessions, and 4 processes of 64 sessions fail far less often. 256 sessions of 90 s in one process are at the limit of its CPU: 1 of 4 such runs failed.

## Pacing and what counts as late

`benchmarks/duplex/client.py` schedules packet N of a session at the session start plus N times 80 ms on an absolute clock. A packet is late when its send start differs from that schedule by more than one packet, 80 ms. A recording with a late packet is not a valid server capacity measurement, see the `pacing violations` row in [full_duplex_bench.md](full_duplex_bench.md). A run is called pacing-unqualified here when any of its packets is late.

The deviation of a packet is its `start_s` in `input-send-receipts.json` minus the `start_s` of packet 0 of the same session minus N times 80 ms, the same definition the repository checks use. `start_s` is taken when the client records the event, before serialization and before `websocket.send()`, so it is when the client was ready to send, not when bytes left the host. The probe therefore times `websocket.send()` on its own to look for back-pressure.

## Cause

Two kinds of evidence point at the trace writes.

- An external sampler read `/proc/<pid>/schedstat`, `stat` and `wchan` of every client process every 5 ms, without running inside the measured process. During event-loop stalls over 100 ms the process was mostly in uninterruptible sleep (state D) in the kernel function `folio_wait_bit_common`, with some `do_get_write_access` (ext4 journal), and spent no time waiting for a CPU. In the 90 s runs before the change, 52 of the stalls over 100 ms were of this kind.
- At 256 sessions the traces grow by about 28 MB/s, 110 kB/s per session, almost all of it base64 audio. The host keeps dirty pages for 30 s (`dirty_expire_centisecs` 3000) and then writes them back to a 7,200 rpm disk. Of 18 runs with a disk stall, the first stall came 27–36 s after the start in 10 and 63–69 s after it in 8.

Ruled out, each with repeated runs: a CPU quota (none, `nr_throttled` stayed 0), CPU migration (pinning the mock and every client to its own CPU did not help), the Python garbage collector (no collection over 100 ms overlapped a stall), the volume of server replies (0, 1, 3 or 6 replies per packet all showed stalls), and WebSocket back-pressure (`websocket.send()` p99 stayed at or below 0.25 ms). Keeping the trace in memory until the session ended removed the disk stalls, which confirms the cause, but its growing in-memory buffers caused CPU stalls of their own, and all 8 such runs still failed, so it is not the fix.

## Change

A `TraceWriter` thread, one per client process, writes the trace and receipts of every session. The event loop only queues the serialized line. The thread writes each session's queued lines in one batch every 0.5 s, and gathers the sessions that end within 50 ms of each other into one batch. The session closes its trace only after every line is on disk, as before, and a failed write still raises from `run_session`. It now raises when the session ends rather than at the failed event.

Visible differences:

- `input-send-receipts.json` is written without indentation. Readers parse it with `json.loads`, so only its bytes and hash change. The indented dump runs in pure Python and held the GIL from the pacing loop.
- While a session records, its trace on disk lags by up to 0.5 s.
- `run_session` returns about 50 ms later, while its last batch is written.
- Pacing jitter at low load is larger: for 128 sessions in one process over 30 s, the pooled p99 deviation rose from 5.0 ms to 16.6 ms and the median run maximum from 14 ms to 37 ms. Each batch takes the GIL from the loop for a moment. This trades a bounded jitter well inside 80 ms for stalls of up to 1.3 s.

The design was chosen by measurement. The disk load on the host changed over time: at 1 × 256 sessions and 30 s, the unchanged client passed 5 of 5 runs on one day and 1 of 3 on the next. The rows marked interleaved ran in one batch in random order; the others ran on their own and are weaker evidence.

| Variant, 30 s unless noted | 4 × 64 unqualified | 1 × 256 unqualified |
|---|---|---|
| Unchanged client (interleaved) | 3 of 3 | 2 of 3 |
| One thread per session, a write per event, 90 s | 1 of 4 | 4 of 4, deviations up to 71 s, every session incomplete |
| One thread per session, batches every 0.5 s (3 of the runs interleaved) | 1 of 11 | 11 of 11 |
| One thread per session, batches every 2 s (interleaved) | 3 of 3 | 3 of 3 |
| One thread per process, prototype (interleaved) | 0 of 3 | 0 of 3 |
| One thread per process, every session end wakes it, receipts in executor threads | 5 of 5 | 0 of 5 |
| One thread per process, 50 ms end batches, indented receipts (5 of the 4 × 64 runs interleaved) | 2 of 8 | 0 of 3 |
| One thread per process, 50 ms end batches, compact receipts, this change (5 of the 4 × 64 runs interleaved) | 0 of 10 | 0 of 5 |

With a thread per session, the loop waited for the GIL, which those threads took hundreds of times a second, and a 256 session process needed more than one core. With one thread, the remaining stalls came when many sessions ended together and the loop again waited for the GIL. Gathering the ends into batches and dumping the receipts compactly in the writer thread removed them in the runs measured.

## Results

P99 values pool the packet deviations across all runs of one layout: P processes of S sessions each, every session streaming silence of 16 kHz PCM16 in 80 ms packets to the mock server, which replies to each packet with one audio delta, one text delta and one `sglang.unit.done`. Every session of every run here completed. The maximum is the median over runs of each run's largest deviation, with the range in brackets. The unchanged client's 90 s runs and the changed client's 90 s runs were recorded hours apart, not interleaved.

### 90 s sessions

| Layout | Unchanged: unqualified runs | late packets | p99 (ms) | maximum (ms) | Changed: unqualified runs | late packets | p99 (ms) | maximum (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 × 64 | 1 of 3 | 7 | 11.7 | 49 [33–102] | 0 of 4 | 0 | 14.8 | 40 [38–48] |
| 1 × 128 | 6 of 7 | 3,571 | 27.3 | 158 [74–584] | 0 of 4 | 0 | 20.0 | 51 [49–55] |
| 1 × 256 | 3 of 3 | 53,835 | 864.8 | 700 [655–1,170] | 1 of 4 | 618 | 40.8 | 79 [67–112] |
| 4 × 32 | 2 of 3 | 495 | 8.0 | 476 [45–598] | 0 of 4 | 0 | 8.8 | 43 [34–53] |
| 4 × 64 | 7 of 7 | 11,047 | 27.4 | 901 [407–1,296] | 1 of 4 | 44 | 17.6 | 58 [49–138] |

Stalls over 100 ms in disk waits fell from 52 to 0. The remaining late packets have other causes. In 1 × 256 the process was running on its CPU through the stalls and used 0.92 cores, so one process has no headroom for 256 sessions. In 4 × 64 all 44 late packets fell at 90.2–90.3 s in one run, while the sessions of each process were ending, and the loop was waiting for the GIL.

### 30 s sessions

| Layout | Unchanged: unqualified runs | late packets | p99 (ms) | maximum (ms) | Changed: unqualified runs | late packets | p99 (ms) | maximum (ms) |
|---|---|---|---|---|---|---|---|---|
| 1 × 64 | 0 of 5 | 0 | 12.3 | 25 [15–28] | 0 of 5 | 0 | 14.2 | 36 [35–37] |
| 1 × 128 | 0 of 5 | 0 | 5.0 | 14 [7–14] | 0 of 5 | 0 | 16.6 | 37 [36–49] |
| 1 × 256 | 2 of 8 | 18,389 | 322.5 | 40 [19–755] | 0 of 5 | 0 | 30.4 | 54 [47–80] |
| 4 × 32 | 0 of 5 | 0 | 8.0 | 34 [33–36] | 0 of 5 | 0 | 6.9 | 30 [12–31] |
| 4 × 64 | 12 of 23 | 1,823 | 16.2 | 81 [26–750] | 0 of 5 | 0 | 17.4 | 41 [41–58] |

A 30 s session ends near the 30 s writeback age, so 30 s runs see fewer disk stalls than 90 s runs, which is why the 90 s table is the stronger test.

### Guidance

On a host like this one, keep a client process at or below 128 sessions and spread more sessions over several processes, for example 4 processes of 32. Above 128 sessions per process, or at 4 × 64, check the pacing of every run, as `benchmarks/duplex/serving.py` does for every level: runs in both layouts went past 80 ms here. These limits come from one host and the mock server, see the limitations.

## Recording path

The study above drove `run_session` directly. The recorder that produces benchmark runs, `run_samples` in `benchmarks/duplex/v15_runner.py`, does more on the same process, and on a 2 CPU host it still missed the 80 ms tolerance at 24 and 32 sessions with the writer thread in place.

### Causes

- Post-processing ran in threads of the recording process. When a session ended, `replay_run`, the `report.json` write and output reconstruction parsed the whole trace and validated it with pydantic. Long C calls and generation 2 garbage collections over the large heap held the GIL for 100 ms and more, the largest collection pause measured was 145 ms, and every open session missed a packet slot at the same moment. The recorder used only 0.15–0.36 cores, so it was not short of CPU.
- `run.json` was dumped with indentation on the event loop at every session end, about 5 ms at 64 sessions and growing with the run.
- The client scheduled packets from a start time taken before its send loop, while the pacing check measures from the first append. A late first packet therefore made every later packet look early by the same amount.
- Every sent event was serialized twice, once for the socket and once for the trace, and every received audio event was decoded and encoded again for the trace.

Checked and left unchanged: `websocket.send()` back-pressure, p99 at or below 0.25 ms as measured above; per-packet logging, there is none; the receive loop, whose run without yielding is bounded by the 16 frame queue; and the sleep, which was already on an absolute deadline.

### Changes

- Post-processing runs in a pool of spawned processes, at most 4 and a quarter of the CPUs the recorder may use, started before the first session and run at `nice` 10. Workers return small summaries. `run_samples` takes a `postprocess` executor for tests and closes the pool it owns.
- `run.json` checkpoints while recording are coalesced and written by a thread from a snapshot taken on the loop. The final write is unchanged.
- The garbage collector freezes the objects that exist before recording starts, so collections do not walk them.
- Packet N is scheduled at the actual send of packet 0 plus N times 80 ms, the origin the pacing check uses.
- A sent event is encoded once and its trace line built from that frame. A received frame is written to the trace as it arrived unless it is not ASCII or contains a line break, in which case it is encoded again.
- `run.json` gains `load.client_loop`: the largest event-loop lag seen by a 10 ms monitor and the number of lags over 80 ms, so a run can show whether the client stalled.

### Results

`pacing_probe.py native-mock` serves the native protocol and streams 22.05 kHz audio back for every input unit, in bursts of `--burst-units` units; `record` drives `run_samples` with a synthetic dataset of 4 subsets, 20–60 s inputs and both variants of each sample. Each run records 2 × concurrency sessions. Host: 2 CPUs and 7 GB of memory shared by the mock, the recorder and its workers. P99 is the largest of the per-run p99 values.

| Tree | Concurrency, burst | Unqualified runs | Sessions with a violation | Late packets | p99 (ms) | Maximum (ms) |
|---|---|---|---|---|---|---|
| Before | 24, 1 | 3 of 3 | 66 of 144 | 81 | 22.8 | 125 [119–136] |
| After | 24, 1 | 0 of 3 | 0 of 144 | 0 | 3.8 | 33 [24–42] |
| After | 24, 25 | 0 of 2 | 0 of 96 | 0 | 8.0 | 27 [25–28] |
| Before | 32, 1 | 3 of 3 | 53 of 192 | 74 | 24.1 | 100 [90–134] |
| After | 32, 1 | 0 of 3 | 0 of 192 | 0 | 5.2 | 22 [16–33] |
| Before | 32, 25 | 1 of 1 | 11 of 64 | 16 | 27.3 | 109 |
| After | 32, 25 | 0 of 2 | 0 of 128 | 0 | 10.2 | 28 [26–30] |
| Before | 64, 1 | 3 of 3 | 348 of 384 | 3,200 | 191.3 | 169 [168–445] |
| After | 64, 1 | 0 of 3 | 0 of 384 | 0 | 11.0 | 42 [27–52] |
| After | 64, 25 | 0 of 2 | 0 of 256 | 0 | 18.5 | 59 [39–80] |
| After | 128, 1 | 0 of 1 | 0 of 256 | 0 | 17.8 | 53 |

The recorder process used 0.07–0.24 cores after the change, not counting its post-processing workers. The 80 ms maximum at 64 sessions with bursts came while the mock's own loop stalled for 180 ms on the shared CPUs, so on this host 64 sessions with bursty replies is near the limit. A unit test makes replay hold the GIL for 1.5 s if it runs in the recording process: post-processing in threads fails it with a late packet, and the default process pool passes it.

## Tool

[benchmarks/duplex/pacing_probe.py](../../benchmarks/duplex/pacing_probe.py) has five subcommands.

- `mock` serves the native session protocol with no model. It replies to every input packet with `--replies-per-append` events, 3 by default: an audio delta, text deltas and a `sglang.unit.done`. It records when each packet arrives and its own event-loop lag, and writes these statistics on SIGINT.
- `run` opens `--concurrency` sessions in one process through `run_session` and streams `--duration` seconds of silence. It writes raw samples to `samples.json` and a summary to `summary.json`, then deletes the session traces, which reach about 10 MB per 90 s session.
- `native-mock` serves the native session protocol used by `run_samples` and streams audio back for every input unit at `--output-sample-rate`, 22.05 kHz by default, grouped in bursts of `--burst-units` units, with transcript deltas.
- `record` runs `run_samples` against a server with a synthetic dataset and writes a summary of pacing violations, late packets, pacing and loop-lag percentiles, garbage collection pauses and CPU use.
- `merge` pools the raw samples of `run` processes that streamed at the same time. Percentiles are over the pooled packets. `perf_counter` is shared by the processes of one host, so overlaps and late-packet times line up across processes.

The summary reports pacing deviation percentiles, packets over 80 ms, sessions with a violation, incomplete sessions, `websocket_send_*` and `append_span_*` durations, event-loop lag, CPU per process in cores, peak resident memory and the peak number of overlapping input streams. `--inject-stall-ms` blocks the event loop once halfway through the run: with 8 sessions, a 60 ms stall raised no violation and a 150 ms stall raised one in every session.

```bash
python -m benchmarks.duplex.pacing_probe mock --stats-output mock.json > mock.log &
mock_pid=$!
until grep -q listening mock.log; do sleep 0.2; done
run_pids=()
for process in 0 1 2 3; do
    python -m benchmarks.duplex.pacing_probe run --concurrency 32 --duration 90 --output out/p$process > /dev/null &
    run_pids+=($!)
done
wait "${run_pids[@]}"
kill -INT $mock_pid
wait $mock_pid
python -m benchmarks.duplex.pacing_probe merge out/p0 out/p1 out/p2 out/p3 --output out/merged
```

The recording path is probed the same way:

```bash
python -m benchmarks.duplex.pacing_probe native-mock --port 8766 --stats-output mock.json > mock.log &
mock_pid=$!
until grep -q listening mock.log; do sleep 0.2; done
python -m benchmarks.duplex.pacing_probe record --url ws://127.0.0.1:8766/v1/realtime --concurrency 32 --per-subset 8 --output out/record
kill -INT $mock_pid
wait $mock_pid
```

The 5 ms process sampler and the run scripts used for this study are not in the repository.

## Environment

- Intel Xeon Platinum 8331C at 2.5 GHz, 96 logical CPUs, in a container with no CPU quota.
- Traces on ext4 on a Seagate ST8000NM000A 7,200 rpm disk; `dirty_expire_centisecs` 3000, `dirty_ratio` 20, no memory limit on the cgroup.
- Repository commit `1d6d0be42` plus this change; Python 3.10.21, websockets 16.1.1, pydantic 2.14.1, numpy 2.2.6.
- Client and mock on the same host over loopback.

## Limitations

- One host. A host whose trace disk keeps up, such as one with an NVMe drive or a larger write cache, may never have shown the disk stalls; the change does not depend on the disk.
- The mock server is a single process and used up to 0.67 cores at 256 sessions. In the 45 runs of the changed client, the largest arrival deviation the mock saw was 7 ms above the client's largest send deviation at the median and 52 ms above it at most, so the mock adds delay of its own on top of the client's and cannot hide a client stall. A real server's larger replies and host load are not covered. A run against a real server on a GPU host is still needed.
- Four runs per cell at 90 s show large differences, not rare events. The 1 of 4 failures at 1 × 256 and 4 × 64 say only that those layouts are near the limit.
- Sessions up to 90 s and 256 sessions per host were tested.
