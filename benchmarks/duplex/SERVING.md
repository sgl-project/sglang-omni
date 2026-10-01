# Realtime serving benchmark

Run against a server exposing `/v1/realtime`:

```bash
python -m benchmarks.duplex.serving \
  --url ws://127.0.0.1:8097/v1/realtime \
  --audio input.wav --profile nemotron --concurrencies 1,2,4,8
```

The input is normalized to mono 16 kHz PCM16 using the existing duplex audio
loader. Each concurrency level starts only after its sessions finish negotiation.
All configured sessions use one monotonic start deadline and fixed 80 ms input
deadlines. A late sender keeps at least 80 ms between later append starts. Each
session has its own JSONL wire trace and input-send-receipts.json; the latter
records scheduled, send-start, and send-completion times. Failed attempts retain
their own traces and count toward the requested concurrency.

The summary reports client send lateness (send start minus deadline), TTFA (first
audio receipt minus common start), gaps between received audio events, and PCM
output duration divided by input duration. Timing distributions include p50,
p75, p95, p99 and max, with linear interpolation. Missing TTFA and output timing
have no samples; missing output has zero coverage and remains in the success
denominator.

Late-send rate is the fraction of sent input frames starting more than 20 ms
after their deadline. The threshold is a load-generator diagnostic, not a
server SLO. Strict 80 ms minimum spacing means small client scheduling delays
can accumulate into send lateness over a long run.

For consecutive output packets, gap excess is the positive part of the receive
gap minus the preceding packet's decoded PCM duration. Output drift at packet
i is its receive time minus the first audio receive time and the duration of
all earlier packets. The first drift is zero; negative values mean audio arrived
ahead of that ideal schedule. Final drift is the last packet's drift, not an
input-to-output causal latency. These quantities use actual decoded PCM lengths,
not an assumed model-frame size. No deadline-miss SLO is defined in v0.

Playback underrun is a simple client simulation. Playback starts 80 ms after
the first audio receipt by default (`--startup-reserve-ms` changes this), using
all audio received during that reserve. It consumes PCM time until the common
input deadline plus the reserve. Each interval that exhausts the buffer counts
as an underrun. A session without audio, or whose first audio arrives after the
input window, is assigned one full-input-duration underrun. Underrun ratio is
total underrun duration divided by the fixed input observation duration; the
aggregate ratio includes all attempted sessions, including failures. This does
not model network jitter, device buffering, or the server's internal stages.
These observations alone do not establish a sustainable concurrency threshold.

## MiniCPM-o

MiniCPM-o works in one-second units and stays silent while it listens, so the
continuous-output metrics above do not apply. With `--profile
minicpmo-native-pr2377` the benchmark scores each unit instead:

```bash
python -m sglang_omni.cli serve \
  --config examples/full_duplex/minicpmo-parity.yaml \
  --model-path /path/to/MiniCPM-o-4_5 --enable-realtime --port 8000

python -m benchmarks.duplex.serving \
  --url ws://127.0.0.1:8000/v1/realtime \
  --audio question-1.wav question-2.wav question-3.wav \
  --profile minicpmo-native-pr2377 \
  --concurrencies 1,2,4,8 --warmup-runs 1 --timeout-s 180
```

Set `max_sessions` in the config to at least the highest concurrency. The parity
config samples greedily, so every run asks the server for the same work. Several
recordings are handed to the sessions in turn; with a single recording every
session speaks in the same second. Recordings need silence after the question,
or the model has no room to answer. `--warmup-runs` sends single sessions before
the first level, because the first reply after start-up is slower.

| Column | Meaning |
|---|---|
| `units` / `speak` | Units the server finished, and those in which the model produced text or audio |
| `miss` | Units finished more than one unit length after they could start; units never finished count as missed |
| `lag` | `sglang.unit.done` receipt minus the send of the packet that completes the unit |
| `speak p95` | The same lag over speaking units only |
| `reply` | First audio of a reply minus the send that completed the unit in which the reply began |
| `underrun` | Silence a player would insert inside replies, divided by the audio received |
| `late sends` | Input packets sent more than 20 ms late; a high value means the load generator, not the server, fell behind |

A miss rate near zero means the server keeps up with real time at that
concurrency. Underrun is simulated per reply: playback starts
`--startup-reserve-ms` after the reply's first audio, and silence between
replies is not counted. A session that never speaks is a success; one that
leaves units unfinished is not.
