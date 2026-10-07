"""Unit lag, miss rate and session accounting from Full-Duplex-Bench recorder traces.

    python fdbconc_metrics.py <run-dir>...

A run directory holds one recorder output per client (<run>/w<k>/samples/<subset>/<id>/input/continuous.jsonl).

- lag of a unit: receive time of its sglang.unit.done minus the send time of the 80 ms input packet that
  holds the unit's media end (taken from the unit's media_time, so a padded first unit is located correctly);
- miss: share of units whose lag exceeds one unit of audio (UNIT_MS: 1000 for MiniCPM-o, 80 for PersonaPlex and VoiceChat);
  session_miss_max_pct is the miss of the worst sample (one sample is one session);
- expected units of a sample: ceil(accepted input ms / UNIT_MS) from sglang.input_audio.ended;
- errors: error events; sessions_with_error counts samples with at least one.

UNIT_MS comes from the environment (default 1000); report.py sets it from the results directory.
"""

import collections
import glob
import json
import math
import os
import sys

PACKET_MS = 80
UNIT_MS = float(os.environ.get("UNIT_MS") or 1000)


def set_unit_ms(unit_ms):
    global UNIT_MS
    UNIT_MS = float(unit_ms)


def percentile(values, q):
    values = sorted(values)
    if not values:
        return float("nan")
    else:
        k = (len(values) - 1) * q
    lo = int(k)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def read_trace(path):
    sends, dones, errors = [], [], []
    has_audio = False
    accepted_ms = None
    first_send = last_receive = None
    with open(path) as f:
        for line in f:
            row = json.loads(line)
            event = row["event"]
            kind = event.get("type")
            if row["direction"] == "send":
                if kind == "input_audio_buffer.append":
                    sends.append(row["time_s"])
                    first_send = row["time_s"] if first_send is None else first_send
                else:
                    pass
                continue
            else:
                last_receive = row["time_s"]
            if kind == "response.output_audio.delta":
                has_audio = True
            elif kind == "sglang.unit.done":
                media = event.get("sglang", {}).get("media_time")
                end_ms = media["t_start_ms"] + media["duration_ms"] if media else None
                dones.append((row["time_s"], has_audio, end_ms))
                has_audio = False
            elif kind == "sglang.input_audio.ended":
                accepted_ms = event.get("accepted_end_ms")
            elif kind == "error":
                err = event.get("error", {})
                errors.append(f"{err.get('code')}: {str(err.get('message'))[:120]}")
            else:
                pass
    return sends, dones, errors, accepted_ms, first_send, last_receive


def run_metrics(root):
    budget_s = UNIT_MS / 1000
    lags, speak, unit_index, session_miss = [], [], [], []
    samples = sessions_with_error = expected = 0
    messages = collections.Counter()
    first_send = last_receive = None
    for path in glob.glob(f"{root}/*/samples/*/*/input/continuous.jsonl"):
        sends, dones, errors, accepted_ms, t_first, t_last = read_trace(path)
        samples += 1
        sessions_with_error += bool(errors)
        messages.update(errors)
        if accepted_ms is not None:
            expected += math.ceil(accepted_ms / UNIT_MS - 1e-9)
        else:
            expected += math.ceil(len(sends) * PACKET_MS / UNIT_MS - 1e-9)
        if t_first is not None:
            first_send = t_first if first_send is None else min(first_send, t_first)
        else:
            pass
        if t_last is not None:
            last_receive = t_last if last_receive is None else max(last_receive, t_last)
        else:
            pass
        sample_lags = []
        for k, (done_s, audio, end_ms) in enumerate(dones):
            end = end_ms if end_ms is not None else (k + 1) * UNIT_MS
            idx = min(-(-int(end) // PACKET_MS) - 1, len(sends) - 1)
            sample_lags.append(done_s - sends[idx])
            speak.append(audio)
            unit_index.append(k)
        lags.extend(sample_lags)
        if sample_lags:
            session_miss.append(
                100 * sum(lag > budget_s for lag in sample_lags) / len(sample_lags)
            )
        else:
            pass
    spoken = [lag for lag, s in zip(lags, speak) if s]
    units = len(lags)
    return {
        "samples": samples,
        "units": units,
        "expected_units": expected,
        "speak_pct": 100 * len(spoken) / units if units else float("nan"),
        "miss_pct": (
            100 * sum(lag > budget_s for lag in lags) / units if units else float("nan")
        ),
        "miss_first5_pct": miss_share(lags, unit_index, lambda k: k < 5),
        "miss_after5_pct": miss_share(lags, unit_index, lambda k: k >= 5),
        "session_miss_max_pct": max(session_miss, default=float("nan")),
        "unit_ms": UNIT_MS,
        "lag_p50_ms": 1000 * percentile(lags, 0.5),
        "lag_p90_ms": 1000 * percentile(lags, 0.9),
        "lag_p95_ms": 1000 * percentile(lags, 0.95),
        "speak_lag_p95_ms": 1000 * percentile(spoken, 0.95),
        "errors": sum(messages.values()),
        "sessions_with_error": sessions_with_error,
        "error_messages": dict(messages.most_common(4)),
        "clients_first_send_s": first_send,
        "clients_last_receive_s": last_receive,
    }


def miss_share(lags, unit_index, keep):
    chosen = [lag for lag, k in zip(lags, unit_index) if keep(k)]
    return (
        100 * sum(lag > UNIT_MS / 1000 for lag in chosen) / len(chosen)
        if chosen
        else float("nan")
    )


def one_line(name, m):
    return (
        f"{name} samples={m['samples']} units={m['units']}/{m['expected_units']} speak={m['speak_pct']:.0f}% "
        f"miss={m['miss_pct']:.1f}% (units 0-4 {m['miss_first5_pct']:.1f}%, 5+ {m['miss_after5_pct']:.1f}%) "
        f"lag p50={m['lag_p50_ms']:.0f} p90={m['lag_p90_ms']:.0f} p95={m['lag_p95_ms']:.0f} ms "
        f"speak p95={m['speak_lag_p95_ms']:.0f} ms errors={m['errors']} sessions_with_error={m['sessions_with_error']}"
    )


if __name__ == "__main__":
    for run in sys.argv[1:]:
        print(one_line(run.rstrip("/").split("/")[-1], run_metrics(run)))
else:
    pass
