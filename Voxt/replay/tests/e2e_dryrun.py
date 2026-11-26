#!/usr/bin/env python3
"""End-to-end self-test of replay-sink and replay-driver without touching the live system.

The driver runs with --dry-run (no keyboard events) and --write-audio-dir (the padded
audio goes to files, paced in real time). A fake dictation app in this script watches
the sink log for the driver's "<id> keyup" mark and types scripted text into the sink
through the sink's own socket. With --play-blackhole, one more segment plays to the
"BlackHole 2ch" device instead of writing a file.

Run from Voxt/replay after ./build.sh:  python3 tests/e2e_dryrun.py [--play-blackhole]
Takes about 40 s.
"""

import argparse
import json
import math
import os
import random
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from pathlib import Path

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SINK = os.path.join(HERE, "build/ReplaySink.app/Contents/MacOS/replay-sink")
DRIVER = os.path.join(HERE, "build/replay-driver")
RATE = 16000
DEADLINE_BASE = 3.0
MS = 1_000_000

# note (luojiaxuan): Fake app output per segment: (seconds after the keyup mark, text to type).
SCRIPTS = {
    "a_complete": [(0.3, "hello"), (0.6, " world")],
    "b_empty": [],
    # note (luojiaxuan): Keeps typing past its deadline (keyup + 3 s + 1.2 s): still writing -> timeout, and
    # the last characters land in the next segment.
    "c_timeout": [(0.2 + 0.25 * k, "x") for k in range(20)],
    "d_after_late": [(0.4, "done")],
}
DURATIONS = {
    "a_complete": 1.0,
    "b_empty": 0.8,
    "c_timeout": 1.2,
    "d_after_late": 1.0,
    "e_blackhole": 0.5,
}


def write_wav(path, seconds, seed):
    rng = random.Random(seed)
    n = int(seconds * RATE)
    samples = [
        int(3000 * math.sin(2 * math.pi * 440 * i / RATE)) + rng.randint(-500, 500)
        for i in range(n)
    ]
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(RATE)
        w.writeframes(struct.pack(f"<{n}h", *samples))
    return samples


def read_wav(path):
    with wave.open(path, "rb") as w:
        frames = w.readframes(w.getnframes())
        return (
            w.getnchannels(),
            w.getframerate(),
            struct.unpack(f"<{len(frames) // 2}h", frames),
        )


class SinkClient:
    def __init__(self, path):
        self.sock = socket.socket(socket.AF_UNIX)
        self.sock.connect(path)
        self.file = self.sock.makefile("rw", encoding="utf-8")

    def send(self, command):
        self.file.write(command + "\n")
        self.file.flush()
        return self.file.readline().rstrip("\n")


def fake_app(log_path, socket_path, stop):
    client = SinkClient(socket_path)
    with open(log_path, encoding="utf-8") as log:
        pending = ""
        while not stop.is_set():
            chunk = log.read()
            if not chunk:
                time.sleep(0.01)
                continue
            pending += chunk
            *lines, pending = pending.split("\n")
            for line in lines:
                event = json.loads(line)
                label = event.get("label", "")
                if event.get("event") == "mark" and label.endswith(" keyup"):
                    seg = label[: -len(" keyup")]
                    start = time.monotonic()
                    for delay, text in SCRIPTS.get(seg, []):
                        time.sleep(max(0.0, start + delay - time.monotonic()))
                        client.send("type " + text)


def check(cond, message, failures):
    print(("ok   " if cond else "FAIL ") + message)
    if not cond:
        failures.append(message)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--play-blackhole", action="store_true")
    parser.add_argument("--keep", action="store_true", help="keep the work directory")
    args = parser.parse_args()

    work = tempfile.mkdtemp(prefix="replay-e2e-")
    os.chdir(
        work
    )  # note (luojiaxuan): the socket path is relative: sun_path is limited to 103 bytes
    os.makedirs("audio")
    ids = list(SCRIPTS) + (["e_blackhole"] if args.play_blackhole else [])
    sources = {}
    with open("manifest.jsonl", "w") as f:
        for k, seg in enumerate(ids):
            sources[seg] = write_wav(f"{seg}.wav", DURATIONS[seg], k)
            f.write(
                json.dumps(
                    {
                        "id": seg,
                        "wav": f"{seg}.wav",
                        "lang": "en",
                        "reference": "hello world",
                    }
                )
                + "\n"
            )
    if args.play_blackhole:
        with open("manifest_file.jsonl", "w") as f, open("manifest.jsonl") as src:
            f.writelines(line for line in src if '"e_blackhole"' not in line)
        with open("manifest_play.jsonl", "w") as f, open("manifest.jsonl") as src:
            f.writelines(line for line in src if '"e_blackhole"' in line)

    sink = subprocess.Popen(
        [SINK, "--log", "sink.log", "--socket", "sink.sock", "--no-activate"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        if os.path.exists("sink.sock"):
            break
        time.sleep(0.05)
    stop = threading.Event()
    threading.Thread(
        target=fake_app, args=("sink.log", "sink.sock", stop), daemon=True
    ).start()

    common = [
        "--out",
        "results.jsonl",
        "--sink-socket",
        "sink.sock",
        "--sink-log",
        "sink.log",
        "--key",
        "right_option",
        "--deadline-base-s",
        str(DEADLINE_BASE),
        "--mem-app",
        "ReplaySink",
        "--mem-out",
        "mem.jsonl",
        "--dry-run",
    ]
    failures = []
    try:
        manifest = "manifest_file.jsonl" if args.play_blackhole else "manifest.jsonl"
        subprocess.run(
            [DRIVER, "run", "--manifest", manifest, "--write-audio-dir", "audio"]
            + common,
            check=True,
        )
        if args.play_blackhole:
            subprocess.run(
                [DRIVER, "run", "--manifest", "manifest_play.jsonl"] + common,
                check=True,
            )
        rows = {
            r["id"]: r
            for r in map(json.loads, Path("results.jsonl").read_text().splitlines())
        }

        check(
            sorted(rows) == sorted(ids), f"one row per segment ({len(rows)})", failures
        )
        check(
            not any(r["driver_fault"] for r in rows.values()),
            "no driver faults",
            failures,
        )
        for seg, r in rows.items():
            padded = r["padded_duration_s"]
            check(
                abs(padded - DURATIONS[seg] - 2.0) < 1e-6,
                f"{seg}: padded = 1.5 + {DURATIONS[seg]} + 0.5 s",
                failures,
            )
            played = (r["t_audio_end"] - r["t_audio_start"]) / 1e9
            check(
                abs(played - padded) < 0.06,
                f"{seg}: real-time audio took {played:.3f} s for {padded:.3f} s",
                failures,
            )
            expected = (DEADLINE_BASE + r["audio_duration_s"]) * 1e9
            check(
                abs(r["deadline"] - r["t_stop"] - expected) < 2,
                f"{seg}: deadline = t_stop + 3 s + audio",
                failures,
            )
            check(
                r["t_clear"]
                < r["t_keydown"]
                < r["t_audio_start"]
                < r["t_audio_end"]
                <= r["t_stop"],
                f"{seg}: clear < keydown < audio < stop",
                failures,
            )
            check(
                r["mem_samples"] > 0
                and r["mem_peak_phys_footprint"] > 0
                and r["mem_read_failures"] == 0,
                f"{seg}: memory peak {r['mem_peak_phys_footprint']} over {r['mem_samples']} samples",
                failures,
            )

        a = rows["a_complete"]
        first, last = (a["t_first"] - a["t_stop"]) / MS, (
            a["t_last"] - a["t_stop"]
        ) / MS
        check(
            a["status"] == "complete"
            and a["final_text"] == "hello world"
            and a["n_changes"] == 2,
            f"a_complete: {a['status']} {a['final_text']!r} in {a['n_changes']} changes",
            failures,
        )
        check(
            300 <= first < 450 and 600 <= last < 750,
            f"a_complete: t_first {first:.0f} ms, t_last {last:.0f} ms "
            "after t_stop (typed at +300 and +600 ms after the keyup mark)",
            failures,
        )
        b = rows["b_empty"]
        check(
            b["status"] == "empty" and b["t_first"] is None and b["t_last"] is None,
            "b_empty: empty",
            failures,
        )
        c = rows["c_timeout"]
        check(c["status"] == "timeout", f"c_timeout: {c['status']}", failures)
        d = rows["d_after_late"]
        check(
            d["n_changes_before_stop"] >= 1
            and d["t_first"] < d["t_stop"]
            and d["final_text"].endswith("done"),
            f"d_after_late: {d['n_changes_before_stop']} late changes before t_stop, text {d['final_text']!r}",
            failures,
        )

        for seg in SCRIPTS:
            channels, rate, data = read_wav(f"audio/{seg}.wav")
            src = sources[seg]
            pre, post = int(1.5 * RATE), int(0.5 * RATE)
            left, right = data[0::2], data[1::2]
            ok = (
                channels == 2
                and rate == RATE
                and len(left) == pre + len(src) + post
                and left == right
                and not any(left[:pre])
                and not any(left[pre + len(src) :])
                and list(left[pre : pre + len(src)]) == src
            )
            check(
                ok,
                f"audio/{seg}.wav: 1.5 s silence + source samples unchanged + 0.5 s silence, two equal channels",
                failures,
            )

        before = len(rows)
        subprocess.run(
            [DRIVER, "run", "--manifest", manifest, "--write-audio-dir", "audio"]
            + common,
            check=True,
        )
        check(
            len(Path("results.jsonl").read_text().splitlines()) == before,
            "a second run skips finished segments",
            failures,
        )

        events = [
            json.loads(line) for line in Path("sink.log").read_text().splitlines()
        ]
        clears = [e["label"] for e in events if e["event"] == "clear"]
        check(
            clears == ids,
            f"the sink log has one clear marker per segment, in order ({len(clears)})",
            failures,
        )
        procs = [
            json.loads(line)
            for line in Path("mem.jsonl").read_text().splitlines()
            if '"event":"proc"' in line
        ]
        check(
            any("/ReplaySink.app/" in p["path"] for p in procs),
            "memory sampler found the ReplaySink process",
            failures,
        )
    finally:
        stop.set()
        try:
            SinkClient("sink.sock").send("quit")
        except OSError:
            # note (luojiaxuan): A sink that already exited (failed to start or crashed) has no socket left;
            # the wait below reaps it either way.
            pass
        sink.wait(timeout=5)
        print(f"work directory: {work}")
        if not args.keep and not failures:
            shutil.rmtree(work)
    print(f"{len(failures)} failures")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
