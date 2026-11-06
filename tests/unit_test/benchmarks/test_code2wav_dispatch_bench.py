# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Code2Wav dispatch component benchmark (#1236 harness).

CPU-only via ``--fake``; no checkpoint. The drain-toggle test guards the
--drain-inbox off semantics: it must skip only the b90d859 drain block while
leaving the coalesced wait/floor pump live — upstream drift that adds readers
of ``_can_batch_stream_chunks`` inside ``_next_message`` would break exactly
these assertions.
"""

from __future__ import annotations

import json
import math

import numpy as np

from benchmarks.eval.benchmark_code2wav_batching import main
from benchmarks.metrics.waveform_tolerance import compare_waveforms, tolerance_failures


def _run(tmp_path, extra_args):
    out = tmp_path / "out.json"
    rc = main(
        [
            "--fake",
            "--streams",
            "4",
            "--windows",
            "3",
            "--modes",
            "aligned",
            "--repeats",
            "1",
            "--warmup-runs",
            "0",
            "--output-json",
            str(out),
            *extra_args,
        ]
    )
    return rc, json.loads(out.read_text())


def test_fake_smoke_with_equivalence_gate(tmp_path) -> None:
    rc, payload = _run(
        tmp_path,
        [
            "--arms",
            "serial-eager,batched",
            "--wait-ms",
            "8",
            "--floor",
            "2",
            "--compare-waveforms",
        ],
    )
    assert rc == 0
    assert len(payload["runs"]) == 2
    for run in payload["runs"]:
        assert "invalid" not in run
        assert run["chunks_per_request_mean"] > 0
        assert run["forward_calls"] > 0
    # FakeVocoder is batch-invariant, so lockstep equivalence is exact.
    assert payload["equivalence"], "equivalence pass did not run"
    for report in payload["equivalence"]:
        assert report["failures"] == {}


def test_drain_toggle_skips_only_the_drain_block(tmp_path) -> None:
    rc, payload = _run(
        tmp_path,
        [
            "--arms",
            "batched",
            "--wait-ms",
            "8",
            "--floor",
            "2",
            "--drain-inbox",
            "on,off",
        ],
    )
    assert rc == 0
    on = next(r for r in payload["runs"] if r["inbox_drain"] == "on")
    off = next(r for r in payload["runs"] if r["inbox_drain"] == "off")
    # drain-on wakes enter the drain block (histogram populated).
    assert on["wake_drain_histogram"]
    # drain-off never enters the drain block...
    assert off["wake_drain_histogram"] == {}
    # ...but the coalesced pump policy stays live: due streams still fire
    # through the wait/floor policy rather than decoding serially on arrival.
    fires = off["fire_reasons"]
    assert fires.get("floor", 0) + fires.get("deadline", 0) > 0
    if fires.get("floor", 0):
        assert max(int(k) for k in off["batch_histogram"]) > 1


def test_tail_frames_reach_the_final_flush(tmp_path) -> None:
    # serial-eager processes messages one at a time, so the 5-frame tail
    # never reaches the decode threshold and can only decode at EOS.
    rc, payload = _run(tmp_path, ["--arms", "serial-eager", "--tail-frames", "5"])
    assert rc == 0
    run = payload["runs"][0]
    assert run["final_flush"]["calls"] == 4
    assert run["final_flush"]["frames"] > 0


def test_waveform_tolerance_passes_identical() -> None:
    ref = np.sin(np.linspace(0.0, 20.0, 4800)).astype(np.float32)
    comparison = compare_waveforms(ref, ref.copy())
    assert comparison.snr_db == math.inf
    assert (
        tolerance_failures(
            comparison,
            min_snr_db=40.0,
            max_peak_diff=0.2,
            max_exceed_fraction=0.01,
        )
        == []
    )


def test_waveform_tolerance_flags_degradation() -> None:
    ref = np.sin(np.linspace(0.0, 20.0, 4800)).astype(np.float32)
    noisy = ref + 0.3
    comparison = compare_waveforms(ref, noisy)
    failures = tolerance_failures(
        comparison,
        min_snr_db=40.0,
        max_peak_diff=0.2,
        max_exceed_fraction=0.01,
    )
    assert failures
    truncated = compare_waveforms(ref, ref[:-1])
    assert truncated.length_mismatch
