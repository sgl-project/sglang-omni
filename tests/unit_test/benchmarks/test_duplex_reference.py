# SPDX-License-Identifier: Apache-2.0
"""Exercise reference scoring, optional external formulas, and resumable evidence."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import types
import wave
from pathlib import Path

import numpy as np
import pytest
import soundfile

from benchmarks.duplex import (
    reference_asr,
    reference_behavior,
    reference_core,
    reference_source,
    reference_summary,
    reference_timing,
)
from benchmarks.eval import benchmark_duplex_reference as cli_module

REFERENCE_PATH = os.environ.get("FDB_REFERENCE_SOURCE")
REF = Path(REFERENCE_PATH) if REFERENCE_PATH else None
SR = 16000


def speech_runs(wav: np.ndarray) -> list[tuple[int, int]]:
    """Sample-index runs where |x| > 0.05 (synthetic speech is constant amplitude)."""
    active = np.abs(np.asarray(wav, dtype=np.float64)) > 0.05
    edges = np.flatnonzero(np.diff(np.concatenate(([0], active.astype(int), [0]))))
    return [(int(s), int(e)) for s, e in zip(edges[::2], edges[1::2])]


@pytest.fixture
def fake_silero(monkeypatch):
    module = types.ModuleType("silero_vad")
    module.get_speech_timestamps = lambda w, model, sampling_rate: [
        {"start": s, "end": e} for s, e in speech_runs(w.numpy())
    ]
    monkeypatch.setitem(sys.modules, "silero_vad", module)
    return lambda: "fake-silero-model"


class FakeParakeet:
    def __init__(self, overshoot_s: float = 0.0):
        self.calls, self.overshoot_s = [], overshoot_s

    def transcribe(self, paths, timestamps):
        assert timestamps is True and len(paths) == 1
        data, sr = soundfile.read(paths[0])
        self.calls.append(hashlib.sha256(data.tobytes()).hexdigest())
        words = [
            {"word": f"w{i}", "start": s / sr, "end": e / sr + self.overshoot_s}
            for i, (s, e) in enumerate(speech_runs(data))
        ]
        return [types.SimpleNamespace(timestamp={"word": words})]


@pytest.fixture
def fake_nemo(monkeypatch):
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["nemo"].collections = sys.modules["nemo.collections"]
    sys.modules["nemo.collections"].asr = sys.modules["nemo.collections.asr"]


def write_wav(path: Path, spans: list[tuple[float, float]]) -> None:
    audio = np.zeros(3 * SR, dtype=np.float32)
    for start_s, end_s in spans:
        audio[int(start_s * SR) : int(end_s * SR)] = 0.3
    path.parent.mkdir(parents=True, exist_ok=True)
    soundfile.write(path, audio, SR, subtype="PCM_16")


def variant(eligible=True, reasons=(), **extra):
    return {"eligible": eligible, "reasons": list(reasons), **extra}


def build_trees(tmp: Path) -> dict[str, Path]:
    """Two engines share identical inputs; outputs and one clean eligibility differ."""
    inputs = {
        "user_interruption/1": {
            "input.wav": [(0.5, 1.5), (2.0, 2.5)],
            "clean_input.wav": [(0.5, 1.5)],
        },
        "background_speech/2": {
            "input.wav": [(0.2, 0.8)],
            "clean_input.wav": [(0.3, 0.9)],
        },
    }
    outputs = {
        "sgl": {
            "user_interruption/1": {
                "output.wav": [(1.0, 2.2)],
                "clean_output.wav": [(1.8, 2.4)],
            },
            "background_speech/2": {
                "output.wav": [(1.2, 3.0)],
                "clean_output.wav": [(1.0, 1.6)],
            },
        },
        "vllm": {
            "user_interruption/1": {"output.wav": [], "clean_output.wav": [(1.7, 2.3)]},
            "background_speech/2": {
                "output.wav": [(1.1, 1.9)],
                "clean_output.wav": [(1.0, 2.0)],
            },
        },
    }
    trees = {}
    for engine, outs in outputs.items():
        tree = tmp / "trees" / engine
        samples = []
        for sid, files in inputs.items():
            for name, spans in {**files, **outs[sid]}.items():
                write_wav(tree / sid / name, spans)
            clean_ok = not (engine == "vllm" and sid == "background_speech/2")
            samples.append(
                {
                    "sample_id": sid,
                    "event_span_s": (
                        [2.0, 2.5] if sid.startswith("user") else [0.2, 0.8]
                    ),
                    "variants": {
                        "overlap": variant(
                            flags=["eof_speech"] if sid.startswith("back") else []
                        ),
                        "clean": variant(
                            clean_ok, [] if clean_ok else ["capture_window_invalid"]
                        ),
                    },
                }
            )
        (tree / "reference-manifest.json").write_text(json.dumps({"samples": samples}))
        trees[engine] = tree
    return trees


def tree_digest(root: Path) -> dict[str, str]:
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*"))
        if p.is_file()
    }


def cli(phase: str, tmp: Path, trees: dict[str, Path], *extra: str):
    argv = [
        phase,
        "--reference-source",
        str(REF),
        "--out",
        str(tmp / "out"),
        *[f"--tree={k}={v}" for k, v in trees.items()],
        *extra,
    ]
    args = cli_module.build_parser().parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    paths = reference_source.verify_reference(args.reference_source)
    return (
        args,
        cli_module.open_engines(args),
        paths,
        reference_core.HashCache(args.out / "hash-cache.json"),
    )


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_official_timing_formulas_match_retained_fixtures(fake_silero):
    paths = reference_source.verify_reference(REF)
    mod, bridge = reference_source.load_official_timing(paths["timing"], fake_silero)
    assert bridge["vad_branch"] == "get_speech_timestamps"
    assert bridge["torch_hub_load_calls"] == [
        {
            "repo_or_dir": "snakers4/silero-vad",
            "model": "silero_vad",
            "args": [],
            "kwargs": {"trust_repo": True, "onnx": False},
        }
    ]
    assert (mod.SR, mod.USER_MERGE_GAP, mod.MODEL_MERGE_GAP) == (16000, 0.6, 0.5)
    merge_intervals = (  # Note (wenyao): pinned API name; noqa: leading-underscore
        mod._merge
    )
    assert merge_intervals([(0.0, 1.0), (1.5, 2.0)], 0.6) == [(0.0, 2.0)]
    assert merge_intervals([(0.0, 1.0), (1.61, 2.0)], 0.6) == [(0.0, 1.0), (1.61, 2.0)]
    assert merge_intervals([(0.0, 1.0), (1.5, 2.0)], 0.5) == [(0.0, 2.0)]
    assert merge_intervals([(0.0, 1.0), (1.51, 2.0)], 0.5) == [(0.0, 1.0), (1.51, 2.0)]
    assert mod.overlaps([(3.0, 5.0)], [(0.0, 8.0)]) == [[3.0, 5.0]]
    assert mod.response_gaps([(3.0, 5.0)], [(0.0, 8.0)]) == []
    assert mod.response_gaps([(0.0, 1.0)], [(1.0, 2.0)]) == []
    assert mod.response_gaps([(0.0, 1.0)], [(1.001, 2.0)]) == [[1.0, 1.001]]
    assert mod.overlaps([(0.0, 3.0)], [(1.0, 3.0), (2.0, 3.0)]) == [[2.0, 3.0]]
    assert mod.response_gaps([(0.0, 1.0), (1.5, 2.0)], [(3.0, 4.0)]) == [[2.0, 3.0]]
    assert mod.overlaps([(0.0, 1.23456)], [(0.11111, 5.0)]) == [[0.111, 1.235]]


def test_soundfile_bridge_matches_stdlib_pcm16_decoding(tmp_path):
    path = tmp_path / "a.wav"
    rng = np.random.default_rng(0)
    soundfile.write(
        path, rng.uniform(-1, 1, 4000).astype(np.float32), SR, subtype="PCM_16"
    )
    with wave.open(str(path)) as w:
        ints = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
    got = reference_source.soundfile_load_wav(SR)(path)
    assert got.dtype.is_floating_point and tuple(got.shape) == (4000,)
    assert np.array_equal(got.numpy(), ints.astype(np.float32) / 32768.0)


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_reference_source_pin_rejects_modified_checkout(tmp_path):
    copy = tmp_path / "ref"
    for rel, _ in reference_core.REFERENCE_FILES.values():
        (copy / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REF / rel, copy / rel)
    reference_source.verify_reference(copy)
    timing = copy / reference_core.REFERENCE_FILES["timing"][0]
    timing.write_text(
        timing.read_text().replace("USER_MERGE_GAP = 0.6", "USER_MERGE_GAP = 0.3")
    )
    with pytest.raises(SystemExit, match="pinned"):
        reference_source.verify_reference(copy)


EXPECTED_USER_MSG = (
    "\n        {\n"
    '            "input_clean": {"text":"hi","chunks":[{"text":"hi","timestamp":[0.5,0.9]}]},\n'
    '            "input_noisy": {"text":"hi ñ","chunks":[{"text":"hi","timestamp":[0.5,0.9]},{"text":"ñ","timestamp":[1.0,1.2]}]},\n'
    '            "output_clean": {"text":"","chunks":[]},\n'
    '            "output_noisy": {"text":"ok","chunks":[{"text":"ok","timestamp":[1.3,1.6]}]}\n'
    "        }\n        "
)


def write_transcripts(sample: Path) -> None:
    docs = {
        "clean_input.json": {
            "text": "hi",
            "chunks": [{"text": "hi", "timestamp": [0.5, 0.9]}],
        },
        "input.json": {
            "text": "hi ñ",
            "chunks": [
                {"text": "hi", "timestamp": [0.5, 0.9]},
                {"text": "ñ", "timestamp": [1.0, 1.2]},
            ],
        },
        "clean_output.json": {"text": "", "chunks": []},
        "output.json": {
            "text": "ok",
            "chunks": [{"text": "ok", "timestamp": [1.3, 1.6]}],
        },
    }
    sample.mkdir(parents=True, exist_ok=True)
    for name, doc in docs.items():
        (sample / name).write_text(json.dumps(doc, indent=4))


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_behavior_request_bytes_match_official_template(tmp_path):
    paths = reference_source.verify_reference(REF)
    official = reference_source.load_official_behavior(
        paths["behavior"], paths["instruction"]
    )
    write_transcripts(tmp_path)
    request = reference_behavior.build_request(official, tmp_path)
    system, user = request["body"]["messages"]
    assert request["body"]["model"] == "gpt-4o-2024-08-06" and request["seeds"] == [
        1,
        2,
        3,
    ]
    assert set(request["body"]) == {
        "model",
        "messages",
    }
    assert system == {
        "role": "system",
        "content": (REF / reference_core.REFERENCE_FILES["instruction"][0])
        .read_bytes()
        .decode(),
    }
    assert user == {"role": "user", "content": EXPECTED_USER_MSG}
    assert all(label in system["content"] for label in reference_core.C_LABELS)
    assert (
        reference_behavior.build_request(official, tmp_path)["request_hash"]
        == request["request_hash"]
    )
    (tmp_path / "output.json").write_text(json.dumps({"text": "ok!", "chunks": []}))
    assert (
        reference_behavior.build_request(official, tmp_path)["request_hash"]
        != request["request_hash"]
    )


def response(content, model="gpt-4o-2024-08-06"):
    return {
        "id": "x",
        "model": model,
        "system_fingerprint": "fp_1",
        "choices": [{"message": {"role": "assistant", "content": content}}],
    }


@pytest.fixture
def official_behavior():
    if REF is None:
        pytest.skip("Set FDB_REFERENCE_SOURCE to the pinned external checkout")
    paths = reference_source.verify_reference(REF)
    return reference_source.load_official_behavior(
        paths["behavior"], paths["instruction"]
    )


def test_judge_retry_is_bounded_with_retained_errors(official_behavior):
    seeds, sleeps = [], []

    def failing(body, seed):
        seeds.append(seed)
        raise ConnectionError(f"boom {seed}")

    result = reference_behavior.run_judgment(
        official_behavior, {"model": "m"}, [1, 2, 3], failing, 5.0, sleeps.append
    )
    assert result["status"] == "failed" and result["label"] is None
    assert seeds == [1, 2, 3] and sleeps == [5.0, 5.0]
    assert [a["error"] for a in result["attempts"]] == [
        f"ConnectionError: boom {s}" for s in (1, 2, 3)
    ]


def test_judge_parse_retry_then_label_validation(official_behavior):
    replies = iter(
        [response("no json here"), response('ok {"behaviour": ["C_RESUME"]}')]
    )
    result = reference_behavior.run_judgment(
        official_behavior, {}, [1, 2, 3], lambda b, s: next(replies), 0, lambda s: None
    )
    assert result["status"] == "valid" and result["label"] == "C_RESUME"
    assert [a["seed"] for a in result["attempts"]] == [1, 2] and result[
        "system_fingerprint"
    ] == "fp_1"
    assert "ValueError" in result["attempts"][0]["error"]
    for content in (
        '{"behaviour": ["C_FOO"]}',
        '{"behaviour": ["C_RESUME", "C_UNKNOWN"]}',
        '{"behaviour": "C_RESUME"}',
    ):
        calls = []
        r = reference_behavior.run_judgment(
            official_behavior,
            {},
            [1, 2, 3],
            lambda b, s: calls.append(s) or response(content),
            0,
            lambda s: None,
        )
        assert r["status"] == "invalid_label" and calls == [1]
    r = reference_behavior.run_judgment(
        official_behavior,
        {},
        [1, 2, 3],
        lambda b, s: response('{"behaviour": ["C_RESUME"]}', model="gpt-4o-mini"),
        0,
        lambda s: None,
    )
    assert r["status"] == "model_mismatch"


def test_manifest_requires_reasons_and_supports_projection(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "sample_id": "a/1",
                        "variants": {
                            "overlap": variant(),
                            "clean": {"eligible": False, "reasons": []},
                        },
                    }
                ]
            }
        )
    )
    with pytest.raises(ValueError, match="reasons"):
        reference_core.load_manifest(bad, None)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps({"pairs": [{"id": "a/1", "ok": [True, True]}]}))
    proj = tmp_path / "proj.py"
    proj.write_text(
        "def project(doc):\n    return {'samples': [{'sample_id': p['id'], 'variants': "
        "{'overlap': {'eligible': p['ok'][0]}, 'clean': {'eligible': p['ok'][1]}}} for p in doc['pairs']]}\n"
    )
    assert reference_core.load_manifest(raw, proj)["samples"][0]["category"] == "a"


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_phases_end_to_end_resume_and_denominators(
    tmp_path, fake_silero, fake_nemo, monkeypatch
):
    trees = build_trees(tmp_path)
    before = {k: tree_digest(v) for k, v in trees.items()}
    nemo = tmp_path / "model.nemo"
    nemo.write_bytes(b"checkpoint")
    model = FakeParakeet()

    args, engines, paths, hashes = cli(
        "asr", tmp_path, trees, "--nemo", str(nemo), "--device", "cpu"
    )
    counts = reference_asr.run_asr(args, engines, paths, hashes, model=model)
    eligible_files = {
        hashes.get(e.source_audio(sid, f))
        for e in engines
        for sid in e.samples
        for v, files in reference_core.VARIANTS.items()
        if e.eligible(sid, v)
        for f in files
    }
    assert counts["ok"] == len(model.calls) == len(eligible_files) < 14
    out = tmp_path / "out" / "engines"
    silent = json.loads(
        (out / "vllm/samples/user_interruption/1/output.json").read_text()
    )
    assert silent == {
        "text": "",
        "chunks": [],
    }
    doc = json.loads((out / "sgl/samples/user_interruption/1/input.json").read_text())
    assert doc == {
        "text": "w0 w1",
        "chunks": [
            {"text": "w0", "timestamp": [0.5, 1.5]},
            {"text": "w1", "timestamp": [2.0, 2.5]},
        ],
    }
    assert not (out / "vllm/samples/background_speech/2/clean_output.json").exists()

    args, engines, paths, hashes = cli(
        "asr", tmp_path, trees, "--nemo", str(nemo), "--device", "cpu"
    )
    assert reference_asr.run_asr(args, engines, paths, hashes, model=model)["ok"] == 0
    assert len(model.calls) == len(eligible_files)

    args, engines, paths, hashes = cli("timing", tmp_path, trees)
    counts = reference_timing.run_timing(args, engines, paths, hashes, fake_silero)
    assert counts == {"ok": 7}
    sgl1 = json.loads(
        (out / "sgl/samples/user_interruption/1/latency_intervals.json").read_text()
    )
    assert sgl1 == {
        "latency_stop_list": [[1.0, 2.2]],
        "latency_resp_list": [],
    }
    vllm1 = json.loads(
        (out / "vllm/samples/user_interruption/1/latency_intervals.json").read_text()
    )
    assert vllm1 == {"latency_stop_list": [], "latency_resp_list": []}
    sgl2 = json.loads(
        (out / "sgl/samples/background_speech/2/latency_intervals.json").read_text()
    )
    assert sgl2 == {"latency_stop_list": [], "latency_resp_list": [[0.8, 1.2]]}
    clean = json.loads(
        (
            out / "sgl/samples/user_interruption/1/clean/latency_intervals.json"
        ).read_text()
    )
    assert clean == {"latency_stop_list": [], "latency_resp_list": [[1.5, 1.8]]}
    receipt = json.loads(
        (
            out / "sgl/samples/background_speech/2/receipts/timing-overlap.json"
        ).read_text()
    )
    assert receipt["labels"]["model_speech_at_output_end"] is True
    assert receipt["user_segments"] == [[0.2, 0.8]] and receipt["raw_vad_samples"][
        "user"
    ] == [[3200, 12800]]
    args, engines, paths, hashes = cli("timing", tmp_path, trees)
    assert reference_timing.run_timing(args, engines, paths, hashes, fake_silero) == {
        "reused": 7
    }

    intervals = out / "sgl/samples/user_interruption/1/latency_intervals.json"
    retained = intervals.read_bytes()
    intervals.write_text('{"latency_stop_list": [], "latency_resp_list": []}')
    with pytest.raises(ValueError, match="Timing intervals changed"):
        reference_timing.run_timing(args, engines, paths, hashes, fake_silero)
    summary_args, _, _, _ = cli("summarize", tmp_path, trees)
    with pytest.raises(ValueError, match="Timing intervals changed"):
        reference_summary.run_summarize(summary_args, engines, paths)
    intervals.write_bytes(retained)

    args, engines, paths, _ = cli("prepare-judge", tmp_path, trees)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    counts = reference_behavior.run_prepare_judge(args, engines, paths)
    assert counts == {"prepared": 3, "blocked_variant_ineligible": 1}

    args, engines, paths, _ = cli(
        "judge", tmp_path, trees, "--judge", "claude-opus-5-5"
    )
    with pytest.raises(SystemExit, match="exactly gpt-4o-2024-08-06"):
        reference_behavior.run_judge(
            args, engines, paths, transport=lambda b, s: pytest.fail("called")
        )
    (out / "sgl/samples/background_speech/2/output.json").write_text(
        '{"text": "x", "chunks": []}'
    )
    sent = []

    def transport(body, seed):
        sent.append(body["messages"][1]["content"])
        return response('{"behaviour": ["C_RESPOND"]}' if len(sent) == 1 else "garbage")

    args, engines, paths, _ = cli(
        "judge", tmp_path, trees, "--judge", "gpt-4o-2024-08-06", "--retry-sleep-s", "0"
    )
    counts = reference_behavior.run_judge(args, engines, paths, transport=transport)
    assert (
        counts == {"stale_request": 1, "valid": 1, "failed": 1} and len(sent) == 1 + 3
    )
    assert not any('"output_noisy": {"text":"x"' in s for s in sent)
    args, engines, paths, _ = cli(
        "judge", tmp_path, trees, "--judge", "gpt-4o-2024-08-06"
    )
    assert reference_behavior.run_judge(
        args, engines, paths, transport=lambda b, s: pytest.fail("resent")
    ) == {"stale_request": 1, "reused": 2, "reused_failed": 1}

    args, engines, paths, _ = cli("summarize", tmp_path, trees, "--bootstrap", "200")
    summary = reference_summary.run_summarize(args, engines, paths)
    vllm_all = summary["engines"]["vllm"]["all"]
    assert vllm_all["timing_supplementary_clean"]["status"] == {
        "ineligible": 1,
        "ok": 1,
    }
    assert vllm_all["timing_supplementary_clean"]["ineligible_reasons"] == {
        "capture_window_invalid": 1
    }
    stop = vllm_all["timing_official_overlap"]["official_all_intervals"]["stop"]
    assert (
        stop["samples"],
        stop["zero_interval_samples"],
        stop["interval_n"],
        stop["mean_s"],
    ) == (2, 2, 0, None)
    assert stop["bootstrap_pooled_mean"]["ci95"] is None
    sgl_stop = summary["engines"]["sgl"]["all"]["timing_official_overlap"][
        "official_all_intervals"
    ]["stop"]
    assert (
        sgl_stop["samples"],
        sgl_stop["zero_interval_samples"],
        sgl_stop["interval_n"],
    ) == (2, 1, 1)
    assert (
        sgl_stop["mean_s"] == pytest.approx(1.2)
        and sgl_stop["bootstrap_pooled_mean"]["unit_n"] == 2
    )
    assert vllm_all["behavior"]["status"]["variant_ineligible"] == 1
    assert vllm_all["asr_files"]["output.wav:ok_empty_transcript"] == 1
    sgl_all = summary["engines"]["sgl"]["all"]
    assert sgl_all["behavior"]["status"] == {"stale_request": 1, "valid": 1}
    assert sgl_all["timing_official_overlap"]["manifest_flags"] == {"eof_speech": 1}
    assert "event_selected_non_official" in sgl_all["timing_official_overlap"]
    assert (
        reference_summary.run_summarize(args, engines, paths)["engines"]
        == summary["engines"]
    )
    assert {k: tree_digest(v) for k, v in trees.items()} == before


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_asr_bounds_violation_is_retained_not_fixed(tmp_path, fake_nemo):
    trees = {"sgl": build_trees(tmp_path)["sgl"]}
    nemo = tmp_path / "model.nemo"
    nemo.write_bytes(b"checkpoint")
    args, engines, paths, hashes = cli(
        "asr",
        tmp_path,
        trees,
        "--nemo",
        str(nemo),
        "--device",
        "cpu",
        "--only",
        "background_speech/2",
    )
    reference_asr.run_asr(
        args, engines, paths, hashes, model=FakeParakeet(overshoot_s=0.2)
    )
    receipts = tmp_path / "out/engines/sgl/samples/background_speech/2/receipts"
    bad = json.loads((receipts / "asr-output.json").read_text())
    assert bad["status"] == "invalid_transcript" and "outside" in bad["errors"][0]
    assert json.loads((receipts / "asr-input.json").read_text())["status"] == "ok"
    ready, blocked = reference_behavior.behavior_units(
        engines[0], ["background_speech/2"]
    )
    assert ready == [] and blocked == {"background_speech/2": "asr_invalid_transcript"}


def test_cluster_bootstrap_resamples_samples_not_intervals():
    boot = reference_summary.cluster_bootstrap([[1.0, 1.0, 1.0], [3.0], []], 500, "s")
    assert boot["unit_n"] == 3 and boot["replicates"] == 500
    lo, hi = boot["ci95"]
    assert 1.0 <= lo < 1.5 < hi <= 3.0
    assert reference_summary.cluster_bootstrap([[], []], 10, "s")["ci95"] is None


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_asr_cache_keeps_official_bytes_through_judge_payload(
    tmp_path, fake_nemo, official_behavior
):
    """Pipeline ASR->cache->materialize->prepare-judge must equal the direct official path."""
    trees = {"sgl": build_trees(tmp_path)["sgl"]}
    nemo = tmp_path / "model.nemo"
    nemo.write_bytes(b"checkpoint")
    only = ["--only", "user_interruption/1"]
    args, engines, paths, hashes = cli(
        "asr", tmp_path, trees, "--nemo", str(nemo), "--device", "cpu", *only
    )
    reference_asr.run_asr(args, engines, paths, hashes, model=FakeParakeet())
    args, engines, paths, _ = cli("prepare-judge", tmp_path, trees, *only)
    assert reference_behavior.run_prepare_judge(args, engines, paths) == {"prepared": 1}
    sample = tmp_path / "out/engines/sgl/samples/user_interruption/1"
    prepared = json.loads((sample / "judge/request.json").read_text())

    direct = tmp_path / "direct/user_interruption"
    shutil.copytree(trees["sgl"] / "user_interruption", direct)
    asr = reference_core.load_module(paths["asr"], "fdb_asr_direct")
    asr.nemo_asr = reference_asr.local_model_hub(FakeParakeet(), "cpu")
    for name in reference_core.AUDIO_FILES:
        asr.get_time_aligned_transcription(str(direct), "default", name)
    for name in ("input.json", "clean_input.json", "output.json", "clean_output.json"):
        assert (sample / name).read_bytes() == (direct / "1" / name).read_bytes()
    expected = reference_behavior.build_request(official_behavior, direct / "1")
    assert (
        prepared["body"] == expected["body"]
        and prepared["request_hash"] == expected["request_hash"]
    )
    assert (
        '"input_noisy": {"text":"w0 w1","chunks":[{"text":"w0","timestamp":[0.5,1.5]}'
        in prepared["body"]["messages"][1]["content"]
    )


def test_engine_resume_freezes_projection_and_projected_manifest(tmp_path, monkeypatch):
    manifest = tmp_path / "tree/reference-manifest.json"
    manifest.parent.mkdir(parents=True)
    manifest.write_text(
        json.dumps(
            {
                "samples": [
                    {
                        "id": "a/1",
                        "variants": {
                            "overlap": {"window": {"valid": True}},
                            "clean": {"window": {"valid": False, "reasons": ["x"]}},
                        },
                    }
                ]
            }
        )
    )
    proj = tmp_path / "proj.py"
    proj.write_text(
        "def project(doc):\n"
        "    for row in doc['samples']:\n"
        "        row['sample_id'] = row['id']\n"
        "        for variant in row['variants'].values():\n"
        "            variant.update(eligible=variant['window']['valid'], "
        "reasons=variant['window'].get('reasons', []))\n"
        "    return doc\n"
    )
    out = tmp_path / "out"
    engine = reference_core.Engine(out, "sgl", manifest.parent, manifest, proj)
    assert engine.eligible("a/1", "overlap") and not engine.eligible("a/1", "clean")
    receipt = json.loads((out / "engines/sgl/manifest-receipt.json").read_text())
    assert receipt["projection_sha256"] == reference_core.sha256_file(proj)
    assert receipt["projected_manifest_sha256"] == reference_core.canonical_hash(
        engine.manifest
    )
    reference_core.Engine(out, "sgl", manifest.parent, manifest, proj)

    with pytest.raises(SystemExit, match="^sgl: projection, projection_sha256 changed"):
        reference_core.Engine(out, "sgl", manifest.parent, manifest, None)
    moved = tmp_path / "moved.py"
    shutil.copyfile(proj, moved)
    with pytest.raises(SystemExit, match="^sgl: projection changed"):
        reference_core.Engine(out, "sgl", manifest.parent, manifest, moved)
    proj.write_text(proj.read_text() + "\n# edited\n")
    with pytest.raises(SystemExit, match="^sgl: projection_sha256 changed"):
        reference_core.Engine(out, "sgl", manifest.parent, manifest, proj)
    shutil.copyfile(moved, proj)
    real = reference_core.load_manifest

    def drifted(path, projection):
        doc = real(path, projection)
        doc["samples"][0]["variants"]["clean"] = {"eligible": True, "reasons": []}
        return doc

    monkeypatch.setattr(reference_core, "load_manifest", drifted)
    with pytest.raises(SystemExit, match="^sgl: projected_manifest_sha256 changed"):
        reference_core.Engine(out, "sgl", manifest.parent, manifest, proj)


@pytest.mark.parametrize(
    "status",
    [
        "failed",
        "reused_failed",
        "blocked_stale_request",
        "asr_audio_changed",
        "reused_model_mismatch",
    ],
)
def test_cli_retained_failures_exit_nonzero(tmp_path, monkeypatch, status):
    monkeypatch.setattr(cli_module, "verify_reference", lambda path: {})
    monkeypatch.setattr(cli_module, "open_engines", lambda args: [])
    monkeypatch.setattr(cli_module, "run_prepare_judge", lambda *args: {status: 1})
    assert (
        cli_module.main(
            [
                "prepare-judge",
                "--reference-source",
                str(tmp_path),
                "--out",
                str(tmp_path / "out"),
                "--tree",
                f"model={tmp_path / 'source'}",
            ]
        )
        == 1
    )
