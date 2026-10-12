# SPDX-License-Identifier: Apache-2.0
"""Exercise the Full-Duplex-Bench v1.0 reference export, ASR, evaluation and judge ledger."""

from __future__ import annotations

import contextlib
import hashlib
import io
import json
import os
import shutil
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import soundfile
from pydantic import JsonValue

from benchmarks.duplex import reference_v10
from benchmarks.duplex.reference_core import JUDGE_MAX_ATTEMPTS, V10_REFERENCE_FILES
from benchmarks.duplex.reference_source import verify_reference
from benchmarks.duplex.v10_dataset import SUBSET_TASKS
from benchmarks.eval.benchmark_duplex_v10 import main
from tests.unit_test.benchmarks.test_duplex_reference_audio import (
    make_constant_pcm,
    make_input_pcm,
    make_session_trace,
)

REFERENCE_PATH = os.environ.get("FDB_REFERENCE_SOURCE")
REF = Path(REFERENCE_PATH) if REFERENCE_PATH else None
SAMPLE_RATE = 16000
INPUT_SAMPLES = 2 * SAMPLE_RATE
INTERRUPT_END_S = 0.5
ANNOTATIONS = {
    "synthetic_pause_handling": ("pause.json", [{"timestamp": [0.2, 0.4]}]),
    "candor_turn_taking": ("turn_taking.json", [{"timestamp": [0.4, 0.4]}]),
    "synthetic_user_interruption": (
        "interrupt.json",
        [
            {
                "timestamp": [0.2, INTERRUPT_END_S],
                "context": "tell me about tea",
                "interrupt": "what about coffee",
            }
        ],
    ),
}
# note (luojiaxuan): speech plays over [0.1, 0.3] and [0.6, 1.8] s of the 2 s window.
OUTPUT_DELTAS = [
    (0.1, make_constant_pcm(0.2, SAMPLE_RATE, 3000)),
    (0.6, make_constant_pcm(1.2, SAMPLE_RATE, 3000)),
]


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def build_capture(
    root: Path, sample_ids: list[str], unrecorded: str
) -> tuple[Path, Path]:
    """Dataset plus one v1.0 run; the unrecorded sample stays invalid without a trace."""
    dataset, run = root / "dataset", root / "run"
    input_pcm = make_input_pcm(INPUT_SAMPLES)
    samples = []
    for sample_id in [*sample_ids, unrecorded]:
        subset = sample_id.split("/")[0]
        sample_dir = dataset / sample_id
        sample_dir.mkdir(parents=True)
        soundfile.write(
            str(sample_dir / "input.wav"),
            np.frombuffer(input_pcm, "<i2"),
            SAMPLE_RATE,
            subtype="PCM_16",
        )
        paths = {"input": f"{sample_id}/input.wav"}
        sha256 = {"input": sha256_bytes((sample_dir / "input.wav").read_bytes())}
        if subset in ANNOTATIONS:
            name, entries = ANNOTATIONS[subset]
            (sample_dir / name).write_text(json.dumps(entries))
            paths["annotation"] = f"{sample_id}/{name}"
            sha256["annotation"] = sha256_bytes((sample_dir / name).read_bytes())
        else:
            pass
        directory = Path("samples") / sample_id / "input"
        if sample_id == unrecorded:
            status, recorded_input = "invalid", None
        else:
            (run / directory).mkdir(parents=True)
            (run / directory / "input.pcm").write_bytes(input_pcm)
            (run / directory / "continuous.jsonl").write_text(
                make_session_trace(input_pcm, audio_deltas=OUTPUT_DELTAS)
            )
            status, recorded_input = "pass", {"sha256": sha256_bytes(input_pcm)}
        samples.append(
            {
                "id": sample_id,
                "subset": subset,
                "task": SUBSET_TASKS[subset],
                "paths": paths,
                "sha256": sha256,
                "variants": {
                    "input": {
                        "directory": str(directory),
                        "status": status,
                        "input": recorded_input,
                        "source": {"file": paths["input"], "sha256": sha256["input"]},
                    }
                },
            }
        )
    run.mkdir(exist_ok=True)
    (run / "manifest.json").write_text(
        json.dumps({"kind": "full-duplex-bench-v1.0", "profile": "fixture"})
    )
    (run / "run.json").write_text(
        json.dumps({"status": "complete", "samples": samples})
    )
    return dataset, run


def export_tree(tmp_path: Path) -> tuple[Path, Path, Path, dict[str, JsonValue]]:
    dataset, run = build_capture(
        tmp_path,
        [
            "synthetic_pause_handling/1",
            "synthetic_user_interruption/1",
            "icc_backchannel/1",
        ],
        unrecorded="candor_turn_taking/1",
    )
    tree = tmp_path / "tree"
    stdout = io.StringIO()
    with contextlib.redirect_stdout(stdout):
        code = main(
            ["reference-export", "--run", str(run), "--dataset-root", str(dataset)]
            + ["--out", str(tree)]
        )
    assert code == 0
    return dataset, run, tree, json.loads(stdout.getvalue())


class SpanParakeet:
    """One word spanning the loud samples of the audio it is fed."""

    def __init__(self) -> None:
        self.sample_counts: list[int] = []

    def transcribe(
        self, paths: list[str], timestamps: bool
    ) -> list[types.SimpleNamespace]:
        assert timestamps is True and len(paths) == 1
        audio, sample_rate = soundfile.read(paths[0])
        self.sample_counts.append(len(audio))
        loud = np.flatnonzero(np.abs(audio) > 0.05)
        words = (
            [
                {
                    "word": "ok",
                    "start": loud[0] / sample_rate,
                    "end": (loud[-1] + 1) / sample_rate,
                }
            ]
            if len(loud)
            else []
        )
        return [types.SimpleNamespace(timestamp={"word": words})]


def install_nemo_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in ("nemo", "nemo.collections", "nemo.collections.asr"):
        monkeypatch.setitem(sys.modules, name, types.ModuleType(name))
    sys.modules["nemo"].collections = sys.modules["nemo.collections"]
    sys.modules["nemo.collections"].asr = sys.modules["nemo.collections.asr"]


def judge_response(content: str) -> types.SimpleNamespace:
    return types.SimpleNamespace(
        model="gpt-4-turbo-2024-04-09",
        usage=types.SimpleNamespace(prompt_tokens=300, completion_tokens=40),
        choices=[
            types.SimpleNamespace(
                finish_reason="stop", message=types.SimpleNamespace(content=content)
            )
        ],
    )


class FakeOpenAI:
    def __init__(self, content: str) -> None:
        self.requests: list[dict[str, JsonValue]] = []
        self.content = content
        self.chat = types.SimpleNamespace(completions=self)

    def create(self, **request: JsonValue) -> types.SimpleNamespace:
        self.requests.append(request)
        return judge_response(self.content)


def test_export_writes_reference_layout_and_manifest(tmp_path: Path) -> None:
    dataset, run, tree, counts = export_tree(tmp_path)

    assert counts["synthetic_user_interruption"] == {"selected": 1, "eligible": 1}
    assert counts["candor_turn_taking"] == {"selected": 1, "eligible": 0}
    assert counts["candor_pause_handling"] == {"selected": 0, "eligible": 0}
    assert sorted(path.name for path in tree.iterdir()) == [
        "icc_backchannel",
        "manifest.json",
        "synthetic_pause_handling",
        "synthetic_user_interruption",
    ]
    interruption = tree / "synthetic_user_interruption" / "1"
    assert sorted(path.name for path in interruption.iterdir()) == [
        "interrupt.json",
        "output.wav",
    ]
    assert [path.name for path in (tree / "icc_backchannel" / "1").iterdir()] == [
        "output.wav"
    ]
    assert (interruption / "interrupt.json").read_bytes() == (
        dataset / "synthetic_user_interruption" / "1" / "interrupt.json"
    ).read_bytes()
    info = soundfile.info(str(interruption / "output.wav"))
    assert (info.samplerate, info.channels, info.subtype, info.frames) == (
        SAMPLE_RATE,
        1,
        "PCM_16",
        INPUT_SAMPLES,
    )
    manifest = json.loads((tree / "manifest.json").read_text())
    rows = {row["sample_id"]: row for row in manifest["samples"]}
    assert rows["synthetic_user_interruption/1"]["eligible"] is True
    assert rows["synthetic_user_interruption/1"]["files"]["output.wav"] == (
        sha256_bytes((interruption / "output.wav").read_bytes())
    )
    assert rows["candor_turn_taking/1"]["eligible"] is False
    assert rows["candor_turn_taking/1"]["reasons"]

    with pytest.raises(ValueError, match="captured completely in two runs"):
        reference_v10.export(
            [run, shutil.copytree(run, tmp_path / "run2")], dataset, tmp_path / "dup"
        )
    with pytest.raises(FileExistsError):
        reference_v10.export([run], dataset, tree)


def test_result_block_parsing_covers_every_task_format() -> None:
    pause = "evaluate: 100%\n----\n[Result]\nAverage take turn:  0.25\n----\n"
    interruption = (
        "Processing x ...\n----\n[Result]\nAverage rating:  4.5\n"
        "Average take turn:  0.99\nAverage latency:  1.13\n----\n"
    )
    backchannel = (
        "JSD: 1\n----\n[Result]\nJSD - Mean: 0.7735 ± 0.0455\n"
        "TOR - Mean: 0.3455 ± 0.4755\nFrequency - Mean: 0.0676 ± 0.0462\n\n"
        "[Raw Counts]\nNumber of samples: 55\n----\n"
    )
    assert reference_v10.parse_result_block(pause) == {"Average take turn": 0.25}
    assert reference_v10.parse_result_block(interruption) == {
        "Average rating": 4.5,
        "Average take turn": 0.99,
        "Average latency": 1.13,
    }
    assert reference_v10.parse_result_block(backchannel) == {
        "JSD mean": 0.7735,
        "JSD std": 0.0455,
        "TOR mean": 0.3455,
        "TOR std": 0.4755,
        "Frequency mean": 0.0676,
        "Frequency std": 0.0462,
        "Number of samples": 55.0,
    }
    with pytest.raises(ValueError, match="one \\[Result\\] block"):
        reference_v10.parse_result_block("Traceback\n")
    with pytest.raises(ValueError, match="unrecognized"):
        reference_v10.parse_result_block("[Result]\nsomething odd\n")


def test_judge_ledger_caps_identical_requests_and_records_exchanges(
    tmp_path: Path,
) -> None:
    client = FakeOpenAI("I cannot rate this.")
    ledger_path = tmp_path / "judge" / "ledger.jsonl"
    judge = reference_v10.JudgeLedger(
        client, ledger_path, "http://127.0.0.1:30000/v1", "qwen3.8-27b"
    )
    request = {
        "model": "gpt-4-turbo",
        "messages": [{"role": "user", "content": "rate"}],
        "seed": 0,
    }
    for _ in range(JUDGE_MAX_ATTEMPTS):
        judge.chat.completions.create(**request)
    with pytest.raises(RuntimeError, match="no parsable rating"):
        judge.chat.completions.create(**request)
    judge.chat.completions.create(
        **{**request, "messages": [{"role": "user", "content": "rate again"}]}
    )

    assert [sent["model"] for sent in client.requests] == ["qwen3.8-27b"] * 4
    exchanges = [json.loads(line) for line in ledger_path.read_text().splitlines()]
    assert [exchange["attempt"] for exchange in exchanges] == [1, 2, 3, 1]
    assert {exchange["requested_model"] for exchange in exchanges} == {"gpt-4-turbo"}
    assert exchanges[0]["content"] == "I cannot rate this."
    assert exchanges[0]["prompt_tokens"] == 300
    summary = judge.summary()
    assert summary["official"] is False
    assert (summary["requests"], summary["distinct_requests"], summary["retries"]) == (
        4,
        2,
        2,
    )
    assert summary["returned_models"] == {"gpt-4-turbo-2024-04-09": 4}


def test_served_model_requires_base_url() -> None:
    with pytest.raises(SystemExit) as error:
        main(
            ["reference-evaluate", "--tree", "tree", "--reference-source", "source"]
            + ["--served-model", "qwen3.8-27b"]
        )
    assert error.value.code == 2


@pytest.mark.skipif(
    REF is None, reason="Set FDB_REFERENCE_SOURCE to the pinned external checkout"
)
def test_reference_asr_crops_interruption_and_evaluation_runs_pinned_code(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _, _, tree, _ = export_tree(tmp_path)
    paths = verify_reference(REF, V10_REFERENCE_FILES)
    install_nemo_stubs(monkeypatch)
    nemo_path = tmp_path / "parakeet.nemo"
    nemo_path.write_bytes(b"checkpoint")
    parakeet = SpanParakeet()

    counts = reference_v10.transcribe(tree, paths, nemo_path, None, "cpu", parakeet)

    assert counts == {
        "synthetic_pause_handling": 1,
        "synthetic_user_interruption": 1,
        "icc_backchannel": 1,
    }
    cropped_samples = INPUT_SAMPLES - int(INTERRUPT_END_S * SAMPLE_RATE)
    assert parakeet.sample_counts == [INPUT_SAMPLES, cropped_samples, INPUT_SAMPLES]
    interruption = json.loads(
        (tree / "synthetic_user_interruption" / "1" / "output.json").read_text()
    )
    pause = json.loads(
        (tree / "synthetic_pause_handling" / "1" / "output.json").read_text()
    )
    assert interruption["text"] == "ok"
    assert interruption["chunks"][0]["timestamp"] == pytest.approx([0.6, 1.8])
    assert pause["chunks"][0]["timestamp"] == pytest.approx([0.1, 1.8])
    with pytest.raises(SystemExit, match="exists"):
        reference_v10.transcribe(tree, paths, nemo_path, None, "cpu", parakeet)

    client = FakeOpenAI(
        "Analysis: It answers the coffee question.\nI would rate the AI's response as 4."
    )
    judge = reference_v10.JudgeLedger(
        client, tree / "judge" / "ledger.jsonl", None, None
    )
    summary = reference_v10.evaluate(
        tree, paths, ["synthetic_user_interruption"], judge
    )

    result = summary["subsets"]["synthetic_user_interruption"]
    assert result["result"] == pytest.approx(
        {"Average rating": 4.0, "Average take turn": 1.0, "Average latency": 0.1}
    )
    assert (result["selected"], result["evaluated"]) == (1, 1)
    assert result["judge"]["official"] is True
    assert result["judge"]["requests"] == 1 and result["judge"]["retries"] == 0
    assert client.requests[0]["model"] == "gpt-4-turbo"
    assert client.requests[0]["seed"] == 0
    with pytest.raises(SystemExit, match="already evaluated"):
        reference_v10.evaluate(tree, paths, ["synthetic_user_interruption"], judge)
    with pytest.raises(SystemExit, match="not exported"):
        reference_v10.evaluate(tree, paths, ["candor_turn_taking"], None)

    summary = reference_v10.evaluate(tree, paths, ["synthetic_pause_handling"], None)
    assert summary["subsets"]["synthetic_pause_handling"]["result"] == {
        "Average take turn": 1.0
    }
    assert set(summary["subsets"]) == {
        "synthetic_user_interruption",
        "synthetic_pause_handling",
    }
