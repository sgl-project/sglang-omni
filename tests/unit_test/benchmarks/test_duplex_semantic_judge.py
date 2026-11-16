# SPDX-License-Identifier: Apache-2.0
"""Verify the semantic judge's control gate while it runs beside the grading batches."""

from __future__ import annotations

import copy
import hashlib
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
from pydantic import JsonValue

from benchmarks.duplex import semantic_judge

ENGINE = "sgl"
PAIRS = 17
WAIT_S = 5
Reply = Callable[[dict[str, JsonValue]], dict[str, JsonValue]]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path: Path, document: JsonValue) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document))


def build_semantic_inputs(root: Path) -> list[str]:
    reference_audio, scores, dataset = (
        root / "reference-audio",
        root / "scores",
        root / "dataset",
    )
    samples = []
    for number in range(1, PAIRS + 1):
        sample_id = f"user_interruption/{number}"
        samples.append(
            {
                "sample_id": sample_id,
                "category": "user_interruption",
                "variants": {
                    "overlap": {
                        "reasons": [],
                        "window": {"T_s": 8.0},
                        "source": {"directory": str(root / "capture" / sample_id)},
                        "boundary": {"overlap_end_s": 2.0},
                    },
                    "clean": {"reasons": []},
                },
            }
        )
        write(
            dataset / sample_id / "metadata.json",
            {
                "context_text": "Name a bird.",
                "current_turn_text": "Actually, name a mammal.",
                "timestamps": [1, 2],
            },
        )
        sample_scores = scores / "engines" / ENGINE / "samples" / sample_id
        for role in semantic_judge.TRANSCRIPT_ROLES:
            audio = reference_audio / sample_id / f"{role}.wav"
            audio.parent.mkdir(parents=True, exist_ok=True)
            audio.write_bytes(f"{sample_id} {role}".encode())
            text = f"{role} text of pair {number}."
            write(
                sample_scores / f"{role}.json",
                {"text": text, "chunks": [{"text": text, "timestamp": [3, 6]}]},
            )
            write(
                sample_scores / "receipts" / f"asr-{role}.json",
                {
                    "status": "ok",
                    "audio_sha256": sha256(audio),
                    "transcript_sha256": sha256(sample_scores / f"{role}.json"),
                },
            )
    write(reference_audio / "reference-manifest.json", {"samples": samples})
    return [
        "--reference-audio",
        str(reference_audio),
        "--scores",
        str(scores),
        "--engine",
        ENGINE,
        "--dataset-root",
        str(dataset),
        "--base-url",
        "http://127.0.0.1:1/v1",
        "--model-id",
        "Qwen/Qwen3.8-27B",
        "--model-revision",
        "a" * 40,
        "--served-model",
        "judge",
    ]


def packets(body: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    cases = body["messages"][-1]["content"].split("Cases, in input order:\n", 1)[1]
    return [json.loads(line) for line in cases.splitlines()]


def is_control(body: dict[str, JsonValue]) -> bool:
    return packets(body)[0]["sample_id"].startswith("s")


def expected_controls() -> list[dict[str, JsonValue]]:
    return semantic_judge.read_json(semantic_judge.CONTROL_EXPECTED_PATH)["assessments"]


def failing_controls() -> list[dict[str, JsonValue]]:
    assessments = copy.deepcopy(expected_controls())
    assessments[0]["axes"]["interaction_handling"]["status"] = "fail"
    return assessments


def graded(packet: dict[str, JsonValue]) -> dict[str, JsonValue]:
    """Pair numbers 3k+1 accept, 3k+2 fail on relevance and 3k stay unresolved."""
    number = int(packet["sample_id"][1:])
    if number % 3 == 0:
        axis = {"status": "unresolved", "reason": "", "source": "none", "quote": ""}
        verdicts = dict.fromkeys(semantic_judge.AXES, axis)
    else:
        verdicts = {
            axis: {
                "status": (
                    "fail" if number % 3 == 2 and axis == "relevance" else "accept"
                ),
                "reason": "Quoted from the reply.",
                "source": "output_asr_text",
                "quote": packet["output_asr_text"],
            }
            for axis in semantic_judge.AXES
        }
    return {"sample_id": packet["sample_id"], "axes": verdicts}


def completion(assessments: list[dict[str, JsonValue]]) -> dict[str, JsonValue]:
    return {
        "choices": [
            {
                "finish_reason": "stop",
                "message": {"content": json.dumps({"assessments": assessments})},
            }
        ]
    }


def scripted(
    body: dict[str, JsonValue], controls: list[dict[str, JsonValue]]
) -> dict[str, JsonValue]:
    if is_control(body):
        return completion(controls)
    else:
        return completion([graded(packet) for packet in packets(body)])


def install_client(monkeypatch: pytest.MonkeyPatch, reply: Reply) -> None:
    def create(**body: JsonValue) -> SimpleNamespace:
        document = reply(body)
        return SimpleNamespace(model_dump=lambda **_: document)

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace()))
    client.chat.completions.create = create
    monkeypatch.setattr(semantic_judge, "OpenAI", lambda **_: client)


def output_documents(out: Path) -> dict[str, JsonValue]:
    def without_start_times(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return {
                key: without_start_times(item)
                for key, item in value.items()
                if key != "started_at"
            }
        elif isinstance(value, list):
            return [without_start_times(item) for item in value]
        else:
            return value

    return {
        str(path.relative_to(out)): without_start_times(semantic_judge.read_json(path))
        for path in sorted(out.rglob("*.json"))
    }


def test_control_request_overlaps_first_batch(tmp_path, monkeypatch):
    argv = build_semantic_inputs(tmp_path)
    control_sent, batch_sent = threading.Event(), threading.Event()
    overlapped = {}

    def reply(body):
        if is_control(body):
            control_sent.set()
            overlapped["control saw a batch"] = batch_sent.wait(WAIT_S)
        elif packets(body)[0]["sample_id"] == "p0001":
            batch_sent.set()
            overlapped["batch saw the control"] = control_sent.wait(WAIT_S)
        else:
            pass
        return scripted(body, expected_controls())

    install_client(monkeypatch, reply)
    assert semantic_judge.main([*argv, "--out", str(tmp_path / "out")]) == 0
    assert overlapped == {"control saw a batch": True, "batch saw the control": True}


def ordered_reply(order: str, batch_count: int) -> Reply:
    """Answer the control request before every batch, or after all of them."""
    controls_done, batches_done = threading.Event(), threading.Event()
    answered, lock = [], threading.Lock()

    def reply(body):
        if is_control(body):
            if order == "batches first":
                assert batches_done.wait(WAIT_S)
            else:
                pass
            document = scripted(body, expected_controls())
            controls_done.set()
        else:
            if order == "controls first":
                assert controls_done.wait(WAIT_S)
            else:
                pass
            document = scripted(body, expected_controls())
            with lock:
                answered.append(packets(body)[0]["sample_id"])
                if len(answered) == batch_count:
                    batches_done.set()
                else:
                    pass
        return document

    return reply


def test_passing_controls_give_the_sequential_outputs(tmp_path, monkeypatch, capsys):
    argv = build_semantic_inputs(tmp_path)
    batch_count = -(-PAIRS // semantic_judge.MAX_BATCH_PAIRS)
    runs = {}
    for order in ("controls first", "batches first"):
        install_client(monkeypatch, ordered_reply(order, batch_count))
        out = tmp_path / order.replace(" ", "-")
        assert semantic_judge.main([*argv, "--out", str(out)]) == 0
        runs[order] = (output_documents(out), capsys.readouterr().out)
    assert runs["controls first"][0] == runs["batches first"][0]
    assert runs["controls first"][1] == runs["batches first"][1].replace(
        "batches-first", "controls-first"
    )
    documents, stdout = runs["controls first"]
    assert stdout.startswith("controls: 18/18 axis statuses match\n")
    assert documents["controls/comparison.json"]["mismatches"] == 0
    assert sorted(
        name.split("/")[1] for name in documents if "assessments" in name
    ) == [f"b{index:03d}" for index in range(1, batch_count + 1)]
    joint = documents["summary.json"]["overall_joint"]
    assert joint == {
        "selected": PAIRS,
        "accepted": len([n for n in range(1, PAIRS + 1) if n % 3 == 1]),
        "failed": len([n for n in range(1, PAIRS + 1) if n % 3 == 2]),
        "unresolved": len([n for n in range(1, PAIRS + 1) if n % 3 == 0]),
    }
    assert [row["joint"] for row in documents["quality-outcomes.json"]] == [
        {0: "unresolved", 1: "accept", 2: "fail"}[number % 3]
        for number in range(1, PAIRS + 1)
    ]


def signalling_executor(queue_cancelled: threading.Event) -> type[ThreadPoolExecutor]:
    class Executor(ThreadPoolExecutor):
        """Signals once the queued batches are cancelled, then waits as usual."""

        def shutdown(self, wait=True, *, cancel_futures=False):
            super().shutdown(wait=False, cancel_futures=cancel_futures)
            if cancel_futures:
                queue_cancelled.set()
            else:
                pass
            super().shutdown(wait=wait)

    return Executor


def test_failing_controls_exit_1_and_keep_no_batch(tmp_path, monkeypatch, capsys):
    argv = build_semantic_inputs(tmp_path)
    batch_sent, queue_cancelled = threading.Event(), threading.Event()
    sent = []

    def reply(body):
        if is_control(body):
            assert batch_sent.wait(WAIT_S)
            return completion(failing_controls())
        else:
            sent.append(packets(body)[0]["sample_id"])
            batch_sent.set()
            assert queue_cancelled.wait(WAIT_S)
            return scripted(body, failing_controls())

    monkeypatch.setattr(
        semantic_judge, "ThreadPoolExecutor", signalling_executor(queue_cancelled)
    )
    install_client(monkeypatch, reply)
    out = tmp_path / "out"
    assert semantic_judge.main([*argv, "--out", str(out), "--concurrency", "1"]) == 1
    captured = capsys.readouterr()
    assert captured.out == (
        "controls: 17/18 axis statuses match\n"
        "  s01 interaction_handling: expected accept, got fail\n"
    )
    assert captured.err == (
        "Control check failed: this judge configuration is not accepted.\n"
    )
    assert sent == ["p0001"]
    assert not (out / "batches").exists()
    assert not (out / "quality-outcomes.json").exists()
    assert not (out / "summary.json").exists()


def test_rerun_after_a_refused_judge_equals_a_fresh_run(tmp_path, monkeypatch):
    argv = build_semantic_inputs(tmp_path)
    out, fresh = tmp_path / "out", tmp_path / "fresh"

    def refused(body):
        raise ConnectionRefusedError(111, "Connection refused")

    install_client(monkeypatch, refused)
    assert semantic_judge.main([*argv, "--out", str(out)]) == 1
    assert not (out / "batches").exists()
    install_client(monkeypatch, lambda body: scripted(body, expected_controls()))
    assert semantic_judge.main([*argv, "--out", str(out)]) == 0
    assert semantic_judge.main([*argv, "--out", str(fresh)]) == 0
    assert output_documents(out) == output_documents(fresh)


@pytest.mark.parametrize("error", [KeyboardInterrupt, OSError])
def test_control_error_cancels_queued_batches(tmp_path, monkeypatch, error):
    argv = build_semantic_inputs(tmp_path)
    batch_sent, queue_cancelled = threading.Event(), threading.Event()
    sent = []

    def broken_controls(*_):
        assert batch_sent.wait(WAIT_S)
        raise error

    def reply(body):
        sent.append(packets(body)[0]["sample_id"])
        batch_sent.set()
        assert queue_cancelled.wait(WAIT_S)
        return scripted(body, expected_controls())

    monkeypatch.setattr(
        semantic_judge, "ThreadPoolExecutor", signalling_executor(queue_cancelled)
    )
    monkeypatch.setattr(semantic_judge, "run_controls", broken_controls)
    install_client(monkeypatch, reply)
    out = tmp_path / "out"
    with pytest.raises(error):
        semantic_judge.main([*argv, "--out", str(out), "--concurrency", "1"])
    assert sent == ["p0001"]
    assert not (out / "batches").exists()


@pytest.mark.parametrize("passing,code", [(True, 0), (False, 1)])
def test_controls_only_sends_no_batches(tmp_path, monkeypatch, passing, code):
    argv = build_semantic_inputs(tmp_path)
    controls = expected_controls() if passing else failing_controls()

    def reply(body):
        assert is_control(body)
        return completion(controls)

    install_client(monkeypatch, reply)
    out = tmp_path / "out"
    assert semantic_judge.main([*argv, "--out", str(out), "--controls-only"]) == code
    assert sorted(path.name for path in out.iterdir()) == [
        "controls",
        "judge-config.json",
    ]
