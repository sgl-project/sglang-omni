# SPDX-License-Identifier: Apache-2.0

import base64
import hashlib
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest
from pydantic import ValidationError
from test_duplex_oracle import trace_fixture

from benchmarks.duplex import artifacts


@pytest.fixture
def recorded_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    pcm = b"\x01\x00" * 160
    (tmp_path / "input.pcm").write_bytes(pcm)
    manifest = {
        "schema_version": 1,
        "profile": "nemotron-voicechat-pr2188",
        "source": {"harness_git_head": "recorded-revision"},
        "server": {
            "revision": "e1b9c9c674b1187918593257906ee6e8cc6a13da",
            "revision_source": "operator_supplied",
        },
        "config": {
            "packet_ms": 80,
            "timeout_s": 90,
            "unsupported": ["concurrent_sessions"],
        },
        "input": {
            "file": "input.pcm",
            "sha256": hashlib.sha256(pcm).hexdigest(),
            "sample_rate": 16000,
        },
        "cases": [
            {
                "id": "continuous",
                "scenario": "continuous",
                "trace_file": "continuous.jsonl",
            },
            {
                "id": "continuous-repeat",
                "scenario": "continuous",
                "trace_file": "continuous-repeat.jsonl",
            },
        ],
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    for case in manifest["cases"]:
        records = [
            {
                "direction": "send",
                "time_s": 0.5,
                "event": {
                    "type": "input_audio_buffer.append",
                    "audio": base64.b64encode(pcm).decode(),
                },
            },
            {"direction": "receive", "time_s": 1.0, "event": {"type": "fixture"}},
        ]
        (tmp_path / case["trace_file"]).write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )

    def evaluate(records: list[dict], scenario: str, profile: str) -> dict:
        return {
            "status": "pass",
            "violations": [],
            "coverage": {"input_output_overlap": True},
            "metrics": {"first_audio_packet_s": 0.2, "input_output_overlap": True},
        }

    monkeypatch.setattr(artifacts, "evaluate_trace", evaluate)
    return tmp_path


def test_replay_accounts_for_every_selected_case(recorded_run: Path) -> None:
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["passed"] == 2
    assert result["summary"]["failed"] == 0
    assert result["summary"]["not_exercised"] == 0
    assert result["recorded_source"] == {"harness_git_head": "recorded-revision"}
    assert result["server"]["revision_source"] == "operator_supplied"
    assert result["config"]["unsupported"] == ["concurrent_sessions"]
    assert result["summary"]["qualified_metrics"] == {
        "first_audio_packet_s": {"n": 2, "mean": 0.2, "min": 0.2, "max": 0.2}
    }
    assert artifacts.replay_run(recorded_run) == result


@pytest.mark.parametrize(
    "defect", ["missing", "truncated", "bad_direction", "nan_clock", "non_object"]
)
def test_bad_trace_retains_selected_denominator(
    recorded_run: Path, defect: str
) -> None:
    path = recorded_run / "continuous-repeat.jsonl"
    if defect == "missing":
        path.unlink()
    elif defect == "truncated":
        with path.open("a") as handle:
            handle.write('{"direction":')
    elif defect == "bad_direction":
        path.write_text(json.dumps({"direction": "invalid", "time_s": 1, "event": {}}))
    elif defect == "nan_clock":
        path.write_text(
            json.dumps({"direction": "receive", "time_s": float("nan"), "event": {}})
        )
    else:
        path.write_text("[]\n")
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["passed"] == 1
    assert result["summary"]["failed"] == 1
    assert result["cases"][1]["artifact_errors"]
    assert result["summary"]["qualified_metrics"]["first_audio_packet_s"]["n"] == 1


@pytest.mark.parametrize("defect", ["missing", "modified", "odd_bytes", "empty"])
def test_bad_pcm_fails_all_affected_cases(recorded_run: Path, defect: str) -> None:
    path = recorded_run / "input.pcm"
    if defect == "missing":
        path.unlink()
    elif defect == "modified":
        path.write_bytes(b"\x02\x00" * 160)
    elif defect == "odd_bytes":
        path.write_bytes(b"\x01")
    else:
        path.write_bytes(b"")
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["failed"] == 2
    assert result["summary"]["qualified_metrics"] == {}
    assert all(case["artifact_errors"] for case in result["cases"])


@pytest.mark.parametrize(
    "audio", [base64.b64encode(b"\x02\x00" * 160).decode(), "not base64", None]
)
def test_trace_audio_must_match_persisted_input(
    recorded_run: Path, audio: str | None
) -> None:
    path = recorded_run / "continuous.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[0]["event"]["audio"] = audio
    path.write_text("".join(json.dumps(record) + "\n" for record in records))
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["passed"] == 1
    assert result["summary"]["failed"] == 1
    assert (
        "sent audio does not match persisted input PCM"
        in result["cases"][0]["artifact_errors"]
    )


def test_partial_send_preserves_client_failure_and_identity_diagnostic(
    recorded_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = recorded_run / "continuous.jsonl"
    path.write_text(
        json.dumps(
            {
                "direction": "error",
                "time_s": 1.0,
                "event": {"message": "connection timeout"},
            }
        )
        + "\n"
    )
    monkeypatch.setattr(
        artifacts,
        "evaluate_trace",
        lambda records, scenario, profile: {
            "status": "fail",
            "violations": ["client error: connection timeout"],
            "coverage": {"input_output_overlap": False},
            "metrics": {},
        },
    )
    result = artifacts.replay_run(recorded_run)
    assert result["cases"][0]["status"] == "fail"
    assert "client error: connection timeout" in result["cases"][0]["violations"]
    assert (
        "sent audio does not match persisted input PCM"
        in result["cases"][0]["artifact_errors"]
    )


@pytest.mark.parametrize("field", ["input", "trace"])
@pytest.mark.parametrize("path_kind", ["relative", "absolute", "symlink"])
def test_artifact_paths_cannot_escape_run_directory(
    recorded_run: Path, field: str, path_kind: str
) -> None:
    outside = recorded_run.parent / f"outside-{field}"
    outside.write_bytes(b"outside")
    if path_kind == "relative":
        path_value = f"../{outside.name}"
    elif path_kind == "absolute":
        path_value = str(outside)
    else:
        (recorded_run / "link").symlink_to(outside)
        path_value = "link"
    manifest_path = recorded_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if field == "input":
        manifest["input"]["file"] = path_value
    else:
        manifest["cases"][0]["trace_file"] = path_value
    manifest_path.write_text(json.dumps(manifest))
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert "escapes run directory" in " ".join(result["cases"][0]["artifact_errors"])
    assert result["cases"][0]["status"] == "fail"


@pytest.mark.parametrize("field", ["id", "trace_file"])
def test_duplicate_selected_identity_fails_without_dropping_cases(
    recorded_run: Path, field: str
) -> None:
    manifest_path = recorded_run / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["cases"][1][field] = manifest["cases"][0][field]
    manifest_path.write_text(json.dumps(manifest))
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["failed"] == 2


def test_not_exercised_is_not_pass_or_qualified_timing(
    recorded_run: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = recorded_run / "continuous-repeat.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    records[-1]["time_s"] = 2.0
    path.write_text("".join(json.dumps(record) + "\n" for record in records))

    def evaluate(records: list[dict], scenario: str, profile: str) -> dict:
        unexercised = records[-1]["time_s"] == 2.0
        return {
            "status": "not_exercised" if unexercised else "pass",
            "violations": [],
            "coverage": {"input_output_overlap": not unexercised},
            "metrics": {"first_audio_packet_s": 0.4 if unexercised else 0.2},
        }

    monkeypatch.setattr(artifacts, "evaluate_trace", evaluate)
    result = artifacts.replay_run(recorded_run)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["passed"] == 1
    assert result["summary"]["not_exercised"] == 1
    assert result["summary"]["qualified_metrics"]["first_audio_packet_s"]["mean"] == 0.2
    assert (
        result["summary"]["diagnostic_metrics"]["first_audio_packet_s"]["mean"] == 0.4
    )
    assert result["cases"][1]["coverage"] == {"input_output_overlap": False}


def test_replay_grades_real_traces_with_the_real_oracle(tmp_path: Path) -> None:
    pcm = b"\x00\x00" * 2560
    (tmp_path / "input.pcm").write_bytes(pcm)
    cases = [
        {
            "id": "continuous",
            "scenario": "continuous",
            "trace_file": "continuous.jsonl",
        },
        {
            "id": "continuous-repeat",
            "scenario": "continuous",
            "trace_file": "continuous-repeat.jsonl",
        },
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "nemotron-voicechat-pr2188",
                "source": {"harness_git_head": "recorded-revision"},
                "server": {
                    "revision": "e1b9c9c674b1187918593257906ee6e8cc6a13da",
                    "revision_source": "operator_supplied",
                },
                "config": {"packet_ms": 80},
                "input": {
                    "file": "input.pcm",
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                    "sample_rate": 16000,
                },
                "cases": cases,
            }
        )
    )
    for case in cases:
        records = trace_fixture()
        if case["id"] == "continuous-repeat":
            records.insert(
                0,
                {
                    "direction": "admission",
                    "time_s": 99.0,
                    "event": {
                        "type": "connection_denied",
                        "http_status": 503,
                        "attempt": 1,
                    },
                },
            )
        (tmp_path / case["trace_file"]).write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    result = artifacts.replay_run(tmp_path)
    assert result["summary"]["selected"] == 2
    assert result["summary"]["passed"] == 2
    assert result["cases"][1]["metrics"]["admission_denials"] == 1
    assert [case["violations"] for case in result["cases"]] == [[], []]
    assert result["cases"][0]["coverage"] == {"input_output_overlap": True}
    assert result["cases"][1]["coverage"] == {"input_output_overlap": True}
    assert result["summary"]["qualified_metrics"]["input_audio_s"] == {
        "n": 2,
        "mean": pytest.approx(0.16),
        "min": pytest.approx(0.16),
        "max": pytest.approx(0.16),
    }
    assert result["summary"]["diagnostic_metrics"] == {}


def test_replay_counts_healthy_case_beside_uncorrelated_receipt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    pcm = b"\x00\x00" * 2560
    (tmp_path / "input.pcm").write_bytes(pcm)
    cases = [
        {
            "id": "continuous",
            "scenario": "continuous",
            "trace_file": "continuous.jsonl",
        },
        {
            "id": "continuous-repeat",
            "scenario": "continuous",
            "trace_file": "continuous-repeat.jsonl",
        },
    ]
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "profile": "nemotron-voicechat-pr2188",
                "source": {"harness_git_head": "recorded-revision"},
                "server": {
                    "revision": "e1b9c9c674b1187918593257906ee6e8cc6a13da",
                    "revision_source": "operator_supplied",
                },
                "config": {"packet_ms": 80},
                "input": {
                    "file": "input.pcm",
                    "sha256": hashlib.sha256(pcm).hexdigest(),
                    "sample_rate": 16000,
                },
                "cases": cases,
            }
        )
    )
    for case in cases:
        records = trace_fixture()
        if case["id"] == "continuous-repeat":
            next(
                record["event"]
                for record in records
                if record["event"]["type"] == "sglang.input_audio.accepted"
            ).pop("client_event_id")
        (tmp_path / case["trace_file"]).write_text(
            "".join(json.dumps(record) + "\n" for record in records)
        )
    monkeypatch.setattr(sys, "argv", ["artifacts", str(tmp_path)])
    with pytest.raises(SystemExit) as stopped:
        artifacts.main()
    assert stopped.value.code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["summary"]["selected"] == 2
    assert report["summary"]["passed"] == 1
    assert report["summary"]["failed"] == 1
    assert report["cases"][0]["status"] == "pass"
    assert (
        "sglang.input_audio.accepted: missing matching prior client event"
        in report["cases"][1]["violations"]
    )
    assert report["summary"]["qualified_metrics"]["input_audio_s"]["n"] == 1


def test_replay_rejects_unknown_manifest_version(recorded_run: Path) -> None:
    path = recorded_run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["schema_version"] = 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="schema_version"):
        artifacts.replay_run(recorded_run)


@pytest.mark.parametrize(
    "status,exit_code", [("pass", 0), ("fail", 1), ("not_exercised", 1)]
)
def test_cli_succeeds_only_when_every_case_passes(
    recorded_run: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture,
    status: str,
    exit_code: int,
) -> None:
    monkeypatch.setattr(
        artifacts,
        "evaluate_trace",
        lambda records, scenario, profile: {
            "status": status,
            "violations": [],
            "coverage": {},
            "metrics": {},
        },
    )
    monkeypatch.setattr(sys, "argv", ["artifacts", str(recorded_run)])
    with pytest.raises(SystemExit) as stopped:
        artifacts.main()
    assert stopped.value.code == exit_code
    assert json.loads(capsys.readouterr().out)["summary"]["selected"] == 2


def test_cli_reports_unreadable_manifest_as_failed_run(
    recorded_run: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture
) -> None:
    (recorded_run / "manifest.json").write_text('{"schema_version":')
    monkeypatch.setattr(sys, "argv", ["artifacts", str(recorded_run)])
    with pytest.raises(SystemExit) as stopped:
        artifacts.main()
    assert stopped.value.code == 1
    assert json.loads(capsys.readouterr().out)["status"] == "fail"


REVISION = "e1b9c9c674b1187918593257906ee6e8cc6a13da"


def test_server_identity_records_claims_and_served_models() -> None:
    class Models(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            body = json.dumps({"data": [{"id": "nemotron-voicechat"}]}).encode()
            self.send_response(200 if self.path == "/v1/models" else 404)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Models)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        port = server.server_address[1]
        identity = artifacts.server_identity(
            f"ws://127.0.0.1:{port}/v1/realtime?intent=x",
            revision=REVISION,
            model="nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
            model_revision="0" * 40,
            runtime="sha256:abc",
        )
    finally:
        server.shutdown()
    assert identity == {
        "revision": REVISION,
        "revision_source": "operator_supplied",
        "model": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
        "model_revision": "0" * 40,
        "runtime": "sha256:abc",
        "identity_source": "operator_supplied",
        "served_models": {
            "url": f"http://127.0.0.1:{port}/v1/models",
            "ids": ["nemotron-voicechat"],
        },
    }


def test_server_identity_records_unreachable_models_endpoint() -> None:
    identity = artifacts.server_identity(
        "wss://127.0.0.1:1/v1/realtime", revision=REVISION, model="m"
    )
    assert identity["served_models"]["url"] == "https://127.0.0.1:1/v1/models"
    assert identity["served_models"]["error"]
    assert identity["model_revision"] is None
    assert identity["runtime"] is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"revision": "main", "model": "m"}, "server revision"),
        ({"revision": REVISION, "model": ""}, "model must be nonempty"),
        ({"revision": REVISION, "model": "m", "model_revision": "v1"}, "model rev"),
    ],
)
def test_server_identity_rejects_unpinned_claims(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        artifacts.server_identity("ws://127.0.0.1:1/v1/realtime", **kwargs)


def test_legacy_cancel_manifest_requires_its_recorded_harness(
    recorded_run: Path,
) -> None:
    path = recorded_run / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["cases"][1]["scenario"] = "cancel_resume"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValidationError, match="scenario"):
        artifacts.replay_run(recorded_run)
