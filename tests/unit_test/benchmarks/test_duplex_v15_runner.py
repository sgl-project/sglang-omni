# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import base64
import itertools
import json
from pathlib import Path

import numpy as np
import pytest
import soundfile
import websockets
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex import v15_runner
from benchmarks.duplex.client import PACKET_MS
from benchmarks.duplex.v15_audio import normalize_audio, reconstruct_output
from benchmarks.duplex.v15_dataset import SUBSETS, discover_samples, inventory
from benchmarks.duplex.v15_runner import run_pairs
from tests.unit_test.benchmarks.test_duplex_client import FIXTURE_PCM, DuplexPeer

SERVER_REVISION = "e1b9c9c674b1187918593257906ee6e8cc6a13da"
SERVER = {
    "revision": SERVER_REVISION,
    "revision_source": "operator_supplied",
    "model": "nvidia/NVIDIA-NemotronLabs-VoiceChat-11B",
}
FIXTURE_SAMPLES = np.frombuffer(FIXTURE_PCM, dtype="<i2")
EXTRA_TEXT = {
    "user_interruption": {"current_turn_text": "wait, actually"},
    "user_backchannel": {"backchannel_text": "yeah"},
    "talking_to_other": {"current_turn_text": "one second"},
    "background_speech": {"background_text": "it is raining"},
}


def write_sample(root: Path, subset: str, name: str, **overrides) -> Path:
    sample_dir = root / subset / name
    sample_dir.mkdir(parents=True)
    for filename in ("input.wav", "clean_input.wav"):
        soundfile.write(
            str(sample_dir / filename), FIXTURE_SAMPLES, 16000, subtype="PCM_16"
        )
    metadata = {"context_text": "hello", **EXTRA_TEXT[subset], "timestamps": [0.1, 0.3]}
    (sample_dir / "metadata.json").write_text(
        json.dumps({**metadata, **overrides.get("metadata", {})})
    )
    for filename in ("input.json", "clean_input.json"):
        (sample_dir / filename).write_text(
            json.dumps(
                {
                    "text": "hello",
                    "chunks": [{"text": "hello", "timestamp": [0.0, 0.1]}],
                }
            )
        )
    return sample_dir


def write_dataset(root: Path, names: tuple[str, ...] = ("1",)) -> None:
    for subset in SUBSETS:
        for name in names:
            write_sample(root, subset, name)


def test_discovery_covers_four_subsets_deterministically(tmp_path: Path) -> None:
    write_dataset(tmp_path, ("10", "2"))
    (tmp_path / "__MACOSX" / "user_interruption").mkdir(parents=True)
    (tmp_path / "user_interruption" / ".DS_Store").write_text("junk")

    samples = discover_samples(tmp_path)

    assert [s.id for s in samples] == [
        f"{subset}/{name}" for subset in SUBSETS for name in ("2", "10")
    ]
    assert all(not s.errors for s in samples)
    backchannel = next(s for s in samples if s.subset == "user_backchannel")
    assert backchannel.metadata["backchannel_text"] == "yeah"
    assert "current_turn_text" not in backchannel.metadata
    assert backchannel.event_span_s == [0.1, 0.3]
    assert backchannel.paths["clean_input"] == "user_backchannel/2/clean_input.wav"
    assert set(backchannel.transcripts) == {
        "input_transcript",
        "clean_input_transcript",
    }
    assert backchannel.audio["input"]["duration_s"] == pytest.approx(0.64)
    assert inventory(tmp_path) == {
        "declared": {
            "user_interruption": 200,
            "user_backchannel": 99,
            "talking_to_other": 100,
            "background_speech": 100,
        },
        "observed": {subset: 2 for subset in SUBSETS},
        "ignored": ["__MACOSX", "user_interruption/.DS_Store"],
    }
    assert [s.id for s in discover_samples(tmp_path, max_per_subset=1)] == [
        f"{subset}/2" for subset in SUBSETS
    ]


def test_selection_rejects_unknown_duplicate_and_ambiguous_requests(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        discover_samples(tmp_path, ["user_backchannel/1", "user_backchannel/1"])
    with pytest.raises(ValueError, match="unknown"):
        discover_samples(tmp_path, ["user_backchannel/99"])
    with pytest.raises(ValueError, match="unknown"):
        discover_samples(tmp_path, ["1"])
    with pytest.raises(ValueError, match="mutually exclusive"):
        discover_samples(tmp_path, ["user_backchannel/1"], max_per_subset=1)
    assert [s.id for s in discover_samples(tmp_path, ["background_speech/1"])] == [
        "background_speech/1"
    ]


def test_selected_missing_and_malformed_samples_keep_their_errors(
    tmp_path: Path,
) -> None:
    write_dataset(tmp_path)
    (tmp_path / "user_interruption" / "1" / "clean_input.wav").unlink()
    write_sample(tmp_path, "user_backchannel", "2", metadata={"timestamps": [0.3, 0.1]})
    write_sample(tmp_path, "talking_to_other", "2", metadata={"timestamps": [0.1, 9.0]})
    write_sample(
        tmp_path, "background_speech", "2", metadata={"timestamps": [float("nan"), 0.2]}
    )
    (write_sample(tmp_path, "background_speech", "3") / "input.json").write_text("{")

    samples = {s.id: s for s in discover_samples(tmp_path)}

    assert len(samples) == 8
    assert samples["user_interruption/1"].errors == ["missing clean_input.wav"]
    assert "not ordered" in samples["user_backchannel/2"].errors[0]
    assert "exceeds input.wav duration" in samples["talking_to_other/2"].errors[0]
    assert "finite" in samples["background_speech/2"].errors[0]
    assert "input.json is not valid JSON" in samples["background_speech/3"].errors[0]
    assert samples["background_speech/2"].event_span_s is None


def test_source_read_failure_stays_with_its_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    write_dataset(tmp_path)
    unreadable = (tmp_path / "user_backchannel" / "1" / "input.wav").resolve()
    real_read_bytes = Path.read_bytes

    def failing_read_bytes(path: Path) -> bytes:
        if path.resolve() == unreadable:
            raise PermissionError(13, "Permission denied", str(path))
        return real_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", failing_read_bytes)
    samples = {s.id: s for s in discover_samples(tmp_path)}

    assert len(samples) == 4
    failed = samples["user_backchannel/1"]
    assert failed.errors == [
        "input.wav unreadable: [Errno 13] Permission denied: "
        f"'{tmp_path / 'user_backchannel' / '1' / 'input.wav'}'"
    ]
    assert "input" not in failed.paths and "input" not in failed.sha256
    assert "clean_input" in failed.sha256
    assert all(not s.errors for i, s in samples.items() if i != "user_backchannel/1")


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_normalization_rejects_nonfinite_float_samples(
    tmp_path: Path, bad: float
) -> None:
    path = tmp_path / "bad.wav"
    soundfile.write(str(path), np.array([0.1, bad, 0.2]), 16000, subtype="FLOAT")
    with pytest.raises(ValueError, match="1 non-finite source samples"):
        normalize_audio(path)


def test_normalization_declares_downmix_and_resampling(tmp_path: Path) -> None:
    exact = tmp_path / "exact.wav"
    soundfile.write(str(exact), FIXTURE_SAMPLES, 16000, subtype="PCM_16")
    pcm, record = normalize_audio(exact)
    assert pcm == FIXTURE_PCM
    assert record["pcm16"] == "exact" and record["resample"] is None

    stereo = tmp_path / "stereo.wav"
    left = np.full(15360, 0.5)
    soundfile.write(
        str(stereo), np.stack([left, -left / 2], axis=1), 24000, subtype="FLOAT"
    )
    pcm, record = normalize_audio(stereo)
    samples = np.frombuffer(pcm, dtype="<i2")
    assert len(samples) == 10240
    assert record["downmix"] == "mean of 2 channels"
    assert record["resample"] == {
        "method": "scipy.signal.resample_poly",
        "up": 2,
        "down": 3,
    }
    assert record["source_frames"] == 15360 and record["frames"] == 10240
    assert np.median(samples) == pytest.approx(0.125 * 32768, abs=2)


def write_trace(path: Path, records: list[dict], tail: str = "") -> None:
    path.write_text("".join(json.dumps(r) + "\n" for r in records) + tail)


def audio_delta(time_s: float, samples: int, value: int) -> dict:
    return {
        "direction": "receive",
        "time_s": time_s,
        "event": {
            "type": "response.output_audio.delta",
            "response_id": "response_0",
            "delta": base64.b64encode(
                np.full(samples, value, "<i2").tobytes()
            ).decode(),
        },
    }


def append(time_s: float, seq: int) -> dict:
    return {
        "direction": "send",
        "time_s": time_s,
        "event": {
            "type": "input_audio_buffer.append",
            "sglang": {"seq": seq, "t_start_ms": seq * PACKET_MS},
        },
    }


APPEND = append(10.0, 0)
# Note (wenyao): The oracle does not gate pacing, so a late append can still pass it.
LATE_APPEND_S = 0.2


def test_playout_keeps_initial_delay_and_mid_stream_gap(tmp_path: Path) -> None:
    write_trace(
        tmp_path / "continuous.jsonl",
        [
            {
                "direction": "receive",
                "time_s": 9.0,
                "event": {"type": "session.updated"},
            },
            APPEND,
            append(10.09, 1),
            audio_delta(10.5, 2205, 1),
            audio_delta(10.55, 2205, 2),
            {
                "direction": "receive",
                "time_s": 10.6,
                "event": {
                    "type": "response.output_audio_transcript.delta",
                    "response_id": "response_0",
                    "delta": "hi",
                },
            },
            audio_delta(11.5, 1000, 3),
        ],
    )

    summary = reconstruct_output(tmp_path)

    playout = json.loads((tmp_path / "playout.json").read_text())
    assert summary["errors"] == []
    assert playout["kind"] == "simulated_zero_buffer_client_playout"
    assert all("epoch" not in chunk for chunk in playout["chunks"])
    assert [c["receive_sample"] for c in playout["chunks"]] == [11025, 12128, 33075]
    assert [c["media_start_sample"] for c in playout["chunks"]] == [0, 2205, 4410]
    assert [c["playout_start_sample"] for c in playout["chunks"]] == [
        11025,
        13230,
        33075,
    ]
    assert summary["initial_delay_s"] == pytest.approx(0.5)
    media, rate = soundfile.read(str(tmp_path / "output-media.wav"), dtype="int16")
    played, _ = soundfile.read(str(tmp_path / "output-playout.wav"), dtype="int16")
    assert rate == 22050 and len(media) == 5410
    assert len(played) == 33075 + 1000
    assert not played[:11025].any() and not played[15435:33075].any()
    assert (played[13230:15435] == 2).all() and (played[33075:] == 3).all()
    timing = playout["input_timing"]
    assert timing["tolerance_s"] == PACKET_MS / 1000 and timing["within_tolerance"]
    assert [a["trace_line"] for a in timing["appends"]] == [2, 3]
    assert [a["source_start_s"] for a in timing["appends"]] == [0.0, 0.08]
    assert timing["appends"][1]["send_deviation_s"] == pytest.approx(0.01)
    assert timing["max_abs_deviation_s"] == pytest.approx(0.01)
    transcript = json.loads((tmp_path / "transcript.json").read_text())
    assert transcript["independent_asr"] is False
    assert transcript["text"] == {"response.output_audio_transcript.delta": "hi"}
    assert transcript["events"][0]["receipt_s"] == pytest.approx(0.6)


def test_truncated_or_silent_traces_are_reconstruction_failures(tmp_path: Path) -> None:
    truncated = audio_delta(10.5, 1, 1)
    truncated["event"]["delta"] = base64.b64encode(b"\x01\x00\x02").decode()
    write_trace(
        tmp_path / "continuous.jsonl", [APPEND, truncated], tail='{"direction": '
    )
    errors = reconstruct_output(tmp_path)["errors"]
    assert "trace line 2: truncated PCM16 audio delta" in errors
    assert any(error.startswith("trace line 3 unreadable") for error in errors)
    assert "no model output audio" in errors
    assert not (tmp_path / "output-media.wav").exists()


async def serve_pairs(handler, dataset: Path, output: Path, **kwargs) -> dict:
    async with websockets.serve(handler, "127.0.0.1", 0) as server:
        port = server.sockets[0].getsockname()[1]
        return await run_pairs(
            dataset,
            url=f"ws://127.0.0.1:{port}/v1/realtime",
            output=output,
            server=SERVER,
            dataset_revision="fixture",
            timeout_s=5.0,
            **kwargs,
        )


def test_run_pairs_records_both_variants_over_a_real_websocket(tmp_path: Path) -> None:
    dataset, output = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    stereo = dataset / "user_backchannel" / "1" / "clean_input.wav"
    left = np.full(15360, 0.25)
    soundfile.write(str(stereo), np.stack([left, left], axis=1), 24000, subtype="FLOAT")
    (dataset / "talking_to_other" / "1" / "metadata.json").write_text("[]")
    peers: list[DuplexPeer] = []

    async def handler(websocket: ServerConnection) -> None:
        peers.append(DuplexPeer())
        await peers[-1].handler(websocket)

    result = asyncio.run(serve_pairs(handler, dataset, output))

    assert result["status"] == "complete"
    assert json.loads((output / "run.json").read_text()) == result
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["dataset"]["inventory"]["observed"] == {s: 1 for s in SUBSETS}
    assert {
        "benchmarks/duplex/v15_audio.py",
        "benchmarks/duplex/v15_runner.py",
    } <= set(manifest["source"]["files_sha256"])
    summary = result["summary"]
    assert summary["samples_selected"] == 4 and summary["variants_selected"] == 8
    assert summary["variants"]["pass"] == 6 and summary["variants"]["invalid"] == 2
    assert summary["pairs_qualified"] == 3
    assert len(peers) == 6
    assert not [e for p in peers for e in p.received if e["type"] == "response.cancel"]

    samples = {s["id"]: s for s in result["samples"]}
    invalid = samples["talking_to_other/1"]
    assert invalid["errors"] == ["metadata.json must hold a JSON object"]
    assert {v["status"] for v in invalid["variants"].values()} == {"invalid"}
    backchannel = samples["user_backchannel/1"]
    assert backchannel["metadata"]["backchannel_text"] == "yeah"
    clean = backchannel["variants"]["clean"]
    assert clean["normalization"]["downmix"] == "mean of 2 channels"
    assert clean["source"]["file"] == "user_backchannel/1/clean_input.wav"
    variant_dir = output / clean["directory"]
    sent_pcm = (variant_dir / "input.pcm").read_bytes()
    assert bytes(peers[3].pcm) == sent_pcm
    assert (
        soundfile.read(str(variant_dir / "input.wav"), dtype="int16")[0].tobytes()
        == sent_pcm
    )
    for name in (
        "continuous.jsonl",
        "report.json",
        "output-media.wav",
        "output-playout.wav",
    ):
        assert clean["files"][name] == f"{clean['directory']}/{name}"
    assert clean["protocol_verdict"] == "pass" and clean["qualified"] is True
    media = soundfile.read(str(variant_dir / "output-media.wav"), dtype="int16")[0]
    emitted = sum(
        len(base64.b64decode(e["delta"])) // 2
        for e in peers[3].sent
        if e["type"] == "response.output_audio.delta"
    )
    assert len(media) == emitted
    assert clean["output"]["playout_duration_s"] >= clean["output"]["media_duration_s"]
    assert clean["input_timing"]["within_tolerance"] is True
    assert clean["input_timing"]["max_abs_deviation_s"] <= PACKET_MS / 1000


def test_run_pairs_attempts_the_clean_variant_after_overlap_failures(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, output = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    connections = itertools.count()
    real_run_session = v15_runner.run_session
    calls = itertools.count()

    async def flaky_run_session(*args, **kwargs) -> None:
        if next(calls) == 2:
            raise RuntimeError("recorder crashed")
        await real_run_session(*args, **kwargs)

    async def handler(websocket: ServerConnection) -> None:
        mode = "disconnect" if next(connections) == 0 else "healthy"
        await DuplexPeer(mode).handler(websocket)

    monkeypatch.setattr(v15_runner, "run_session", flaky_run_session)
    result = asyncio.run(
        serve_pairs(
            handler,
            dataset,
            output,
            sample_ids=["user_interruption/1", "user_backchannel/1"],
        )
    )

    first, second = result["samples"]
    assert first["variants"]["overlap"]["status"] == "fail"
    assert first["variants"]["overlap"]["protocol_verdict"] == "fail"
    assert first["variants"]["overlap"]["qualified"] is False
    assert first["variants"]["clean"]["status"] == "pass"
    assert second["variants"]["overlap"]["status"] == "error"
    assert second["variants"]["overlap"]["errors"] == ["RuntimeError: recorder crashed"]
    assert second["variants"]["clean"]["status"] == "pass"
    assert result["summary"]["variants_selected"] == 4
    assert result["summary"]["pairs_qualified"] == 0
    assert (
        (output / first["variants"]["overlap"]["directory"] / "continuous.jsonl")
        .stat()
        .st_size
    )


def test_run_pairs_disqualifies_a_protocol_pass_with_a_late_append(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, output = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    real_run_session = v15_runner.run_session

    async def late_run_session(url: str, pcm: bytes, *, trace_path: Path, **kwargs):
        await real_run_session(url, pcm, trace_path=trace_path, **kwargs)
        records = [json.loads(line) for line in trace_path.read_text().splitlines()]
        late = [
            index
            for index, record in enumerate(records)
            if record["event"].get("type") == "input_audio_buffer.append"
        ][3]
        for record in records[late:]:
            record["time_s"] += LATE_APPEND_S
        write_trace(trace_path, records)

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer().handler(websocket)

    monkeypatch.setattr(v15_runner, "run_session", late_run_session)
    result = asyncio.run(
        serve_pairs(handler, dataset, output, sample_ids=["user_interruption/1"])
    )

    for state in result["samples"][0]["variants"].values():
        assert state["protocol_verdict"] == "pass"
        assert state["status"] == "fail" and state["qualified"] is False
        assert state["input_timing"]["within_tolerance"] is False
        assert state["input_timing"]["max_abs_deviation_s"] >= LATE_APPEND_S
        assert any("input pacing deviation" in error for error in state["errors"])
        timing = json.loads((output / state["directory"] / "playout.json").read_text())[
            "input_timing"
        ]
        deviations = [a["send_deviation_s"] for a in timing["appends"]]
        assert max(abs(d) for d in deviations[:3]) <= PACKET_MS / 1000
        assert deviations[3] >= LATE_APPEND_S
    assert result["summary"]["pairs_qualified"] == 0


def test_run_pairs_contains_a_preparation_failure_to_its_variant(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, output = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    real_normalize = v15_runner.normalize_audio
    broken = (dataset / "user_interruption" / "1" / "input.wav").resolve()

    def flaky_normalize(path: Path):
        if path == broken:
            raise soundfile.LibsndfileError(0, "decoder failed: ")
        return real_normalize(path)

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer().handler(websocket)

    monkeypatch.setattr(v15_runner, "normalize_audio", flaky_normalize)
    result = asyncio.run(serve_pairs(handler, dataset, output))

    manifest = json.loads((output / "manifest.json").read_text())
    assert [s["id"] for s in manifest["samples"]] == [f"{s}/1" for s in SUBSETS]
    failed = result["samples"][0]["variants"]["overlap"]
    assert failed["status"] == "error" and failed["qualified"] is False
    assert failed["errors"][0].startswith("preparation failed: LibsndfileError")
    assert failed["source"]["sha256"] == result["samples"][0]["sha256"]["input"]
    assert result["summary"]["variants_selected"] == 8
    assert result["summary"]["variants"]["pass"] == 7
    assert result["samples"][0]["variants"]["clean"]["status"] == "pass"


def test_run_pairs_accounts_a_nonfinite_clean_input_and_runs_its_pair(
    tmp_path: Path,
) -> None:
    dataset, output = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    samples = FIXTURE_SAMPLES / 32768
    samples[100] = np.nan
    soundfile.write(
        str(dataset / "user_interruption" / "1" / "clean_input.wav"),
        samples,
        16000,
        subtype="FLOAT",
    )

    async def handler(websocket: ServerConnection) -> None:
        await DuplexPeer().handler(websocket)

    result = asyncio.run(
        serve_pairs(handler, dataset, output, sample_ids=["user_interruption/1"])
    )

    variants = result["samples"][0]["variants"]
    assert variants["overlap"]["status"] == "pass"
    assert variants["clean"]["status"] == "error"
    assert variants["clean"]["qualified"] is False
    assert "non-finite" in variants["clean"]["errors"][0]
    assert not (output / variants["clean"]["directory"] / "input.pcm").exists()
    assert result["summary"]["variants_selected"] == 2


def test_run_pairs_accounts_inputs_the_deadline_cannot_meet(tmp_path: Path) -> None:
    dataset, output = tmp_path / "data", tmp_path / "run"
    write_dataset(dataset)
    result = asyncio.run(
        run_pairs(
            dataset,
            url="ws://127.0.0.1:1/v1/realtime",
            output=output,
            server=SERVER,
            dataset_revision="fixture",
            timeout_s=0.5,
            max_per_subset=1,
        )
    )
    assert result["summary"]["variants"]["invalid"] == 8
    assert "not below timeout" in result["samples"][0]["variants"]["clean"]["errors"][0]
    assert (output / "manifest.json").is_file()
    with pytest.raises(ValueError, match="unknown"):
        asyncio.run(
            run_pairs(
                dataset,
                url="ws://127.0.0.1:1/v1/realtime",
                output=tmp_path / "never",
                server=SERVER,
                dataset_revision="fixture",
                timeout_s=5.0,
                sample_ids=["user_interruption/404"],
            )
        )
    assert not (tmp_path / "never").exists()
