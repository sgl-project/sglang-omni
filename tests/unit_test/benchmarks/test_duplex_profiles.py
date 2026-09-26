import asyncio
import base64
import copy
import itertools
import json
import types
from pathlib import Path

import pytest
import soundfile
import websockets
from websockets.asyncio.server import ServerConnection

from benchmarks.duplex.artifacts import replay_run
from benchmarks.duplex.oracle import evaluate_trace
from benchmarks.duplex.v15_audio import reconstruct_output
from benchmarks.duplex.v15_runner import run_pairs
from benchmarks.duplex.v15_transcribe import transcribe_run
from tests.unit_test.benchmarks.test_duplex_oracle import GRANTED
from tests.unit_test.benchmarks.test_duplex_v15_runner import write_dataset

MINICPMO = "minicpmo-native-pr2377"


def minicpmo_trace() -> list[dict]:
    records = []

    def add(time_s: float, direction: str, kind: str, **event: object) -> None:
        records.append(
            {
                "time_s": time_s,
                "direction": direction,
                "event": {"type": kind, "event_id": f"e{len(records)}", **event},
            }
        )

    grant = copy.deepcopy(GRANTED)
    grant.update(native_unit_ms=1000, first_unit_ms=1000)
    grant["output_audio_format"]["rate"] = 24000
    grant["output_modalities"] = ["audio", "text"]
    add(0, "receive", "session.created", session={"id": "session_A"})
    add(0.01, "send", "session.update", event_id="configure")
    add(
        0.02,
        "receive",
        "session.updated",
        client_event_id="configure",
        session={
            "id": "session_A",
            "sglang": {"granted": grant},
            "audio": {"output": {"format": {"type": "audio/pcm", "rate": 24000}}},
        },
    )
    for seq in range(51):
        add(
            0.1 + seq * 0.08,
            "send",
            "input_audio_buffer.append",
            event_id=f"input{seq}",
            audio=base64.b64encode(bytes(2560)).decode(),
            sglang={"seq": seq, "t_start_ms": seq * 80},
        )
        add(
            0.101 + seq * 0.08,
            "receive",
            "sglang.input_audio.accepted",
            seq=seq,
            accepted_end_ms=(seq + 1) * 80,
            client_event_id=f"input{seq}",
        )
    for index, (start, samples) in enumerate(((3.11, 7200), (4.12, 2400))):
        response_id = f"response{index}"
        add(
            start,
            "receive",
            "response.created",
            response={"id": response_id, "status": "in_progress"},
        )
        add(
            start + 0.01,
            "receive",
            "response.output_audio.delta",
            response_id=response_id,
            item_id=f"item{index}",
            output_index=0,
            content_index=0,
            delta=base64.b64encode(b"\x01\x00" * samples).decode(),
        )
        add(
            start + 0.02,
            "receive",
            "response.output_audio.done",
            response_id=response_id,
        )
        add(
            start + 0.03,
            "receive",
            "response.done",
            response={
                "id": response_id,
                "status": "completed",
                "status_details": {"reason": "stop"},
            },
        )
    add(4.18, "send", "sglang.input_audio.end", event_id="end")
    add(
        4.19,
        "receive",
        "sglang.input_audio.ended",
        client_event_id="end",
        accepted_end_ms=4080,
        tail_policy="pad",
    )
    for index, duration in enumerate((1000, 1000, 1000, 1000, 80)):
        add(
            min(1.11 + index, 4.20),
            "receive",
            "sglang.unit.done",
            unit_id=f"unit_{index}",
            sglang={
                "media_time": {"t_start_ms": index * 1000, "duration_ms": duration}
            },
        )
    add(
        4.21,
        "receive",
        "sglang.input_audio.drained",
        client_event_id="end",
        accepted_end_ms=4080,
        consumed_ms=4080,
        discarded_ms=0,
        padding_ms=920,
    )
    add(4.22, "send", "session.close", event_id="close")
    add(
        4.23,
        "receive",
        "session.closed",
        client_event_id="close",
        reason="client_closed",
    )
    return sorted(records, key=lambda record: record["time_s"])


def test_minicpmo_accepts_natural_turns_and_variable_audio() -> None:
    result = evaluate_trace(minicpmo_trace(), scenario="continuous", profile=MINICPMO)
    assert result["violations"] == []
    assert result["status"] == "pass"
    assert result["metrics"]["output_audio_s"] == pytest.approx(0.4)
    assert result["metrics"]["input_audio_s"] == pytest.approx(4.08)


@pytest.mark.parametrize(
    "kind",
    [
        "sglang.input_audio.accepted",
        "sglang.unit.done",
        "response.done",
        "sglang.input_audio.drained",
    ],
)
def test_minicpmo_still_rejects_missing_protocol_receipts(kind: str) -> None:
    trace = minicpmo_trace()
    trace.pop(next(i for i, r in enumerate(trace) if r["event"]["type"] == kind))
    assert (
        evaluate_trace(trace, scenario="continuous", profile=MINICPMO)["status"]
        == "fail"
    )


def test_minicpmo_is_not_accepted_as_voicechat() -> None:
    assert evaluate_trace(minicpmo_trace(), scenario="continuous")["status"] == "fail"


def test_minicpmo_silence_is_a_valid_protocol_observation() -> None:
    trace = [
        r for r in minicpmo_trace() if not r["event"]["type"].startswith("response.")
    ]
    result = evaluate_trace(trace, scenario="continuous", profile=MINICPMO)
    assert result["status"] == "pass"
    assert result["coverage"]["input_output_overlap"] is False


def test_minicpmo_reconstruction_uses_24khz(tmp_path: Path) -> None:
    (tmp_path / "continuous.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in minicpmo_trace())
    )
    result = reconstruct_output(tmp_path, profile=MINICPMO)
    audio, rate = soundfile.read(tmp_path / "output-media.wav")
    assert result["errors"] == []
    assert rate == 24000
    assert len(audio) / rate == pytest.approx(0.4)
    assert result["sample_rate"] == rate


def test_minicpmo_silence_reconstructs_the_input_window(tmp_path: Path) -> None:
    trace = [
        r for r in minicpmo_trace() if not r["event"]["type"].startswith("response.")
    ]
    (tmp_path / "continuous.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in trace)
    )
    result = reconstruct_output(tmp_path, profile=MINICPMO)
    assert result["errors"] == []
    assert soundfile.info(tmp_path / "output-media.wav").frames == 0
    audio, rate = soundfile.read(tmp_path / "output-playout.wav")
    assert rate == 24000
    assert len(audio) / rate == pytest.approx(4.08)
    assert not audio.any()


def test_minicpmo_pair_capture_and_replay(tmp_path: Path) -> None:
    async def exercise() -> None:
        async def handler(ws: ServerConnection) -> None:
            ids = itertools.count()

            async def send(kind: str, **payload: object) -> None:
                await ws.send(
                    json.dumps(
                        {"type": kind, "event_id": f"server{next(ids)}", **payload}
                    )
                )

            await send("session.created", session={"id": "mini"})
            total = 0
            async for raw in ws:
                event = json.loads(raw)
                if event["type"] == "session.update":
                    assert event["session"]["output_modalities"] == ["audio", "text"]
                    grant = copy.deepcopy(GRANTED)
                    grant.update(
                        native_unit_ms=1000,
                        first_unit_ms=1000,
                        output_modalities=["audio", "text"],
                    )
                    grant["output_audio_format"]["rate"] = 24000
                    await send(
                        "session.updated",
                        client_event_id=event["event_id"],
                        session={"id": "mini", "sglang": {"granted": grant}},
                    )
                elif event["type"] == "input_audio_buffer.append":
                    total += len(base64.b64decode(event["audio"]))
                    await send(
                        "sglang.input_audio.accepted",
                        client_event_id=event["event_id"],
                        seq=event["sglang"]["seq"],
                        accepted_end_ms=total / 32,
                    )
                elif event["type"] == "sglang.input_audio.end":
                    await send(
                        "sglang.input_audio.ended",
                        client_event_id=event["event_id"],
                        accepted_end_ms=total / 32,
                        tail_policy="pad",
                    )
                    await send(
                        "response.created",
                        response={"id": "answer", "status": "in_progress"},
                    )
                    await send(
                        "response.output_audio.delta",
                        response_id="answer",
                        item_id="item",
                        content_index=0,
                        output_index=0,
                        delta=base64.b64encode(b"\x01\x00" * 480).decode(),
                    )
                    await send("response.output_audio.done", response_id="answer")
                    await send(
                        "response.done",
                        response={
                            "id": "answer",
                            "status": "completed",
                            "status_details": {"reason": "stop"},
                        },
                    )
                    await send(
                        "sglang.unit.done",
                        unit_id="unit_0",
                        sglang={
                            "media_time": {"t_start_ms": 0, "duration_ms": total / 32}
                        },
                    )
                    await send(
                        "sglang.input_audio.drained",
                        client_event_id=event["event_id"],
                        accepted_end_ms=total / 32,
                        consumed_ms=total / 32,
                        discarded_ms=0,
                        padding_ms=1000 - total / 32,
                    )
                elif event["type"] == "session.close":
                    await send(
                        "session.closed",
                        client_event_id=event["event_id"],
                        reason="client_closed",
                    )
                    await ws.close()

        dataset = tmp_path / "dataset"
        write_dataset(dataset)
        output = tmp_path / "recording"
        async with websockets.serve(handler, "127.0.0.1", 0) as server:
            port = server.sockets[0].getsockname()[1]
            result = await run_pairs(
                dataset,
                url=f"ws://127.0.0.1:{port}",
                output=output,
                server={"revision": "a" * 40},
                dataset_revision="pinned",
                timeout_s=5,
                sample_ids=["user_interruption/1"],
                profile=MINICPMO,
            )
        assert result["summary"]["pairs_qualified"] == 1
        assert json.loads((output / "manifest.json").read_text())["profile"] == MINICPMO
        for variant in result["samples"][0]["variants"].values():
            directory = output / variant["directory"]
            assert replay_run(directory)["summary"]["passed"] == 1
            assert variant["output"]["media_duration_s"] == pytest.approx(0.02)
            assert soundfile.info(directory / "output-media.wav").samplerate == 24000
            records = [
                json.loads(line)
                for line in (directory / "continuous.jsonl").read_text().splitlines()
            ]
            appends = [
                r
                for r in records
                if r["direction"] == "send"
                and r["event"]["type"] == "input_audio_buffer.append"
            ]
            end = next(
                r
                for r in records
                if r["direction"] == "send"
                and r["event"]["type"] == "sglang.input_audio.end"
            )
            assert end["time_s"] - appends[0]["time_s"] >= 0.64
            receipts = json.loads((directory / "input-send-receipts.json").read_text())[
                "appends"
            ]
            assert [r["event_id"] for r in receipts] == [
                r["event"]["event_id"] for r in appends
            ]
            assert all(r["start_s"] <= r["completed_s"] for r in receipts)

    asyncio.run(exercise())
    samples = []

    def transcribe(audio, **options):
        samples.append(len(audio))
        return {"text": "", "segments": []}

    weights = tmp_path / "test-model.pt"
    weights.write_bytes(b"test weights")
    result = transcribe_run(
        tmp_path / "recording",
        output=tmp_path / "transcripts",
        model=types.SimpleNamespace(transcribe=transcribe),
        model_path=weights,
        device="cpu",
        timeline="media",
    )
    assert samples == [320, 320]
    assert all(v["status"] == "transcribed" for v in result["variants"])
