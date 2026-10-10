"""Local protocol verification without a model or GPU."""

import argparse
import asyncio
import base64
import json
import socket
import sys
import wave
from pathlib import Path
from typing import Literal

import pytest
from aiohttp import web
from websockets.asyncio.server import ServerConnection, serve

from benchmarks.eval.benchmark_asr_seedtts import main as run_http
from benchmarks.eval.benchmark_nemotron_native import run
from benchmarks.eval.nemotron_native_client import measure


@pytest.mark.parametrize(
    "fault", ["none", "duplicate", "duration", "error", "empty-reference"]
)
def test_wire_contract(
    fault: Literal["none", "duplicate", "duration", "error", "empty-reference"],
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        sequences: list[int] = []

        async def server(socket: ServerConnection) -> None:
            await socket.send(json.dumps({"type": "session.created"}))
            configuration = json.loads(await socket.recv())
            assert configuration["session"] == {"output_modalities": ["text"]}
            await socket.send(json.dumps({"type": "session.updated"}))
            audio = bytearray()
            async for raw in socket:
                event = json.loads(raw)
                if event["type"] == "input_audio_buffer.append":
                    sequences.append(event["sglang"]["seq"])
                    audio.extend(base64.b64decode(event["audio"]))
                    if event["sglang"]["seq"] == 0:
                        await socket.send(
                            json.dumps(
                                {
                                    "type": "response.output_text.delta",
                                    "response_id": "r1",
                                    "delta": "hello",
                                }
                            )
                        )
                    else:
                        pass
                elif event["type"] == "sglang.input_audio.end":
                    if fault == "error":
                        await socket.send(
                            json.dumps({"type": "error", "error": "injected"})
                        )
                        return
                    else:
                        final = {
                            "type": "response.output_text.done",
                            "response_id": "r1",
                            "text": "hello",
                        }
                        await socket.send(json.dumps(final))
                    if fault == "duplicate":
                        await socket.send(json.dumps(final))
                    else:
                        pass
                    await socket.send(
                        json.dumps(
                            {
                                "type": "sglang.input_audio.drained",
                                "consumed_ms": (
                                    0 if fault == "duration" else len(audio) / 32
                                ),
                            }
                        )
                    )
                elif event["type"] == "session.close":
                    await socket.send(json.dumps({"type": "session.closed"}))
                    return
                else:
                    raise AssertionError(event)

        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            result = await measure(
                f"ws://127.0.0.1:{port}", b"\0\0" * 1000, "test", "hello", paced=True
            )
        assert sequences == [0, 1, 2, 3]
        if fault in {"none", "empty-reference"}:
            assert result.error is None
            assert result.text == "hello"
            assert (
                result.eos_to_final_seconds is not None
                and result.eos_to_final_seconds >= 0
            )
            assert result.wall_seconds >= result.audio_seconds
            assert json.loads(result.events[-1]["raw_json"])["type"] == "session.closed"
        else:
            assert result.error is not None
        sequences.clear()
        with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 1000)
        meta = tmp_path / "meta.lst"
        reference = "Mhm." if fault == "empty-reference" else "hello"
        meta.write_text(f"sample|{reference}|audio.wav|{reference}\n")
        output = tmp_path / "results"
        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            arguments = argparse.Namespace(
                meta=str(meta),
                dataset_revision=None,
                max_samples=1,
                score_language="en",
                output=output,
                url=f"ws://127.0.0.1:{port}",
                model="fake",
                concurrencies=[1, 4],
                repeats=1,
                warmup=fault == "empty-reference",
                packet_milliseconds=20,
                burst=True,
                timeout_seconds=1,
            )
            if fault in {"none", "empty-reference"}:
                await run(arguments)
            else:
                with pytest.raises(RuntimeError, match="measured requests failed"):
                    await run(arguments)
        summary = json.loads((output / "summary.json").read_text())[0]
        raw = json.loads((output / "c1-r1.jsonl").read_text())
        if fault == "none":
            assert summary["success"] == 1 and summary["wer_percent"] == 0
            assert raw["score"]["hyp_norm"] == "hello"
        elif fault == "empty-reference":
            assert summary["success"] == 1
            assert summary["wer_evaluated"] == 0 and summary["wer_skipped"] == 1
            assert summary["wer_percent"] is None
            assert raw["error"] is None
            assert raw["score"]["error"] == "Empty reference after normalization"
        else:
            assert summary["success"] == 0 and summary["wer_percent"] is None
            assert raw["error"] and not raw["score"]["is_success"]

    asyncio.run(scenario())


def test_timeout_retains_received_events() -> None:
    async def scenario() -> None:
        async def server(socket: ServerConnection) -> None:
            await socket.send(json.dumps({"type": "session.created"}))
            await socket.wait_closed()

        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            result = await measure(
                f"ws://127.0.0.1:{port}", b"", "timeout", "", timeout_seconds=0.05
            )
        assert result.error.startswith("TimeoutError")
        assert json.loads(result.events[0]["raw_json"])["type"] == "session.created"

    asyncio.run(scenario())


@pytest.mark.parametrize("request_language", [None, "auto"])
def test_http_separates_request_language_and_scoring(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, request_language: str | None
) -> None:
    with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(b"\0\0" * 1000)
    metadata = tmp_path / "meta.lst"
    metadata.write_text("sample|hello|audio.wav|hello\n", encoding="utf-8")
    output = tmp_path / "results.json"
    expected_language = "en" if request_language is None else request_language

    async def scenario() -> None:
        languages: list[str] = []

        async def transcribe(request: web.Request) -> web.Response:
            form = await request.post()
            assert form["language"] == expected_language
            languages.append(str(form["language"]))
            return web.json_response({"text": "hello"})

        application = web.Application()
        application.router.add_post("/v1/audio/transcriptions", transcribe)
        runner = web.AppRunner(application)
        await runner.setup()
        try:
            with socket.socket() as server_socket:
                server_socket.bind(("127.0.0.1", 0))
                port = server_socket.getsockname()[1]
                site = web.SockSite(runner, server_socket)
                await site.start()
                monkeypatch.setattr(
                    sys,
                    "argv",
                    [
                        "benchmark_asr_seedtts",
                        "--meta",
                        str(metadata),
                        "--output",
                        str(output),
                        "--model-path",
                        "test",
                        "--port",
                        str(port),
                        "--concurrencies",
                        "1,4",
                        "--repeats",
                        "1",
                        "--disable-resource-monitor",
                        "--save-raw-dir",
                        str(tmp_path / "raw"),
                        "--warmup",
                        "--lang",
                        "en",
                    ],
                )
                if request_language is not None:
                    monkeypatch.setattr(
                        sys, "argv", [*sys.argv, "--request-language", request_language]
                    )
                else:
                    pass
                await asyncio.to_thread(run_http)
                await site.stop()
        finally:
            await runner.cleanup()
        assert languages == [expected_language] * 4

    asyncio.run(scenario())
    results = json.loads(output.read_text())
    assert results["config"]["lang"] == "en"
    assert results["config"]["request_language"] == expected_language
    assert results["config"]["warmup"] is True
    for result in results["results"]:
        assert result["total"] == result["evaluated"] == 1
        assert result["corpus_wer"]["mean"] == 0
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == 2
