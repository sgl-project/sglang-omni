"""Local protocol verification without a model or GPU."""

import argparse
import asyncio
import base64
import json
import socket
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from aiohttp import web
from websockets.asyncio.server import ServerConnection, serve

from benchmarks.dataset.seedtts import SampleInput
from benchmarks.eval.benchmark_asr_realtime import (
    load_pcm16,
    run_asr_realtime_once,
    run_sweep,
)
from benchmarks.eval.benchmark_asr_seedtts import main as run_http
from benchmarks.eval.benchmark_asr_seedtts import run_asr_seedtts_once
from benchmarks.realtime_asr.client import run_session
from benchmarks.realtime_asr.metrics import check_invariants, latency_metrics
from benchmarks.realtime_asr.replay import load_trace, replay_trace

NativeFault = Literal[
    "none", "duplicate", "duration", "identity", "delta", "error", "empty-reference"
]


@dataclass(kw_only=True)
class NativeTestServer:
    fault: NativeFault = "none"
    sequences: list[int] = field(default_factory=list)
    captured_audio: list[bytes] = field(default_factory=list)

    async def handle(self, socket: ServerConnection) -> None:
        await socket.send(json.dumps({"type": "session.created"}))
        configuration = json.loads(await socket.recv())
        assert configuration["session"] == {"output_modalities": ["text"]}
        await socket.send(json.dumps({"type": "session.updated"}))
        audio = bytearray()
        async for raw in socket:
            event = json.loads(raw)
            if event["type"] == "input_audio_buffer.append":
                self.sequences.append(event["sglang"]["seq"])
                await socket.send(
                    json.dumps(
                        {
                            "type": "sglang.input_audio.accepted",
                            "seq": event["sglang"]["seq"],
                        }
                    )
                )
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
                self.captured_audio.append(bytes(audio))
                if self.fault == "error":
                    await socket.send(
                        json.dumps({"type": "error", "error": "injected"})
                    )
                    return
                else:
                    final = {
                        "type": "response.output_text.done",
                        "response_id": "wrong" if self.fault == "identity" else "r1",
                        "text": "wrong" if self.fault == "delta" else "hello",
                    }
                    await socket.send(json.dumps(final))
                if self.fault == "duplicate":
                    await socket.send(json.dumps(final))
                else:
                    pass
                await socket.send(
                    json.dumps(
                        {
                            "type": "response.done",
                            "response": {"status": "completed"},
                        }
                    )
                )
                await socket.send(
                    json.dumps(
                        {
                            "type": "sglang.input_audio.drained",
                            "consumed_ms": (
                                0 if self.fault == "duration" else len(audio) / 32
                            ),
                        }
                    )
                )
            elif event["type"] == "session.close":
                await socket.send(json.dumps({"type": "session.closed"}))
                return
            else:
                raise AssertionError(event)


@pytest.mark.parametrize(
    "fault",
    ["none", "duplicate", "duration", "identity", "delta", "error", "empty-reference"],
)
def test_wire_contract(
    fault: NativeFault,
    tmp_path: Path,
) -> None:
    async def scenario() -> None:
        server = NativeTestServer(fault=fault)

        async with serve(server.handle, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            trace = await run_session(
                f"ws://127.0.0.1:{port}",
                b"\0\0" * 1000,
                protocol="native",
                packet_ms=20,
                paced=True,
            )
        assert server.sequences == [0, 1, 2, 3]
        if fault in {"none", "empty-reference"}:
            assert check_invariants(trace) == []
            assert trace.completed_text == "hello"
            metrics = latency_metrics(trace)
            assert metrics["final_latency_s"][0] >= 0
            assert metrics["first_text_s"] is not None
            assert trace.done_sent_s - trace.first_send_s >= trace.audio_duration_s
            assert trace.received[-1].type == "session.closed"
        else:
            assert check_invariants(trace)
        server.sequences.clear()
        with wave.open(str(tmp_path / "audio.wav"), "wb") as audio:
            audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
            audio.writeframes(b"\0\0" * 1000)
        reference = "Mhm." if fault == "empty-reference" else "hello"
        output = tmp_path / "results"
        async with serve(server.handle, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            result = await run_asr_realtime_once(
                [
                    SampleInput(
                        "sample", reference, str(tmp_path / "audio.wav"), reference
                    )
                ],
                host="127.0.0.1",
                port=port,
                protocol="native",
                packet_ms=20,
                paced=False,
                timeout_s=1,
                trace_dir=str(output),
            )
        summary = result["summary"]
        record = result["per_sample"][0]
        saved = load_trace(Path(record["trace_file"]))
        assert saved.trace.protocol == "native"
        assert replay_trace(saved)["metrics"] == latency_metrics(saved.trace)
        if fault == "none":
            assert summary["success"] == summary["evaluated"] == 1
            assert summary["corpus_wer"] == 0
            assert replay_trace(saved)["verdict"] == "pass"
        elif fault == "empty-reference":
            assert summary["success"] == 1 and summary["evaluated"] == 0
            assert summary["corpus_wer"] is None
            assert record["wer_error"] == "Empty reference after normalization"
        else:
            assert summary["success"] == 0 and summary["corpus_wer"] is None
            assert summary["first_text_s"] is None
            assert replay_trace(saved)["verdict"] == "fail"

    asyncio.run(scenario())


def test_timeout_retains_received_events() -> None:
    async def scenario() -> None:
        async def server(socket: ServerConnection) -> None:
            await socket.send(json.dumps({"type": "session.created"}))
            await socket.wait_closed()

        async with serve(server, "127.0.0.1", 0) as listener:
            port = listener.sockets[0].getsockname()[1]
            result = await run_session(
                f"ws://127.0.0.1:{port}", b"", protocol="native", timeout_s=0.05
            )
        assert result.error.startswith("timeout")
        assert result.received[0].type == "session.created"

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
                if request_language is None:
                    legacy_result = await asyncio.to_thread(
                        asyncio.run,
                        run_asr_seedtts_once(
                            [
                                SampleInput(
                                    "sample", "hello", str(tmp_path / "audio.wav"), ""
                                )
                            ],
                            host="127.0.0.1",
                            port=port,
                            concurrency=1,
                            model_path="test",
                            lang="en",
                        ),
                    )
                    assert legacy_result["summary"]["evaluated"] == 1
                else:
                    pass
                await site.stop()
        finally:
            await runner.cleanup()
        assert languages == [expected_language] * (5 if request_language is None else 4)

    asyncio.run(scenario())
    results = json.loads(output.read_text())
    assert results["config"]["lang"] == "en"
    assert results["config"]["request_language"] == expected_language
    assert results["config"]["warmup"] is True
    for result in results["results"]:
        assert result["total"] == result["evaluated"] == 1
        assert result["corpus_wer"]["mean"] == 0
    assert len(list((tmp_path / "raw").glob("*.jsonl"))) == 2


@pytest.mark.parametrize("fail_warmup", [False, True])
def test_native_sweep_preserves_pcm_and_excludes_warmup(
    tmp_path: Path, fail_warmup: bool
) -> None:
    pcm = b"\x00\x80\xff\x7f" * 160
    audio_path = tmp_path / "audio.wav"
    with wave.open(str(audio_path), "wb") as audio:
        audio.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        audio.writeframes(pcm)
    assert load_pcm16(str(audio_path), preserve_pcm=True) == pcm
    with patch(
        "benchmarks.eval.benchmark_asr_realtime._load_wav_mono_16k",
        return_value=torch.tensor([-1.0, 32767 / 32768] * 160),
    ):
        assert load_pcm16(str(audio_path)) == b"\x01\x80\xfe\x7f" * 160
    server = NativeTestServer(fault="error" if fail_warmup else "none")

    async def scenario() -> None:
        async with serve(server.handle, "127.0.0.1", 0) as listener:
            arguments = argparse.Namespace(
                host="127.0.0.1",
                port=listener.sockets[0].getsockname()[1],
                protocol="native",
                mode=None,
                lang="en",
                model_path="test",
                packet_ms=20,
                no_pace=True,
                trailing_silence_ms=1000,
                timeout_s=2,
                http_baseline=False,
                concurrencies=[1, 4],
                repeats=2,
                warmup=True,
                output=str(tmp_path / "results.json"),
                trace_dir=str(tmp_path / "traces"),
            )
            samples = [SampleInput("sample", "hello", str(audio_path), "")]
            if fail_warmup:
                with pytest.raises(RuntimeError, match="Warmup failed"):
                    await run_sweep(arguments, samples)
                saved = load_trace(next((tmp_path / "traces").glob("*/000000.json")))
                assert replay_trace(saved)["verdict"] == "fail"
                assert len(server.captured_audio) == 1
                return
            else:
                results = await run_sweep(arguments, samples)
        assert len(results) == 4
        assert [result["config"]["repeat"] for result in results] == [1, 2, 1, 2]
        assert all(not result["config"]["warmup"] for result in results)
        assert all(result["summary"]["success"] == 1 for result in results)
        assert server.captured_audio == [pcm] * 6
        manifests = [
            json.loads(path.read_text())
            for path in (tmp_path / "traces").glob("*/manifest.json")
        ]
        assert sum(manifest["config"]["warmup"] for manifest in manifests) == 2

    asyncio.run(scenario())
