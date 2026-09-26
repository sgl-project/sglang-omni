# SPDX-License-Identifier: Apache-2.0
"""Run native VoiceChat sessions, or expose them on the shared WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Protocol

import numpy as np
import uvicorn
from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from sglang_omni.client.client import Client
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    NemotronVoiceChatDuplexPipelineConfig,
)
from sglang_omni.models.nemotron_voicechat.payload_types import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from sglang_omni.models.nemotron_voicechat.realtime import deployment
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto.session import SessionLimits
from sglang_omni.serve.openai_api import create_app
from sglang_omni.serve.realtime.control import Drained, Failure
from sglang_omni.serve.realtime.manager import RealtimeDeployment
from sglang_omni.serve.realtime.output import AudioDelta, TextDelta
from sglang_omni.serve.realtime.runtime import SessionRuntime
from sglang_omni.utils.audio import load_audio

WARMUP_TIMEOUT_S = 120
INPUT_FRAME_SAMPLES = 1280
OUTPUT_FRAME_SAMPLES = 1764
PCM16_SAMPLE_BYTES = 2
INPUT_FRAME_BYTES = INPUT_FRAME_SAMPLES * PCM16_SAMPLE_BYTES
FRAME_DURATION_S = INPUT_FRAME_SAMPLES / INPUT_SAMPLE_RATE
UNPACED_YIELD_S = 0.01


class WorkerStop(Protocol):
    async def __call__(self) -> None: ...


def mount_example_ui(app: FastAPI) -> None:
    """Keep the demo and its WebSocket on the same origin (no CORS setup)."""
    assets = Path(__file__).resolve().parent / "voicechat_ui"
    app.mount("/voicechat-assets", StaticFiles(directory=assets), name="voicechat-ui")

    @app.get("/", include_in_schema=False)
    async def voicechat_ui() -> FileResponse:
        return FileResponse(
            assets / "index.html", headers={"Cache-Control": "no-store"}
        )


def close_workers_on_shutdown(app: FastAPI, stop: WorkerStop) -> None:
    """Close GPU workers before Uvicorn re-raises a termination signal."""
    lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def managed_lifespan(app: FastAPI) -> AsyncIterator[None]:
        try:
            async with lifespan(app):
                yield
        finally:
            await stop()

    app.router.lifespan_context = managed_lifespan


async def warmup_realtime(realtime_deployment: RealtimeDeployment) -> None:
    """Warm both first-frame and continuation kernels before accepting a mic."""
    runtime = SessionRuntime(
        "nemotron-voicechat",
        realtime_deployment.capabilities,
        realtime_deployment.adapter_factory,
        realtime_deployment.limits,
    )

    async def consume() -> None:
        async for envelope in runtime.outputs():
            if isinstance(envelope.event, Failure):
                raise RuntimeError(envelope.event.message)  # noqa: TRY004
            else:
                pass
            if isinstance(envelope.event, Drained):
                return
            else:
                pass
        raise RuntimeError("warmup ended without a drain receipt")

    task = asyncio.create_task(consume())
    try:
        await runtime.update({"output_modalities": ["audio"]}, "warmup-configure")
        await runtime.append(bytes(INPUT_FRAME_BYTES * 2), 0, None, "warmup-audio")
        await runtime.end("warmup-end")
        await asyncio.wait_for(task, WARMUP_TIMEOUT_S)
    finally:
        await runtime.close("warmup_finished")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def run(arguments: argparse.Namespace) -> None:
    config = NemotronVoiceChatDuplexPipelineConfig(model_path=arguments.model_path)
    if arguments.talker_attention:
        config.stages[2].engine.attention_backend = arguments.talker_attention
    else:
        pass
    runner = MultiProcessPipelineRunner(config)
    await runner.start(timeout=900)
    try:
        client = Client(runner.coordinator)
        if not arguments.no_warmup:
            print("Warming up VoiceChat kernels...", flush=True)
            await warmup_realtime(
                deployment(
                    client,
                    session_limits=SessionLimits(operation_timeout_s=WARMUP_TIMEOUT_S),
                )
            )
        else:
            pass
        realtime_deployment = deployment(client)
        if arguments.serve:
            app = create_app(
                client,
                model_name="nemotron-voicechat",
                realtime_deployment=realtime_deployment,
            )
            close_workers_on_shutdown(app, runner.stop)
            mount_example_ui(app)
            print(f"VoiceChat UI: http://localhost:{arguments.port}", flush=True)
            await uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=arguments.port)
            ).serve()
            return
        else:
            pass
        if not arguments.audio:
            raise ValueError("--audio is required unless --serve is set")
        else:
            pass
        audio = load_audio(
            arguments.audio,
            source_name="VoiceChat",
            target_sample_rate=INPUT_SAMPLE_RATE,
            mono=False,
        )[0]
        if arguments.seconds:
            audio = audio[: int(arguments.seconds * INPUT_SAMPLE_RATE)]
        else:
            pass
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        runtime = SessionRuntime(
            "nemotron-voicechat",
            realtime_deployment.capabilities,
            realtime_deployment.adapter_factory,
            realtime_deployment.limits,
        )
        audio_packets: list[bytes] = []
        text_deltas: list[str] = []
        packet_receipt_times: list[float] = []
        started = time.perf_counter()

        async def consume() -> None:
            async for envelope in runtime.outputs():
                event = envelope.event
                if isinstance(event, AudioDelta):
                    audio_packets.append(event.pcm)
                    packet_receipt_times.append(time.perf_counter() - started)
                elif isinstance(event, TextDelta):
                    text_deltas.append(event.text)
                elif isinstance(event, Failure):
                    raise RuntimeError(str(event))  # noqa: TRY004
                elif isinstance(event, Drained):
                    break
                else:
                    pass

        task = asyncio.create_task(consume())
        try:
            await runtime.update({"output_modalities": ["audio", "text"]}, "configure")
            for offset in range(0, len(pcm), INPUT_FRAME_BYTES):
                await runtime.append(
                    pcm[offset : offset + INPUT_FRAME_BYTES],
                    offset // INPUT_FRAME_BYTES,
                    None,
                    str(offset),
                )
                if arguments.paced:
                    await asyncio.sleep(FRAME_DURATION_S)
                else:
                    await asyncio.sleep(UNPACED_YIELD_S)
            await runtime.end("eos")
            await asyncio.wait_for(task, 120)
        finally:
            await runtime.close("example_finished")
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            else:
                pass
        expected_samples = (
            (len(pcm) // PCM16_SAMPLE_BYTES + INPUT_FRAME_SAMPLES - 1)
            // INPUT_FRAME_SAMPLES
        ) * OUTPUT_FRAME_SAMPLES
        actual_samples = sum(map(len, audio_packets)) // 2
        if actual_samples != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} audio samples, received {actual_samples}"
            )
        else:
            pass
        with wave.open(arguments.out, "wb") as output_wave:
            output_wave.setnchannels(1)
            output_wave.setsampwidth(2)
            output_wave.setframerate(OUTPUT_SAMPLE_RATE)
            output_wave.writeframes(b"".join(audio_packets))
        report = {
            "text": "".join(text_deltas),
            "audio_samples": sum(map(len, audio_packets)) // 2,
            "input_samples": len(pcm) // 2,
            "first_audio_s": packet_receipt_times[0] if packet_receipt_times else None,
            "elapsed_s": time.perf_counter() - started,
        }
        print(json.dumps(report, indent=2), flush=True)
        Path(arguments.out + ".json").write_text(json.dumps(report, indent=2))
    finally:
        await runner.stop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio")
    parser.add_argument("--out", default="voicechat-duplex.wav")
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--paced", action="store_true")
    parser.add_argument(
        "--serve",
        action="store_true",
        help="serve the microphone UI and WebSocket on localhost",
    )
    parser.add_argument(
        "--no-warmup",
        action="store_true",
        help="skip startup warmup (requires populated kernel caches)",
    )
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--talker-attention", choices=["triton", "torch_native"])
    asyncio.run(run(parser.parse_args()))
else:
    pass
