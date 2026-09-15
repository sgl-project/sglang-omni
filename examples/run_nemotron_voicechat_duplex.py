# SPDX-License-Identifier: Apache-2.0
"""Run native VoiceChat sessions, or expose them on the shared WebSocket API."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
import wave
from pathlib import Path


def mount_example_ui(app):
    """Keep the demo and its WebSocket on the same origin (no CORS setup)."""
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles

    assets = Path(__file__).resolve().parent / "voicechat_ui"
    app.mount("/voicechat-assets", StaticFiles(directory=assets), name="voicechat-ui")

    @app.get("/", include_in_schema=False)
    async def voicechat_ui():
        return FileResponse(
            assets / "index.html", headers={"Cache-Control": "no-store"}
        )


async def warmup_realtime(dep):
    """Warm both first-frame and continuation kernels before accepting a mic."""
    from sglang_omni.serve.realtime.control import Drained, Failure
    from sglang_omni.serve.realtime.runtime import SessionRuntime

    runtime = SessionRuntime(
        "nemotron-voicechat", dep.capabilities, dep.adapter_factory, dep.limits
    )

    async def consume():
        async for envelope in runtime.outputs():
            if isinstance(envelope.event, Failure):
                raise RuntimeError(envelope.event.message)  # noqa: TRY004
            if isinstance(envelope.event, Drained):
                return
        raise RuntimeError("warmup ended without a drain receipt")

    task = asyncio.create_task(consume())
    try:
        await runtime.update({"output_modalities": ["audio"]}, "warmup-configure")
        await runtime.append(b"\0\0" * 2560, 0, None, "warmup-audio")
        await runtime.end("warmup-end")
        await asyncio.wait_for(task, 90)
    finally:
        await runtime.close("warmup_finished")
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def run(args):
    import numpy as np

    from sglang_omni.client import Client
    from sglang_omni.models.nemotron_voicechat.duplex_config import (
        NemotronVoiceChatDuplexPipelineConfig,
    )
    from sglang_omni.models.nemotron_voicechat.realtime import deployment
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.serve.realtime.control import Drained, Failure
    from sglang_omni.serve.realtime.output import AudioDelta, TextDelta
    from sglang_omni.serve.realtime.runtime import SessionRuntime
    from sglang_omni.utils.audio import load_audio

    config = NemotronVoiceChatDuplexPipelineConfig(model_path=args.model_path)
    if args.talker_attention:
        config.stages[2].engine.attention_backend = args.talker_attention
    runner = MultiProcessPipelineRunner(config)
    await runner.start(timeout=900)
    try:
        client = Client(runner.coordinator)
        dep = deployment(client)
        if args.serve:
            import uvicorn

            from sglang_omni.serve.openai_api import create_app

            if not args.no_warmup:
                print(
                    "Warming up VoiceChat before opening the microphone UI...",
                    flush=True,
                )
                await warmup_realtime(dep)
            app = create_app(
                client, model_name="nemotron-voicechat", realtime_deployment=dep
            )
            mount_example_ui(app)
            print(f"VoiceChat UI: http://localhost:{args.port}", flush=True)
            await uvicorn.Server(
                uvicorn.Config(app, host="127.0.0.1", port=args.port)
            ).serve()
            return
        if not args.audio:
            raise ValueError("--audio is required unless --serve is set")
        audio = load_audio(
            args.audio, source_name="VoiceChat", target_sample_rate=16000, mono=False
        )[0]
        if args.seconds:
            audio = audio[: int(args.seconds * 16000)]
        pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
        runtime = SessionRuntime(
            "nemotron-voicechat", dep.capabilities, dep.adapter_factory, dep.limits
        )
        chunks, text, receipts = [], [], []
        started = time.perf_counter()

        async def consume():
            async for envelope in runtime.outputs():
                event = envelope.event
                if isinstance(event, AudioDelta):
                    chunks.append(event.pcm)
                    receipts.append(time.perf_counter() - started)
                elif isinstance(event, TextDelta):
                    text.append(event.text)
                elif isinstance(event, Failure):
                    raise RuntimeError(str(event))  # noqa: TRY004
                elif isinstance(event, Drained):
                    break

        task = asyncio.create_task(consume())
        try:
            await runtime.update({"output_modalities": ["audio", "text"]}, "configure")
            for offset in range(0, len(pcm), 2560):
                await runtime.append(
                    pcm[offset : offset + 2560], offset // 2560, None, str(offset)
                )
                if args.paced:
                    await asyncio.sleep(0.08)
                else:
                    await asyncio.sleep(0.01)
            await runtime.end("eos")
            await asyncio.wait_for(task, 120)
        finally:
            await runtime.close("example_finished")
            if not task.done():
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
        expected_samples = ((len(pcm) // 2 + 1279) // 1280) * 1764
        actual_samples = sum(map(len, chunks)) // 2
        if actual_samples != expected_samples:
            raise RuntimeError(
                f"Expected {expected_samples} audio samples, received {actual_samples}"
            )
        with wave.open(args.out, "wb") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(22050)
            f.writeframes(b"".join(chunks))
        report = {
            "text": "".join(text),
            "audio_samples": sum(map(len, chunks)) // 2,
            "input_samples": len(pcm) // 2,
            "first_audio_s": receipts[0] if receipts else None,
            "elapsed_s": time.perf_counter() - started,
        }
        print(json.dumps(report, indent=2), flush=True)
        Path(args.out + ".json").write_text(json.dumps(report, indent=2))
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
        "--no-warmup", action="store_true", help="skip startup warmup when serving"
    )
    parser.add_argument("--port", type=int, default=8097)
    parser.add_argument("--talker-attention", choices=["triton", "torch_native"])
    asyncio.run(run(parser.parse_args()))
