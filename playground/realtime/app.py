# SPDX-License-Identifier: Apache-2.0
"""Serve the MiniCPM demo with a transparent SGLang-Omni realtime connection."""

import argparse
import asyncio
import base64
import io
import json
import logging
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import soundfile as sf
import yaml
from aiohttp import ClientError, ClientSession, ClientTimeout, WSMsgType, web
from PIL import Image
from scipy.signal import resample_poly

ROOT = Path(__file__).parent
FRONTEND = ROOT / "frontend"
logger = logging.getLogger(__name__)


async def warmup(api_base: str) -> None:
    """Exercise audio, vision and speech before accepting browser sessions."""
    reference = load_reference("assets/minicpmo/ref_audio/ref_en_dlc_1.wav")
    samples = np.frombuffer(base64.b64decode(reference["data"]), dtype="<f4")
    reference_wav = io.BytesIO()
    sf.write(reference_wav, samples, 16000, format="WAV", subtype="PCM_16")
    audio = np.zeros(8 * 16000, dtype="<i2")
    length = min(len(samples), 4 * 16000)
    audio[:length] = np.clip(samples[:length] * 32767, -32768, 32767).astype("<i2")
    image = io.BytesIO()
    Image.new("RGB", (448, 448), "white").save(image, format="JPEG")
    async with ClientSession() as client:
        async with client.get(f"{api_base}/v1/realtime/capabilities") as response:
            response.raise_for_status()
            capabilities = await response.json()
        if not capabilities.get("native_full_duplex"):
            raise RuntimeError("Playground warmup requires native full duplex")
        else:
            pass
        modes = (
            (False, True) if "image" in capabilities["input_modalities"] else (False,)
        )
        for with_image in modes:
            mode = "video" if with_image else "audio"
            logger.info(f"Warming up {mode} conversation and speech generation")
            received_audio = False
            drained = False
            async with asyncio.timeout(180):
                async with client.ws_connect(
                    f"{api_base}/v1/realtime", max_msg_size=16 * 1024 * 1024
                ) as websocket:
                    async for message in websocket:
                        if message.type != WSMsgType.TEXT:
                            raise RuntimeError(
                                f"Warmup connection ended: {message.type}"
                            )
                        else:
                            event = json.loads(message.data)
                        kind = event["type"]
                        if kind == "session.created":
                            await websocket.send_json(
                                {
                                    "type": "session.update",
                                    "event_id": "warmup-config",
                                    "session": {
                                        "instructions": "You are a helpful voice assistant. Reply briefly to what you hear.",
                                        "output_modalities": ["audio"],
                                        "sglang": {
                                            "sampling": {
                                                "greedy": True,
                                                "force_listen_count": 0,
                                            },
                                            "reference_audio": {
                                                "media_type": "audio/wav",
                                                "data": base64.b64encode(
                                                    reference_wav.getvalue()
                                                ).decode(),
                                            },
                                        },
                                    },
                                }
                            )
                        elif kind == "session.updated":
                            if with_image:
                                await websocket.send_json(
                                    {
                                        "type": "sglang.input_image.append",
                                        "event_id": "warmup-image",
                                        "image": base64.b64encode(
                                            image.getvalue()
                                        ).decode(),
                                        "sglang": {"t_ms": 0},
                                    }
                                )
                            else:
                                pass
                            for sequence, offset in enumerate(
                                range(0, len(audio), 1280)
                            ):
                                await websocket.send_json(
                                    {
                                        "type": "input_audio_buffer.append",
                                        "event_id": f"warmup-audio-{sequence}",
                                        "audio": base64.b64encode(
                                            audio[offset : offset + 1280].tobytes()
                                        ).decode(),
                                        "sglang": {
                                            "seq": sequence,
                                            "t_start_ms": offset / 16,
                                        },
                                    }
                                )
                            await websocket.send_json(
                                {
                                    "type": "sglang.input_audio.end",
                                    "event_id": "warmup-end",
                                }
                            )
                        elif kind == "response.output_audio.delta":
                            received_audio = received_audio or bool(event["delta"])
                        elif kind == "sglang.input_audio.drained":
                            drained = True
                            await websocket.send_json(
                                {"type": "session.close", "event_id": "warmup-close"}
                            )
                        elif kind == "session.closed":
                            break
                        elif kind == "error":
                            raise RuntimeError(f"Warmup failed: {event['error']}")
                        else:
                            pass
            if not received_audio or not drained:
                raise RuntimeError(f"{mode} warmup did not complete speech generation")
            else:
                logger.info(f"{mode} warmup complete")


@lru_cache(maxsize=16)
def load_reference(relative_path: str) -> dict:
    audio, rate = sf.read(ROOT / relative_path, dtype="float32", always_2d=True)
    divisor = math.gcd(rate, 16000)
    samples = resample_poly(audio.mean(axis=1), 16000 // divisor, rate // divisor)
    return {
        "data": base64.b64encode(samples.astype("<f4").tobytes()).decode(),
        "name": Path(relative_path).name,
        "duration": len(samples) / 16000,
    }


def create_app(api_base: str) -> web.Application:
    app = web.Application()

    async def prepare(application: web.Application) -> None:
        await warmup(api_base)

    app.on_startup.append(prepare)

    async def capabilities(request: web.Request) -> web.Response:
        try:
            async with ClientSession(timeout=ClientTimeout(total=10)) as client:
                async with client.get(
                    f"{api_base}/v1/realtime/capabilities"
                ) as response:
                    return web.Response(
                        body=await response.read(),
                        status=response.status,
                        content_type="application/json",
                    )
        except (ClientError, TimeoutError) as error:
            return web.json_response({"error": str(error)}, status=502)

    async def realtime(request: web.Request) -> web.WebSocketResponse:
        browser = web.WebSocketResponse(max_msg_size=16 * 1024 * 1024)
        await browser.prepare(request)
        try:
            async with ClientSession() as client:
                async with client.ws_connect(
                    f"{api_base}/v1/realtime", max_msg_size=16 * 1024 * 1024
                ) as upstream:

                    async def send_input() -> None:
                        async for message in browser:
                            if message.type == WSMsgType.TEXT:
                                await upstream.send_str(message.data)
                            else:
                                break

                    async def receive_output() -> None:
                        async for message in upstream:
                            if message.type == WSMsgType.TEXT:
                                await browser.send_str(message.data)
                            else:
                                break

                    tasks = [
                        asyncio.create_task(send_input()),
                        asyncio.create_task(receive_output()),
                    ]
                    try:
                        await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    finally:
                        for task in tasks:
                            task.cancel()
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                    for result in results:
                        if isinstance(result, Exception):
                            raise result
        except (ClientError, TimeoutError, ConnectionError) as error:
            if not browser.closed:
                await browser.send_json(
                    {"type": "error", "error": {"message": str(error)}}
                )
        finally:
            await browser.close()
        return browser

    async def index(request: web.Request) -> web.FileResponse:
        return web.FileResponse(FRONTEND / "orb" / "index.html")

    app.router.add_get("/v1/realtime/capabilities", capabilities)
    app.router.add_get("/v1/realtime", realtime)
    app.router.add_get("/", index)

    presets = {}
    for mode in ("audio_duplex", "omni"):
        presets[mode] = []
        for path in sorted(
            (ROOT / "assets" / "minicpmo" / "presets" / mode).glob("*.yaml")
        ):
            preset = yaml.safe_load(path.read_text())
            reference_path = preset.pop("ref_audio_path", None)
            if reference_path:
                preset["ref_audio"] = {
                    "path": reference_path,
                    "name": Path(reference_path).name,
                    "data": None,
                }
            presets[mode].append(preset)
        presets[mode].sort(key=lambda preset: preset.get("order", 999))

    async def list_presets(request: web.Request) -> web.Response:
        return web.json_response(presets)

    async def preset_audio(request: web.Request) -> web.Response:
        preset = next(
            (
                entry
                for entry in presets.get(request.match_info["mode"], [])
                if entry["id"] == request.match_info["preset_id"]
            ),
            None,
        )
        if preset is None:
            raise web.HTTPNotFound()
        if "ref_audio" not in preset:
            return web.json_response({})
        reference = await asyncio.to_thread(load_reference, preset["ref_audio"]["path"])
        return web.json_response({"ref_audio": reference})

    app.router.add_get("/orb", index)
    app.router.add_get("/api/presets", list_presets)
    app.router.add_get("/api/presets/{mode}/{preset_id}/audio", preset_audio)
    app.router.add_static("/static/", FRONTEND)
    return app


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api-base", default="http://127.0.0.1:8000")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    web.run_app(create_app(args.api_base.rstrip("/")), host=args.host, port=args.port)
