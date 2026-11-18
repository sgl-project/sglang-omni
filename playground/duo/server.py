# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Serve the Duo browser client through SGLang's realtime protocol."""

import argparse
import asyncio
import base64
import io
import json
import logging
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from typing import NotRequired, TypedDict

import av
import numpy as np
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from PIL import Image, ImageOps

FRAMES_PER_BLOCK = 17
FPS = 24
INPUT_SAMPLES = round(16000 * FRAMES_PER_BLOCK / FPS)
BLOCK_MS = 1000 * FRAMES_PER_BLOCK / FPS
ROOT = Path(__file__).resolve().parent
logger = logging.getLogger(__name__)


class MediaTime(TypedDict):
    t_start_ms: float


class MediaMetadata(TypedDict):
    media_time: MediaTime


class EngineMediaPacket(TypedDict):
    type: str
    response_id: str
    delta: str
    sglang: MediaMetadata


class VideoFrame(TypedDict):
    pts: int
    data: str


class VideoBlock(TypedDict):
    chunk_index: int
    codec: str
    frames: list[VideoFrame]
    timings: dict[str, float]
    frame_offset: NotRequired[int]
    block_complete: NotRequired[bool]


class BrowserMediaPacket(TypedDict):
    type: str
    response_id: str
    delta: str
    video: VideoBlock


@dataclass
class MediaBlock:
    images: dict[int, str] = field(default_factory=dict)
    audio: str | None = None
    response_id: str = ""


class MediaAssembler:
    def __init__(self) -> None:
        self.blocks: dict[int, MediaBlock] = {}
        self.next_block = 0

    def append(self, packet: EngineMediaPacket) -> list[BrowserMediaPacket]:
        event_type = packet["type"]
        if event_type not in (
            "sglang.response.video.delta",
            "response.output_audio.delta",
        ):
            return []
        else:
            pass
        timestamp_ms = packet["sglang"]["media_time"]["t_start_ms"]
        if event_type == "sglang.response.video.delta":
            frame_index = round(timestamp_ms * FPS / 1000)
            block_index, local_index = divmod(frame_index, FRAMES_PER_BLOCK)
            block = self.blocks.setdefault(block_index, MediaBlock())
            block.images[local_index] = packet["delta"]
        else:
            block_index = round(timestamp_ms / BLOCK_MS)
            block = self.blocks.setdefault(block_index, MediaBlock())
            block.audio = packet["delta"]
        block.response_id = packet["response_id"]
        ready = []
        while self.next_block in self.blocks:
            block = self.blocks[self.next_block]
            if block.audio is None or len(block.images) != FRAMES_PER_BLOCK:
                break
            else:
                pass
            ready.append(
                {
                    "type": "response.output_audio.delta",
                    "response_id": block.response_id,
                    "delta": block.audio,
                    "video": {
                        "chunk_index": self.next_block,
                        "codec": "jpeg",
                        "frames": [
                            {
                                "pts": self.next_block * FRAMES_PER_BLOCK + index,
                                "data": block.images[index],
                            }
                            for index in range(FRAMES_PER_BLOCK)
                        ],
                        "timings": {},
                    },
                }
            )
            self.blocks.pop(self.next_block)
            self.next_block += 1
        return ready


def split_browser_packet(packet: BrowserMediaPacket) -> list[BrowserMediaPacket]:
    """Partition aligned stereo PCM and frames without changing media samples."""
    audio = base64.b64decode(packet["delta"], validate=True)
    frames = packet["video"]["frames"]
    sample_count = len(audio) // 4
    parts: list[BrowserMediaPacket] = []
    for start_frame in range(0, len(frames), 4):
        stop_frame = min(start_frame + 4, len(frames))
        start_sample = round(start_frame * sample_count / len(frames))
        stop_sample = round(stop_frame * sample_count / len(frames))
        parts.append(
            {
                "type": packet["type"],
                "response_id": packet["response_id"],
                "delta": base64.b64encode(
                    audio[start_sample * 4 : stop_sample * 4]
                ).decode(),
                "video": {
                    **packet["video"],
                    "frames": frames[start_frame:stop_frame],
                    "timings": dict(packet["video"]["timings"]),
                    "frame_offset": start_frame,
                    "block_complete": stop_frame == len(frames),
                },
            }
        )
    return parts


class LiveH264Encoder:
    """Encode ordered JPEG frames with the browser gateway's low-delay AVC profile."""

    def __init__(self) -> None:
        self.jpeg = av.CodecContext.create("mjpeg", "r")
        self.jpeg.thread_count = 1
        self.codec: av.CodecContext | None = None
        self.avc_codec: str | None = None

    def encode(self, frames: list[VideoFrame]) -> list[tuple[bytes, bool]]:
        packets = []
        for frame in frames:
            image = self.jpeg.decode(
                av.Packet(base64.b64decode(frame["data"], validate=True))
            )[0]
            image = image.reformat(
                format="yuv420p", src_color_range=2, dst_color_range=1
            )
            if self.codec is None:
                self.codec = av.CodecContext.create("libx264", "w")
                self.codec.width, self.codec.height = image.width, image.height
                self.codec.pix_fmt = "yuv420p"
                self.codec.time_base = Fraction(1, FPS)
                self.codec.framerate = Fraction(FPS, 1)
                self.codec.thread_count = 2
                self.codec.options = {
                    "preset": "ultrafast",
                    "tune": "zerolatency",
                    "crf": "20",
                    "profile": "main",
                    "level": "3.1",
                    "x264-params": "keyint=48:scenecut=0:repeat-headers=1:bframes=0:cabac=1",
                }
            else:
                pass
            # JPEG decoding marks every frame I; let x264 select inter frames.
            image.pict_type = av.video.frame.PictureType.NONE
            image.pts = frame["pts"]
            image.time_base = Fraction(1, FPS)
            emitted = self.codec.encode(image)
            assert len(emitted) == 1
            packet = emitted[0]
            raw = bytes(packet)
            if self.avc_codec is None:
                start = raw.index(b"\x00\x00\x00\x01\x67") + 5
                self.avc_codec = "avc1." + raw[start : start + 3].hex()
            else:
                pass
            packets.append((raw, packet.is_keyframe))
        return packets


def encode_browser_packet(
    packet: BrowserMediaPacket,
    encoder: LiveH264Encoder | None = None,
    encoded_frames: list[tuple[bytes, bool]] | None = None,
) -> bytes:
    started = time.perf_counter()
    audio = base64.b64decode(packet["delta"], validate=True)
    if encoded_frames is not None:
        encoded = encoded_frames
    elif encoder is None:
        encoded = [
            (base64.b64decode(frame["data"], validate=True), False)
            for frame in packet["video"]["frames"]
        ]
    else:
        encoded = encoder.encode(packet["video"]["frames"])
    images = [raw for raw, keyframe in encoded]
    video = {
        "chunk_index": packet["video"]["chunk_index"],
        "codec": "jpeg" if encoder is None else "h264",
        "timings": dict(packet["video"]["timings"]),
        "frames": [
            {"pts": frame["pts"], "byte_length": len(raw), "keyframe": keyframe}
            for frame, (raw, keyframe) in zip(packet["video"]["frames"], encoded)
        ],
    }
    if "frame_offset" in packet["video"]:
        video["frame_offset"] = packet["video"]["frame_offset"]
        video["block_complete"] = packet["video"]["block_complete"]
    else:
        pass
    if encoder is not None:
        video["avc_codec"] = encoder.avc_codec
    else:
        pass
    video["timings"]["gateway_encode_ms"] = (
        packet["video"]["timings"].get("gateway_encode_wait_ms", 0)
        + (time.perf_counter() - started) * 1000
    )
    header = {
        "type": packet["type"],
        "response_id": packet["response_id"],
        "audio_byte_length": len(audio),
        "video": video,
    }
    encoded_header = json.dumps(header, separators=(",", ":"), allow_nan=False).encode()
    return b"".join(
        (struct.pack("<I", len(encoded_header)), encoded_header, audio, *images)
    )


class DuplexDemo:
    def __init__(
        self, backend_url: str, sample_directory: Path, portrait_directory: Path
    ) -> None:
        self.backend_url = backend_url.rstrip("/")
        self.sample_directory = sample_directory
        self.portrait_directory = portrait_directory
        self.presets = json.loads((ROOT / "personas.json").read_text())
        self.sample_frames = sorted((sample_directory / "frames").glob("*.jpg"))
        self.sample_audio = (sample_directory / "audio.pcm").read_bytes()
        self.block_count = min(
            len(self.sample_frames) // FRAMES_PER_BLOCK,
            len(self.sample_audio) // (INPUT_SAMPLES * 2),
        )
        self.busy = False

    async def status(self, request: web.Request) -> web.Response:
        try:
            async with ClientSession(timeout=ClientTimeout(total=3)) as client:
                async with client.get(
                    self.backend_url + "/duo/configuration"
                ) as response:
                    ready = response.status == 200
                    configuration = await response.json() if ready else {}
        except (OSError, asyncio.TimeoutError):
            ready = False
            configuration = {}
        return web.json_response(
            {
                "ready": ready,
                "busy": self.busy,
                "blocks": self.block_count,
                "media_reset_supported": configuration.get(
                    "realization_kv_reset_supported", False
                ),
                "system_prompt_prefix": configuration.get("system_prompt_prefix", ""),
                "system_prompt_suffix": configuration.get("system_prompt_suffix", ""),
                "media_reset_lookahead_blocks": configuration.get(
                    "video_lookahead_blocks", 0
                ),
            }
        )

    async def presets_response(self, request: web.Request) -> web.Response:
        return web.json_response(self.presets)

    async def portrait(self, request: web.Request) -> web.FileResponse:
        preset = request.match_info["preset"]
        if preset not in self.presets:
            raise web.HTTPNotFound()
        else:
            return web.FileResponse(self.portrait_directory / preset / "portrait.jpg")

    async def example(self, request: web.Request) -> web.FileResponse:
        return web.FileResponse(self.sample_directory / "example.mp4")

    async def page(self, request: web.Request) -> web.FileResponse:
        name = request.match_info.get("name", "index.html")
        if name not in {
            "index.html",
            "app.js",
            "style.css",
            "capture.js",
            "catchup.js",
            "input_catchup.js",
            "video_decoding.js",
        }:
            raise web.HTTPNotFound()
        else:
            return web.FileResponse(ROOT / name)

    async def socket(self, request: web.Request) -> web.WebSocketResponse:
        if self.busy:
            raise web.HTTPConflict(text="已有会话进行中")
        else:
            self.busy = True
        binary_output = request.query.get("binary") == "1"
        stream_output = binary_output and request.query.get("stream") == "1"
        browser = web.WebSocketResponse(
            max_msg_size=32 << 20, heartbeat=20, compress=False
        )
        encoder = (
            LiveH264Encoder()
            if binary_output and request.query.get("video") == "h264"
            else None
        )
        encoding_executor = (
            ThreadPoolExecutor(max_workers=1, thread_name_prefix="duo-video-encode")
            if encoder is not None
            else None
        )
        try:
            await browser.prepare(request)
            setup = await browser.receive_json(timeout=30)
            preset = setup.get("preset", "seaside")
            if (
                preset not in self.presets
                or setup.get("prompt_mode", "fields") != "fields"
            ):
                raise ValueError("请选择人物预设，使用人物、环境和初始姿态字段")
            else:
                pass
            description = {
                name: setup.get(name, self.presets[preset][name])
                for name in ("appearance", "environment", "initial_pose")
            }
            seed = setup.get("seed", 42)
            if type(seed) is not int or not 0 <= seed <= 4294967295:
                raise ValueError("随机种子必须是 0 到 4294967295 之间的整数")
            else:
                pass
            reference_bytes = (
                base64.b64decode(setup["image"], validate=True)
                if setup.get("image")
                else (self.portrait_directory / preset / "portrait.jpg").read_bytes()
            )
            with Image.open(io.BytesIO(reference_bytes)) as image:
                output = io.BytesIO()
                ImageOps.fit(image.convert("RGB"), (576, 768)).save(
                    output, format="JPEG", quality=95
                )
                reference = base64.b64encode(output.getvalue()).decode()
            await browser.send_json(
                {
                    "type": "demo.preparing",
                    "message": "正在准备 SGLang 会话",
                    "elapsed_s": 0,
                }
            )
            async with ClientSession() as client:
                async with client.get(
                    self.backend_url + "/duo/configuration"
                ) as response:
                    configuration = await response.json()
                video_lookahead_blocks = configuration.get("video_lookahead_blocks", 0)
                async with client.ws_connect(
                    self.backend_url + "/v1/realtime", max_msg_size=64 << 20
                ) as engine:
                    created = await engine.receive_json()
                    if created["type"] != "session.created":
                        raise RuntimeError(str(created))
                    else:
                        pass
                    await engine.send_json(
                        {
                            "type": "session.update",
                            "event_id": "setup",
                            "session": {
                                "output_modalities": ["audio", "video"],
                                "audio": {
                                    "input": {
                                        "format": {"type": "audio/pcm", "rate": 16000}
                                    },
                                    "output": {
                                        "format": {"type": "audio/pcm", "rate": 32000}
                                    },
                                },
                                "sglang": {
                                    "max_slice_nums": 2,
                                    "realization": {
                                        "decode": "duo_5_17",
                                        "description": description,
                                        "reference_image": reference,
                                        "seed": seed,
                                    },
                                },
                            },
                        }
                    )
                    configured = await engine.receive_json()
                    if configured["type"] != "session.updated":
                        raise RuntimeError(str(configured))
                    else:
                        await browser.send_json(created)
                    submitted: list[float] = []
                    input_forward_seconds: list[float] = []
                    assembler = MediaAssembler()
                    ended = False
                    reset_generation = 0
                    reset_blocks: dict[int, int] = {}

                    async def upload() -> None:
                        nonlocal ended, reset_generation
                        frames: list[str] = []
                        samples: list[np.ndarray] = []
                        async for message in browser:
                            input_received = time.perf_counter()
                            if message.type != WSMsgType.TEXT:
                                break
                            else:
                                packet = json.loads(message.data)
                            event_type = packet["type"]
                            if event_type in (
                                "input_audio_buffer.append",
                                "demo.record_audio",
                            ):
                                frames.extend(packet.get("video_frames", []))
                                if packet.get("audio"):
                                    audio_bytes = base64.b64decode(packet["audio"])
                                    audio_format = packet.get("format", "pcm_f32le")
                                    if audio_format == "pcm_s16le":
                                        pcm_samples = np.frombuffer(
                                            audio_bytes, dtype="<i2"
                                        )
                                    elif audio_format == "pcm_f32le":
                                        pcm_samples = (
                                            np.clip(
                                                np.frombuffer(audio_bytes, dtype="<f4"),
                                                -1,
                                                1,
                                            )
                                            * 32767
                                        ).astype("<i2")
                                    else:
                                        raise ValueError(
                                            "Unsupported input audio format"
                                        )
                                    samples.append(pcm_samples)
                                else:
                                    pass
                                continue
                            elif event_type == "demo.sample_block":
                                index = int(packet["index"])
                                if not 0 <= index < self.block_count:
                                    raise ValueError(
                                        "Sample block is outside the available recording"
                                    )
                                else:
                                    pass
                                frames = [
                                    base64.b64encode(path.read_bytes()).decode()
                                    for path in self.sample_frames[
                                        index * 17 : (index + 1) * 17
                                    ]
                                ]
                                samples = [
                                    (
                                        np.frombuffer(
                                            self.sample_audio[
                                                index
                                                * INPUT_SAMPLES
                                                * 2 : (index + 1)
                                                * INPUT_SAMPLES
                                                * 2
                                            ],
                                            dtype="<i2",
                                        ).astype(np.float32)
                                        / 32768
                                        * 32767
                                    ).astype("<i2")
                                ]
                            elif event_type == "demo.client_stats":
                                logger.info(
                                    f"Duo client playback: session={created['session']['id']} metrics={packet}"
                                )
                                continue
                            elif event_type == "demo.reset_media":
                                if not configuration.get(
                                    "realization_kv_reset_supported", False
                                ):
                                    await browser.send_json(
                                        {
                                            "type": "error",
                                            "error": {
                                                "message": "服务端未启用手动 DiT KV 重置"
                                            },
                                        }
                                    )
                                    continue
                                if (
                                    ended
                                    or len(submitted) - assembler.next_block
                                    > video_lookahead_blocks
                                ):
                                    raise ValueError(
                                        "请等待已提交的视频块返回后重置 Realization KV"
                                    )
                                else:
                                    reset_generation += 1
                                reset_blocks[len(submitted)] = reset_generation
                                await engine.send_json(
                                    {
                                        "type": "session.update",
                                        "event_id": f"realization-reset-{reset_generation}",
                                        "session": {
                                            "sglang": {
                                                "realization": {
                                                    "kv_reset_generation": reset_generation,
                                                }
                                            }
                                        },
                                    }
                                )
                                continue
                            elif event_type == "session.close":
                                if not ended:
                                    await engine.send_json(
                                        {
                                            "type": "sglang.input_audio.end",
                                            "event_id": "end",
                                        }
                                    )
                                    ended = True
                                else:
                                    pass
                                return
                            elif event_type != "input_audio_buffer.commit":
                                continue
                            else:
                                pass
                            if len(frames) != FRAMES_PER_BLOCK:
                                raise ValueError(
                                    f"Expected 17 camera frames, received {len(frames)}"
                                )
                            else:
                                pass
                            sequence = len(submitted)
                            for index, image in enumerate(frames):
                                await engine.send_json(
                                    {
                                        "type": "sglang.input_image.append",
                                        "event_id": f"image-{sequence}-{index}",
                                        "image": image,
                                        "sglang": {
                                            "t_ms": sequence * INPUT_SAMPLES / 16
                                            + index * 1000 / FPS
                                        },
                                    }
                                )
                            audio = (
                                np.concatenate(samples)
                                if samples
                                else np.zeros(INPUT_SAMPLES, dtype="<i2")
                            )
                            audio = np.pad(
                                audio[:INPUT_SAMPLES],
                                (0, max(0, INPUT_SAMPLES - len(audio))),
                            )
                            pcm = audio.astype("<i2", copy=False).tobytes()
                            forwarded_at = time.perf_counter()
                            submitted.append(forwarded_at)
                            input_forward_seconds.append(forwarded_at - input_received)
                            await engine.send_json(
                                {
                                    "type": "input_audio_buffer.append",
                                    "event_id": f"audio-{sequence}",
                                    "audio": base64.b64encode(pcm).decode(),
                                    "sglang": {
                                        "seq": sequence,
                                        "t_start_ms": sequence * INPUT_SAMPLES / 16,
                                    },
                                }
                            )
                            frames, samples = [], []
                            if (
                                event_type == "demo.sample_block"
                                and int(packet["index"]) + 1 == self.block_count
                            ):
                                await engine.send_json(
                                    {
                                        "type": "sglang.input_audio.end",
                                        "event_id": "end",
                                    }
                                )
                                ended = True
                            else:
                                pass

                    async def download() -> None:
                        pending_frames: dict[int, str] = {}
                        encoding_futures: dict[
                            int, asyncio.Future[list[tuple[bytes, bool]]]
                        ] = {}
                        next_encoding_frame = 0
                        async for message in engine:
                            if message.type != WSMsgType.TEXT:
                                break
                            else:
                                packet = json.loads(message.data)
                            if (
                                encoder is not None
                                and packet["type"] == "sglang.response.video.delta"
                            ):
                                timestamp_ms = packet["sglang"]["media_time"][
                                    "t_start_ms"
                                ]
                                frame_index = round(timestamp_ms * FPS / 1000)
                                pending_frames[frame_index] = packet["delta"]
                                while next_encoding_frame in pending_frames:
                                    encoding_futures[
                                        next_encoding_frame
                                    ] = asyncio.get_running_loop().run_in_executor(
                                        encoding_executor,
                                        encoder.encode,
                                        [
                                            {
                                                "pts": next_encoding_frame,
                                                "data": pending_frames.pop(
                                                    next_encoding_frame
                                                ),
                                            }
                                        ],
                                    )
                                    next_encoding_frame += 1
                            else:
                                pass
                            for output in assembler.append(packet):
                                index = output["video"]["chunk_index"]
                                output["video"]["timings"]["submit_to_av_ms"] = (
                                    time.perf_counter() - submitted[index]
                                ) * 1000
                                if index in reset_blocks:
                                    output["video"]["timings"][
                                        "realization_kv_reset_generation"
                                    ] = reset_blocks.pop(index)
                                else:
                                    pass
                                output["video"]["timings"][
                                    "gateway_input_forward_ms"
                                ] = (input_forward_seconds[index] * 1000)
                                av_ready = time.perf_counter()
                                if binary_output:
                                    parts = (
                                        split_browser_packet(output)
                                        if stream_output
                                        else [output]
                                    )
                                    for part in parts:
                                        encoded_frames = None
                                        if encoder is not None:
                                            encode_wait_started = time.perf_counter()
                                            frame_packets = await asyncio.gather(
                                                *(
                                                    encoding_futures.pop(frame["pts"])
                                                    for frame in part["video"]["frames"]
                                                )
                                            )
                                            encoded_frames = [
                                                packets[0] for packets in frame_packets
                                            ]
                                            part["video"]["timings"][
                                                "gateway_encode_wait_ms"
                                            ] = (
                                                time.perf_counter()
                                                - encode_wait_started
                                            ) * 1000
                                        else:
                                            pass
                                        part["video"]["timings"][
                                            "gateway_av_to_encode_ms"
                                        ] = (time.perf_counter() - av_ready) * 1000
                                        encoded = await asyncio.to_thread(
                                            encode_browser_packet,
                                            part,
                                            encoder,
                                            encoded_frames,
                                        )
                                        send_started = time.perf_counter()
                                        await browser.send_bytes(encoded)
                                        logger.info(
                                            f"Duo gateway delivery: session={created['session']['id']} block={index} frame_offset={part['video'].get('frame_offset', 0)} frames={len(part['video']['frames'])} codec={'jpeg' if encoder is None else 'h264'} bytes={len(encoded)} send_ms={(time.perf_counter() - send_started) * 1000:.3f}"
                                        )
                                else:
                                    await browser.send_json(output)
                            if packet["type"] == "error":
                                await browser.send_json(packet)
                                return
                            elif packet["type"] == "session.updated":
                                await browser.send_json(packet)
                            elif packet["type"] == "sglang.input_audio.drained":
                                await engine.send_json(
                                    {"type": "session.close", "event_id": "close"}
                                )
                            elif packet["type"] == "session.closed":
                                # Let the browser drain its scheduled audio and video before closing.
                                return
                            else:
                                pass

                    tasks = [
                        asyncio.create_task(upload()),
                        asyncio.create_task(download()),
                    ]
                    try:
                        done, pending = await asyncio.wait(
                            tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            task.result()
                        if tasks[0] in done and not tasks[1].done():
                            await asyncio.wait_for(tasks[1], 120)
                        elif tasks[1] in done and not tasks[0].done():
                            await tasks[0]
                        else:
                            pass
                    finally:
                        for task in tasks:
                            task.cancel()
                        await asyncio.gather(*tasks, return_exceptions=True)
        except (ValueError, RuntimeError, TimeoutError) as error:
            if not browser.closed:
                await browser.send_json(
                    {"type": "error", "error": {"message": str(error)}}
                )
            else:
                pass
        finally:
            if encoding_executor is not None:
                encoding_executor.shutdown(wait=True, cancel_futures=True)
            else:
                pass
            self.busy = False
        return browser


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-url", required=True)
    parser.add_argument("--sample-directory", type=Path, required=True)
    parser.add_argument("--portrait-directory", type=Path, required=True)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=18081)
    options = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    demo = DuplexDemo(
        options.backend_url, options.sample_directory, options.portrait_directory
    )
    application = web.Application()
    application.add_routes(
        [
            web.get("/", demo.page),
            web.get("/status", demo.status),
            web.get("/presets", demo.presets_response),
            web.get("/ws", demo.socket),
            web.get("/portraits/{preset}.jpg", demo.portrait),
            web.get("/example.mp4", demo.example),
            web.get("/{name}", demo.page),
        ]
    )
    web.run_app(application, host=options.host, port=options.port)


if __name__ == "__main__":
    main()
