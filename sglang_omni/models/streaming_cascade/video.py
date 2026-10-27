# SPDX-License-Identifier: Apache-2.0
"""The openai_video kind: the scene a video model of an SGLang diffusion server shoots, one clip per call of the generator, each continuing from the last frame of the one before."""

import asyncio
import base64
import io
import logging
import math
from collections.abc import AsyncIterator
from typing import Literal

import httpx
from pydantic import Field

from sglang_omni.models.streaming_cascade.base import (
    ComponentConfig,
    StreamComponent,
    bearer_headers,
    stop_running,
)
from sglang_omni.models.streaming_cascade.tools import ToolLifecycle
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import TOOL_CALL, VIDEO, VIDEO_CLIP, VIDEO_PROGRESS
from sglang_omni.proto.tools import (
    ToolCall,
    ToolResponse,
    tool_calls_of,
    tool_response_chunk,
)
from sglang_omni.scheduling.session import ChunkEmitter

logger = logging.getLogger(__name__)

VIDEO_MIME_TYPE = "video/mp4"
KEYFRAME_MIME_TYPE = "image/png"
JOB_STATUS_DONE = "completed"
JOB_STATUS_FAILED = "failed"
DELETE_TIMEOUT_S = 5.0


class OpenAIVideoConfig(ComponentConfig):
    """A scene of clips a video model makes behind the OpenAI-compatible video job API of an SGLang diffusion server (`POST /v1/videos`, polled until done), one clip of `seconds` per call of the tool named `tool` that the generator offers. The call's `prompt` says what happens next; its `new_scene` starts from nothing instead of the last frame of the clip before, which the node keeps as the next clip's first keyframe. The clip's file is published in pieces of at most `piece_bytes` as soon as it is decoded, its progress meanwhile; the tool's result answers the call afterwards. A clip past `timeout_s` answers with an error."""

    kind: Literal["openai_video"] = "openai_video"
    base_url: str = Field(pattern=r"^https?://[^/]+")
    model: str
    api_key_env: str = "OPENAI_API_KEY"
    tool: str = Field(default="generate_video", min_length=1)
    seconds: int = Field(default=4, ge=1)
    short_edge: int = Field(default=384, ge=64)
    aspect_ratio: str = Field(default="16:9", min_length=1)
    num_inference_steps: int | None = Field(default=None, gt=0)
    poll_interval_s: float = Field(default=0.25, gt=0)
    piece_bytes: int = Field(default=256 * 1024, gt=0)
    timeout_s: float = Field(default=300, gt=0)

    @property
    def lifecycle(self) -> ToolLifecycle:
        """The calls of the tool end the reply that makes them and survive its interruption: a clip takes longer than a reply, and the scene continues from it."""
        return ToolLifecycle(awaited=False, on_interrupt="retain")


def video_progress_chunk(
    call_id: str, clip: int, status: str, percent: int
) -> TimedChunk:
    return TimedChunk(
        VIDEO,
        0,
        0,
        0,
        {"call_id": call_id, "clip": clip, "status": status, "percent": percent},
        format=VIDEO_PROGRESS,
    )


def video_clip_chunks(
    call_id: str,
    clip: int,
    data: bytes,
    piece_bytes: int,
    *,
    duration_s: float,
    width: int,
    height: int,
    fps: float,
    prompt: str,
) -> list[TimedChunk]:
    """The file of one clip in pieces of at most piece_bytes, each saying how many there are."""
    pieces = max(1, math.ceil(len(data) / piece_bytes))
    return [
        TimedChunk(
            VIDEO,
            0,
            0,
            0,
            {
                "call_id": call_id,
                "clip": clip,
                "piece": piece,
                "pieces": pieces,
                "mime_type": VIDEO_MIME_TYPE,
                "data": data[piece * piece_bytes : (piece + 1) * piece_bytes],
                "duration_s": duration_s,
                "width": width,
                "height": height,
                "fps": fps,
                "prompt": prompt,
            },
            format=VIDEO_CLIP,
        )
        for piece in range(pieces)
    ]


def clip_facts(data: bytes) -> tuple[float, int, int, float]:
    """The duration in seconds, the width, the height and the frame rate of the video file, from its headers."""
    import av

    with av.open(io.BytesIO(data)) as container:
        stream = container.streams.video[0]
        duration_s = (
            float(stream.duration * stream.time_base)
            if stream.duration is not None
            else (
                float(container.duration / av.time_base)
                if container.duration is not None
                else 0.0
            )
        )
        fps = float(stream.average_rate) if stream.average_rate else 0.0
        return duration_s, stream.width, stream.height, fps


def last_frame(data: bytes) -> bytes:
    """The last frame of the video file as PNG; decoding the whole file is the only way to it."""
    import av

    with av.open(io.BytesIO(data)) as container:
        frame = None
        for frame in container.decode(container.streams.video[0]):
            pass
        if frame is None:
            raise ValueError("the video file has no frame")
        else:
            pass
        png = io.BytesIO()
        frame.to_image().save(png, format="PNG")
        return png.getvalue()


class OpenAIVideo(StreamComponent):
    """Shoots the Session's scene: the calls of the node's tool queue up and run one at a time, since each clip starts from the frame the one before ended on.

    A call ends the reply that made it (its tool is unawaited) and the
    clip is published in pieces the moment it is decoded, before the
    tool_response that lets the dialog report it in a turn of its own; the
    pieces and the progress of the clip carry no turn. An `interrupt`
    changes nothing: a clip in generation is retained, and the scene
    continues from it. The end of the input drops the queued calls and
    waits for the clip in generation, and closing the Session deletes its
    job.
    """

    def __init__(self, config: OpenAIVideoConfig) -> None:
        self.config = config
        self.client = httpx.AsyncClient(
            timeout=config.timeout_s, headers=bearer_headers(config.api_key_env)
        )
        self.emit: ChunkEmitter | None = None
        self.queued: asyncio.Queue[ToolCall | None] = asyncio.Queue()
        self.worker: asyncio.Task[None] | None = None
        # The calls so far, numbered as the clips of the scene in call order, the PNG of the last frame of the latest clip, and the job of the clip in generation.
        self.shot = 0
        self.numbers: dict[str, int] = {}
        self.keyframe: bytes | None = None
        self.job_id: str | None = None

    def can_replace(self) -> bool:
        return self.job_id is None and self.queued.empty()

    async def process(
        self, inputs: AsyncIterator[TimedChunk], emit: ChunkEmitter
    ) -> None:
        self.emit = emit
        self.worker = asyncio.create_task(self.work())
        try:
            async for chunk in inputs:
                if chunk.format == TOOL_CALL:
                    for call in tool_calls_of(chunk):
                        if call.name == self.config.tool:
                            self.shot += 1
                            self.numbers[call.id] = self.shot
                            emit(video_progress_chunk(call.id, self.shot, "queued", 0))
                            self.queued.put_nowait(call)
                        else:
                            pass
                else:
                    pass
            # The end of the input: the queued calls are dropped, the clip in generation finishes within its timeout.
            while not self.queued.empty():
                self.queued.get_nowait()
            self.queued.put_nowait(None)
            await self.worker
        finally:
            await stop_running([self.worker])

    async def close(self) -> None:
        await stop_running([self.worker] if self.worker is not None else [])
        if self.job_id is not None:
            try:
                await asyncio.wait_for(
                    self.client.delete(f"{self.base_url}/videos/{self.job_id}"),
                    DELETE_TIMEOUT_S,
                )
            except (httpx.HTTPError, TimeoutError) as error:
                logger.info(f"video job {self.job_id} was not deleted: {error!r}")
            self.job_id = None
        else:
            pass
        await self.client.aclose()

    @property
    def base_url(self) -> str:
        return self.config.base_url.rstrip("/")

    async def work(self) -> None:
        assert self.emit is not None
        while (call := await self.queued.get()) is not None:
            clip = self.numbers.pop(call.id)
            try:
                response = await asyncio.wait_for(
                    self.shoot(call, clip), self.config.timeout_s
                )
            except TimeoutError:
                response = {
                    "error": f"{call.name} gave no clip within {self.config.timeout_s} s"
                }
                await self.abandon()
            except (httpx.HTTPError, ValueError, KeyError, TypeError) as error:
                logger.warning(f"clip {clip} of call {call.id} failed", exc_info=error)
                response = {"error": str(error) or type(error).__name__}
                await self.abandon()
            if "error" in response:
                self.emit(video_progress_chunk(call.id, clip, "failed", 0))
            else:
                pass
            logger.info(f"video {call.id} {call.name}({call.arguments}) -> {response}")
            self.emit(
                tool_response_chunk(
                    [ToolResponse(id=call.id, name=call.name, response=response)], None
                )
            )

    async def abandon(self) -> None:
        """Delete the job of a clip that failed or timed out, so the server stops working on it."""
        if self.job_id is not None:
            try:
                await asyncio.wait_for(
                    self.client.delete(f"{self.base_url}/videos/{self.job_id}"),
                    DELETE_TIMEOUT_S,
                )
            except (httpx.HTTPError, TimeoutError) as error:
                logger.info(f"video job {self.job_id} was not deleted: {error!r}")
            self.job_id = None
        else:
            pass

    async def shoot(self, call: ToolCall, clip: int) -> dict[str, object]:
        """Make the call's clip: queue the job, publish its progress, then its file in pieces, and keep its last frame for the next clip."""
        assert self.emit is not None
        arguments = call.arguments if isinstance(call.arguments, dict) else {}
        prompt = arguments.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string")
        else:
            prompt = prompt.strip()
        keyframe = None if arguments.get("new_scene") else self.keyframe
        self.emit(video_progress_chunk(call.id, clip, "generating", 0))
        created = await self.client.post(
            f"{self.base_url}/videos", json=self.request(prompt, keyframe)
        )
        created.raise_for_status()
        job = created.json()
        self.job_id = str(job["id"])
        percent = 0
        while job.get("status") not in (JOB_STATUS_DONE, JOB_STATUS_FAILED):
            await asyncio.sleep(self.config.poll_interval_s)
            polled = await self.client.get(f"{self.base_url}/videos/{self.job_id}")
            polled.raise_for_status()
            job = polled.json()
            if (
                job.get("status") not in (JOB_STATUS_DONE, JOB_STATUS_FAILED)
                and int(job.get("progress") or 0) != percent
            ):
                percent = int(job.get("progress") or 0)
                self.emit(video_progress_chunk(call.id, clip, "generating", percent))
            else:
                pass
        if job.get("status") == JOB_STATUS_FAILED:
            raise ValueError(f"the video job failed: {job.get('error')}")
        else:
            pass
        content = await self.client.get(
            f"{self.base_url}/videos/{self.job_id}/content", follow_redirects=True
        )
        content.raise_for_status()
        self.job_id = None
        data = content.content
        duration_s, width, height, fps = await asyncio.to_thread(clip_facts, data)
        for piece in video_clip_chunks(
            call.id,
            clip,
            data,
            self.config.piece_bytes,
            duration_s=duration_s,
            width=width,
            height=height,
            fps=fps,
            prompt=prompt,
        ):
            self.emit(piece)
        self.keyframe = await asyncio.to_thread(last_frame, data)
        self.emit(video_progress_chunk(call.id, clip, "done", 100))
        return {
            "clip": clip,
            "status": "playing for the user",
            "seconds": round(duration_s, 1),
            "prompt": prompt,
            "continues": keyframe is not None,
        }

    def request(self, prompt: str, keyframe: bytes | None) -> dict[str, object]:
        """The job request: text to video and audio for a new scene, the keyframe as the first frame of the clip otherwise."""
        body: dict[str, object] = {
            "model": self.config.model,
            "prompt": prompt,
            "seconds": self.config.seconds,
            "task": "t2va" if keyframe is None else "fl2va",
            "conditions": (
                []
                if keyframe is None
                else [
                    {
                        "type": "image",
                        "uri": f"data:{KEYFRAME_MIME_TYPE};base64,{base64.b64encode(keyframe).decode('ascii')}",
                        "role": "keyframe",
                        "frame_index": 0,
                    }
                ]
            ),
            "target": {
                "short_edge": self.config.short_edge,
                "aspect_ratio": (
                    self.config.aspect_ratio if keyframe is None else "auto"
                ),
                "duration_seconds": float(self.config.seconds),
            },
            "num_outputs_per_prompt": 1,
        }
        if self.config.num_inference_steps is not None:
            body["num_inference_steps"] = self.config.num_inference_steps
        else:
            pass
        return body
