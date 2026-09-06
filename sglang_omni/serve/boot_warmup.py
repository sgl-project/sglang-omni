# SPDX-License-Identifier: Apache-2.0
"""Boot-time full-pipeline warmup burst.

A warmup driven from a client can only reach batch-1 shapes: it issues one
request and waits for the reply. Kernels, CUDA-graph buckets and allocator
blocks first touched at batch>1 are therefore still cold when the first real
admission wave arrives, and that wave pays the first-touch cost in its
time-to-first-audio. This module pays it before the HTTP port opens, by
driving a burst of synthetic speech requests through the same in-process
``Client`` the API adapters use.
"""

from __future__ import annotations

import asyncio
import logging
import tempfile
import time
import wave
from contextlib import aclosing
from pathlib import Path

import numpy as np

from sglang_omni.client.client import Client
from sglang_omni.client.types import GenerateRequest, Message, SamplingParams

logger = logging.getLogger(__name__)

_SAMPLE_RATE = 16000
_REFERENCE_SECONDS = 4.0
_BASE_TONE_HZ = 220.0
# Note (wenyao): Talker startup and vocoder initial chunks finish within
# a few frames, so this short burst reaches their steady-state shapes.
_TALKER_NEW_TOKENS = 32
_MAX_NEW_TOKENS = 64
_TIMEOUT_S = 180.0


def _write_reference_wav(path: Path, tone_hz: float) -> None:
    frames = int(_SAMPLE_RATE * _REFERENCE_SECONDS)
    t = np.arange(frames, dtype=np.float32) / _SAMPLE_RATE
    samples = (8000.0 * np.sin(2.0 * np.pi * tone_hz * t)).astype("<i2")
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(samples.tobytes())


def _build_request(model_name: str, reference_wav: str, index: int) -> GenerateRequest:
    # Note (wenyao): Reusing reference audio or text hits the encoder/prefix
    # caches, preventing the whole warmup burst from reaching those stages.
    prompt = (
        f'Listen to the audio above. The speaker is reading: "warmup sample '
        f'number {index}". Now please read the following text out loud in the '
        f"same voice and style: This is warmup utterance {index} of the "
        f"synthetic boot-time burst."
    )
    return GenerateRequest(
        model=model_name,
        messages=[Message(role="user", content=prompt)],
        sampling=SamplingParams(temperature=0.0, max_new_tokens=_MAX_NEW_TOKENS),
        extra_params={
            "talker_min_new_tokens": _TALKER_NEW_TOKENS,
            "talker_max_new_tokens": _TALKER_NEW_TOKENS,
        },
        stream=True,
        max_tokens=_MAX_NEW_TOKENS,
        output_modalities=["text", "audio"],
        metadata={"audios": [reference_wav], "audio_config": {"format": "wav"}},
    )


async def _drain(client: Client, request: GenerateRequest, request_id: str) -> bool:
    try:
        stream = client.generate(request, request_id=request_id)
        async with aclosing(stream):
            async for _ in stream:
                pass
    except Exception:
        logger.warning("Boot warmup request %s failed", request_id, exc_info=True)
        return False
    return True


async def run_boot_warmup(
    client: Client,
    *,
    model_name: str,
    num_requests: int,
) -> None:
    """Drive ``num_requests`` synthetic speech requests, all in flight at once.

    The burst width is the point: it is what puts each stage on a batch>1
    shape. Failures are logged and swallowed -- a warmup that cannot run must
    not stop the server from serving.
    """
    if num_requests <= 0:
        return

    started = time.perf_counter()
    logger.info("Boot warmup: issuing %d synthetic speech request(s)", num_requests)
    try:
        reference_dir = tempfile.TemporaryDirectory(
            prefix="sglang-omni-boot-warmup-", ignore_cleanup_errors=True
        )
    except OSError:
        logger.warning(
            "Boot warmup skipped: could not create temporary inputs", exc_info=True
        )
        return
    with reference_dir as tmpdir:
        try:
            requests = []
            for index in range(num_requests):
                path = Path(tmpdir) / f"reference-{index}.wav"
                _write_reference_wav(path, _BASE_TONE_HZ + 20.0 * index)
                requests.append(_build_request(model_name, str(path), index))
        except Exception:
            logger.warning(
                "Boot warmup skipped: could not build the synthetic inputs",
                exc_info=True,
            )
            return

        try:
            outcomes = await asyncio.wait_for(
                asyncio.gather(
                    *(
                        _drain(client, request, f"boot-warmup-{index}")
                        for index, request in enumerate(requests)
                    )
                ),
                timeout=_TIMEOUT_S,
            )
        except asyncio.TimeoutError:
            logger.warning(
                "Boot warmup did not finish within %.0fs; serving anyway", _TIMEOUT_S
            )
            return

    completed = sum(outcomes)
    logger.info(
        "Boot warmup complete: %d/%d request(s) in %.1fs",
        completed,
        num_requests,
        time.perf_counter() - started,
    )
    if completed == 0:
        logger.warning(
            "Boot warmup: every request failed; the first admission wave will "
            "still pay the cold-start ramp"
        )
