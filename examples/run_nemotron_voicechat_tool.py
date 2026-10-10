# SPDX-License-Identifier: Apache-2.0
"""Run VoiceChat with an external tool node as one two-node session graph.

The user's recording streams into the VoiceChat node in real time. When the
thinker writes a tool call on its function channel, the graph routes it to the
tool node, and the tool response is routed back into the thinker.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time
import wave

import numpy as np

from sglang_omni.admission import QueueFullError
from sglang_omni.client.client import Client
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    NemotronVoiceChatToolPipelineConfig,
)
from sglang_omni.models.nemotron_voicechat.payload_types import (
    INPUT_SAMPLE_RATE,
    OUTPUT_SAMPLE_RATE,
)
from sglang_omni.pipeline.graph import GraphSession
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import SessionLimits, TimedChunk
from sglang_omni.utils.audio import load_audio

INPUT_FRAME_SAMPLES = 1280
FRAME_DURATION_MS = INPUT_FRAME_SAMPLES * 1000 / INPUT_SAMPLE_RATE
PCM16_SCALE = 32767
# The first frames compile kernels and capture graphs; later frames take milliseconds.
FIRST_FRAME_TIMEOUT_S = 300


async def run(arguments: argparse.Namespace) -> None:
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
    # The model answers while it hears silence, so leave room after the last request.
    audio = np.concatenate(
        [audio, np.zeros(int(arguments.tail_seconds * INPUT_SAMPLE_RATE))]
    )
    frame_count = -(-audio.shape[0] // INPUT_FRAME_SAMPLES)
    audio = np.pad(audio, (0, frame_count * INPUT_FRAME_SAMPLES - audio.shape[0]))
    pcm = (np.clip(audio, -1, 1) * PCM16_SCALE).astype("<i2").tobytes()
    frame_bytes = INPUT_FRAME_SAMPLES * 2

    config = NemotronVoiceChatToolPipelineConfig(model_path=arguments.model_path)
    assert config.graph is not None
    runner = MultiProcessPipelineRunner(config)
    await runner.start(timeout=900)
    try:
        graph = await GraphSession.open(
            Client(runner.coordinator),
            config.graph,
            OmniRequest(inputs=None),
            limits=SessionLimits(operation_timeout_s=FIRST_FRAME_TIMEOUT_S),
        )
        reply_pcm: list[bytes] = []
        reply_text: list[str] = []
        started = time.perf_counter()

        async def consume() -> None:
            async for output in graph.outputs():
                elapsed_s = time.perf_counter() - started
                if output.modality == "tool_call":
                    print(
                        f"[{elapsed_s:6.2f}s] tool call: {output.payload}", flush=True
                    )
                elif isinstance(output.payload, dict):
                    reply_pcm.append(output.payload["pcm"])
                    reply_text.append(output.payload["text"])
                    if output.eos:
                        return
                    else:
                        pass
                else:
                    pass

        reader = asyncio.create_task(consume())
        for sequence in range(frame_count):
            chunk = TimedChunk(
                "audio",
                sequence * FRAME_DURATION_MS,
                FRAME_DURATION_MS,
                sequence,
                pcm[sequence * frame_bytes : (sequence + 1) * frame_bytes],
                format="pcm16",
                eos=sequence == frame_count - 1,
            )
            while True:
                try:
                    await graph.append(chunk)
                    break
                except QueueFullError:
                    # The first frames compile kernels; wait for the graph to drain.
                    await asyncio.sleep(FRAME_DURATION_MS / 1000)
            # Real-time pacing keeps the tool round trip on the same clock as the user.
            await asyncio.sleep(
                max(
                    0.0,
                    (sequence + 1) * FRAME_DURATION_MS / 1000
                    - (time.perf_counter() - started),
                )
            )
        await reader
        await graph.close()
    finally:
        await runner.stop()
    print(f"text: {''.join(reply_text)}")
    with wave.open(arguments.out, "wb") as output_file:
        output_file.setnchannels(1)
        output_file.setsampwidth(2)
        output_file.setframerate(OUTPUT_SAMPLE_RATE)
        output_file.writeframes(b"".join(reply_pcm))
    print(f"audio written to {arguments.out}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--audio", required=True, help="channel 0 is the user")
    parser.add_argument("--out", default="voicechat_tool_reply.wav")
    parser.add_argument("--seconds", type=float, default=None)
    parser.add_argument("--tail-seconds", type=float, default=8.0)
    # Stage processes inherit this level; INFO shows each tool result.
    logging.basicConfig(level=logging.INFO)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()
