# SPDX-License-Identifier: Apache-2.0
"""Stream audio files through one Session of a streaming cascade launch config."""

import argparse
import asyncio
import wave
from dataclasses import replace
from pathlib import Path

import numpy as np

from sglang_omni.config.manager import ConfigManager
from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
from sglang_omni.proto import OmniRequest
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    COMMIT,
    CONTROL,
    PCM16,
    PCM16_BYTES_PER_SAMPLE,
    SESSION_CONTROL,
    SESSION_INPUT_SAMPLE_RATE_HZ,
)

INPUT_BYTES_PER_MS = PCM16_BYTES_PER_SAMPLE * SESSION_INPUT_SAMPLE_RATE_HZ // 1000


def read_pcm16(path: Path) -> bytes:
    with wave.open(str(path), "rb") as source:
        if (source.getnchannels(), source.getsampwidth(), source.getframerate()) != (
            1,
            2,
            16000,
        ):
            raise ValueError(f"{path} must be mono 16-bit PCM at 16000 Hz")
        else:
            return source.readframes(source.getnframes())


async def run(
    config_path: Path,
    audio_paths: list[Path],
    output_path: Path,
    *,
    commit: bool = False,
) -> None:
    config = ConfigManager.from_file(str(config_path)).config
    unit_bytes = config.unit_ms * INPUT_BYTES_PER_MS
    runner = MultiProcessPipelineRunner(config)
    await runner.start()
    try:
        coordinator = runner.coordinator
        identity = await coordinator.open_session(
            OmniRequest(None),
            limits=replace(
                config.session_limits(), operation_timeout_s=config.timeout_s * 2 + 10
            ),
        )
        outputs = coordinator.session_outputs(identity)
        acknowledged: asyncio.Queue[int] = asyncio.Queue()
        with wave.open(str(output_path), "wb") as output:
            output.setparams((1, 2, 24000, 0, "NONE", "not compressed"))

            async def receive() -> None:
                async for chunk in outputs:
                    if chunk.kind == "input_done":
                        acknowledged.put_nowait(chunk.input_seq)
                    elif chunk.modality == AUDIO and isinstance(chunk.payload, bytes):
                        pcm = np.frombuffer(chunk.payload, dtype="<f4")
                        output.writeframes(
                            (np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes()
                        )
                    else:
                        print(chunk.payload, flush=True)

            async def append(chunk: TimedChunk) -> None:
                await coordinator.append_session(identity, chunk)
                await acknowledged.get()

            async def send() -> None:
                sequence = 0
                start_ms = 0.0
                for path in audio_paths:
                    pcm = read_pcm16(path)
                    for offset in range(0, len(pcm), unit_bytes):
                        audio = pcm[offset : offset + unit_bytes]
                        duration_ms = len(audio) / INPUT_BYTES_PER_MS
                        await append(
                            TimedChunk(
                                AUDIO,
                                start_ms,
                                duration_ms,
                                sequence,
                                audio,
                                format=PCM16,
                            )
                        )
                        await asyncio.sleep(duration_ms / 1000)
                        sequence += 1
                        start_ms += duration_ms
                    if commit:
                        await append(
                            TimedChunk(
                                CONTROL,
                                start_ms,
                                0,
                                sequence,
                                {"type": COMMIT},
                                format=SESSION_CONTROL,
                            )
                        )
                        sequence += 1
                    else:
                        pass
                await append(
                    TimedChunk(
                        AUDIO, start_ms, 0, sequence, b"", format=PCM16, eos=True
                    )
                )

            receiver = asyncio.create_task(receive())
            sender = asyncio.create_task(send())
            try:
                done, _ = await asyncio.wait(
                    {sender, receiver}, return_when=asyncio.FIRST_COMPLETED
                )
                for task in done:
                    task.result()
            finally:
                sender.cancel()
                receiver.cancel()
                await asyncio.gather(sender, receiver, return_exceptions=True)
                await outputs.aclose()
                await coordinator.close_session(identity)
    finally:
        await runner.stop()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "audio",
        type=Path,
        nargs="+",
        help="16 kHz mono PCM WAV files, streamed back to back",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).parent
        / "profiles"
        / "qwen3-asr-qwen3-tts-vivian"
        / "launch.yaml",
    )
    parser.add_argument("--output", type=Path, default=Path("cascade-response.wav"))
    parser.add_argument(
        "--commit",
        action="store_true",
        help="End an utterance with a commit after each file, for a config without the VAD node",
    )
    arguments = parser.parse_args()
    asyncio.run(
        run(
            arguments.config,
            arguments.audio,
            arguments.output,
            commit=arguments.commit,
        )
    )


if __name__ == "__main__":
    main()
