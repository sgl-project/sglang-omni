# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni speech pipeline: text + audio output.

Usage::

    python examples/run_qwen3_omni_speech.py \
        --prompt "Tell me about what makes a beautiful sunset."

    # With custom GPU mapping:
    python examples/run_qwen3_omni_speech.py \
        --prompt "Hello, how are you?" \
        --gpu-thinker 0 --gpu-talker 1 --gpu-code-predictor 2

    # Save audio to file:
    python examples/run_qwen3_omni_speech.py \
        --prompt "Read me a bedtime story." \
        --output audio.wav
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import time

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--model-path", type=str, default="Qwen/Qwen3-Omni-30B-A3B-Instruct"
    )
    parser.add_argument(
        "--prompt", type=str, default="Hello! Tell me something interesting."
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a friendly assistant. Speak naturally and warmly.",
    )
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save audio to WAV file (default: print result only)",
    )
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--gpu-code-predictor", type=int, default=2)
    parser.add_argument("--gpu-code2wav", type=int, default=0)
    parser.add_argument("--gpu-image-encoder", type=int, default=0)
    parser.add_argument("--gpu-audio-encoder", type=int, default=0)
    parser.add_argument("--timeout", type=float, default=300.0)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    from sglang_omni.models.qwen3_omni.config import Qwen3OmniSpeechPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    # Build GPU placement from CLI args
    gpu_placement = {
        "thinker": args.gpu_thinker,
        "talker_ar": args.gpu_talker,
        "code_predictor": args.gpu_code_predictor,
        "code2wav": args.gpu_code2wav,
    }

    config = Qwen3OmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
        gpu_placement=gpu_placement,
    )
    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting 9-stage speech pipeline...")
    await runner.start(timeout=600)

    try:
        request = {
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": args.prompt},
            ],
            "images": [],
            "videos": [],
            "audios": [],
        }

        t0 = time.time()
        result = await asyncio.wait_for(
            runner.coordinator.submit(
                "speech-request",
                OmniRequest(
                    inputs=request,
                    params={
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                    },
                ),
            ),
            timeout=args.timeout,
        )
        duration = time.time() - t0
        logger.info("Pipeline completed in %.2fs", duration)

        # Extract and save audio if requested
        if args.output and isinstance(result, dict):
            _save_audio(result, args.output)

    finally:
        await runner.stop()


def _save_audio(result: dict, output_path: str) -> None:
    """Extract audio waveform from pipeline result and save as WAV."""
    import wave

    import numpy as np

    for stage_name, payload in result.items():
        data = getattr(payload, "data", None)
        if not isinstance(data, dict):
            continue
        waveform = data.get("audio_waveform")
        if waveform is None:
            continue

        import torch

        if isinstance(waveform, bytes):
            # code2wav serializes as raw bytes + shape/dtype metadata
            dtype_str = data.get("audio_waveform_dtype", "float32")
            shape = data.get("audio_waveform_shape", [-1])
            waveform = np.frombuffer(waveform, dtype=np.dtype(dtype_str)).reshape(shape)
        elif isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().float().numpy()

        waveform = waveform.squeeze()
        sample_rate = data.get("sample_rate", 24000)

        # Normalize and convert to int16
        peak = max(abs(waveform.max()), abs(waveform.min()), 1e-8)
        waveform_int16 = (waveform / peak * 32767).astype(np.int16)

        with wave.open(output_path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(sample_rate)
            f.writeframes(waveform_int16.tobytes())

        logger.info(
            "Audio saved: %s (%.2fs, %d Hz)",
            output_path,
            len(waveform_int16) / sample_rate,
            sample_rate,
        )
        return

    logger.warning("No audio waveform found in pipeline result")


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
