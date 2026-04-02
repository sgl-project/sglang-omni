# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o speech pipeline: vision + audio input/output.

MiniCPM-o extends MiniCPM-V with full audio support:
- Audio input via Whisper encoder
- Audio output via CosyVoice/CosyVoice2 vocoder

Supports both MiniCPM-o 2.6 and 4.5:
- 2.6: SigLIP-400M + Whisper + CosyVoice + MiniCPM-3.0
- 4.5: SigLIP2-400M + Whisper-medium + CosyVoice2 + Qwen3-8B

Usage::

    # Basic text-to-speech
    python examples/run_minicpm_o_speech.py \\
        --prompt "Hello! Tell me about quantum computing."

    # With image input
    python examples/run_minicpm_o_speech.py \\
        --prompt "Describe what you see in this image." \\
        --image tests/data/cars.jpg

    # With audio input
    python examples/run_minicpm_o_speech.py \\
        --prompt "Please transcribe and respond to this audio." \\
        --audio tests/data/cough.wav

    # Save output audio
    python examples/run_minicpm_o_speech.py \\
        --prompt "Tell me a short story." \\
        --output story.wav

    # Custom GPU mapping
    python examples/run_minicpm_o_speech.py \\
        --prompt "Hello!" \\
        --gpu-image 0 --gpu-audio 0 --gpu-llm 1 --gpu-vocoder 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import os
import time
from pathlib import Path

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Model
    parser.add_argument(
        "--model-path",
        type=str,
        default="openbmb/MiniCPM-o-2_6",
        help="Hugging Face model id or local path (supports 2.6 and 4.5)",
    )

    # Input options
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello! Tell me something interesting about AI.",
        help="Text prompt for the model",
    )
    parser.add_argument(
        "--system",
        type=str,
        default="You are a helpful assistant. Respond naturally and conversationally.",
        help="System prompt",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to input image (optional)",
    )
    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="Path to input audio file (optional)",
    )

    # Generation parameters
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)

    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save output audio to WAV file (default: print result only)",
    )
    parser.add_argument(
        "--output-sample-rate",
        type=int,
        default=22050,
        help="Output audio sample rate (default: 22050 for CosyVoice)",
    )

    # GPU placement
    parser.add_argument("--gpu-image", type=int, default=0, help="GPU for image encoder")
    parser.add_argument("--gpu-audio", type=int, default=0, help="GPU for audio encoder")
    parser.add_argument("--gpu-llm", type=int, default=0, help="GPU for LLM backbone")
    parser.add_argument("--gpu-vocoder", type=int, default=0, help="GPU for vocoder")

    # Pipeline options
    parser.add_argument(
        "--relay-backend",
        type=str,
        default="shm",
        choices=["shm", "nixl"],
        help="Relay backend for inter-stage data transfer",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Pipeline execution timeout in seconds",
    )

    return parser.parse_args()


def _load_image(image_path: str) -> str:
    """Load image and convert to base64 data URL."""
    import base64
    import mimetypes

    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    mime_type = mimetypes.guess_type(str(path))[0] or "image/jpeg"
    with open(path, "rb") as f:
        b64_data = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64_data}"


def _load_audio(audio_path: str) -> dict:
    """Load audio file and return as dict for pipeline."""
    path = Path(audio_path)
    if not path.exists():
        raise FileNotFoundError(f"Audio not found: {audio_path}")

    return {"path": str(path)}


def _save_audio(result: dict, output_path: str, sample_rate: int) -> None:
    """Extract audio waveform from pipeline result and save as WAV."""
    import wave

    import numpy as np

    # Search through pipeline result for audio data
    for stage_name, payload in result.items():
        data = getattr(payload, "data", None)
        if not isinstance(data, dict):
            continue

        waveform = data.get("audio_waveform") or data.get("audio_samples")
        if waveform is None:
            continue

        import torch

        if isinstance(waveform, bytes):
            # Deserialize from raw bytes
            dtype_str = data.get("audio_waveform_dtype", "float32")
            shape = data.get("audio_waveform_shape", [-1])
            waveform = np.frombuffer(waveform, dtype=np.dtype(dtype_str)).reshape(shape)
        elif isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().float().numpy()

        waveform = waveform.squeeze()
        actual_sr = data.get("sample_rate", sample_rate)

        # Normalize and convert to int16
        peak = max(abs(waveform.max()), abs(waveform.min()), 1e-8)
        waveform_int16 = (waveform / peak * 32767).astype(np.int16)

        with wave.open(output_path, "w") as f:
            f.setnchannels(1)
            f.setsampwidth(2)
            f.setframerate(actual_sr)
            f.writeframes(waveform_int16.tobytes())

        duration = len(waveform_int16) / actual_sr
        logger.info(f"Audio saved: {output_path} ({duration:.2f}s, {actual_sr} Hz)")
        return

    logger.warning("No audio waveform found in pipeline result")


async def main_async(args: argparse.Namespace) -> None:
    from sglang_omni.models.minicpm_v.components.common import (
        get_minicpm_version,
        has_audio_support,
    )
    from sglang_omni.models.minicpm_v.config import MiniCPMOPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    # Verify model supports audio
    if not has_audio_support(args.model_path):
        raise ValueError(
            f"Model {args.model_path} does not support audio. "
            "Use MiniCPM-o (not MiniCPM-V) for audio capabilities."
        )

    version = get_minicpm_version(args.model_path)
    logger.info(f"Loading MiniCPM-o version {version}")

    # Build pipeline config
    config = MiniCPMOPipelineConfig(model_path=args.model_path)

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting MiniCPM-o speech pipeline...")
    await runner.start(timeout=600)

    try:
        # Build request with optional multimodal inputs
        messages = [
            {"role": "system", "content": args.system},
            {"role": "user", "content": args.prompt},
        ]

        request_inputs: dict = {
            "messages": messages,
            "images": [],
            "audios": [],
        }

        # Add image if provided
        if args.image:
            image_data = _load_image(args.image)
            request_inputs["images"].append(image_data)
            logger.info(f"Added image: {args.image}")

        # Add audio if provided
        if args.audio:
            audio_data = _load_audio(args.audio)
            request_inputs["audios"].append(audio_data)
            logger.info(f"Added audio: {args.audio}")

        t0 = time.time()
        result = await asyncio.wait_for(
            runner.coordinator.submit(
                "speech-request",
                OmniRequest(
                    inputs=request_inputs,
                    params={
                        "max_new_tokens": args.max_new_tokens,
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    },
                ),
            ),
            timeout=args.timeout,
        )
        duration = time.time() - t0
        logger.info(f"Pipeline completed in {duration:.2f}s")

        # Extract text response
        if isinstance(result, dict):
            for stage_name, payload in result.items():
                data = getattr(payload, "data", None)
                if isinstance(data, dict):
                    text = data.get("text") or data.get("response_text")
                    if text:
                        print(f"\n{'='*60}")
                        print(f"Response: {text}")
                        print(f"{'='*60}\n")
                        break

        # Save audio if requested
        if args.output and isinstance(result, dict):
            _save_audio(result, args.output, args.output_sample_rate)

    finally:
        await runner.stop()


def main() -> None:
    mp.set_start_method("spawn", force=True)
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
