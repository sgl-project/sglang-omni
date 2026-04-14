# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni speech pipeline: text + audio output.

The talker is a self-contained MingOmniTalker with its own LLM + CFM + AudioVAE,
generating speech from the thinker's decoded text.

Usage::

    python examples/run_ming_omni_speech.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --prompt "请给我讲一个故事。"

    # With custom GPU placement:
    python examples/run_ming_omni_speech.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --prompt "你好，今天天气怎么样？" \
        --gpu-thinker 0 --gpu-talker 1

    # Save audio to file:
    python examples/run_ming_omni_speech.py \
        --model-path inclusionAI/Ming-flash-omni-2.0 \
        --prompt "朗读一首古诗。" \
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
        "--model-path",
        type=str,
        default="inclusionAI/Ming-flash-omni-2.0",
        help="Hugging Face model id or local path",
    )
    parser.add_argument("--prompt", type=str, default="你好！给我讲一个有趣的事情。")
    parser.add_argument(
        "--system",
        type=str,
        default="你是一个友好的AI助手。请用自然、温暖的语气说话。",
    )
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument(
        "--relay-backend", type=str, default="shm", choices=["nixl", "shm"]
    )
    parser.add_argument("--audio-path", type=str, default=None, help="Audio input path")
    parser.add_argument(
        "--output",
        type=str,
        default="./output_audio.wav",
        help="Save audio to WAV file (default: ./output_audio.wav)",
    )
    parser.add_argument(
        "--voice", type=str, default="DB30", help="Voice ID for the talker"
    )
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=300.0)
    parser.add_argument("--cpu-offload-gb", type=float, default=0)
    parser.add_argument("--mem-fraction-static", type=float, default=None)
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    from sglang_omni.models.ming_omni.config import MingOmniSpeechPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    gpu_placement = {
        "thinker": args.gpu_thinker,
        "talker": args.gpu_talker,
    }

    overrides = {}
    if args.cpu_offload_gb:
        overrides["cpu_offload_gb"] = args.cpu_offload_gb
    if args.mem_fraction_static is not None:
        overrides["mem_fraction_static"] = args.mem_fraction_static

    config = MingOmniSpeechPipelineConfig(
        model_path=args.model_path,
        relay_backend=args.relay_backend,
        gpu_placement=gpu_placement,
        server_args_overrides=overrides if overrides else None,
    )
    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting Ming-Omni speech pipeline...")
    await runner.start(timeout=600)

    try:
        # Build message content
        if args.audio_path:
            content = [
                {"type": "audio_url", "audio_url": {"url": args.audio_path}},
                {"type": "text", "text": args.prompt},
            ]
        else:
            content = args.prompt

        request = {
            "messages": [
                {"role": "system", "content": args.system},
                {"role": "user", "content": content},
            ],
            "audios": [args.audio_path] if args.audio_path else [],
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

        if args.output and isinstance(result, dict):
            _save_audio(result, args.output)

    finally:
        await runner.stop()


def _save_audio(result: dict, output_path: str) -> None:
    """Extract audio waveform from pipeline result and save as WAV."""
    import wave

    import numpy as np

    for stage_name, payload in result.items():
        # Multi-terminal coordinator returns {stage_name: data_dict}
        # where data_dict is the raw dict (not a StagePayload object).
        if isinstance(payload, dict):
            data = payload
        else:
            data = getattr(payload, "data", None)
        if not isinstance(data, dict):
            continue
        waveform = data.get("audio_waveform")
        if waveform is None:
            continue

        import torch

        if isinstance(waveform, bytes):
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
