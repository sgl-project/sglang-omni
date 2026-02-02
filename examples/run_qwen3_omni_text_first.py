# SPDX-License-Identifier: Apache-2.0
"""Text-first split pipeline for Qwen3-Omni."""

from __future__ import annotations

import argparse
import asyncio
import time

from sglang_omni.config import PipelineRunner, compile_pipeline
from sglang_omni.models.qwen3_omni import create_text_first_pipeline_config
from sglang_omni.models.qwen3_omni.pipeline.stages import preload_torch_weights
from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.proto import OmniRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-path",
        "--model-id",
        dest="model_path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Local model path or Hugging Face model id",
    )
    parser.add_argument("--prompt", type=str, default="Describe both the image and the audio content in detail.")
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--thinker-max-seq-len", type=int, default=81920)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--frontend-device", type=str, default="cpu")
    parser.add_argument("--image-device", type=str, default="cuda:3")
    parser.add_argument("--audio-device", type=str, default="cuda:3")
    parser.add_argument("--thinker-device", type=str, default="cuda:3")
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Only load models from the local HF cache (no downloads).",
    )
    parser.add_argument("--image-path", type=str, default=None)
    parser.add_argument("--audio-path", type=str, default=None)
    parser.add_argument("--audio-target-sr", type=int, default=16000)
    parser.add_argument(
        "--backend",
        type=str,
        default="hf",
        choices=["hf", "torch", "torch_native", "native"],
        help="Model backend: hf (default) or torch-native.",
    )
    return parser.parse_args()


async def main_async(args: argparse.Namespace) -> None:
    model_path = resolve_model_path(
        args.model_path, local_files_only=args.local_files_only
    )
    t0 = time.perf_counter()
    if args.backend.lower() in {"torch", "torch_native", "native"}:
        print("[timing] preloading weights ...")
        preload_torch_weights(
            model_path=str(model_path),
            image_device=args.image_device,
            audio_device=args.audio_device,
            thinker_device=args.thinker_device,
            dtype=args.dtype,
        )
        print(f"[timing] preload done  ({time.perf_counter() - t0:.1f}s)")
    config = create_text_first_pipeline_config(
        model_path=str(model_path),
        frontend_device=args.frontend_device,
        image_device=args.image_device,
        audio_device=args.audio_device,
        thinker_device=args.thinker_device,
        thinker_max_seq_len=args.thinker_max_seq_len,
        dtype=args.dtype,
        backend=args.backend,
    )
    print(f"[timing] compiling pipeline ...")
    coordinator, stages = compile_pipeline(config)
    print(f"[timing] compile done  ({time.perf_counter() - t0:.1f}s)")
    runner = PipelineRunner(coordinator, stages)

    print(f"[timing] starting pipeline ...")
    await runner.start()
    print(f"[timing] pipeline started  ({time.perf_counter() - t0:.1f}s)")
    try:
        images = [args.image_path] if args.image_path else []
        audios = [args.audio_path] if args.audio_path else []
        request = {
            "messages": [
                {"role": "user", "content": args.prompt},
            ],
            "images": images,
            "audios": audios,
            "audio_target_sr": args.audio_target_sr,
        }
        result = await coordinator.submit(
            "qwen3-omni-text-first",
            OmniRequest(
                inputs=request,
                params={
                    "max_new_tokens": args.max_new_tokens,
                    "temperature": args.temperature,
                },
            ),
        )
        print(result)
    finally:
        await runner.stop()


def main() -> None:
    args = parse_args()
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
