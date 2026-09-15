# SPDX-License-Identifier: Apache-2.0
"""Generate and verify Cosmos3-Super images through native, SDK, or HTTP.

Run this file from the repository root. The defaults are intentionally shared
by all three modes so their decoded pixels can be compared directly.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import time
from pathlib import Path
from typing import Any

MODEL_ID = "nvidia/Cosmos3-Super"
MODEL_REVISION = "fe77b66696d645f663b8f27e942b3b43e4629e23"
MODEL_SPEC = f"{MODEL_ID}@{MODEL_REVISION}"
DEFAULT_CONFIG = Path(__file__).parent / "configs/cosmos3_super_t2i_1xa100_bf16.yaml"
DEFAULT_PROMPT = (
    "A small red warehouse robot carefully folds a blue cloth on a clean "
    "wooden workbench, soft daylight, realistic industrial photography."
)
DEFAULT_NEGATIVE_PROMPT = "blurry, distorted, low quality, text, watermark"
SUPPORTED_RESOLUTIONS = {
    (1280, 720),
    (720, 1280),
    (832, 480),
    (480, 832),
    (1024, 1024),
    (640, 640),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("direct", "sdk", "http"))
    parser.add_argument("--model-path", default=MODEL_SPEC)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--server-url", default="http://127.0.0.1:30010")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/cosmos3_super")
    )
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--negative-prompt", default=DEFAULT_NEGATIVE_PROMPT)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--steps", type=int, default=35)
    parser.add_argument("--guidance-scale", type=float, default=6.0)
    parser.add_argument("--flow-shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--startup-timeout", type=float, default=7200.0)
    return parser.parse_args()


def generation_params(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": 1,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "use_resolution_template": False,
        "use_system_prompt": False,
        "use_guardrails": False,
    }


def verify_image(
    path: Path,
    *,
    args: argparse.Namespace,
    mode: str,
    latency_s: float,
    native_metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    from PIL import Image, ImageStat

    with Image.open(path) as image:
        image.load()
        image = image.convert("RGB")
        if image.size != (args.width, args.height):
            raise RuntimeError(
                f"Expected {args.width}x{args.height}, got {image.width}x{image.height}"
            )
        stats = ImageStat.Stat(image)
        channel_stddev = [round(value, 4) for value in stats.stddev]
        entropy = round(image.entropy(), 4)
        if entropy < 1.0 or max(channel_stddev) < 2.0:
            raise RuntimeError("Generated image is nearly uniform")
        pixel_digest = hashlib.sha256(image.tobytes()).hexdigest()

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    _, separator, declared_revision = args.model_path.partition("@")
    manifest = {
        "mode": mode,
        "model": MODEL_ID,
        "declared_model_spec": args.model_path,
        "checkpoint_revision": declared_revision if separator else None,
        "prompt": args.prompt,
        "negative_prompt": args.negative_prompt,
        "width": args.width,
        "height": args.height,
        "num_frames": 1,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "latency_s": round(latency_s, 3),
        "bytes": path.stat().st_size,
        "sha256": digest,
        "pixel_sha256": pixel_digest,
        "entropy": entropy,
        "channel_stddev": channel_stddev,
        "path": str(path.resolve()),
    }
    if native_metadata:
        manifest["native"] = json.loads(json.dumps(native_metadata, default=str))
    manifest_path = path.with_suffix(path.suffix + ".json")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)
    return manifest


def _native_model_args(model_path: str) -> dict[str, Any]:
    repo_id, separator, revision = model_path.partition("@")
    return {
        "model_path": repo_id,
        "revision": revision if separator else None,
        "served_model_name": repo_id,
        "num_gpus": 1,
        "tp_size": 1,
        "performance_mode": "memory",
        "warmup_mode": "off",
        "attention_backend": "torch_sdpa",
        "enable_torch_compile": False,
        "layerwise_offload_components": ["dit"],
        "dit_layerwise_resident_layers": 24,
    }


def http_payload(args: argparse.Namespace) -> dict[str, Any]:
    """Build JSON for raw HTTP; OpenAI SDK ``extra_body`` is client-only."""
    return {
        "model": MODEL_ID,
        "prompt": args.prompt,
        "size": f"{args.width}x{args.height}",
        "n": 1,
        "num_inference_steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "flow_shift": args.flow_shift,
        "seed": args.seed,
        "response_format": "b64_json",
        "negative_prompt": args.negative_prompt,
        "num_frames": 1,
        "use_resolution_template": False,
        "use_system_prompt": False,
        "use_guardrails": False,
    }


def run_direct(args: argparse.Namespace) -> list[dict[str, Any]]:
    from sglang.multimodal_gen.runtime.entrypoints.diffusion_generator import (
        DiffGenerator,
    )
    from sglang.multimodal_gen.runtime.server_args import ServerArgs

    args.output_dir.mkdir(parents=True, exist_ok=True)
    generator = DiffGenerator.from_server_args(
        ServerArgs.from_kwargs(**_native_model_args(args.model_path))
    )
    manifests = []
    try:
        for index in range(args.repeats):
            params = generation_params(args)
            params.update(
                save_output=True,
                return_file_paths_only=True,
                output_path=str(args.output_dir.resolve()),
                output_file_name=f"direct-{index}",
            )
            started = time.perf_counter()
            result = generator.generate(sampling_params_kwargs=params)
            latency = time.perf_counter() - started
            if isinstance(result, list):
                result = result[0] if result else None
            if result is None or not result.output_file_path:
                raise RuntimeError("Native SGLang returned no saved image")
            manifests.append(
                verify_image(
                    Path(result.output_file_path),
                    args=args,
                    mode="direct",
                    latency_s=latency,
                    native_metadata={
                        "generation_time_s": result.generation_time,
                        "peak_memory_mb": result.peak_memory_mb,
                        "metrics": result.metrics,
                    },
                )
            )
    finally:
        generator.shutdown()
    return manifests


async def run_sdk(args: argparse.Namespace) -> list[dict[str, Any]]:
    from sglang_omni.client import Client, GenerateRequest, SamplingParams
    from sglang_omni.config.manager import ConfigManager
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner

    config = ConfigManager.from_file(str(args.config)).config
    config.model_path = args.model_path
    config.stages[0].factory.output_dir = str(args.output_dir)
    runner = MultiProcessPipelineRunner(config)
    started = time.perf_counter()
    await runner.start(timeout=args.startup_timeout)
    print(
        f"Omni SDK pipeline ready in {time.perf_counter() - started:.3f}s", flush=True
    )
    manifests = []
    try:
        client = Client(runner.coordinator)
        params = generation_params(args)
        params.pop("prompt")
        for index in range(args.repeats):
            request = GenerateRequest(
                model=MODEL_ID,
                prompt=args.prompt,
                sampling=SamplingParams(seed=args.seed),
                stage_params={"generation": params},
                stream=False,
            )
            started = time.perf_counter()
            result = await client.completion(
                request, request_id=f"cosmos3-super-sdk-{index}"
            )
            latency = time.perf_counter() - started
            if not result.media or not result.media[0].get("path"):
                raise RuntimeError("Omni SDK returned no saved image")
            manifests.append(
                verify_image(
                    Path(result.media[0]["path"]),
                    args=args,
                    mode="sdk",
                    latency_s=latency,
                    native_metadata=result.media[0],
                )
            )
    finally:
        await runner.stop()
    return manifests


def run_http(args: argparse.Namespace) -> list[dict[str, Any]]:
    import requests

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifests = []
    for index in range(args.repeats):
        payload = http_payload(args)
        started = time.perf_counter()
        response = requests.post(
            f"{args.server_url.rstrip('/')}/v1/images/generations",
            json=payload,
            timeout=args.startup_timeout,
        )
        latency = time.perf_counter() - started
        response.raise_for_status()
        response_payload = response.json()
        item = response_payload["data"][0]
        encoded = item.get("b64_json")
        if not encoded:
            raise RuntimeError("HTTP response did not contain data[0].b64_json")
        path = args.output_dir / f"http-{index}.png"
        path.write_bytes(base64.b64decode(encoded, validate=True))
        manifests.append(
            verify_image(
                path,
                args=args,
                mode="http",
                latency_s=latency,
                native_metadata={
                    key: response_payload[key]
                    for key in ("inference_time_s", "peak_memory_mb")
                    if key in response_payload
                },
            )
        )
    return manifests


def main() -> None:
    args = parse_args()
    if args.repeats < 1:
        raise SystemExit("--repeats must be positive")
    if (args.width, args.height) not in SUPPORTED_RESOLUTIONS:
        supported = ", ".join(
            f"{width}x{height}" for width, height in sorted(SUPPORTED_RESOLUTIONS)
        )
        raise SystemExit(
            f"Cosmos3 does not support {args.width}x{args.height}; choose {supported}"
        )
    if args.mode == "direct":
        run_direct(args)
    elif args.mode == "sdk":
        asyncio.run(run_sdk(args))
    else:
        run_http(args)


if __name__ == "__main__":
    main()
