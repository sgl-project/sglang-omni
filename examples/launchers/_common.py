# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import asyncio
import logging
import time
import wave
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LauncherPreset:
    description: str
    build_parser: Callable[[], argparse.ArgumentParser]
    run: Callable[[argparse.Namespace], None]
    spawn: bool = False
    default_log_level: str = "INFO"


def parser(description: str) -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )


def add_model_path(
    target: argparse.ArgumentParser,
    default: str,
) -> None:
    target.add_argument(
        "--model-path",
        type=str,
        default=default,
        help="Hugging Face model id or local path",
    )


def add_mem_fraction(
    target: argparse.ArgumentParser,
    help_text: str,
) -> None:
    target.add_argument(
        "--mem-fraction-static",
        type=float,
        default=None,
        help=help_text,
    )


def add_server_args(
    target: argparse.ArgumentParser,
    *,
    model_name: str | None,
) -> None:
    target.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind host (default: 0.0.0.0).",
    )
    target.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server bind port (default: 8000).",
    )
    target.add_argument(
        "--model-name",
        type=str,
        default=model_name,
        help="Model name exposed by /v1/models.",
    )


def add_offline_args(
    target: argparse.ArgumentParser,
    *,
    prompt: str,
    system: str,
    max_new_tokens: int,
    output: str | None,
) -> None:
    target.add_argument("--prompt", type=str, default=prompt, help="User prompt.")
    target.add_argument("--system", type=str, default=system, help="System prompt.")
    target.add_argument(
        "--max-new-tokens",
        type=int,
        default=max_new_tokens,
        help=f"Maximum generated tokens (default: {max_new_tokens}).",
    )
    target.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    output_help = "Output WAV path; omit to skip saving audio."
    if output is not None:
        output_help = f"Output WAV path (default: {output})."
    target.add_argument("--output", type=str, default=output, help=output_help)
    target.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="Pipeline request timeout in seconds (default: 300).",
    )


def validate_fraction(flag_name: str, value: float | None) -> None:
    if value is not None and not 0.0 < value < 1.0:
        raise ValueError(f"{flag_name} must be > 0 and < 1, got {value}")


def apply_stage_factory_updates(
    config: Any,
    *,
    stage_name: str,
    updates: dict[str, object] | None = None,
    server_arg_updates: dict[str, object] | None = None,
) -> None:
    for stage in config.stages:
        if stage.name != stage_name:
            continue
        factory_args = dict(stage.factory_args or {})
        if updates:
            factory_args.update(updates)
        if server_arg_updates:
            overrides = dict(factory_args.get("server_args_overrides") or {})
            overrides.update(server_arg_updates)
            factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args
        return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def set_stage_gpu(
    config: Any,
    stage_name: str,
    gpu_id: int | list[int],
) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.gpu = gpu_id
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def set_stage_tp_size(config: Any, stage_name: str, tp_size: int) -> None:
    for stage in config.stages:
        if stage.name == stage_name:
            stage.tp_size = int(tp_size)
            stage.parallelism = stage.parallelism.model_copy(
                update={"tp": int(tp_size)}
            )
            return
    raise ValueError(
        f"Stage {stage_name!r} not found in config {type(config).__name__}"
    )


def save_audio(result: dict[str, Any], output_path: str) -> None:
    import numpy as np
    import torch

    for payload in result.values():
        data = payload if isinstance(payload, dict) else payload.data
        if not isinstance(data, dict):
            continue
        waveform = data.get("audio_waveform")
        if waveform is None:
            continue
        if isinstance(waveform, bytes):
            dtype = np.dtype(data.get("audio_waveform_dtype", "float32"))
            shape = data.get("audio_waveform_shape", [-1])
            waveform = np.frombuffer(waveform, dtype=dtype).reshape(shape)
        elif isinstance(waveform, torch.Tensor):
            waveform = waveform.cpu().float().numpy()
        waveform = waveform.squeeze()
        sample_rate = data.get("sample_rate", 24000)
        peak = max(abs(waveform.max()), abs(waveform.min()), 1e-8)
        waveform_int16 = (waveform / peak * 32767).astype(np.int16)
        with wave.open(output_path, "w") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(waveform_int16.tobytes())
        logger.info(
            "Audio saved: %s (%.2fs, %d Hz)",
            output_path,
            len(waveform_int16) / sample_rate,
            sample_rate,
        )
        return
    logger.warning("No audio waveform found in pipeline result")


async def run_speech_request(
    config: Any,
    *,
    request: dict[str, object],
    max_new_tokens: int,
    temperature: float,
    timeout: float,
    output: str | None,
    label: str,
) -> None:
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    runner = MultiProcessPipelineRunner(config)
    logger.info("Starting %s speech pipeline...", label)
    await runner.start(timeout=600)
    try:
        started = time.time()
        result = await asyncio.wait_for(
            runner.coordinator.submit(
                "speech-request",
                OmniRequest(
                    inputs=request,
                    params={
                        "max_new_tokens": max_new_tokens,
                        "temperature": temperature,
                    },
                ),
            ),
            timeout=timeout,
        )
        logger.info("Pipeline completed in %.2fs", time.time() - started)
        if output and isinstance(result, dict):
            save_audio(result, output)
    finally:
        await runner.stop()


def run_async(
    handler: Callable[[argparse.Namespace], Any],
    args: argparse.Namespace,
) -> None:
    asyncio.run(handler(args))
