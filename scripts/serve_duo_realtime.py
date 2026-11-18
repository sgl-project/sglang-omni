#!/usr/bin/env python3
"""Serve the Duo live pipeline through Omni's shared realtime API.

This launcher intentionally keeps the model placement explicit: Intent, Realization,
VAE, and optional ASR each receive one device.  The websocket protocol and
unit scheduling stay in the shared ``/v1/realtime`` implementation.
"""

from __future__ import annotations

import argparse
import asyncio
import gc
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

import msgspec
import uvicorn
from fastapi import FastAPI
from sglang.srt.utils import freeze_gc

from sglang_omni.models.duo.realtime import (
    build_duo_realtime_deployment,
    load_duo_live_resources,
    warmup_duo_realtime,
)
from sglang_omni.serve.openai_api import create_app


class _RealtimeOnlyClient:
    """Small health surface required by the shared OpenAI app.

    Realization work is executed by the deployment adapter, so no coordinator client is
    needed for this standalone launcher.  Keeping this object local also makes
    it clear that `/v1/realtime` is the serving boundary for this prototype.
    """

    def health(self) -> dict[str, bool]:
        return {"running": True}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--codec-model-path", dest="model_path", required=True)
    parser.add_argument(
        "--intent-model-path",
        dest="intent_model_path",
        required=True,
    )
    parser.add_argument(
        "--intent-adapter-path",
        dest="intent_adapter_path",
        default="",
    )
    parser.add_argument("--raven-lora-path", default="")
    parser.add_argument("--trainables-path", default="")
    parser.add_argument(
        "--realization-checkpoint",
        dest="realization_checkpoint",
        help="Duo Realization DCP checkpoint directory (5-to-17)",
    )
    parser.add_argument(
        "--realization-backend",
        dest="realization_backend",
        choices=("reference", "native"),
        default="reference",
    )
    parser.add_argument(
        "--realization-stage",
        dest="realization_stage",
        choices=("causal", "dmd"),
        default="causal",
    )
    parser.add_argument(
        "--realization-tp-size",
        dest="realization_tp_size",
        type=int,
        choices=(1, 2, 4, 8),
        default=1,
    )
    parser.add_argument(
        "--realization-sp-mode",
        dest="realization_sp_mode",
        choices=("ulysses", "ring"),
        default="ulysses",
    )
    parser.add_argument(
        "--realization-enable-cuda-graph",
        dest="realization_enable_cuda_graph",
        action="store_true",
    )
    parser.add_argument(
        "--realization-optimize-copy-sync",
        dest="realization_optimize_copy_sync",
        action="store_true",
    )
    parser.add_argument("--realization-merge-fixed-lora", action="store_true")
    parser.add_argument("--realization-fp8-block-linears", action="store_true")
    parser.add_argument(
        "--realization-fp8-direct-kernels",
        dest="realization_fp8_direct_kernels",
        action="store_true",
    )
    parser.add_argument(
        "--realization-modelopt-checkpoint",
        dest="realization_modelopt_checkpoint",
        help="Released per-channel FP8 Realization directory",
    )
    parser.add_argument(
        "--realization-fp8-attention-calibration",
        dest="realization_fp8_attention_calibration",
    )
    parser.add_argument(
        "--realization-fp8-attention-library",
        dest="realization_fp8_attention_library",
        help="Optional SM90 FP8 full-API FA3 shared library",
    )
    parser.add_argument("--realization-cuda-graph-max-entries", type=int, default=8)
    parser.add_argument(
        "--realization-cuda-graph-min-history-tokens", type=int, default=0
    )
    parser.add_argument(
        "--realization-sp-size",
        dest="realization_sp_size",
        type=int,
        choices=(1, 2, 4),
        default=1,
    )
    parser.add_argument(
        "--realization-worker-devices",
        dest="realization_worker_devices",
        type=int,
        nargs="+",
    )
    parser.add_argument(
        "--kernel-backend", choices=("sglang", "torch"), default="sglang"
    )
    parser.add_argument("--height", type=int, default=576)
    parser.add_argument("--width", type=int, default=768)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--flow-shift", type=float, default=1.0)
    parser.add_argument("--asr-model-path")
    parser.add_argument(
        "--asr-base-url", help="SGLang-Omni ASR service URL including /v1"
    )
    parser.add_argument("--asr-model-name", default="duo-asr")
    parser.add_argument(
        "--intent-backend",
        dest="intent_backend",
        choices=("reference", "native"),
        default="reference",
    )
    parser.add_argument("--intent-device", dest="intent_device", default="cuda:0")
    parser.add_argument(
        "--intent-tp-size",
        dest="intent_tp_size",
        type=int,
        choices=(1, 2, 4),
        default=1,
    )
    parser.add_argument("--intent-quantization", choices=("fp8",), default=None)
    parser.add_argument(
        "--intent-worker-devices",
        dest="intent_worker_devices",
        type=int,
        nargs="+",
    )
    parser.add_argument("--intent-vision-device")
    parser.add_argument(
        "--realization-device", dest="realization_device", default="cuda:1"
    )
    parser.add_argument(
        "--codec-device", "--vae-device", dest="vae_device", default="cuda:2"
    )
    parser.add_argument(
        "--codec-decode-backend",
        "--vae-decode-backend",
        dest="vae_decode_backend",
        choices=("native", "solarx"),
        default="native",
    )
    parser.add_argument("--vae-no-cuda-graph", action="store_true")
    parser.add_argument(
        "--codec-attention-backend",
        "--vae-attention-backend",
        dest="vae_attention_backend",
        choices=("torch_sdpa", "fa", "torch_cudnn_sdpa"),
        default="torch_sdpa",
    )
    parser.add_argument(
        "--codec-worker-devices",
        "--vae-worker-devices",
        dest="vae_worker_devices",
        type=int,
        nargs="+",
        help="two or four GPUs for native tiled 5-to-17 video VAE decode",
    )
    parser.add_argument("--asr-device", default="cuda:3")
    parser.add_argument("--max-blocks", type=int, default=360)
    parser.add_argument("--max-context-tokens", type=int, default=32768)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument(
        "--intent-kv-window-tokens",
        dest="intent_kv_window_tokens",
        type=int,
        help="native Intent prefix + recent complete chunks + reply budget; defaults to max-context-tokens",
    )
    parser.add_argument(
        "--realization-reference-window-chunks",
        dest="realization_reference_window_chunks",
        type=int,
        default=0,
        help="maximum current plus historical Realization blocks; zero uses only the token budget",
    )
    parser.add_argument(
        "--realization-kv-window-tokens",
        dest="realization_kv_window_tokens",
        type=int,
        default=65536,
        help="DCP reference/shared + clean history + current plan/media attention tokens",
    )
    parser.add_argument(
        "--realization-kv-workspace-capacity-tokens",
        dest="realization_kv_workspace_capacity_tokens",
        type=int,
        help="Reserve attention scratch independently of the logical Realization KV window",
    )
    parser.add_argument(
        "--overlap-intent-codec",
        "--overlap-intent-vae",
        dest="overlap_intent_codec",
        action="store_true",
    )
    parser.add_argument("--warmup-sample-directory", type=Path)
    parser.add_argument("--warmup-reference-image", type=Path, action="append")
    parser.add_argument(
        "--realization-kv-reset-interval-blocks",
        type=int,
        default=None,
        help="Reset DiT history after this many generated blocks. Omit to allow manual reset.",
    )
    parser.add_argument("--warmup-blocks", type=int, default=40)
    parser.add_argument("--warmup-description", type=Path)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8089)
    parser.add_argument(
        "--no-cuda-graph",
        action="store_true",
        help="disable native Intent and Raven Realization CUDA graphs (DCP Realization runs eager)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (args.warmup_sample_directory is None) != (args.warmup_reference_image is None):
        raise ValueError("Warmup requires both sample directory and reference image")
    else:
        pass
    if args.warmup_description is not None and args.warmup_sample_directory is None:
        raise ValueError("Warmup description requires warmup sample inputs")
    else:
        pass
    resources = load_duo_live_resources(
        model_path=args.model_path,
        intent_model_path=args.intent_model_path,
        intent_adapter_path=args.intent_adapter_path,
        raven_lora_path=args.raven_lora_path,
        trainables_path=args.trainables_path,
        intent_device=args.intent_device,
        intent_tp_size=args.intent_tp_size,
        intent_quantization=args.intent_quantization,
        intent_worker_devices=args.intent_worker_devices,
        intent_vision_device=args.intent_vision_device,
        realization_device=args.realization_device,
        vae_device=args.vae_device,
        vae_decode_backend=args.vae_decode_backend,
        vae_use_graph=not args.vae_no_cuda_graph,
        vae_attention_backend=args.vae_attention_backend,
        vae_worker_devices=args.vae_worker_devices,
        asr_device=args.asr_device,
        asr_model_path=args.asr_model_path,
        asr_base_url=args.asr_base_url,
        asr_model_name=args.asr_model_name,
        max_blocks=args.max_blocks,
        overlap_intent_codec=args.overlap_intent_codec,
        max_context_tokens=args.max_context_tokens,
        max_new_tokens=args.max_new_tokens,
        intent_kv_window_tokens=args.intent_kv_window_tokens,
        realization_kv_window_tokens=args.realization_kv_window_tokens,
        realization_reference_window_chunks=args.realization_reference_window_chunks,
        realization_kv_reset_interval_blocks=args.realization_kv_reset_interval_blocks,
        realization_kv_workspace_capacity_tokens=args.realization_kv_workspace_capacity_tokens,
        realization_enable_cuda_graph=args.realization_enable_cuda_graph,
        realization_optimize_copy_sync=args.realization_optimize_copy_sync,
        realization_merge_fixed_lora=args.realization_merge_fixed_lora,
        realization_fp8_block_linears=args.realization_fp8_block_linears,
        realization_modelopt_checkpoint=args.realization_modelopt_checkpoint,
        realization_fp8_direct_kernels=args.realization_fp8_direct_kernels,
        realization_fp8_attention_calibration=args.realization_fp8_attention_calibration,
        realization_fp8_attention_library=args.realization_fp8_attention_library,
        realization_cuda_graph_max_entries=args.realization_cuda_graph_max_entries,
        realization_cuda_graph_min_history_tokens=args.realization_cuda_graph_min_history_tokens,
        enable_cuda_graph=not args.no_cuda_graph,
        realization_checkpoint=args.realization_checkpoint,
        realization_stage=args.realization_stage,
        realization_sp_size=args.realization_sp_size,
        realization_tp_size=args.realization_tp_size,
        realization_sp_mode=args.realization_sp_mode,
        realization_worker_devices=args.realization_worker_devices,
        kernel_backend=args.kernel_backend,
        realization_backend=args.realization_backend,
        intent_backend=args.intent_backend,
        height=args.height,
        width=args.width,
        steps=args.steps,
        flow_shift=args.flow_shift,
    )
    deployment = build_duo_realtime_deployment(resources)
    app = create_app(
        _RealtimeOnlyClient(),
        model_name="duo",
        enable_realtime=True,
        realtime_deployment=deployment,
    )

    @app.get("/duo/configuration")
    def realization_configuration() -> dict[str, str | bool | int | None]:
        prefix, suffix = resources.hooks.planner.system_prompt_fragments()
        return {
            "realization_kv_reset_supported": resources.manual_reset_enabled,
            "realization_kv_reset_interval_blocks": resources.realization_kv_reset_interval_blocks,
            "system_prompt_prefix": prefix,
            "system_prompt_suffix": suffix,
            "video_lookahead_blocks": resources.hooks.codecs.video_lookahead_blocks,
        }

    original_lifespan = app.router.lifespan_context

    @asynccontextmanager
    async def lifespan(application: FastAPI) -> AsyncIterator[None]:
        async with original_lifespan(application):
            # Keep thread-local CUDA/cuDNN state warm across realtime sessions.
            with ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="realization-realtime"
            ) as executor:
                asyncio.get_running_loop().set_default_executor(executor)
                if args.warmup_sample_directory is not None:
                    for index, reference_image in enumerate(
                        args.warmup_reference_image
                    ):
                        await warmup_duo_realtime(
                            resources,
                            sample_directory=args.warmup_sample_directory,
                            reference_image=reference_image,
                            blocks=args.warmup_blocks if index == 0 else 1,
                            height=args.height,
                            width=args.width,
                            description=(
                                None
                                if args.warmup_description is None
                                else msgspec.json.decode(
                                    args.warmup_description.read_bytes(),
                                    type=dict[str, str],
                                )
                            ),
                        )
                else:
                    pass
                gc.collect()
                freeze_gc("Realization realtime server")
                try:
                    yield
                finally:
                    shutdown_resources = resources.shutdown
                    resources.shutdown = None
                    if shutdown_resources is not None:
                        shutdown_resources()
                    else:
                        pass

    app.router.lifespan_context = lifespan
    try:
        uvicorn.run(app, host=args.host, port=args.port, log_level="info")
    finally:
        if resources.shutdown is not None:
            resources.shutdown()
        else:
            pass


if __name__ == "__main__":
    main()
