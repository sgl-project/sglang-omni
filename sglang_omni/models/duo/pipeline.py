"""Compose the native ASR stage and the Duo realtime session runtime."""

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path
from uuid import uuid4

from fastapi import APIRouter

from sglang_omni.audio.wav import pcm16_wav
from sglang_omni.client import Client
from sglang_omni.models.duo.config import DuoPipelineConfig
from sglang_omni.models.duo.realtime import (
    build_duo_realtime_deployment,
    load_duo_live_resources,
)
from sglang_omni.serve.transcriptions import build_chunk_generate_request


class PipelineRecognizer:
    def __init__(self, client: Client):
        self.client = client
        self.loop = asyncio.get_running_loop()

    def transcribe(self, pcm16: bytes) -> str:
        request = build_chunk_generate_request(
            pcm16_wav(pcm16, 16000),
            model="duo",
            filename="audio.wav",
            language=None,
            prompt=None,
            temperature=None,
            repetition_penalty=None,
            max_new_tokens=None,
        )
        request_id = f"duo-asr-{uuid4().hex}"

        async def complete():
            try:
                async with asyncio.timeout(60):
                    return await self.client.completion(request, request_id=request_id)
            except TimeoutError:
                await self.client.abort(request_id)
                raise

        result = asyncio.run_coroutine_threadsafe(complete(), self.loop).result()
        return result.text.strip().rstrip("。．.")


def create_deployment(client: Client, config: DuoPipelineConfig):
    root = Path(config.model_path)
    prepared = root / "prepared-duo"
    os.environ.setdefault("TORCHTITAN_LIVE", "1")
    os.environ.setdefault("SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK", "0")
    resources = load_duo_live_resources(
        model_path=str(prepared / "codec"),
        intent_model_path=str(root / "duo-intent-merged"),
        intent_adapter_path=str(root / "Duo-1.0-causal-base/intent_adapter"),
        raven_lora_path="",
        trainables_path="",
        intent_backend="native",
        intent_tp_size=4,
        intent_worker_devices=list(range(4)),
        realization_backend="native",
        realization_stage="dmd",
        steps=4,
        flow_shift=5,
        realization_checkpoint=str(prepared / "realization"),
        realization_device="cuda:0",
        realization_tp_size=8,
        realization_worker_devices=list(range(8)),
        realization_reference_window_chunks=14,
        realization_kv_workspace_capacity_tokens=131072,
        realization_optimize_copy_sync=True,
        realization_kv_reset_interval_blocks=config.realization_kv_reset_interval_blocks,
        realization_modelopt_checkpoint=config.realization_modelopt_checkpoint,
        realization_fp8_direct_kernels=config.realization_modelopt_checkpoint
        is not None,
        vae_device="cuda:4",
        vae_worker_devices=list(range(4, 8)),
        vae_decode_backend="solarx",
        vae_attention_backend="torch_cudnn_sdpa",
        intent_kv_window_tokens=32768,
        overlap_intent_codec=True,
        height=768,
        width=576,
    )
    resources.hooks.asr = PipelineRecognizer(client)
    worker = ThreadPoolExecutor(max_workers=1, thread_name_prefix="duo-block")
    resources.block_executor = worker
    router = APIRouter()

    router.get("/duo/configuration")(resources.configuration)

    def shutdown():
        worker.shutdown()
        if resources.shutdown:
            resources.shutdown()
        else:
            pass

    return replace(
        build_duo_realtime_deployment(resources), router=router, shutdown=shutdown
    )
