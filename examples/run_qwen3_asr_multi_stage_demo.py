# SPDX-License-Identifier: Apache-2.0
"""Multi-stage Qwen3-ASR pipeline demo.

Stage 1: Preprocess (text/audio -> input_ids/features)
Stage 2: Audio Encoder (features -> embeddings)
Stage 3: Thinker (embeddings/ids -> logits/tokens)
Stage 4: Decode (tokens -> text)
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import time

import torch

from sglang_omni import Coordinator, Stage, Worker
from sglang_omni.models.qwen3_asr.io import PipelineState
from sglang_omni.models.qwen3_asr.pipeline.next_stage import (
    AUDIO_STAGE,
    PREPROCESSING_STAGE,
    THINKER_STAGE,
    decode_next,
    encoder_next,
    preprocessing_next,
    thinker_next,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Endpoints for each stage
STAGE_PREPROCESS_ENDPOINT = "tcp://127.0.0.1:17401"
STAGE_AUDIO_ENDPOINT = "tcp://127.0.0.1:17402"
STAGE_THINKER_ENDPOINT = "tcp://127.0.0.1:17403"
STAGE_DECODE_ENDPOINT = "tcp://127.0.0.1:17404"
COORDINATOR_ENDPOINT = "tcp://127.0.0.1:17400"
ABORT_ENDPOINT = "tcp://127.0.0.1:17499"

ENDPOINTS = {
    PREPROCESSING_STAGE: STAGE_PREPROCESS_ENDPOINT,
    AUDIO_STAGE: STAGE_AUDIO_ENDPOINT,
    THINKER_STAGE: STAGE_THINKER_ENDPOINT,
    "decode": STAGE_DECODE_ENDPOINT,
}


def run_preprocess_stage(model_id: str):
    from sglang_omni.models.qwen3_asr.pipeline.stages import (
        create_preprocessing_executor,
    )

    executor = create_preprocessing_executor(model_id)
    worker = Worker(executor, role="preprocess")

    stage = Stage(
        name=PREPROCESSING_STAGE,
        get_next=preprocessing_next,
        recv_endpoint=STAGE_PREPROCESS_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "preprocess_worker", "relay_type": "shm"},
    )
    stage.add_worker(worker)
    asyncio.run(stage.run())


def run_audio_stage(model_id: str, device: str, dtype: str):
    from sglang_omni.models.qwen3_asr.pipeline.stages import (
        create_audio_encoder_executor,
    )

    executor = create_audio_encoder_executor(model_id, device=device, dtype=dtype)
    worker = Worker(executor, role="audio_encoder")

    stage = Stage(
        name=AUDIO_STAGE,
        get_next=encoder_next,
        recv_endpoint=STAGE_AUDIO_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={
            "worker_id": "audio_worker",
            "relay_type": "shm",
            "gpu_id": 0 if "cuda" in device else None,
        },
    )
    stage.add_worker(worker)
    asyncio.run(stage.run())


def run_thinker_stage(model_id: str, device: str, dtype: str):
    from sglang_omni.models.qwen3_asr.pipeline.stages import create_thinker_executor

    executor = create_thinker_executor(model_id, device=device, dtype=dtype)
    worker = Worker(executor, role="thinker")

    stage = Stage(
        name=THINKER_STAGE,
        get_next=thinker_next,
        recv_endpoint=STAGE_THINKER_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={
            "worker_id": "thinker_worker",
            "relay_type": "shm",
            "gpu_id": 0 if "cuda" in device else None,
        },
    )
    stage.add_worker(worker)
    asyncio.run(stage.run())


def run_decode_stage(model_id: str):
    from transformers import AutoTokenizer

    from sglang_omni.executors import PreprocessingExecutor
    from sglang_omni.proto import StagePayload

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

    def decoder(payload: StagePayload) -> StagePayload:
        state = PipelineState.from_dict(payload.data)
        if state.thinker_out and "output_ids" in state.thinker_out:
            output_ids = state.thinker_out["output_ids"]
            text = tokenizer.decode(output_ids, skip_special_tokens=True)
            payload.data = text
        return payload

    executor = PreprocessingExecutor(decoder)
    worker = Worker(executor, role="decode")

    stage = Stage(
        name="decode",
        get_next=decode_next,
        recv_endpoint=STAGE_DECODE_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "decode_worker", "relay_type": "shm"},
    )
    stage.add_worker(worker)
    asyncio.run(stage.run())


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", type=str, default="Qwen3-ASR/Qwen3-ASR-0.6B")
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--dtype", type=str, default="fp16" if torch.cuda.is_available() else "fp32"
    )
    args = parser.parse_args()

    # Start stages in separate processes
    ctx = mp.get_context("spawn")
    processes = []

    stages = [
        (run_preprocess_stage, (args.model_id,)),
        (run_audio_stage, (args.model_id, args.device, args.dtype)),
        (run_thinker_stage, (args.model_id, args.device, args.dtype)),
        (run_decode_stage, (args.model_id,)),
    ]

    for func, fargs in stages:
        p = ctx.Process(target=func, args=fargs)
        p.start()
        processes.append(p)

    # Wait for stages to initialize
    await asyncio.sleep(5)

    coordinator = Coordinator(
        completion_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        entry_stage=PREPROCESSING_STAGE,
    )

    coordinator.register_stage(PREPROCESSING_STAGE, STAGE_PREPROCESS_ENDPOINT)
    coordinator.register_stage(AUDIO_STAGE, STAGE_AUDIO_ENDPOINT)
    coordinator.register_stage(THINKER_STAGE, STAGE_THINKER_ENDPOINT)
    coordinator.register_stage("decode", STAGE_DECODE_ENDPOINT)

    await coordinator.start()
    completion_task = asyncio.create_task(coordinator.run_completion_loop())

    import librosa

    audio_path = "examples/asr_en.wav"
    dummy_audio, _ = librosa.load(audio_path, sr=16000)
    dummy_audio = dummy_audio.tolist()

    from sglang_omni.proto import OmniRequest

    request = OmniRequest(inputs={"text": "Audio recognition", "audio": dummy_audio})

    logger.info("Sending request to pipeline...")
    start_time = time.time()
    result = await coordinator.submit("qwen3-asr-test-001", request)
    elapsed = time.time() - start_time

    logger.info(f"Result received in {elapsed:.2f}s: {result}")

    # Cleanup
    completion_task.cancel()
    await coordinator.stop()
    for p in processes:
        p.terminate()
        p.join()


if __name__ == "__main__":
    asyncio.run(main())
