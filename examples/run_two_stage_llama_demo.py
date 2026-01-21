# SPDX-License-Identifier: Apache-2.0
"""Three-stage Llama 3 8B pipeline demo.

Stage 1: Tokenize prompt text
Stage 2: AR engine generate tokens
Stage 3: Decode tokens to text
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import multiprocessing as mp
import time
from typing import Any

import torch

from sglang_omni import Coordinator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

STAGE1_ENDPOINT = "tcp://127.0.0.1:17001"
STAGE2_ENDPOINT = "tcp://127.0.0.1:17002"
STAGE3_ENDPOINT = "tcp://127.0.0.1:17003"
COORDINATOR_ENDPOINT = "tcp://127.0.0.1:17000"
ABORT_ENDPOINT = "tcp://127.0.0.1:17099"

ENDPOINTS = {
    "tokenize": STAGE1_ENDPOINT,
    "engine": STAGE2_ENDPOINT,
    "decode": STAGE3_ENDPOINT,
}

DTYPE_MAP = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _gpu_id_from_device(device: str) -> int | None:
    if not device.startswith("cuda"):
        return None
    if ":" in device:
        return int(device.split(":")[1])
    return 0


def tokenize_get_next(request_id: str, output: Any) -> str | None:
    return "engine"


def engine_get_next(request_id: str, output: Any) -> str | None:
    return "decode"


def decode_get_next(request_id: str, output: Any) -> str | None:
    return None


def run_tokenize_stage(model_id: str) -> None:
    import logging

    from transformers import AutoTokenizer

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def processor(payload: StagePayload) -> StagePayload:
        text = payload.request.inputs
        input_ids = tokenizer.encode(text, return_tensors="pt")[0]
        payload.data = {"input_ids": input_ids}
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="tokenize")

    stage = Stage(
        name="tokenize",
        get_next=tokenize_get_next,
        recv_endpoint=STAGE1_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "tokenize_worker", "gpu_id": None},
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())


def run_decode_stage(model_id: str) -> None:
    import logging

    from transformers import AutoTokenizer

    from sglang_omni import Stage, Worker
    from sglang_omni.executors import FrontendExecutor
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    def processor(payload: StagePayload) -> StagePayload:
        data = payload.data
        if isinstance(data, dict):
            output_ids = data.get("output_ids", data)
        else:
            output_ids = data
        if hasattr(output_ids, "tolist"):
            output_ids = output_ids.tolist()
        text = tokenizer.decode(output_ids, skip_special_tokens=True)
        payload.data = text
        return payload

    executor = FrontendExecutor(processor)
    worker = Worker(executor, role="decode")

    stage = Stage(
        name="decode",
        get_next=decode_get_next,
        recv_endpoint=STAGE3_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={"worker_id": "decode_worker", "gpu_id": None},
    )
    stage.add_worker(worker)

    asyncio.run(stage.run())


def run_engine_stage(
    model_id: str,
    device: str,
    dtype: torch.dtype,
    max_seq_len: int | None,
    relay_device: str,
) -> None:
    import logging

    from transformers import AutoModelForCausalLM, AutoTokenizer

    from sglang_omni import Stage, Worker
    from sglang_omni.engines.omni import create_ar_engine
    from sglang_omni.executors import EngineExecutor
    from sglang_omni.executors.engine_request_builders import build_ar_request
    from sglang_omni.proto import StagePayload

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype)

    if max_seq_len is None:
        max_seq_len = getattr(model.config, "max_position_embeddings", 2048)

    engine = create_ar_engine(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device=device,
    )

    def result_builder(payload: StagePayload, result: Any) -> StagePayload:
        payload.data = {"output_ids": list(result.output_ids)}
        return payload

    executor = EngineExecutor(engine, build_ar_request, result_builder=result_builder)
    worker = Worker(executor, role="engine")

    stage = Stage(
        name="engine",
        get_next=engine_get_next,
        recv_endpoint=STAGE2_ENDPOINT,
        coordinator_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        endpoints=ENDPOINTS,
        relay_config={
            "worker_id": "engine_worker",
            "gpu_id": _gpu_id_from_device(relay_device),
        },
    )
    stage.add_worker(worker)

    async def run() -> None:
        await engine.start()
        try:
            await stage.run()
        finally:
            await engine.stop()

    asyncio.run(run())


async def run_coordinator(args: argparse.Namespace) -> None:
    from sglang_omni.proto import OmniRequest

    coordinator = Coordinator(
        completion_endpoint=COORDINATOR_ENDPOINT,
        abort_endpoint=ABORT_ENDPOINT,
        entry_stage="tokenize",
    )

    coordinator.register_stage("tokenize", STAGE1_ENDPOINT)
    coordinator.register_stage("engine", STAGE2_ENDPOINT)
    coordinator.register_stage("decode", STAGE3_ENDPOINT)

    await coordinator.start()
    completion_task = asyncio.create_task(coordinator.run_completion_loop())

    try:
        await asyncio.sleep(1.0)

        request = OmniRequest(
            inputs=args.prompt,
            params={
                "max_new_tokens": args.max_new_tokens,
                "temperature": args.temperature,
            },
        )
        result = await coordinator.submit("llama-req-1", request)

        if not isinstance(result, str):
            raise RuntimeError(f"Unexpected result: {result}")
        logger.info("Completion: %s", result)

        await coordinator.shutdown_stages()
        await asyncio.sleep(0.5)
    finally:
        completion_task.cancel()
        try:
            await completion_task
        except asyncio.CancelledError:
            pass
        await coordinator.stop()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Three-stage Llama 3 8B demo")
    parser.add_argument(
        "--model-id",
        default="meta-llama/Meta-Llama-3-8B",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--prompt",
        default="Hello, how are you?",
        help="Input prompt",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=32,
        help="Max new tokens",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        help="Engine device",
    )
    parser.add_argument(
        "--dtype",
        choices=sorted(DTYPE_MAP.keys()),
        default="fp16",
        help="Model dtype",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="Override model max sequence length",
    )
    parser.add_argument(
        "--engine-relay-device",
        default="cpu",
        help="Relay device for engine stage (default: cpu)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dtype = DTYPE_MAP[args.dtype]

    stage1_proc = mp.Process(
        target=run_tokenize_stage,
        name="TokenizerStage",
        args=(args.model_id,),
    )
    stage2_proc = mp.Process(
        target=run_engine_stage,
        name="EngineStage",
        args=(
            args.model_id,
            args.device,
            dtype,
            args.max_seq_len,
            args.engine_relay_device,
        ),
    )
    stage3_proc = mp.Process(
        target=run_decode_stage,
        name="DecodeStage",
        args=(args.model_id,),
    )

    stage1_proc.start()
    stage2_proc.start()
    stage3_proc.start()

    logger.info(
        "Stage processes started: tokenize=%d engine=%d decode=%d",
        stage1_proc.pid,
        stage2_proc.pid,
        stage3_proc.pid,
    )

    try:
        time.sleep(1.0)
        asyncio.run(run_coordinator(args))
    finally:
        stage1_proc.join(timeout=2)
        stage2_proc.join(timeout=2)
        stage3_proc.join(timeout=2)

        if stage1_proc.is_alive():
            stage1_proc.terminate()
            stage1_proc.join(timeout=1)
        if stage2_proc.is_alive():
            stage2_proc.terminate()
            stage2_proc.join(timeout=1)
        if stage3_proc.is_alive():
            stage3_proc.terminate()
            stage3_proc.join(timeout=1)


if __name__ == "__main__":
    main()
