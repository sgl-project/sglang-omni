#!/usr/bin/env python3
"""Validate Ming Omni thinker output consistency across TP configurations.

Usage:
    python scripts/test_ming_tp.py run --tp 1 --cpu-offload-gb 150
    python scripts/test_ming_tp.py run --tp 2 --cpu-offload-gb 40
    python scripts/test_ming_tp.py compare tp1_results.json tp2_results.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import multiprocessing as mp
import os
import sys

logging.basicConfig(
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

TEST_PROMPTS = [
    "What is 1+1?",
    "What is the capital of France?",
    "What is the capital of Japan?",
    "Explain quantum computing in one sentence.",
]


async def run_thinker(
    tp_size: int,
    cpu_offload_gb: int,
    mem_fraction: float,
    output_file: str,
    attention_backend: str | None = None,
):
    from sglang_omni.models.ming_omni.config import MingOmniPipelineConfig
    from sglang_omni.pipeline.mp_runner import MultiProcessPipelineRunner
    from sglang_omni.proto import OmniRequest

    overrides = {
        "tp_size": tp_size,
        "cpu_offload_gb": cpu_offload_gb,
        "mem_fraction_static": mem_fraction,
    }
    if attention_backend is not None:
        overrides["attention_backend"] = attention_backend

    config = MingOmniPipelineConfig(
        model_path="inclusionAI/Ming-flash-omni-2.0",
        relay_backend="shm",
        server_args_overrides=overrides,
    )

    runner = MultiProcessPipelineRunner(config)
    logger.info(
        "Starting pipeline with TP=%d, cpu_offload_gb=%d, attention_backend=%s ...",
        tp_size,
        cpu_offload_gb,
        attention_backend,
    )
    await runner.start(timeout=600)

    results = []
    try:
        for i, prompt in enumerate(TEST_PROMPTS):
            logger.info("[%d/%d] Prompt: %s", i + 1, len(TEST_PROMPTS), prompt)
            request = {
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a friendly AI assistant. Please answer concisely.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "audios": [],
            }
            result = await asyncio.wait_for(
                runner.coordinator.submit(
                    f"tp-test-{i}",
                    OmniRequest(
                        inputs=request,
                        params={"max_new_tokens": 64, "temperature": 0.0},
                    ),
                ),
                timeout=120,
            )
            text = ""
            if isinstance(result, dict):
                for stage_name, payload in result.items():
                    data = (
                        payload
                        if isinstance(payload, dict)
                        else getattr(payload, "data", {})
                    )
                    if isinstance(data, dict) and "text" in data:
                        text = data["text"]
                        break
            assert text, f"Empty output for prompt: {prompt}"
            results.append({"prompt": prompt, "output": text})
            logger.info("  Output: %s", text[:200])
    finally:
        await runner.stop()

    with open(output_file, "w") as f:
        json.dump(
            {"tp_size": tp_size, "results": results}, f, indent=2, ensure_ascii=False
        )
    logger.info("Results saved to %s", output_file)


def compare_outputs(file1: str, file2: str):
    with open(file1) as f:
        data1 = json.load(f)
    with open(file2) as f:
        data2 = json.load(f)

    print(f"\n{'='*60}")
    print(f"Comparing TP={data1['tp_size']} vs TP={data2['tp_size']}")
    print(f"{'='*60}")

    all_match = True
    for r1, r2 in zip(data1["results"], data2["results"]):
        match = r1["output"].strip() == r2["output"].strip()
        status = "MATCH" if match else "MISMATCH"
        if not match:
            all_match = False
        print(f"\n[{status}] Prompt: {r1['prompt']}")
        print(f"  TP={data1['tp_size']}: {r1['output'][:120]}")
        print(f"  TP={data2['tp_size']}: {r2['output'][:120]}")

    print(f"\n{'='*60}")
    if all_match:
        print("ALL OUTPUTS MATCH - TP validation PASSED")
    else:
        print("OUTPUTS DIFFER - TP validation FAILED, needs investigation")
    print(f"{'='*60}")
    return all_match


def main():
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd")

    run_p = sub.add_parser("run")
    run_p.add_argument("--tp", type=int, required=True)
    run_p.add_argument("--cpu-offload-gb", type=int, default=80)
    run_p.add_argument("--mem-fraction", type=float, default=0.80)
    run_p.add_argument("--attention-backend", type=str, default=None)
    run_p.add_argument("--output", type=str, default=None)

    cmp_p = sub.add_parser("compare")
    cmp_p.add_argument("file1")
    cmp_p.add_argument("file2")

    args = parser.parse_args()

    if args.cmd == "run":
        output = args.output or f"tp{args.tp}_results.json"
        asyncio.run(
            run_thinker(
                args.tp,
                args.cpu_offload_gb,
                args.mem_fraction,
                output,
                args.attention_backend,
            )
        )
    elif args.cmd == "compare":
        sys.exit(0 if compare_outputs(args.file1, args.file2) else 1)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
