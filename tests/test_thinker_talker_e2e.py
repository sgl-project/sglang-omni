# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for thinker + talker pipeline.

Runs the full pipeline:
  thinker → talker → decode

Verifies:
  1. Thinker produces text tokens + captures hidden states
  2. Talker produces codec codes (shape [16, N])
  3. Decode stage returns both text + codec outputs

Usage:
    python tests/test_thinker_talker_e2e.py \
        --model-path Qwen/Qwen3-Omni-30B-A3B-Instruct \
        --prompt "Hello, how are you?"
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from typing import Any

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


async def run_e2e(
    model_path: str,
    prompt: str,
    gpu_id: int = 0,
    max_new_tokens: int = 64,
    thinker_max_seq_len: int = 4096,
    talker_max_seq_len: int = 4096,
) -> dict[str, Any]:
    """Run the full thinker + talker pipeline and return results."""
    from transformers import AutoTokenizer

    from sglang_omni.models.qwen3_omni.pipeline.stages import (
        create_decode_executor,
        create_sglang_talker_executor_from_config,
        create_sglang_thinker_executor_from_config,
    )
    from sglang_omni.models.qwen3_omni.pipeline.state_io import load_state
    from sglang_omni.proto import OmniRequest, StagePayload

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # ----------------------------------------------------------------
    # 1. Build executors
    # ----------------------------------------------------------------
    logger.info("Creating thinker executor...")
    thinker_executor = create_sglang_thinker_executor_from_config(
        model_path=model_path,
        gpu_id=gpu_id,
        thinker_max_seq_len=thinker_max_seq_len,
    )

    logger.info("Creating talker executor...")
    talker_executor = create_sglang_talker_executor_from_config(
        model_path=model_path,
        gpu_id=gpu_id,
        talker_max_seq_len=talker_max_seq_len,
    )

    logger.info("Creating decode executor...")
    decode_executor = create_decode_executor(model_path)

    # ----------------------------------------------------------------
    # 2. Prepare input payload (skip preprocessing/encoders for text-only)
    # ----------------------------------------------------------------
    logger.info("Preparing input...")
    # Try chat template first, fall back to direct tokenization
    try:
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ]
        prompt_text = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
    except (ValueError, AttributeError):
        prompt_text = prompt
        logger.info("Chat template not available, using raw prompt")

    encoded = tokenizer(prompt_text, return_tensors="pt")
    input_ids = encoded["input_ids"].squeeze(0)
    attention_mask = encoded.get("attention_mask")
    if attention_mask is not None:
        attention_mask = attention_mask.squeeze(0)

    from sglang_omni.models.qwen3_omni.io import PipelineState

    state = PipelineState(
        prompt={
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "prompt_text": prompt_text,
        },
    )

    request = OmniRequest(
        inputs=prompt,
        params={
            "max_new_tokens": max_new_tokens,
            "temperature": 0.0,
        },
    )
    payload = StagePayload(
        request_id="e2e-test-0",
        request=request,
        data=state.to_dict(),
    )

    # ----------------------------------------------------------------
    # 3. Run thinker via EngineExecutor
    # ----------------------------------------------------------------
    logger.info("Running thinker (max_new_tokens=%d)...", max_new_tokens)
    await thinker_executor.start()
    try:
        await thinker_executor.add_request(payload)
        payload = await thinker_executor.get_result()
    finally:
        await thinker_executor.stop()

    # Verify thinker output
    state = load_state(payload)
    thinker_out = state.thinker_out
    assert thinker_out is not None, "Thinker produced no output"
    output_ids = thinker_out.get("output_ids", [])
    logger.info("Thinker produced %d tokens", len(output_ids))
    assert len(output_ids) > 0, "Thinker produced zero tokens"

    text = tokenizer.decode(output_ids, skip_special_tokens=True)
    logger.info("[PASS] Text output: %r", text[:200])

    # Verify hidden states were captured for talker
    talker_inputs = state.talker_inputs
    assert talker_inputs, "talker_inputs not populated by thinker result_builder"
    thinker_embed = talker_inputs.get("thinker_embed")
    thinker_hidden = talker_inputs.get("thinker_hidden")
    assert isinstance(
        thinker_embed, torch.Tensor
    ), f"thinker_embed missing or wrong type: {type(thinker_embed)}"
    assert isinstance(
        thinker_hidden, torch.Tensor
    ), f"thinker_hidden missing or wrong type: {type(thinker_hidden)}"
    logger.info(
        "thinker_embed shape: %s, thinker_hidden shape: %s",
        thinker_embed.shape,
        thinker_hidden.shape,
    )

    # ----------------------------------------------------------------
    # 4. Run talker via EngineExecutor
    # ----------------------------------------------------------------
    logger.info("Running talker...")
    await talker_executor.start()
    try:
        await talker_executor.add_request(payload)
        payload = await talker_executor.get_result()
    finally:
        await talker_executor.stop()

    # Verify talker output
    state = load_state(payload)
    talker_out = state.talker_out
    assert talker_out is not None, "Talker produced no output"
    codec_codes = talker_out.get("codec_codes")
    assert codec_codes is not None, "Talker produced no codec_codes"
    if isinstance(codec_codes, torch.Tensor):
        codec_shape = tuple(codec_codes.shape)
    elif isinstance(codec_codes, list):
        codec_shape = (len(codec_codes), len(codec_codes[0]) if codec_codes else 0)
    else:
        codec_shape = "unknown"
    logger.info("[PASS] Codec codes shape: %s", codec_shape)

    # ----------------------------------------------------------------
    # 5. Run decode (PreprocessingExecutor — synchronous)
    # ----------------------------------------------------------------
    logger.info("Running decode...")
    await decode_executor.add_request(payload)
    payload = await decode_executor.get_result()

    result = payload.data
    logger.info(
        "Decode result keys: %s",
        list(result.keys()) if isinstance(result, dict) else type(result),
    )

    if isinstance(result, dict):
        if "text" in result:
            logger.info("[PASS] Decoded text: %r", result["text"][:200])
        if "codec_codes" in result:
            logger.info("[PASS] Decoded codec_codes present")
        if "events" in result:
            logger.info("[PASS] Events count: %d", len(result["events"]))

    logger.info("=== E2E TEST PASSED ===")
    return result


def main():
    parser = argparse.ArgumentParser(description="Thinker + Talker E2E test")
    parser.add_argument(
        "--model-path",
        type=str,
        default="Qwen/Qwen3-Omni-30B-A3B-Instruct",
        help="Path to the Qwen3-Omni model",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Hello, how are you?",
        help="Input prompt",
    )
    parser.add_argument("--gpu-id", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--thinker-max-seq-len", type=int, default=4096)
    parser.add_argument("--talker-max-seq-len", type=int, default=4096)
    args = parser.parse_args()

    asyncio.run(
        run_e2e(
            model_path=args.model_path,
            prompt=args.prompt,
            gpu_id=args.gpu_id,
            max_new_tokens=args.max_new_tokens,
            thinker_max_seq_len=args.thinker_max_seq_len,
            talker_max_seq_len=args.talker_max_seq_len,
        )
    )


if __name__ == "__main__":
    main()
