# SPDX-License-Identifier: Apache-2.0
"""Engine smoke tests using Hugging Face models."""

from __future__ import annotations

import asyncio

import pytest
import torch

from sglang_omni.engines.omni import (
    ARRequestData,
    EncoderRequestData,
    create_ar_engine,
    create_encoder_engine,
)


async def _run_encoder_engine() -> None:
    from transformers import AutoModel, AutoTokenizer

    model_path = "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path)

    engine = create_encoder_engine(model=model, tokenizer=tokenizer, device="cuda")
    await engine.start()
    try:
        input_ids = tokenizer.encode("hello", return_tensors="pt")[0]
        data = EncoderRequestData(input_ids=input_ids)
        await engine.add_request("enc-1", data)
        result = await engine.get_result("enc-1")
        assert result.embeddings is not None
    finally:
        await engine.stop()


def test_encoder_engine_runs() -> None:
    asyncio.run(_run_encoder_engine())


async def _run_qwen3_8b_engine() -> None:
    assert torch.cuda.is_available(), "CUDA is required for the Qwen3 8B test."

    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_path = "Qwen/Qwen3-0.6B"
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float16,
    )

    max_seq_len = getattr(model.config, "max_position_embeddings", 2048)
    engine = create_ar_engine(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=max_seq_len,
        device="cuda",
    )
    await engine.start()
    try:
        input_ids = tokenizer.encode("Hello", return_tensors="pt")[0]
        data = ARRequestData(input_ids=input_ids, max_new_tokens=4, temperature=0.0)
        await engine.add_request("qwen-1", data)
        result = await engine.get_result("qwen-1")
        assert result.output_ids
    finally:
        await engine.stop()


@pytest.mark.skip(
    reason="Torch's bug will make this test hang, you can try upgrade cudnn to fix this: pip install nvidia-cudnn-cu12 --upgrade"
)
def test_qwen3_8b_engine_runs() -> None:
    asyncio.run(_run_qwen3_8b_engine())
