# SPDX-License-Identifier: Apache-2.0
"""Factory functions for creating OmniEngine instances."""

from __future__ import annotations

from typing import Any

import torch

from .engine import OmniEngine
from .model_runner import ModelRunner
from .runtime.ar import (
    ARBatchPlanner,
    ARInputPreparer,
    AROutputProcessor,
    ARResourceManager,
)
from .runtime.common import (
    EosIterationController,
    SimpleResourceManager,
    SinglePassIterationController,
)
from .runtime.encoder import (
    EncoderBatchPlanner,
    EncoderInputPreparer,
    EncoderOutputProcessor,
)
from .scheduler import Scheduler


def create_encoder_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_batch_size: int = 32,
    pooling: str = "last",
    device: str = "cuda",
) -> OmniEngine:
    """Create an encoder engine.

    Args:
        model: The encoder model (e.g., BERT, RoBERTa)
        tokenizer: Optional tokenizer (used to get pad_token_id)
        max_batch_size: Maximum batch size for scheduling
        pooling: Pooling strategy - "last", "mean", or "cls"
        device: Device to run on

    Returns:
        OmniEngine configured for encoder models

    Example:
        from transformers import BertModel, BertTokenizer

        model = BertModel.from_pretrained("bert-base-uncased")
        tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

        engine = create_encoder_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Hello world", return_tensors="pt")
        data = EncoderRequestData(input_ids=input_ids[0])

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # Returns embeddings tensor
    """
    # Get pad_token_id from tokenizer if available
    pad_token_id = 0
    if tokenizer is not None:
        pad_token_id = getattr(tokenizer, "pad_token_id", None) or 0

    scheduler = Scheduler(
        batch_planner=EncoderBatchPlanner(max_batch_size=max_batch_size),
        resource_manager=SimpleResourceManager(max_count=max_batch_size),
        iteration_controller=SinglePassIterationController(),
    )

    # Create model runner
    model_runner = ModelRunner(
        model=model,
        input_preparer=EncoderInputPreparer(pad_token_id=pad_token_id),
        output_processor=EncoderOutputProcessor(pooling=pooling),
        device=device,
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)


def create_ar_engine(
    model: torch.nn.Module,
    tokenizer: Any = None,
    max_seq_len: int = 2048,
    device: str = "cuda",
) -> OmniEngine:
    """Create an AR engine (single request, HF KV cache).

    Args:
        model: The causal LM model (e.g., LLaMA, GPT-2)
        tokenizer: Tokenizer (used to get eos_token_id)
        max_seq_len: Maximum sequence length
        device: Device to run on

    Returns:
        OmniEngine configured for AR models

    Example:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        model = AutoModelForCausalLM.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")

        engine = create_ar_engine(model, tokenizer)
        await engine.start()

        # Create request data
        input_ids = tokenizer.encode("Once upon a time", return_tensors="pt")
        data = ARRequestData(
            input_ids=input_ids[0],
            max_new_tokens=256,
            temperature=0.7,
        )

        await engine.add_request("req-1", data)
        result = await engine.get_result("req-1")  # ARRequestData with output_ids

        generated_text = tokenizer.decode(result.output_ids)
    """
    # Get eos_token_id from tokenizer
    eos_token_id = 2
    if tokenizer is not None:
        eos_token_id = getattr(tokenizer, "eos_token_id", None) or 2

    def _stream_adapter(request, output):
        token = output.data
        if isinstance(token, tuple):
            token = token[0]
        if token is None:
            return None
        return int(token)

    scheduler = Scheduler(
        batch_planner=ARBatchPlanner(),
        resource_manager=ARResourceManager(max_count=1),
        iteration_controller=EosIterationController(
            eos_token_id=eos_token_id,
            max_length=max_seq_len,
        ),
        stream_adapter=_stream_adapter,
    )

    # Create model runner
    model_runner = ModelRunner(
        model=model,
        input_preparer=ARInputPreparer(),
        output_processor=AROutputProcessor(),
        device=device,
    )

    return OmniEngine(scheduler=scheduler, model_runner=model_runner)
