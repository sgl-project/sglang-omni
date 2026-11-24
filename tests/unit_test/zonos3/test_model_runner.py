# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.zonos3.model_runner import Zonos3ModelRunner
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor

CODEBOOK_COUNT = 10
CODEBOOK_SIZE = 8
HIDDEN_SIZE = 4
POOL_ROWS = 3


def build_runner(dtype: torch.dtype) -> Zonos3ModelRunner:
    embedders = torch.nn.ModuleList(
        torch.nn.Embedding(CODEBOOK_SIZE, HIDDEN_SIZE, dtype=dtype)
        for _ in range(CODEBOOK_COUNT + 1)
    )
    model = SimpleNamespace(
        config=SimpleNamespace(
            codebook_count=CODEBOOK_COUNT,
            audio_pad_id=0,
            codebook_size=CODEBOOK_SIZE,
            end_of_audio_id=1,
            text_pad_id=2,
        ),
        decode_input_embedding=torch.nn.Embedding(POOL_ROWS, HIDDEN_SIZE, dtype=dtype),
        embedders=embedders,
    )
    worker = Mock(spec=ModelWorker)
    worker.gpu_id = 0
    worker.model_runner = SimpleNamespace(
        model=model,
        req_to_token_pool=SimpleNamespace(req_to_token=torch.zeros(POOL_ROWS, 1)),
    )
    return Zonos3ModelRunner(worker, Mock(spec=SGLangOutputProcessor), model=model)


def test_the_pointer_table_is_absent_without_the_fused_kernel() -> None:
    runner = build_runner(torch.float32)
    assert runner.is_fused_embedding is False
    assert runner.codebook_table_addresses is None


def test_the_pointer_table_is_absent_for_bfloat16_off_cuda() -> None:
    runner = build_runner(torch.bfloat16)
    assert runner.is_fused_embedding is False
    assert runner.codebook_table_addresses is None
