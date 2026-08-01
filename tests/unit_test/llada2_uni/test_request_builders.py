# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from array import array
from types import SimpleNamespace

import torch

from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.models.llada2_uni.request_builders import build_dllm_thinker_request


def test_dllm_thinker_request_uses_upstream_token_array() -> None:
    state = LLaDA2UniPipelineState(
        prompt={"input_ids": torch.tensor([[11, 12, 13]], dtype=torch.long)}
    )
    dllm_config = SimpleNamespace(block_size=2, mask_id=99)

    data = build_dllm_thinker_request(
        state,
        params={"max_new_tokens": 4},
        tokenizer=SimpleNamespace(eos_token_id=2),
        vocab_size=128,
        dllm_config=dllm_config,
        request_id="req-dllm",
    )

    data.req._init_fill_ids_for_dllm()

    assert isinstance(data.req.origin_input_ids, array)
    assert data.req.full_untruncated_fill_ids == array("q", [11, 12, 13, 99, 99])
