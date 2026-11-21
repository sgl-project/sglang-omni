# SPDX-License-Identifier: Apache-2.0
"""Small LLaDA2-Uni fakes for interleaved generation tests."""

from __future__ import annotations

import asyncio

from sglang_omni.models.llada2_uni.components.preprocessor import LLaDA2Preprocessor
from sglang_omni.models.llada2_uni.interleaved import advance_interleaved_state
from sglang_omni.models.llada2_uni.payload_types import LLaDA2UniPipelineState
from sglang_omni.proto import OmniRequest, StagePayload

# The block size the interleaved tests run the thinker with.
BLOCK_SIZE = 4


class Tokenizer:
    mask_token_id = 99
    eos_token_id = 2
    additional_stop_token_ids = [666]

    def convert_tokens_to_ids(self, token: str) -> int:
        return {"<|image|>": 10, "<|/image|>": 11, "<boi>": 12}[token]

    def convert_ids_to_tokens(self, token: int) -> str:
        return f"<|reserved_token_{token - 20}|>"

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return [100 + i for i, _ in enumerate(text.split())]

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(token) for token in ids if token != self.eos_token_id)


def preprocess(request: OmniRequest, interleaved: bool = True) -> StagePayload:
    preprocessor = object.__new__(LLaDA2Preprocessor)
    preprocessor.tokenizer = Tokenizer()
    preprocessor.max_seq_len = 8192
    preprocessor.interleaved = interleaved
    return asyncio.run(
        preprocessor(StagePayload(request_id="story", request=request, data={}))
    )


def advance(state: LLaDA2UniPipelineState, ids: list[int]) -> None:
    state.thinker_out = {"output_ids": ids, "is_final": True}
    advance_interleaved_state(
        state,
        Tokenizer(),
        completed_phase=state.stream_state["interleaved"]["phase"],
        block_size=BLOCK_SIZE,
    )
