# SPDX-License-Identifier: Apache-2.0
"""MiniCPM-o decode-stage result contract."""

from __future__ import annotations

from sglang_omni.models.minicpm_o.merge import build_decode_result
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import SpecTokensDetailsDict


class FakeTokenizer:
    def decode(self, token_ids: list[int], skip_special_tokens: bool = False) -> str:
        return "hello"


SPEC_TOKENS_DETAILS: SpecTokensDetailsDict = {
    "spec_accept_rate": 0.5,
    "spec_accept_length": 2.0,
    "spec_num_correct_drafts": 1,
    "spec_num_proposed_drafts": 2,
    "spec_verify_ct": 1,
    "spec_correct_drafts_histogram": [0, 1],
}


def test_build_decode_result_returns_text_and_usage() -> None:
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(inputs=[], params={"stream": False}),
        data={
            "engine_outputs": {
                "thinker": {
                    "output_ids": [1, 2],
                    "is_final": True,
                    "spec_tokens_details": SPEC_TOKENS_DETAILS,
                }
            }
        },
    )
    result = build_decode_result(
        payload,
        tokenizer=FakeTokenizer(),
        eos_token_id=None,
        is_streaming=False,
    )
    assert result["text"] == "hello"
    assert result["events"][0]["type"] == "text_final"
    assert result["usage"]["completion_tokens"] == 2
    assert result["spec_tokens_details"] == SPEC_TOKENS_DETAILS
