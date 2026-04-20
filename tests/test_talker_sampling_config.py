# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from sglang_omni.models.qwen3_omni.components.talker_executor import (
    TalkerStreamingExecutor,
)
from sglang_omni.proto import OmniRequest, StagePayload


def test_talker_sampling_config_prefers_stage_params() -> None:
    executor = TalkerStreamingExecutor.__new__(TalkerStreamingExecutor)
    executor._codec_vocab_size = 4096
    executor._talker_model = SimpleNamespace(
        config=SimpleNamespace(codec_eos_token_id=2150)
    )
    payload = StagePayload(
        request_id="req-1",
        request=OmniRequest(
            inputs=[],
            params={
                "talker_max_new_tokens": 4096,
                "talker_temperature": 0.9,
                "talker_top_k": 50,
                "talker_top_p": 1.0,
                "talker_repetition_penalty": 1.05,
                "stage_params": {
                    "talker_ar": {
                        "max_new_tokens": 64,
                        "temperature": 0.25,
                        "top_k": 12,
                        "top_p": 0.8,
                        "repetition_penalty": 1.15,
                    }
                },
            },
        ),
        data={},
    )

    config = TalkerStreamingExecutor._resolve_talker_sampling_config(executor, payload)

    assert config["max_new_tokens"] == 64
    assert config["temperature"] == 0.25
    assert config["top_k"] == 12
    assert config["top_p"] == 0.8
    assert config["repetition_penalty"] == 1.15
    assert config["codec_eos_id"] == 2150
    assert 2150 not in config["suppress_tokens"]
