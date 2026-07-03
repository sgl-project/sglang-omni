# SPDX-License-Identifier: Apache-2.0
"""POST /generate — Miles-compatible RL rollout endpoint."""

from __future__ import annotations

from typing import Any

from fastapi.testclient import TestClient

from sglang_omni.client.types import (
    CompletionAudio,
    CompletionResult,
    GenerateRequest,
    UsageInfo,
)
from sglang_omni.serve import create_app
from sglang_omni.serve.openai_api import _build_rollout_generate_request
from sglang_omni.serve.protocol import RolloutGenerateRequest as RolloutRequest


class _RolloutClient:
    """Captures the converted request and returns a canned CompletionResult."""

    def __init__(self, result: CompletionResult) -> None:
        self._result = result
        self.requests: list[GenerateRequest] = []

    def health(self) -> dict[str, Any]:
        return {"running": True}

    async def completion(
        self,
        request: GenerateRequest,
        *,
        request_id: str,
        audio_format: str = "wav",
    ) -> CompletionResult:
        del request_id, audio_format
        self.requests.append(request)
        return self._result


def _text_result() -> CompletionResult:
    return CompletionResult(
        request_id="r1",
        text="hello world",
        finish_reason="stop",
        usage=UsageInfo(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        output_token_logprobs=[[-0.1, 11], [-0.2, 22]],
        weight_version="v3",
    )


def test_generate_returns_miles_meta_info() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": "hi"}],
            "sampling_params": {"temperature": 0.7, "max_new_tokens": 16},
            "stream": False,
            "metadata": {"rollout_id": 42, "group_id": 1},
        },
    )

    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "hello world"
    meta = body["meta_info"]
    assert meta["finish_reason"] == {"type": "stop", "length": None}
    assert meta["completion_tokens"] == 2
    assert meta["prompt_tokens"] == 5
    assert meta["cached_tokens"] == 0
    assert meta["weight_version"] == "v3"
    assert meta["output_token_logprobs"] == [[-0.1, 11], [-0.2, 22]]
    assert meta["request_metadata"] == {"rollout_id": 42, "group_id": 1}


def test_generate_returns_omni_rollout_when_present() -> None:
    result = _text_result()
    result.output_token_logprobs = None
    result.omni_rollout = {
        "version": 1,
        "model_family": "qwen3_omni",
        "stages": ["talker"],
        "total_action_count": 1,
        "action_streams": [],
    }
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_omni_rollout": True,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["meta_info"]["omni_rollout"] == result.omni_rollout


def test_generate_omits_omni_rollout_when_not_requested() -> None:
    result = _text_result()
    result.omni_rollout = {"version": 1, "action_streams": []}
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_omni_rollout": False,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["meta_info"]["omni_rollout"] is None


def test_generate_rejects_missing_omni_rollout_when_requested() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_omni_rollout": True,
        },
    )

    assert resp.status_code == 501
    assert "omni_rollout" in resp.text


def test_generate_omits_logprobs_when_not_requested() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_logprob": False,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["meta_info"]["output_token_logprobs"] is None


def test_generate_preserves_empty_logprob_list_when_requested() -> None:
    result = _text_result()
    result.output_token_logprobs = []
    result.usage = UsageInfo(prompt_tokens=5, completion_tokens=0, total_tokens=5)
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_logprob": True,
        },
    )

    assert resp.status_code == 200
    assert resp.json()["meta_info"]["output_token_logprobs"] == []


def test_generate_rejects_missing_logprobs_when_requested() -> None:
    result = _text_result()
    result.output_token_logprobs = None
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_logprob": True,
        },
    )

    assert resp.status_code == 501
    assert "output_token_logprobs" in resp.text


def test_generate_audio_logprob_error_hints_omni_rollout() -> None:
    result = _text_result()
    result.output_token_logprobs = None
    result.audio = CompletionAudio(id="a1", data="QUJD", transcript="hello world")
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="higgs-audio"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "output_modalities": ["audio"],
        },
    )

    assert resp.status_code == 501
    assert "return_omni_rollout=true" in resp.text


def test_generate_rejects_logprob_length_mismatch() -> None:
    result = _text_result()
    result.output_token_logprobs = [[-0.1, 11]]
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "return_logprob": True,
        },
    )

    assert resp.status_code == 500
    assert "completion_tokens=2" in resp.text


def test_generate_echoes_null_weight_version_without_failing() -> None:
    result = _text_result()
    result.weight_version = None
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
        },
    )

    assert resp.status_code == 200
    assert resp.json()["meta_info"]["weight_version"] is None


def test_generate_emits_audio_block_when_present() -> None:
    result = _text_result()
    result.audio = CompletionAudio(id="a1", data="QUJD", transcript="hello world")
    client = _RolloutClient(result)
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "messages": [{"role": "user", "content": "say hi"}],
            "sampling_params": {},
            "output_modalities": ["audio"],
        },
    )

    assert resp.status_code == 200
    audio = resp.json()["audio"]
    assert audio is not None
    assert audio["data"] == "QUJD"
    assert audio["format"] == "wav"


def test_generate_without_audio_returns_null_audio() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={"prompt": "hi", "sampling_params": {}},
    )

    assert resp.status_code == 200
    assert resp.json()["audio"] is None


def test_generate_rejects_ambiguous_prompt_inputs() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={"prompt": "hi", "input_ids": [1, 2, 3], "sampling_params": {}},
    )

    assert resp.status_code == 400


def test_generate_rejects_streaming_rollout_until_supported() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "stream": True,
        },
    )

    assert resp.status_code == 400
    assert "stream=true" in resp.text
    assert client.requests == []


def test_generate_rejects_unknown_sampling_param() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {"temprature": 0.7},
        },
    )

    assert resp.status_code == 422
    assert "temprature" in resp.text
    assert client.requests == []


def test_generate_rejects_invalid_stop_type() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {"stop": 12},
        },
    )

    assert resp.status_code == 422
    assert "stop" in resp.text
    assert client.requests == []


def test_generate_stage_sampling_bad_key_returns_422() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    resp = tc.post(
        "/generate",
        json={
            "prompt": "hi",
            "sampling_params": {},
            "stage_sampling": {"thinker": {"bad_key": 1}},
        },
    )

    assert resp.status_code == 422
    assert "bad_key" in resp.text
    assert client.requests == []


def test_generate_rejects_message_without_role_or_content() -> None:
    client = _RolloutClient(_text_result())
    tc = TestClient(create_app(client, model_name="qwen3-omni"))

    missing_role = tc.post(
        "/generate",
        json={
            "messages": [{"content": "hi"}],
            "sampling_params": {},
        },
    )
    missing_content = tc.post(
        "/generate",
        json={
            "messages": [{"role": "user"}],
            "sampling_params": {},
        },
    )

    assert missing_role.status_code == 422
    assert "role" in missing_role.text
    assert missing_content.status_code == 422
    assert "content" in missing_content.text
    assert client.requests == []


def test_converter_maps_input_ids_to_prompt_token_ids() -> None:
    req = RolloutRequest(
        input_ids=[1, 2, 3],
        sampling_params={"temperature": 0.5, "max_new_tokens": 8},
        return_logprob=True,
    )
    gen = _build_rollout_generate_request(req)

    assert gen.prompt_token_ids == [1, 2, 3]
    assert gen.prompt is None
    assert gen.messages is None
    assert gen.sampling.temperature == 0.5
    assert gen.sampling.max_new_tokens == 8
    assert gen.stream is False
    assert gen.extra_params["return_logprob"] is True


def test_converter_preserves_prompt_as_raw_rollout_input() -> None:
    from sglang_omni.client import Client

    req = RolloutRequest(prompt="hi", sampling_params={})
    gen = _build_rollout_generate_request(req)
    omni = Client._build_omni_request(gen)

    assert gen.prompt == "hi"
    assert gen.prompt_token_ids is None
    assert gen.messages is None
    assert omni.inputs == "hi"


def test_converter_preserves_messages_as_chat_rollout_input() -> None:
    from sglang_omni.client import Client

    req = RolloutRequest(
        messages=[{"role": "user", "content": "hi"}],
        sampling_params={},
    )
    gen = _build_rollout_generate_request(req)
    omni = Client._build_omni_request(gen)

    assert gen.prompt is None
    assert gen.prompt_token_ids is None
    assert [msg.to_dict() for msg in gen.messages or []] == [
        {"role": "user", "content": "hi"}
    ]
    assert omni.inputs == [{"role": "user", "content": "hi"}]


def test_converter_defaults_rollout_to_text_output_modality() -> None:
    req = RolloutRequest(prompt="hi", sampling_params={})
    gen = _build_rollout_generate_request(req)

    assert gen.output_modalities == ["text"]


def test_converter_preserves_explicit_output_modalities() -> None:
    req = RolloutRequest(
        prompt="hi",
        sampling_params={},
        output_modalities=["text", "audio"],
    )
    gen = _build_rollout_generate_request(req)

    assert gen.output_modalities == ["text", "audio"]


def test_converter_threads_return_logprob_into_omni_params() -> None:
    from sglang_omni.client import Client

    req = RolloutRequest(prompt="hi", sampling_params={}, return_logprob=True)
    gen = _build_rollout_generate_request(req)
    omni = Client._build_omni_request(gen)

    assert omni.params.get("return_logprob") is True


def test_converter_defaults_stream_false_and_logprob_true() -> None:
    req = RolloutRequest(prompt="hi", sampling_params={})
    assert req.stream is False
    assert req.return_logprob is True
    assert req.return_omni_rollout is False
