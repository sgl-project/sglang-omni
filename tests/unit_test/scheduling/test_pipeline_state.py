from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pipeline_state import (
    PipelineStateBase,
    build_usage,
    decode_typed_tensor,
    encode_typed_tensor,
    load_state,
    store_state,
)


@dataclass
class _DummyState(PipelineStateBase):
    value: str = ""

    def to_dict(self) -> dict[str, Any]:
        data = {"value": self.value, "sample_rate": self.sample_rate}
        self.append_usage_fields(data)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "_DummyState":
        return cls(
            value=data.get("value", ""),
            sample_rate=int(data.get("sample_rate", 24000)),
            prompt_tokens=int(data.get("prompt_tokens", 0)),
            completion_tokens=int(data.get("completion_tokens", 0)),
            engine_time_s=float(data.get("engine_time_s", 0.0)),
        )


def test_build_usage_omits_empty_usage() -> None:
    assert build_usage(_DummyState()) is None


def test_build_usage_includes_total_and_rounded_engine_time() -> None:
    state = _DummyState(prompt_tokens=3, completion_tokens=5, engine_time_s=1.23456789)

    assert build_usage(state) == {
        "prompt_tokens": 3,
        "completion_tokens": 5,
        "total_tokens": 8,
        "engine_time_s": 1.234568,
    }


def test_load_and_store_state_round_trip_stage_payload() -> None:
    payload = StagePayload(
        request_id="req",
        request=OmniRequest(inputs={}),
        data={"value": "ok", "prompt_tokens": 2},
    )

    state = load_state(payload, _DummyState)
    state.completion_tokens = 4
    stored = store_state(payload, state)

    assert stored is payload
    assert payload.data == {
        "value": "ok",
        "sample_rate": 24000,
        "prompt_tokens": 2,
        "completion_tokens": 4,
    }


def test_serialize_value_detaches_tensor_to_cpu() -> None:
    tensor = torch.tensor([1, 2], requires_grad=False)

    value = PipelineStateBase.serialize_value(tensor)

    assert isinstance(value, torch.Tensor)
    assert value.device.type == "cpu"
    assert value.tolist() == [1, 2]


def test_tts_pipeline_states_share_base_usage_contract() -> None:
    import dataclasses

    from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
    from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
    from sglang_omni.models.moss_tts.payload_types import MossTTSState
    from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
    from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
    from sglang_omni.models.voxtral_tts.io import VoxtralTTSState

    # Every in-scope TTS model routes its state through PipelineStateBase.
    state_classes = (
        S2ProState,
        HiggsTtsState,
        MossTTSState,
        MossTTSLocalState,
        Qwen3TTSState,
        VoxtralTTSState,
    )
    base_fields = {
        "sample_rate",
        "prompt_tokens",
        "completion_tokens",
        "engine_time_s",
        "schema_version",
    }

    for state_cls in state_classes:
        assert issubclass(state_cls, PipelineStateBase), state_cls.__name__
        field_names = {f.name for f in dataclasses.fields(state_cls)}
        missing = base_fields - field_names
        assert not missing, f"{state_cls.__name__} missing base fields: {missing}"
        assert callable(getattr(state_cls, "to_dict", None)), state_cls.__name__
        assert callable(getattr(state_cls, "from_dict", None)), state_cls.__name__


def test_schema_version_guard_is_opt_in() -> None:
    # Default None -> no tag written, preserving today's behavior.
    data: dict[str, Any] = {}
    PipelineStateBase().append_schema_version(data)
    assert "schema_version" not in data

    # check_schema_version accepts untagged payloads and matching tags.
    PipelineStateBase.check_schema_version({}, 2)
    PipelineStateBase.check_schema_version({"schema_version": 2}, 2)
    with pytest.raises(ValueError):
        PipelineStateBase.check_schema_version({"schema_version": 1}, 2)


def test_schema_version_opt_in_writes_tag() -> None:
    data: dict[str, Any] = {}
    PipelineStateBase(schema_version=3).append_schema_version(data)
    assert data["schema_version"] == 3


def test_base_requires_to_dict_and_from_dict() -> None:
    with pytest.raises(NotImplementedError):
        PipelineStateBase().to_dict()
    with pytest.raises(NotImplementedError):
        PipelineStateBase.from_dict({})


def test_typed_tensor_round_trip_preserves_values() -> None:
    codes = torch.tensor([[1, 2, 3], [4, 5, 6]])

    data = encode_typed_tensor(codes, key="audio_codes")

    assert set(data) == {
        "audio_codes_bytes",
        "audio_codes_shape",
        "audio_codes_dtype",
    }
    restored = decode_typed_tensor(data, key="audio_codes")
    assert restored is not None
    assert restored.dtype == torch.int64
    assert restored.tolist() == [[1, 2, 3], [4, 5, 6]]


def test_typed_tensor_picks_int32_for_large_values() -> None:
    codes = torch.tensor([[70000, 1]])

    data = encode_typed_tensor(codes, key="audio_codes")

    assert data["audio_codes_dtype"] == "int32"
    assert decode_typed_tensor(data, key="audio_codes").tolist() == [[70000, 1]]


def test_typed_tensor_legacy_list_fallback_and_missing() -> None:
    restored = decode_typed_tensor(
        {"audio_codes": [[1, 2], [3, 4]]},
        key="audio_codes",
        legacy_key="audio_codes",
    )
    assert restored.tolist() == [[1, 2], [3, 4]]
    assert decode_typed_tensor({}, key="audio_codes") is None
