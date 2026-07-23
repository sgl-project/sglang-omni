from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
import torch

from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.pipeline_state import (
    DeclarativeStateBase,
    PipelineStateBase,
    build_usage,
    load_state,
    store_state,
    wire,
)
from sglang_omni.scheduling.typed_tensor import decode_typed_tensor, encode_typed_tensor


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

    from sglang_omni.models.audar_tts.payload_types import AudarTTSState
    from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
    from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
    from sglang_omni.models.ming_tts.payload_types import MingTTSState
    from sglang_omni.models.moss_tts.payload_types import MossTTSState
    from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
    from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
    from sglang_omni.models.voxtral_tts.io import VoxtralTTSState

    # Every in-scope TTS model routes its state through PipelineStateBase.
    state_classes = (
        AudarTTSState,
        S2ProState,
        HiggsTtsState,
        MingTTSState,
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
    }

    for state_cls in state_classes:
        assert issubclass(state_cls, PipelineStateBase), state_cls.__name__
        field_names = {f.name for f in dataclasses.fields(state_cls)}
        missing = base_fields - field_names
        assert not missing, f"{state_cls.__name__} missing base fields: {missing}"
        assert callable(getattr(state_cls, "to_dict", None)), state_cls.__name__
        assert callable(getattr(state_cls, "from_dict", None)), state_cls.__name__


def _normalize_payload_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "dtype": str(value.dtype),
            "shape": list(value.shape),
            "data": value.detach().cpu().tolist(),
        }
    if isinstance(value, dict):
        return {key: _normalize_payload_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_payload_value(item) for item in value]
    return value


def _assert_round_trip_preserves_payload(state: PipelineStateBase) -> None:
    before = state.to_dict()
    restored = type(state).from_dict(before)
    after = restored.to_dict()

    assert set(after) == set(before), type(state).__name__
    assert _normalize_payload_value(after) == _normalize_payload_value(before)


def _assert_restored_fields(
    state: PipelineStateBase, overrides: dict[str, Any] | None = None
) -> None:
    """Field-complete check on the *restored object's attributes*.

    Every dataclass field of the restored state must equal the originally
    constructed value; overrides lists only the fields whose representation
    intentionally changes across the round trip (tensor-to-list models).
    Iterating all fields instead of an explicit expected mapping makes it
    impossible to silently omit a field: a to_dict() bug that drops a field
    falls back to the dataclass default on restore, which a dict-vs-dict
    comparison cannot see, and a hand-written expected mapping can forget
    non-default fields like MOSS-Local's sample_rate=48000 (both raised in
    PR #1019 review)."""
    import dataclasses

    overrides = overrides or {}
    field_names = {field.name for field in dataclasses.fields(state)}
    unknown = set(overrides) - field_names
    assert not unknown, f"overrides for unknown fields: {unknown}"

    restored = type(state).from_dict(state.to_dict())
    for field in dataclasses.fields(state):
        expected_value = overrides.get(field.name, getattr(state, field.name))
        actual_value = getattr(restored, field.name)
        assert _normalize_payload_value(actual_value) == _normalize_payload_value(
            expected_value
        ), (
            f"{type(state).__name__}.{field.name}: "
            f"{actual_value!r} != {expected_value!r}"
        )


def test_tts_pipeline_state_round_trips_preserve_payload_fields() -> None:
    from sglang_omni.models.audar_tts.payload_types import AudarTTSState
    from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
    from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState
    from sglang_omni.models.ming_tts.payload_types import MingTTSState
    from sglang_omni.models.moss_tts.payload_types import MossTTSState
    from sglang_omni.models.moss_tts_local.payload_types import MossTTSLocalState
    from sglang_omni.models.qwen3_tts.payload_types import Qwen3TTSState
    from sglang_omni.models.voxtral_tts.io import VoxtralTTSState

    # Each (state, overrides) pair is checked two ways: the
    # to_dict()-vs-to_dict() comparison (wire-format stability across a
    # double round trip), and a field-complete attribute check on the
    # restored object against the constructed values. overrides names only
    # the intentional representation changes: Qwen3-TTS flattens tensors to
    # lists in to_dict() and never rebuilds them in from_dict(); Voxtral does
    # the same only for audio_samples (audio_codes round-trips as a tensor
    # via typed_tensor bytes); MOSS / MOSS-Local / S2-Pro / Higgs round-trip
    # their tensor fields natively. Note the float32 tensor -> Python list
    # conversions derive expected values through the same precision path.
    cases: list[tuple[PipelineStateBase, dict[str, Any]]] = [
        (
            AudarTTSState(
                target_text="target",
                reference_text="reference",
                reference_audio={"bytes": b"wav"},
                prompt="prompt",
                audio_codes=[1, 2, 3],
                generation_kwargs={"temperature": 0.7},
                prompt_tokens=4,
                completion_tokens=6,
                engine_time_s=0.125,
            ),
            {},
        ),
        (
            S2ProState(
                input_ids=[1, 2, 3],
                vq_mask_tokens=[False, True, False],
                vq_parts=[torch.tensor([[1, 2], [3, 4]])],
                output_codes=torch.tensor([[0, 1], [2, 3]]),
                prompt_tokens=3,
                completion_tokens=5,
                engine_time_s=0.125,
                finish_reason="stop",
                audio_samples=[0.1, 0.2],
            ),
            {},
        ),
        (
            HiggsTtsState(
                prompt_token_ids=[10, 11],
                reference_codes_delayed=[[1, 2], [3, 4]],
                target_text="target",
                reference_text="reference",
                reference_waveform=torch.tensor([[[0.1, 0.2]]]),
                reference_code_cache_key="cache-key",
                uploaded_voice_name="voice",
                uploaded_voice_created_at=123,
                top_p=0.9,
                top_k=10,
                seed=7,
                return_logprob=True,
                return_omni_rollout=True,
                output_codes_delayed=[[5, 6], [7, 8]],
                omni_rollout={"tokens": [1, 2], "logprobs": [-0.1, -0.2]},
                prompt_tokens=2,
                completion_tokens=4,
                engine_time_s=0.25,
                audio_samples=torch.tensor([0.3, 0.4]),
            ),
            {},
        ),
        (
            MingTTSState(
                text="hello",
                prompt="prompt",
                instructions="calm",
                language="en",
                voice="voice",
                ref_audio={"path": "ref.wav"},
                ref_text="reference",
                input_ids=[1, 2, 3, 4, 5, 6],
                prompt_text="reference",
                spk_token_positions=[1],
                spk_injection_positions=[2],
                audio_token_position=3,
                prompt_latent_start_position=4,
                prompt_latent_token_count=2,
                spk_emb=torch.zeros(1, 2),
                prompt_latent=torch.zeros(1, 1, 2),
                max_decode_steps=16,
                cfg=1.5,
                sigma=0.2,
                temperature=0.7,
                generated_latents=torch.tensor(
                    [[[0.5, -1.25]], [[2.0, 0.0]]],
                    dtype=torch.float32,
                ),
                generated_last_chunk=[False, True],
                stop_step=1,
                finish_reason="stop",
                prompt_tokens=6,
                completion_tokens=2,
                engine_time_s=0.125,
                sample_rate=44100,
                duration_s=0.5,
                audio_decode_time_s=0.25,
            ),
            {},
        ),
        (
            MossTTSState(
                text="hello",
                ref_audio={"path": "ref.wav"},
                ref_text="ref",
                language="en",
                instructions="calm",
                token_count=6,
                generation_kwargs={"temperature": 0.7},
                delayed_audio_codes=torch.tensor([[1, 2], [3, 4]]),
                assistant_start_length=2,
                prompt_tokens=6,
                completion_tokens=8,
                engine_time_s=0.375,
            ),
            {},
        ),
        (
            MossTTSLocalState(
                text="hello",
                ref_audio={"path": "ref.wav"},
                ref_text="ref",
                language="en",
                instructions="bright",
                token_count=5,
                generation_kwargs={"top_p": 0.8},
                audio_codes=torch.tensor([[1, 2, 3], [4, 5, 6]]),
                prompt_tokens=5,
                completion_tokens=7,
                engine_time_s=0.5,
            ),
            {},
        ),
        (
            Qwen3TTSState(
                text="hello",
                task_type="Instruct",
                task_type_explicit=True,
                language="en",
                voice="voice",
                instructions="fast",
                ref_audio={"path": "ref.wav"},
                ref_text="ref",
                uploaded_voice_name="uploaded",
                uploaded_voice_created_at=456,
                x_vector_only_mode=True,
                non_streaming_mode=True,
                generation_kwargs={"seed": 9},
                seed=9,
                audio_codes=torch.tensor([[1, 2], [3, 4]]),
                ref_code_len=1,
                audio_samples=torch.tensor([0.5, 0.6]),
                prompt_tokens=9,
                completion_tokens=11,
                engine_time_s=0.625,
            ),
            {
                "audio_codes": [[1, 2], [3, 4]],
                "audio_samples": torch.tensor([0.5, 0.6]).tolist(),
            },
        ),
        (
            VoxtralTTSState(
                input_ids=[1, 2],
                voice="voice",
                max_new_tokens=16,
                audio_codes=torch.tensor([[1, 2], [3, 4]]),
                prompt_tokens=2,
                completion_tokens=3,
                engine_time_s=0.75,
                audio_samples=torch.tensor([0.7, 0.8]),
            ),
            {
                "audio_samples": torch.tensor([0.7, 0.8]).tolist(),
            },
        ),
    ]

    for state, overrides in cases:
        _assert_round_trip_preserves_payload(state)
        _assert_restored_fields(state, overrides)


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


def test_typed_tensor_float_round_trip_transports_as_float32() -> None:
    latents = torch.tensor([[0.5, -1.25], [2.0, 0.0]], dtype=torch.bfloat16)

    data = encode_typed_tensor(latents, key="latents")

    assert data["latents_dtype"] == "float32"
    restored = decode_typed_tensor(data, key="latents")
    assert restored.dtype == torch.float32
    assert restored.tolist() == [[0.5, -1.25], [2.0, 0.0]]


def test_typed_tensor_empty_float_round_trip_keeps_shape() -> None:
    data = encode_typed_tensor(torch.empty((0, 2, 3)), key="latents")

    assert data["latents_dtype"] == "float32"
    restored = decode_typed_tensor(data, key="latents")
    assert restored.shape == (0, 2, 3)
    assert restored.dtype == torch.float32


def test_typed_tensor_legacy_list_fallback_and_missing() -> None:
    restored = decode_typed_tensor(
        {"audio_codes": [[1, 2], [3, 4]]},
        key="audio_codes",
        legacy_key="audio_codes",
    )
    assert restored.tolist() == [[1, 2], [3, 4]]
    assert decode_typed_tensor({}, key="audio_codes") is None


def test_declarative_typed_tensor_missing_payload_keeps_default() -> None:
    @dataclass
    class _TypedDefaultState(DeclarativeStateBase):
        audio_codes: Any = wire(default_factory=lambda: [[9, 10]], codec="typed_tensor")

    assert _TypedDefaultState.from_dict({}).audio_codes == [[9, 10]]
    assert _TypedDefaultState.from_dict({"audio_codes": None}).audio_codes is None


@pytest.mark.parametrize(
    "payload",
    [
        {"audio_codes_shape": [1]},
        {"audio_codes_bytes": b"\x01\x00"},
        {"audio_codes_dtype": "uint16"},
        {
            "audio_codes_bytes": b"\x01\x00",
            "audio_codes_dtype": "uint16",
        },
        {"audio_codes_bytes": None, "audio_codes_shape": [1]},
        {
            "audio_codes_bytes": b"\x01\x00",
            "audio_codes_shape": [1],
            "audio_codes_dtype": None,
        },
    ],
)
def test_declarative_typed_tensor_rejects_partial_payload(
    payload: dict[str, Any],
) -> None:
    @dataclass
    class _TypedState(DeclarativeStateBase):
        audio_codes: Any = wire(None, codec="typed_tensor")

    with pytest.raises(ValueError, match="typed_tensor payload"):
        _TypedState.from_dict(payload)


def test_declarative_typed_tensor_allows_omitted_dtype() -> None:
    @dataclass
    class _TypedState(DeclarativeStateBase):
        audio_codes: Any = wire(None, codec="typed_tensor")

    payload = encode_typed_tensor(torch.tensor([1, 2]), key="audio_codes")
    del payload["audio_codes_dtype"]

    assert _TypedState.from_dict(payload).audio_codes.tolist() == [1, 2]


def test_declarative_default_factory_is_lazy() -> None:
    calls = 0

    def make_items() -> list[int]:
        nonlocal calls
        calls += 1
        return [1]

    @dataclass
    class _FactoryState(DeclarativeStateBase):
        items: list[int] = wire(default_factory=make_items, codec="list")

    state = _FactoryState()
    assert calls == 1

    calls_before_to_dict = calls
    payload = state.to_dict()
    assert calls == calls_before_to_dict

    calls_before_restore = calls
    restored = _FactoryState.from_dict(payload)
    assert calls == calls_before_restore
    assert restored.items == [1]


@pytest.mark.parametrize("emit", ["not-non", "with:audio_samples"])
def test_declarative_wire_rejects_unknown_emit_mode(emit: str) -> None:
    with pytest.raises(ValueError, match="unknown wire emit mode"):
        wire(None, emit=emit)


def test_higgs_sample_rate_emission_stays_model_local() -> None:
    from sglang_omni.models.higgs_tts.payload_types import HiggsTtsState

    assert "sample_rate" not in HiggsTtsState().to_dict()
    data = HiggsTtsState(sample_rate=16000, audio_samples=torch.tensor([0.1])).to_dict()
    assert data["sample_rate"] == 16000
