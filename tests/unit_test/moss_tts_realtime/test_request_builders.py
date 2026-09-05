# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sglang_omni.models.moss_tts_realtime import request_builders as rb
from sglang_omni.models.moss_tts_realtime import text_delta
from sglang_omni.models.moss_tts_realtime.payload_types import MossTTSRealtimeState
from sglang_omni.models.moss_tts_realtime.request_state import (
    MossTTSRealtimePendingInput,
    MossTTSRealtimeRequestData,
    MossTTSRealtimeTurnLedger,
    MossTTSRealtimeTurnPhase,
    MossTTSRealtimeTurnState,
)
from sglang_omni.proto import OmniRequest, StagePayload
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_BOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_BOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_EOS_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import MODEL_CONFIG
from tests.unit_test.moss_tts_realtime.runtime_config import (
    REFERENCE_AUDIO_PAD_TOKEN_ID as MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID,
)
from tests.unit_test.moss_tts_realtime.runtime_config import (
    TEXT_PAD_TOKEN_ID as MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID,
)

N_VQ = int(MODEL_CONFIG.rvq)
ROW_WIDTH = N_VQ + 1
CODEBOOK_SIZE = int(MODEL_CONFIG.audio_pad_token)


class FakeTokenizer:
    def __init__(self, *, size: int = 200_000) -> None:
        self.size = size
        self.len_calls = 0
        self.calls: list[tuple[str, bool | None]] = []

    def __len__(self) -> int:
        self.len_calls += 1
        return self.size

    def encode(
        self,
        text: str,
        add_special_tokens: bool | None = None,
    ) -> list[int]:
        self.calls.append((text, add_special_tokens))
        if text == rb._ASSISTANT_TURN_PREFIX:
            return [151645, 198, 151644, 77091, 198]
        return [1000 + (ord(char) % 1000) for char in text]


class FakeProcessor:
    channels = N_VQ
    delay_tokens_len = 12
    audio_channel_pad = MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID
    audio_bos_token = MOSS_TTS_REALTIME_AUDIO_BOS_TOKEN_ID
    audio_eos_token = MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID
    audio_pad_token_id = MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID
    text_pad_token_id = MOSS_TTS_REALTIME_TEXT_PAD_TOKEN_ID

    def __init__(self) -> None:
        self.model_config = MODEL_CONFIG
        self.tokenizer = FakeTokenizer()
        self.ensemble_calls: list[np.ndarray | None] = []
        self.user_calls: list[tuple[str, np.ndarray]] = []

    def make_ensemble(self, voice_codes: np.ndarray | None = None) -> np.ndarray:
        copied = None if voice_codes is None else np.array(voice_codes, copy=True)
        self.ensemble_calls.append(copied)
        frame_count = 0 if copied is None else copied.shape[0]
        rows = np.full(
            (1 + frame_count, ROW_WIDTH),
            MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
            dtype=np.int64,
        )
        rows[0, 0] = 10
        if copied is not None:
            rows[1:, 0] = MOSS_TTS_REALTIME_REFERENCE_AUDIO_PAD_TOKEN_ID
            rows[1:, 1:] = copied
        return rows

    def make_user_prompt(self, text: str, audio_codes: np.ndarray) -> np.ndarray:
        copied = np.array(audio_codes, copy=True)
        self.user_calls.append((text, copied))
        rows = np.full(
            (2, ROW_WIDTH),
            MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID,
            dtype=np.int64,
        )
        rows[:, 0] = [20, 21]
        rows[0, 1:] = copied[0]
        rows[1, 1] = MOSS_TTS_REALTIME_AUDIO_EOS_TOKEN_ID
        return rows


class RecordingAudioEncoder:
    def __init__(self, *, frames: int = 4) -> None:
        self.frames = frames
        self.calls: list[object] = []

    def encode(self, value: object) -> SimpleNamespace:
        self.calls.append(value)
        codec_quantizers = 2 * N_VQ
        codes = torch.arange(codec_quantizers * self.frames, dtype=torch.long).reshape(
            codec_quantizers, 1, self.frames
        )
        return SimpleNamespace(audio_codes=codes.remainder(CODEBOOK_SIZE))


def _codes(frames: int = 4, *, quantizers: int = N_VQ) -> np.ndarray:
    values = np.arange(frames * quantizers, dtype=np.int64)
    return values.reshape(frames, quantizers) % CODEBOOK_SIZE


def _normalize_audio_codes(value: object) -> np.ndarray:
    return rb.normalize_moss_tts_realtime_audio_codes(
        value,
        num_codebooks=N_VQ,
        codebook_size=CODEBOOK_SIZE,
    )


def _payload(
    inputs: object = None,
    *,
    params: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
    data: object = None,
    request_id: str = "req-1",
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs=inputs,
            params=params or {},
            metadata=metadata or {},
        ),
        data={} if data is None else data,
    )


@pytest.fixture(autouse=True)
def _clear_prepared_queue(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(text_delta, "_TOKENIZER_VOCAB_SIZE", None)
    rb.clear_moss_tts_realtime_preprocessing_context()
    yield
    rb.clear_moss_tts_realtime_preprocessing_context()


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("channels", 15),
        ("channels", float(N_VQ)),
        ("delay_tokens_len", 11),
        ("audio_channel_pad", 0),
        ("audio_bos_token", 0),
        ("audio_eos_token", 0),
        ("audio_pad_token_id", 0),
        ("text_pad_token_id", 0),
    ],
)
def test_processor_contract_rejects_mismatched_snapshot_values(
    name: str,
    value: object,
) -> None:
    processor = FakeProcessor()
    setattr(processor, name, value)

    with pytest.raises(ValueError, match=name):
        rb.set_moss_tts_realtime_preprocessing_context(processor=processor)


def test_offline_state_defaults_match_hf_sampling_contract() -> None:
    state = rb.build_moss_tts_realtime_state(
        _payload("hello", request_id="offline"),
        num_codebooks=N_VQ,
    )

    assert state.session_id == "offline:offline"
    assert state.turn_id == "offline"
    assert state.initial_text == "hello"
    assert state.initial_token_ids == []
    assert state.input_done is True
    assert state.keep_session is False
    assert state.generation_kwargs == {
        "max_new_tokens": 1000,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 30,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "repetition_window": 50,
    }
    assert state.stream_metadata == {
        "stream": True,
        "modality": "audio_codes",
        "n_vq": N_VQ,
        # Chunk-level identity stamps keep the vocoder session-keyed even when
        # the terminal payload arrives after the audio chunks.
        "session_id": "offline:offline",
        "turn_id": "offline",
    }


def test_offline_state_ignores_implicit_speech_sampling_defaults() -> None:
    state = rb.build_moss_tts_realtime_state(
        _payload(
            "hello",
            params={
                "temperature": 0.8,
                "top_p": 0.8,
                "top_k": 30,
                "repetition_penalty": 1.1,
            },
            metadata={
                "tts_params": {
                    "voice": "default",
                    "response_format": "wav",
                    "speed": 1.0,
                }
            },
        ),
        num_codebooks=N_VQ,
    )

    assert state.generation_kwargs == {
        "max_new_tokens": 1000,
        "temperature": 0.8,
        "top_p": 0.6,
        "top_k": 30,
        "do_sample": True,
        "repetition_penalty": 1.1,
        "repetition_window": 50,
    }


def test_offline_state_preserves_explicit_speech_sampling_overrides() -> None:
    state = rb.build_moss_tts_realtime_state(
        _payload(
            "hello",
            params={
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 20,
                "repetition_penalty": 1.05,
                "seed": 17,
            },
            metadata={
                "tts_params": {
                    "explicit_generation_params": [
                        "max_new_tokens",
                        "temperature",
                        "top_p",
                        "top_k",
                        "repetition_penalty",
                        "seed",
                    ]
                }
            },
        ),
        num_codebooks=N_VQ,
    )

    assert state.generation_kwargs == {
        "max_new_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.8,
        "top_k": 20,
        "do_sample": True,
        "repetition_penalty": 1.05,
        "repetition_window": 50,
        "seed": 17,
    }


def test_direct_request_sampling_params_remain_authoritative() -> None:
    state = rb.build_moss_tts_realtime_state(
        _payload(
            "hello",
            params={
                "temperature": 0.6,
                "top_p": 0.7,
                "top_k": 15,
                "repetition_penalty": 1.02,
            },
        ),
        num_codebooks=N_VQ,
    )

    assert state.generation_kwargs["temperature"] == 0.6
    assert state.generation_kwargs["top_p"] == 0.7
    assert state.generation_kwargs["top_k"] == 15
    assert state.generation_kwargs["repetition_penalty"] == 1.02


def test_offline_state_preserves_explicit_open_input() -> None:
    state = rb.build_moss_tts_realtime_state(
        _payload("hello", params={"input_done": False}),
        num_codebooks=N_VQ,
    )

    assert state.input_done is False


def test_offline_state_keeps_an_explicit_session() -> None:
    state = rb.build_moss_tts_realtime_state(
        _payload({"text": "hello", "session_id": "session-1"}),
        num_codebooks=N_VQ,
    )

    assert state.session_id == "session-1"
    assert state.keep_session is True


def test_state_payload_round_trip_preserves_new_realtime_fields() -> None:
    original = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        turn_index=2,
        user_text="context",
        user_audio={"audio_path": "user.wav"},
        initial_token_ids=[10, 11],
        input_done=True,
        keep_session=False,
        generation_kwargs={"temperature": 0.0, "do_sample": False},
    )

    restored = rb.build_moss_tts_realtime_state(
        _payload("ignored", data=original.to_dict())
    )

    assert restored.session_id == "session"
    assert restored.turn_id == "turn"
    assert restored.turn_index == 2
    assert restored.user_text == "context"
    assert restored.user_audio == {"audio_path": "user.wav"}
    assert restored.initial_token_ids == [10, 11]
    assert restored.initial_text is None
    assert restored.input_done is True
    assert restored.keep_session is False
    assert restored.generation_kwargs["temperature"] == 0.0
    assert restored.generation_kwargs["do_sample"] is False


@pytest.mark.parametrize(
    "payload",
    [
        _payload({"initial_token_ids": [True]}),
        _payload({"initial_token_ids": [1.5]}),
        _payload({"turn_index": True}),
        _payload("hello", params={"max_new_tokens": True}),
        _payload("hello", params={"temperature": float("nan")}),
        _payload("hello", params={"top_p": "0.6"}),
        _payload("hello", params={"do_sample": "false"}),
        _payload("hello", params={"seed": True}),
        _payload("hello", params={"stream": 1}),
    ],
)
def test_state_builder_rejects_lossy_scalar_coercions(payload: StagePayload) -> None:
    with pytest.raises((TypeError, ValueError)):
        rb.build_moss_tts_realtime_state(payload)


def test_state_builder_rejects_malformed_nested_inputs() -> None:
    with pytest.raises(TypeError, match="request.inputs"):
        rb.build_moss_tts_realtime_state(_payload(["hello"]))
    with pytest.raises(TypeError, match="inputs.user"):
        rb.build_moss_tts_realtime_state(_payload({"user": "not-a-mapping"}))
    with pytest.raises(TypeError, match="inputs.references"):
        rb.build_moss_tts_realtime_state(_payload({"references": {}}))


def test_audio_codes_normalize_supported_codec_layouts() -> None:
    codec_quantizers = 2 * N_VQ
    quantizer_major = (
        np.arange(codec_quantizers * 80, dtype=np.int64).reshape(codec_quantizers, 80)
        % CODEBOOK_SIZE
    )
    expected = quantizer_major[:N_VQ].T

    for value in (
        quantizer_major,
        quantizer_major.T,
        quantizer_major[:, None, :],
        quantizer_major[None, :, :],
        expected,
        expected.T,
    ):
        actual = _normalize_audio_codes(value)
        np.testing.assert_array_equal(actual, expected)
        assert actual.shape == (80, N_VQ)
        assert actual.dtype == np.int64
        assert actual.flags.c_contiguous


@pytest.mark.parametrize(
    "value",
    [
        np.zeros((2, 100), dtype=np.float32),
        np.zeros((1, 1, 100), dtype=np.int64),
        np.zeros((100, 100), dtype=np.int64),
        np.full((2, N_VQ), CODEBOOK_SIZE, dtype=np.int64),
        {"audio_codes": None},
    ],
)
def test_audio_codes_reject_invalid_layout_dtype_or_range(value: object) -> None:
    with pytest.raises((TypeError, ValueError), match="audio|rank|shape"):
        _normalize_audio_codes(value)


def test_raw_waveform_is_encoded_instead_of_misclassified_as_codes() -> None:
    processor = FakeProcessor()
    encoder = RecordingAudioEncoder(frames=5)
    waveform = np.zeros((1, 1, 3200), dtype=np.float32)
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        ref_audio=waveform,
        initial_token_ids=[1],
    )

    prepared = rb.prepare_moss_tts_realtime_state(
        state,
        processor=processor,
        audio_encoder=encoder,
    )

    assert len(encoder.calls) == 1
    assert encoder.calls[0] is waveform
    assert prepared.voice_codes is not None
    assert prepared.voice_codes.shape == (5, N_VQ)


def test_voice_reference_and_user_audio_use_distinct_encoders() -> None:
    processor = FakeProcessor()
    reference_encoder = RecordingAudioEncoder(frames=3)
    audio_encoder = RecordingAudioEncoder(frames=2)
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        ref_audio="voice.wav",
        user_text="user context",
        user_audio="user.wav",
        initial_token_ids=[1],
    )

    prepared = rb.prepare_moss_tts_realtime_state(
        state,
        processor=processor,
        audio_encoder=audio_encoder,
        reference_encoder=reference_encoder,
    )

    assert reference_encoder.calls == ["voice.wav"]
    assert audio_encoder.calls == ["user.wav"]
    assert prepared.voice_codes is not None
    assert prepared.voice_codes.shape == (3, N_VQ)
    assert prepared.user_audio_codes is not None
    assert prepared.user_audio_codes.shape == (2, N_VQ)


def test_falsey_reference_encoder_is_not_replaced() -> None:
    class FalseyAudioEncoder(RecordingAudioEncoder):
        def __bool__(self) -> bool:
            return False

    processor = FakeProcessor()
    reference_encoder = FalseyAudioEncoder(frames=3)
    audio_encoder = RecordingAudioEncoder(frames=2)
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        ref_audio="voice.wav",
        initial_token_ids=[1],
    )

    prepared = rb.prepare_moss_tts_realtime_state(
        state,
        processor=processor,
        audio_encoder=audio_encoder,
        reference_encoder=reference_encoder,
    )

    assert reference_encoder.calls == ["voice.wav"]
    assert audio_encoder.calls == []
    assert prepared.voice_codes is not None
    assert prepared.voice_codes.shape == (3, N_VQ)


def test_later_turn_skips_unused_voice_reference_encode() -> None:
    processor = FakeProcessor()
    reference_encoder = RecordingAudioEncoder()
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn-2",
        turn_index=1,
        ref_audio="voice.wav",
        initial_token_ids=[1],
    )

    prepared = rb.prepare_moss_tts_realtime_state(
        state,
        processor=processor,
        reference_encoder=reference_encoder,
    )

    assert reference_encoder.calls == []
    assert prepared.voice_codes is None
    assert prepared.include_system_prompt is False


def test_preencoded_nested_codes_bypass_audio_encoder() -> None:
    processor = FakeProcessor()
    encoder = RecordingAudioEncoder()
    reference_encoder = RecordingAudioEncoder()
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        ref_audio=_codes(3).tolist(),
        initial_token_ids=[1],
    )

    prepared = rb.prepare_moss_tts_realtime_state(
        state,
        processor=processor,
        audio_encoder=encoder,
        reference_encoder=reference_encoder,
    )

    assert encoder.calls == []
    assert reference_encoder.calls == []
    assert prepared.voice_codes is not None
    np.testing.assert_array_equal(prepared.voice_codes.numpy(), _codes(3))


def test_text_only_and_user_context_prompt_lowering() -> None:
    processor = FakeProcessor()
    voice_codes = _codes(3)
    user_codes = _codes(2)

    text_only = rb.build_moss_tts_realtime_turn_prompt(
        processor=processor,
        voice_codes=voice_codes,
        user_text=None,
        user_audio_codes=None,
        include_system_prompt=True,
    )
    assert text_only.shape == (1 + len(voice_codes) + 5, ROW_WIDTH)
    np.testing.assert_array_equal(text_only[1 : 1 + len(voice_codes), 1:], voice_codes)
    assert text_only[-5:, 0].tolist() == [151645, 198, 151644, 77091, 198]

    later_turn = rb.build_moss_tts_realtime_turn_prompt(
        processor=processor,
        voice_codes=voice_codes,
        user_text="user context",
        user_audio_codes=user_codes,
        include_system_prompt=False,
    )
    assert later_turn.shape == (2, ROW_WIDTH)
    assert len(processor.ensemble_calls) == 1
    assert processor.user_calls[-1][0] == "user context"
    np.testing.assert_array_equal(processor.user_calls[-1][1], user_codes)


def test_prompt_lowering_rejects_incomplete_context_and_lossy_rows() -> None:
    processor = FakeProcessor()
    with pytest.raises(ValueError, match="both user text and user audio"):
        rb.build_moss_tts_realtime_turn_prompt(
            processor=processor,
            voice_codes=None,
            user_text="context",
            user_audio_codes=None,
            include_system_prompt=True,
        )

    processor.make_ensemble = lambda voice_codes=None: np.zeros(
        (1, ROW_WIDTH), dtype=np.float32
    )
    with pytest.raises(TypeError, match="integer dtype"):
        rb.build_moss_tts_realtime_turn_prompt(
            processor=processor,
            voice_codes=None,
            user_text=None,
            user_audio_codes=None,
            include_system_prompt=True,
        )


def test_prepare_uses_text_tokenization_or_preserves_direct_tokens() -> None:
    processor = FakeProcessor()
    text_state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn-text",
        initial_text="Hi",
    )
    direct_state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn-direct",
        turn_index=1,
        initial_token_ids=[7, 8],
    )

    text_prepared = rb.prepare_moss_tts_realtime_state(
        text_state,
        processor=processor,
    )
    direct_prepared = rb.prepare_moss_tts_realtime_state(
        direct_state,
        processor=processor,
    )

    assert text_prepared.initial_token_ids == (1072, 1105)
    assert ("Hi", False) in processor.tokenizer.calls
    assert direct_prepared.initial_token_ids == (7, 8)
    assert text_prepared.include_system_prompt is True
    assert direct_prepared.include_system_prompt is False


def test_prepare_resolves_tokenizer_size_once() -> None:
    processor = FakeProcessor()

    rb.prepare_moss_tts_realtime_state(
        MossTTSRealtimeState(
            session_id="session",
            turn_id="turn",
            initial_text="hello",
        ),
        processor=processor,
    )

    assert processor.tokenizer.len_calls == 1


def test_preprocessing_reuses_process_global_tokenizer_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeProcessor()
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)

    initial_size = processor.tokenizer.size
    assert text_delta.get_moss_tts_realtime_tokenizer_vocab_size() == initial_size
    assert processor.tokenizer.len_calls == 1

    for request_id in ("first", "second"):
        output = rb.preprocess_moss_tts_realtime_payload(
            _payload("hello", request_id=request_id)
        )
        rb.cleanup_prepared_moss_tts_realtime_request(output.request_id)

    assert processor.tokenizer.len_calls == 1

    processor.tokenizer.size += 1
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)
    assert processor.tokenizer.len_calls == 1
    assert text_delta.get_moss_tts_realtime_tokenizer_vocab_size() == initial_size

    monkeypatch.setattr(text_delta, "_TOKENIZER_VOCAB_SIZE", None)
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)
    assert processor.tokenizer.len_calls == 2
    assert (
        text_delta.get_moss_tts_realtime_tokenizer_vocab_size()
        == processor.tokenizer.size
    )


def test_prepare_rejects_direct_token_outside_tokenizer() -> None:
    processor = FakeProcessor()
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        initial_token_ids=[len(processor.tokenizer)],
    )

    with pytest.raises(ValueError, match="exceeds tokenizer size"):
        rb.prepare_moss_tts_realtime_state(state, processor=processor)


def test_row_cache_keys_are_deterministic_and_cover_all_columns() -> None:
    rows = torch.full((2, ROW_WIDTH), MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID)
    rows[:, 0] = 42
    different = rows.clone()
    different[1, 7] = 3

    first = rb.build_moss_tts_realtime_row_cache_key_ids(rows)
    second = rb.build_moss_tts_realtime_row_cache_key_ids(rows.clone())
    changed = rb.build_moss_tts_realtime_row_cache_key_ids(different)

    assert first == second
    assert first[0] == first[1]
    assert changed[0] == first[0]
    assert changed[1] != first[1]
    with pytest.raises(TypeError, match="integer tensor"):
        rb.build_moss_tts_realtime_row_cache_key_ids(rows.float())


def test_prepared_handoff_publish_pop_and_cleanup_lifecycle() -> None:
    processor = FakeProcessor()
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)

    output = rb.preprocess_moss_tts_realtime_payload(_payload("hello"))
    snapshot = rb.moss_tts_realtime_prepared_snapshot()
    assert snapshot.prepared == frozenset({"req-1"})
    assert snapshot.inflight == frozenset()
    assert output.data[rb._MOSS_TTS_REALTIME_PREPARED_MARKER] == "req-1"

    prepared = rb.pop_prepared_moss_tts_realtime_request(output)
    assert prepared is not None
    assert prepared.state.initial_text == "hello"
    assert not rb.moss_tts_realtime_prepared_snapshot().prepared

    with pytest.raises(RuntimeError, match="state is missing"):
        rb.pop_prepared_moss_tts_realtime_request(output)


def test_marker_mismatch_does_not_consume_another_requests_handoff() -> None:
    processor = FakeProcessor()
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)
    output = rb.preprocess_moss_tts_realtime_payload(_payload("hello"))
    wrong = StagePayload(
        request_id="other",
        request=output.request,
        data=dict(output.data),
    )

    with pytest.raises(ValueError, match="must match payload.request_id"):
        rb.pop_prepared_moss_tts_realtime_request(wrong)
    assert rb.moss_tts_realtime_prepared_snapshot().prepared == frozenset({"req-1"})

    rb.cleanup_prepared_moss_tts_realtime_request("req-1")
    assert not rb.moss_tts_realtime_prepared_snapshot().prepared


def test_abort_or_context_reset_during_preprocessing_leaves_no_stale_marker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    processor = FakeProcessor()

    def fake_prepare(
        payload: StagePayload,
        *,
        processor: object,
        audio_encoder: object = None,
        reference_encoder: object = None,
    ) -> rb.MossTTSRealtimePreparedRequest:
        del processor, audio_encoder, reference_encoder
        rb.cleanup_prepared_moss_tts_realtime_request(payload.request_id)
        rows = torch.full((1, ROW_WIDTH), MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID)
        rows[0, 0] = 1
        return rb.MossTTSRealtimePreparedRequest(
            state=MossTTSRealtimeState(session_id="s", turn_id="t"),
            turn_prompt_rows=rows,
            turn_prompt_cache_ids=[1],
            turn_prompt_input_ids=torch.tensor([1]),
            initial_token_ids=(),
            voice_codes=None,
            user_audio_codes=None,
            include_system_prompt=True,
            generation_kwargs={},
        )

    monkeypatch.setattr(rb, "prepare_moss_tts_realtime_request", fake_prepare)
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)
    output = rb.preprocess_moss_tts_realtime_payload(_payload("hello"))

    assert rb._MOSS_TTS_REALTIME_PREPARED_MARKER not in output.data
    snapshot = rb.moss_tts_realtime_prepared_snapshot()
    assert not snapshot.prepared
    assert not snapshot.inflight
    assert not snapshot.aborted

    def reset_prepare(
        payload: StagePayload,
        *,
        processor: object,
        audio_encoder: object = None,
        reference_encoder: object = None,
    ) -> rb.MossTTSRealtimePreparedRequest:
        del payload, processor, audio_encoder, reference_encoder
        rb.clear_moss_tts_realtime_preprocessing_context()
        return fake_prepare_result()

    def fake_prepare_result() -> rb.MossTTSRealtimePreparedRequest:
        rows = torch.full((1, ROW_WIDTH), MOSS_TTS_REALTIME_AUDIO_PAD_TOKEN_ID)
        rows[0, 0] = 1
        return rb.MossTTSRealtimePreparedRequest(
            state=MossTTSRealtimeState(session_id="s", turn_id="t"),
            turn_prompt_rows=rows,
            turn_prompt_cache_ids=[1],
            turn_prompt_input_ids=torch.tensor([1]),
            initial_token_ids=(),
            voice_codes=None,
            user_audio_codes=None,
            include_system_prompt=True,
            generation_kwargs={},
        )

    monkeypatch.setattr(rb, "prepare_moss_tts_realtime_request", reset_prepare)
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)
    reset_output = rb.preprocess_moss_tts_realtime_payload(
        _payload("hello", request_id="reset")
    )
    assert rb._MOSS_TTS_REALTIME_PREPARED_MARKER not in reset_output.data
    assert rb.moss_tts_realtime_prepared_snapshot().context is None


def test_preprocessing_failure_rolls_back_inflight_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_prepare(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise RuntimeError("codec failed")

    monkeypatch.setattr(rb, "prepare_moss_tts_realtime_request", fail_prepare)
    rb.set_moss_tts_realtime_preprocessing_context(processor=FakeProcessor())

    with pytest.raises(RuntimeError, match="codec failed"):
        rb.preprocess_moss_tts_realtime_payload(_payload("hello"))
    snapshot = rb.moss_tts_realtime_prepared_snapshot()
    assert not snapshot.prepared
    assert not snapshot.inflight
    assert not snapshot.aborted


def test_prepared_payload_exposes_tokenized_readiness_and_worker_safe_data() -> None:
    processor = FakeProcessor()
    rb.set_moss_tts_realtime_preprocessing_context(processor=processor)

    output = rb.preprocess_moss_tts_realtime_payload(_payload("abcdefghijkl"))

    assert output.data[rb.MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY] == [
        1097,
        1098,
        1099,
        1100,
        1101,
        1102,
        1103,
        1104,
        1105,
        1106,
        1107,
        1108,
    ]
    data = rb.build_moss_tts_realtime_request_data(
        output,
        model=SimpleNamespace(config=MODEL_CONFIG),
    )

    assert data.req is None
    assert data.initial_token_ids == tuple(
        output.data[rb.MOSS_TTS_REALTIME_PREPARED_INITIAL_TOKEN_IDS_KEY]
    )
    assert data.input_ids.tolist() == rb.build_moss_tts_realtime_row_cache_key_ids(
        data.prompt_rows
    )
    assert not rb.moss_tts_realtime_prepared_snapshot().prepared


def test_terminal_result_drops_internal_prompt_rows() -> None:
    generated_row = (77, *tuple(range(1, N_VQ + 1)))
    state = MossTTSRealtimeState(
        session_id="session",
        turn_id="turn",
        prompt_rows=torch.tensor([generated_row], dtype=torch.long),
    )
    turn = MossTTSRealtimeTurnState(
        session_id="session",
        turn_id="turn",
        request_id="request",
        pending_input=MossTTSRealtimePendingInput(
            max_tokens=1,
            max_bytes=1,
            max_updates=1,
            input_done=True,
        ),
        ledger=MossTTSRealtimeTurnLedger(
            model_config=MODEL_CONFIG,
            appended_rows=[generated_row],
        ),
        phase=MossTTSRealtimeTurnPhase.COMPLETED,
    )
    data = MossTTSRealtimeRequestData(
        state=state,
        turn_state=turn,
        model_config=MODEL_CONFIG,
        generation_row_start=0,
    )

    result = rb.apply_moss_tts_realtime_result(_payload("ignored"), data)

    assert "prompt_rows" not in result.data
    assert result.data["audio_codes"].shape == (1, N_VQ)
