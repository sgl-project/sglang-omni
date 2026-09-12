# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sglang_omni.serve.realtime.semantic_vad import (
    SemanticTurnDetector,
    SemanticVADConfig,
)
from sglang_omni.serve.realtime.smart_turn import (
    SMART_TURN_MODEL_ENV,
    SMART_TURN_MODEL_SHA256,
    SmartTurnEOU,
    load_smart_turn,
)
from sglang_omni.serve.realtime.turn_detector import build_turn_detector
from sglang_omni.serve.realtime.vad import VAD_FRAME_SAMPLES, Emit, VADEvent


class FakeSpeechModel:
    def __init__(self, decisions: list[bool]) -> None:
        self.decisions = list(decisions)
        self.reset_calls = 0

    def predict(self, frame: np.ndarray, sample_rate: int) -> float:
        assert frame.size == VAD_FRAME_SAMPLES and sample_rate == 16000
        return 1.0 if self.decisions.pop(0) else 0.0

    def reset(self) -> None:
        self.reset_calls += 1


class FakeEOU:
    def __init__(self, values: list[float | Exception]) -> None:
        self.values = list(values)
        self.calls = 0

    def predict(self, audio: np.ndarray, sample_rate: int) -> float:
        assert audio.size > 0 and sample_rate == 16000
        self.calls += 1
        value = self.values.pop(0)
        if isinstance(value, Exception):
            raise value
        return value


def _pcm_frames(count: int) -> bytes:
    return b"\x00\x00" * VAD_FRAME_SAMPLES * count


@pytest.mark.parametrize(
    ("eagerness", "confidence", "immediate", "hard_stop", "fallback"),
    [
        ("low", 0.92, 0.99, 3000, 800),
        ("medium", 0.88, 0.97, 2000, 640),
        ("high", 0.80, 0.93, 1200, 512),
    ],
)
def test_eagerness_presets(
    eagerness: str,
    confidence: float,
    immediate: float,
    hard_stop: int,
    fallback: int,
):
    config = SemanticVADConfig.from_eagerness(eagerness)

    assert config.confidence_threshold == confidence
    assert config.immediate_confidence_threshold == immediate
    assert config.max_pause_ms == hard_stop
    assert config.fallback_silence_ms == fallback


@pytest.mark.parametrize(
    ("probability", "silence_frames"),
    [(0.99, 8), (0.90, 20), (0.10, 63)],
)
def test_medium_eagerness_uses_confidence_windows(
    probability: float, silence_frames: int
):
    speech = FakeSpeechModel([True] + [False] * silence_frames)
    eou = FakeEOU([probability])
    detector = SemanticTurnDetector(
        eou,
        SemanticVADConfig.from_eagerness("medium"),
        speech_model=speech,
    )

    emits = detector.process(_pcm_frames(1 + silence_frames))

    assert emits == [
        Emit(VADEvent.SPEECH_STARTED, 0),
        Emit(VADEvent.SPEECH_STOPPED, VAD_FRAME_SAMPLES),
    ]
    assert eou.calls == 1


def test_resumed_speech_discards_pause_decision():
    decisions = [True] + [False] * 5 + [True] + [False] * 8
    eou = FakeEOU([0.1, 0.99])
    detector = SemanticTurnDetector(
        eou,
        SemanticVADConfig.from_eagerness("medium"),
        speech_model=FakeSpeechModel(decisions),
    )

    emits = detector.process(_pcm_frames(len(decisions)))

    assert [event.event_type for event in emits] == [
        VADEvent.SPEECH_STARTED,
        VADEvent.SPEECH_STOPPED,
    ]
    assert emits[-1].sample_offset == 7 * VAD_FRAME_SAMPLES
    assert eou.calls == 2


def test_inference_failure_uses_fixed_silence_fallback(caplog):
    decisions = [True] + [False] * 20
    eou = FakeEOU([RuntimeError("broken model")])
    detector = SemanticTurnDetector(
        eou,
        SemanticVADConfig.from_eagerness("medium"),
        speech_model=FakeSpeechModel(decisions),
    )

    with caplog.at_level(logging.WARNING):
        emits = detector.process(_pcm_frames(len(decisions)))

    assert "fixed-silence fallback" in caplog.text
    assert emits[-1] == Emit(VADEvent.SPEECH_STOPPED, VAD_FRAME_SAMPLES)
    assert eou.calls == 1


def test_detector_factory_normalizes_missing_semantic_model():
    fake_server = object()
    with patch(
        "sglang_omni.serve.realtime.turn_detector.StreamingVAD",
        return_value=fake_server,
    ):
        build = build_turn_detector(
            {"type": "semantic_vad", "eagerness": "high"},
            smart_turn_model=None,
        )

    assert build.detector is fake_server
    assert build.effective_config["type"] == "server_vad"
    assert build.effective_config["eagerness"] is None


def test_semantic_factory_applies_silero_options():
    fake_detector = object()
    with patch(
        "sglang_omni.serve.realtime.turn_detector.SemanticTurnDetector",
        return_value=fake_detector,
    ) as detector_type:
        build = build_turn_detector(
            {
                "type": "semantic_vad",
                "eagerness": "high",
                "threshold": 0.7,
                "prefix_padding_ms": 120,
            },
            smart_turn_model=FakeEOU([0.9]),
        )

    config = detector_type.call_args.args[1]
    assert build.detector is fake_detector
    assert config.speech_threshold == 0.7
    assert config.prefix_padding_ms == 120


def test_smart_turn_uses_normalized_whisper_features():
    feature_extractor = MagicMock()
    features = np.zeros((1, 80, 800), dtype=np.float32)
    feature_extractor.return_value.input_features = features
    session = MagicMock()
    session.run.return_value = [np.array([0.9], dtype=np.float32)]
    model = SmartTurnEOU(session=session, feature_extractor=feature_extractor)

    assert model.predict(np.ones(16000, dtype=np.float32), 16000) == pytest.approx(0.9)
    assert feature_extractor.call_args.kwargs["do_normalize"] is True
    with patch.object(session, "run", return_value=[np.array([1.1])]):
        with pytest.raises(ValueError, match="invalid probability"):
            model.predict(np.ones(16000, dtype=np.float32), 16000)


def test_smart_turn_loader_is_local_and_checksum_pinned(tmp_path, monkeypatch):
    monkeypatch.delenv(SMART_TURN_MODEL_ENV, raising=False)
    assert load_smart_turn() is None

    model_path = tmp_path / "smart-turn-v3.2-cpu.onnx"
    model_path.write_bytes(b"not the pinned model")
    monkeypatch.setenv(SMART_TURN_MODEL_ENV, str(model_path))

    assert len(SMART_TURN_MODEL_SHA256) == 64
    with pytest.raises(ValueError, match="checksum mismatch"):
        load_smart_turn()
