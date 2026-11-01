# SPDX-License-Identifier: Apache-2.0
"""Native encoder stages retain turn ownership while relaying incremental units."""

import threading

import pytest
import torch

from sglang_omni.models.qwen3_omni import native_stages
from sglang_omni.models.qwen3_omni.native_stages import QwenEncoderHooks
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext


class ImageEncoder(torch.nn.Module):
    def __init__(self, cancellation: threading.Event | None = None) -> None:
        super().__init__()
        self.forward_call_count: int = 0
        self.cancellation: threading.Event | None = cancellation

    def forward(self, pixel_values: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_call_count += 1
        if self.cancellation is not None:
            self.cancellation.set()
        else:
            pass
        return {"image_embeds": pixel_values + 10}


class AudioEncoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_call_count: int = 0

    def forward(self, input_features: torch.Tensor) -> dict[str, torch.Tensor]:
        self.forward_call_count += 1
        return {"audio_embeds": input_features + 20}


def encoder_payload(turn: int) -> StagePayload:
    return StagePayload(
        "unit",
        OmniRequest(None),
        {
            "session": {"turn": turn},
            "encoder_inputs": {
                "image_encoder": {"pixel_values": torch.tensor([[1.0, 2.0]])},
                "audio_encoder": {"input_features": torch.tensor([[3.0, 4.0]])},
            },
        },
    )


def encoder_context(cancellation: threading.Event | None = None) -> SessionContext:
    return SessionContext(
        session_identity=SessionIdentity("test"),
        cancelled=cancellation or threading.Event(),
        emit=lambda chunk: None,
    )


def test_encoders_preserve_session_metadata_and_merge_actual_encoder_results() -> None:
    image_model, audio_model = ImageEncoder(), AudioEncoder()
    image = QwenEncoderHooks(image_model, "image_encoder")
    audio = QwenEncoderHooks(audio_model, "audio_encoder")
    session_context = encoder_context()
    for hooks in (image, audio):
        hooks.open(session_context.session_identity, OmniRequest(None))
    unit = encoder_payload(1)
    chunk = TimedChunk("control", 0, 0, 0, {}, format="qwen.unit", turn=1)
    for hooks in (image, audio):
        unit = hooks.append(chunk, unit, session_context)
        assert unit.data["session"] == {"turn": 1}
    merged = unit.data["thinker_inputs"]["model_inputs"]
    assert merged["image_embeds"].tolist() == [[11.0, 12.0]]
    assert merged["audio_embeds"].tolist() == [[23.0, 24.0]]
    assert image_model.forward_call_count == audio_model.forward_call_count == 1
    next_unit = encoder_payload(2)
    for hooks in (image, audio):
        next_unit = hooks.append(chunk, next_unit, session_context)
    assert image_model.forward_call_count == audio_model.forward_call_count == 2
    for hooks in (image, audio):
        hooks.close(session_context.session_identity)


@pytest.mark.parametrize("cancel_before", [False, True])
def test_cancelled_encoder_unit_skips_the_model(cancel_before: bool) -> None:
    cancelled = threading.Event()
    model = ImageEncoder(cancelled)
    hooks = QwenEncoderHooks(model, "image_encoder")
    session_context = encoder_context(cancelled)
    hooks.open(session_context.session_identity, OmniRequest(None))
    if cancel_before:
        cancelled.set()
    else:
        pass
    chunk = TimedChunk("control", 0, 0, 0, {}, format="qwen.unit", turn=1)
    hooks.append(chunk, encoder_payload(1), session_context)
    assert model.forward_call_count == (0 if cancel_before else 1)
    cancelled.clear()
    model.cancellation = None
    hooks.append(chunk, encoder_payload(1), session_context)
    assert model.forward_call_count == (1 if cancel_before else 2)


def test_audio_encoder_scheduler_runs_one_unit_at_a_time(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Overlapping audio encoder units could return another session's embeddings."""
    monkeypatch.setattr(
        native_stages, "Qwen3OmniAudioEncoder", lambda **kwargs: AudioEncoder()
    )
    scheduler = native_stages.create_encoder_scheduler(
        "unused", stage_name="audio_encoder", max_open_sessions=4, device="cpu"
    )
    assert scheduler.max_concurrency == 1
