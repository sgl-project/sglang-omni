# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading

import pytest
import torch

from sglang_omni_v1.models.fishaudio_s2_pro import stages
from sglang_omni_v1.proto import OmniRequest, StagePayload
from sglang_omni_v1.scheduling.messages import IncomingMessage


class _FakeCodec:
    sample_rate = 44100
    frame_length = 4

    def __init__(self) -> None:
        self.calls: list[tuple[int, ...]] = []

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(indices.shape))
        batch_size = indices.shape[0]
        samples = indices.shape[-1] * self.frame_length
        rows = []
        for row in range(batch_size):
            rows.append(torch.full((1, samples), float(row + 1)))
        return torch.stack(rows, dim=0)


def _payload(request_id: str, code_len: int) -> StagePayload:
    output_codes = torch.arange(10 * code_len, dtype=torch.long).reshape(10, code_len)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello"),
        data={"output_codes": output_codes.tolist()},
    )


def _empty_payload(request_id: str) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello"),
        data={},
    )


def _zero_length_payload(request_id: str) -> StagePayload:
    output_codes = torch.empty((10, 0), dtype=torch.long)
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello"),
        data={"output_codes": output_codes.tolist()},
    )


def test_fish_vocoder_uses_simple_scheduler_batch_path(monkeypatch) -> None:
    codec = _FakeCodec()
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "_load_codec", lambda checkpoint, device: codec)

    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=50,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(
            IncomingMessage("req-short", "new_request", _payload("req-short", 2))
        )
        scheduler.inbox.put(
            IncomingMessage("req-long", "new_request", _payload("req-long", 3))
        )

        first = scheduler.outbox.get(timeout=2.0)
        second = scheduler.outbox.get(timeout=2.0)
        outputs = {first.request_id: first.data, second.request_id: second.data}
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)

    assert codec.calls == [(2, 9, 3)]
    assert len(outputs["req-short"].data["audio_data"]) == 8
    assert len(outputs["req-long"].data["audio_data"]) == 12
    assert outputs["req-short"].data["audio_data"] == [1.0] * 8
    assert outputs["req-long"].data["audio_data"] == [2.0] * 12


def test_fish_vocoder_rejects_missing_output_codes(monkeypatch) -> None:
    codec = _FakeCodec()
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "_load_codec", lambda checkpoint, device: codec)

    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=1,
    )
    thread = threading.Thread(target=scheduler.start, daemon=True)
    thread.start()
    try:
        scheduler.inbox.put(
            IncomingMessage("req-empty", "new_request", _empty_payload("req-empty"))
        )
        output = scheduler.outbox.get(timeout=2.0)
    finally:
        scheduler.stop()
        thread.join(timeout=2.0)

    assert output.request_id == "req-empty"
    assert output.type == "error"
    assert isinstance(output.data, ValueError)
    assert "req-empty" in str(output.data)
    assert codec.calls == []


def test_fish_vocoder_batch_rejects_zero_length_output_codes_before_decode(
    monkeypatch,
) -> None:
    codec = _FakeCodec()
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "_load_codec", lambda checkpoint, device: codec)

    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=50,
    )

    assert scheduler._batch_fn is not None
    with pytest.raises(ValueError, match="req-zero"):
        scheduler._batch_fn(
            [
                _payload("req-good", 2),
                _zero_length_payload("req-zero"),
            ]
        )
    assert codec.calls == []
