# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import pytest
import torch

from sglang_omni_v1.models.fishaudio_s2_pro import stages
from sglang_omni_v1.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni_v1.scheduling.messages import IncomingMessage
from tests.unit_test.fixtures.fish_fakes import FakeFishCodec, make_s2pro_payload
from tests.unit_test.pipeline.helpers import run_scheduler


def test_fish_vocoder_batches_and_trims_audio_by_code_length(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preserves batched vocoder decode and per-request trim by code length."""
    codec = FakeFishCodec(frame_length=4)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(stages, "_load_codec", lambda checkpoint, device: codec)
    scheduler = stages.create_vocoder_executor(
        "unused",
        device="cpu",
        max_batch_size=4,
        max_batch_wait_ms=50,
    )

    def payload(request_id: str, code_len: int) -> object:
        return make_s2pro_payload(
            S2ProState(
                output_codes=torch.arange(3 * code_len).reshape(3, code_len),
                prompt_tokens=4,
                completion_tokens=code_len,
                engine_time_s=0.5,
            ),
            request_id=request_id,
        )

    first, second = run_scheduler(
        scheduler,
        [
            IncomingMessage("req-short", "new_request", payload("req-short", 2)),
            IncomingMessage("req-long", "new_request", payload("req-long", 3)),
        ],
        output_count=2,
    )
    outputs = {first.request_id: first.data, second.request_id: second.data}

    assert codec.calls == [(2, 2, 3)]
    assert outputs["req-short"].data["audio_data"] == [1.0] * 8
    assert outputs["req-long"].data["audio_data"] == [2.0] * 12
    assert outputs["req-short"].data["usage"]["total_tokens"] == 6
