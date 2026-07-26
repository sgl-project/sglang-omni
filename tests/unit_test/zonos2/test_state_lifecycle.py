# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.zonos2 import request_builders
from sglang_omni.models.zonos2.engine_builder import Zonos2EngineBuilder
from sglang_omni.models.zonos2.payload_types import (
    FRAME_WIDTH,
    N_CODEBOOKS,
    Zonos2State,
)
from sglang_omni.models.zonos2.request_builders import (
    Zonos2SGLangRequestData,
    make_zonos2_scheduler_adapters,
)
from sglang_omni.models.zonos2.state_pool import Zonos2DecodeStatePool
from sglang_omni.proto import OmniRequest, StagePayload


def _terminal_data(request_id: str = "req-zonos2") -> Zonos2SGLangRequestData:
    payload = StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs={}),
        data=Zonos2State().to_dict(),
    )
    return Zonos2SGLangRequestData(
        prompt_rows=torch.zeros((1, FRAME_WIDTH), dtype=torch.long),
        output_codes=[torch.zeros(N_CODEBOOKS, dtype=torch.long)],
        engine_start_s=time.perf_counter(),
        stage_payload=payload,
    )


def test_result_adapter_releases_terminal_request_state() -> None:
    reset_calls: list[str] = []
    model = SimpleNamespace(reset_request=reset_calls.append)
    _, result_adapter = make_zonos2_scheduler_adapters(model=model)

    result = result_adapter(_terminal_data())

    assert result.data["completion_tokens"] == 1
    assert reset_calls == ["req-zonos2"]


def test_result_adapter_releases_state_when_serialization_fails(monkeypatch) -> None:
    reset_calls: list[str] = []
    model = SimpleNamespace(reset_request=reset_calls.append)
    _, result_adapter = make_zonos2_scheduler_adapters(model=model)

    def fail_result(*_args, **_kwargs):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(request_builders, "apply_sglang_zonos2_result", fail_result)

    with pytest.raises(RuntimeError, match="serialization failed"):
        result_adapter(_terminal_data())

    assert reset_calls == ["req-zonos2"]


def test_engine_builder_abort_callback_releases_request_state() -> None:
    builder = Zonos2EngineBuilder()
    with pytest.raises(AssertionError):
        builder.make_abort_callback()

    reset_calls: list[str] = []
    builder.model = SimpleNamespace(reset_request=reset_calls.append)
    abort_callback = builder.make_abort_callback()
    builder.model = None

    abort_callback("req-zonos2")

    assert reset_calls == ["req-zonos2"]


def test_release_invalidates_cached_active_rows_before_request_id_reuse() -> None:
    model = SimpleNamespace(
        _decode_input_embedding=SimpleNamespace(
            weight=torch.zeros((2, 3), dtype=torch.float32)
        ),
        n_codebooks=2,
    )
    pool = Zonos2DecodeStatePool(model)
    request = SimpleNamespace(request_id="reused-rid")

    first_row = int(pool.prepare_active_rows([request])[0])
    pool.feedback_embeds[first_row].fill_(1)
    pool.release_row(request.request_id)
    free_rows_after_release = len(pool._free_rows)
    pool.release_row(request.request_id)

    assert request.request_id not in pool._rid_to_row
    assert pool._active_ids is None
    assert pool._active_rows is None
    assert torch.count_nonzero(pool.feedback_embeds[first_row]) == 0
    assert len(pool._free_rows) == free_rows_after_release

    reused_row = int(pool.prepare_active_rows([request])[0])

    assert pool._rid_to_row[request.request_id] == reused_row
    assert len(pool._free_rows) == pool.padding_row - 1
