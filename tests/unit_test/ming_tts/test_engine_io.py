# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.ming_tts import engine_io
from sglang_omni.models.ming_tts.engine_io import (
    MingTTSSGLangRequestData,
    make_ming_tts_scheduler_adapters,
)
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.proto import OmniRequest, StagePayload


def _payload() -> StagePayload:
    state = MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2)
    return StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def _result_adapter(reset_request):
    model = SimpleNamespace(patch_size=2, latent_dim=3)
    _, result_adapter = make_ming_tts_scheduler_adapters(
        model=model,
        tokenizer=SimpleNamespace(),
        reset_request=reset_request,
    )
    return result_adapter


def _request_data(
    *,
    generated_latents: torch.Tensor | None = None,
    generated_last_chunk: list[bool] | None = None,
    stop_step: int | None = None,
    finish_reason=None,
    req_finished_reason=None,
) -> MingTTSSGLangRequestData:
    return MingTTSSGLangRequestData(
        req=SimpleNamespace(
            output_ids=[],
            finished_reason=req_finished_reason,
        ),
        state=MingTTSState(text="hello", input_ids=[1, 2, 3], max_decode_steps=2),
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        max_new_tokens=2,
        generated_latents=generated_latents,
        generated_last_chunk=list(generated_last_chunk or []),
        stop_step=stop_step,
        finish_reason=finish_reason,
        stage_payload=_payload(),
    )


def test_ming_tts_result_adapter_serializes_empty_latent_output() -> None:
    reset_requests = []

    payload = _result_adapter(reset_requests.append)(_request_data())
    restored = MingTTSState.from_dict(payload.data)
    latents = restored.generated_latents

    assert latents is not None
    assert latents.shape == (0, 2, 3)
    assert restored.generated_last_chunk == []
    assert restored.completion_tokens == 0
    assert restored.finish_reason == "stop"
    assert reset_requests == ["req-ming-tts"]


def test_ming_tts_result_adapter_prefers_stop_head_finish_reason() -> None:
    data = _request_data(
        generated_latents=torch.ones(1, 2, 3),
        generated_last_chunk=[True],
        stop_step=0,
        finish_reason="length",
    )

    payload = _result_adapter(lambda _: None)(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "stop"
    assert restored.stop_step == 0
    assert restored.completion_tokens == 1


def test_ming_tts_result_adapter_preserves_sglang_length_finish_reason() -> None:
    class FinishedReason:
        def to_json(self):
            return {"type": "length"}

    data = _request_data(
        generated_latents=torch.ones(1, 2, 3),
        generated_last_chunk=[True],
        req_finished_reason=FinishedReason(),
    )

    payload = _result_adapter(lambda _: None)(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "length"
    assert restored.stop_step is None


def test_ming_tts_result_adapter_infers_length_at_max_steps() -> None:
    data = _request_data(
        generated_latents=torch.stack(
            (torch.ones(2, 3), torch.ones(2, 3) * 2),
            dim=0,
        ),
        generated_last_chunk=[False, True],
    )

    payload = _result_adapter(lambda _: None)(data)
    restored = MingTTSState.from_dict(payload.data)

    assert restored.finish_reason == "length"
    assert restored.completion_tokens == 2


def test_ming_tts_result_adapter_resets_state_after_serialization_error(
    monkeypatch,
) -> None:
    reset_requests = []

    def fail_serialization(*_args):
        raise RuntimeError("serialization failed")

    monkeypatch.setattr(engine_io, "store_ming_tts_state", fail_serialization)
    data = _request_data(generated_latents=torch.ones(1, 2, 3))

    with pytest.raises(RuntimeError, match="serialization failed"):
        _result_adapter(reset_requests.append)(data)

    assert reset_requests == ["req-ming-tts"]
