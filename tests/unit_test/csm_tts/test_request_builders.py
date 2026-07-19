# SPDX-License-Identifier: Apache-2.0
"""Unit tests for csm_tts.request_builders (CPU; sglang stubbed via conftest
on machines without it).


"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.csm_tts import request_builders
from sglang_omni.models.csm_tts.payload_types import CsmTtsState
from sglang_omni.models.csm_tts.utils import BACKBONE_CTX
from sglang_omni.proto import OmniRequest, StagePayload


def _state(prompt_len: int = 8, **kwargs) -> CsmTtsState:
    return CsmTtsState(text="hi", prompt_ids=list(range(1, prompt_len + 1)), **kwargs)


def _payload(
    state: CsmTtsState, rid: str = "req-1", params: dict | None = None
) -> StagePayload:
    return StagePayload(
        request_id=rid,
        request=OmniRequest(inputs={}, params=params or {}),
        data=state.to_dict(),
    )


def test_sampling_params_normalize_called_and_seed_kwarg(monkeypatch) -> None:
    """``sampling_params.normalize(tokenizer=None)`` is invoked (stop_strs
    crash upstream otherwise) and the seed rides the ``sampling_seed`` kwarg,
    NOT ``seed``."""
    init_kwargs: list[dict] = []
    normalize_calls: list = []
    real_cls = request_builders.SamplingParams

    class SpyParams(real_cls):
        def __init__(self, **kwargs):
            init_kwargs.append(dict(kwargs))
            super().__init__(**kwargs)

        def normalize(self, tokenizer):
            normalize_calls.append(tokenizer)
            return super().normalize(tokenizer)

    monkeypatch.setattr(request_builders, "SamplingParams", SpyParams)
    state = _state(seed=42, max_new_tokens=10, temperature=0.7, top_k=50)
    data = request_builders.build_sglang_csm_request(state, request_id="r1")

    assert normalize_calls == [None]
    kwargs = init_kwargs[0]
    assert kwargs["sampling_seed"] == 42
    assert "seed" not in kwargs
    assert kwargs["max_new_tokens"] == 10
    assert kwargs["temperature"] == pytest.approx(0.7)
    assert kwargs["top_k"] == 50

    assert data.req.rid == "r1"
    # V1 prefill-manager probe attrs (absence → AttributeError upstream).
    assert data.req._codec_suppress_tokens is None
    assert data.req._input_embeds_are_projected is False
    # Depth set is CSM-private and never rides SamplingParams.
    assert "depth_temperature" not in kwargs
    assert data.depth_temperature == pytest.approx(0.9)
    assert data.depth_top_k == 50


def test_empty_prompt_rejected() -> None:
    with pytest.raises(ValueError, match="prompt_ids"):
        request_builders.build_sglang_csm_request(
            CsmTtsState(text="x", prompt_ids=[]), request_id="r0"
        )


def test_max_new_tokens_clamp_vs_ctx_budget() -> None:
    """Clamp is ``min(cap, BACKBONE_CTX - 1 - len(prompt_ids))`` — against
    max_req_len = 2047, NOT 2048; zero budget raises an
    actionable error."""
    model = SimpleNamespace(reset_request=lambda rid: None)

    # cap binds for short prompts
    rb_cap, _ = request_builders.make_csm_scheduler_adapters(
        model, max_new_tokens_cap=125
    )
    data = rb_cap(_payload(_state(prompt_len=10, max_new_tokens=4096)))
    assert data.max_new_tokens == 125
    assert data.req.sampling_params.max_new_tokens == 125

    # context budget binds near the ceiling: 2047 - len(prompt_ids)
    rb_unc, _ = request_builders.make_csm_scheduler_adapters(
        model, max_new_tokens_cap=None
    )
    data = rb_unc(_payload(_state(prompt_len=BACKBONE_CTX - 50, max_new_tokens=4096)))
    assert data.max_new_tokens == 49  # 2047 - 1998, NOT 50

    # a prompt of 2047 tokens leaves no frame budget at all
    with pytest.raises(ValueError, match="frame budget"):
        rb_unc(_payload(_state(prompt_len=BACKBONE_CTX - 1)))


def test_result_adapter_writes_usage_and_resets() -> None:
    resets: list[str] = []
    model = SimpleNamespace(reset_request=resets.append)
    rb, ra = request_builders.make_csm_scheduler_adapters(model)
    data = rb(_payload(_state(prompt_len=5, max_new_tokens=10), rid="req-x"))
    data.output_frames.append(torch.arange(32, dtype=torch.long))

    out = ra(data)

    assert resets == ["req-x"]
    state = CsmTtsState.from_dict(out.data)
    assert state.completion_frames == 1
    assert state.prompt_tokens == 5
    assert state.output_frames == [list(range(32))]
    assert state.engine_time_s > 0


def test_stream_metadata_only_for_streaming_requests() -> None:
    state = _state()
    streaming = _payload(
        state, params={"stream": True, "initial_codec_chunk_frames": 1}
    )
    metadata = request_builders.build_csm_stream_metadata(streaming, None)
    assert metadata == {
        "modality": "audio_codes",
        "stream": True,
        "num_codebooks": 32,
        "codebook_size": 2051,
        "initial_codec_chunk_frames": 1,
    }
    assert request_builders.build_csm_stream_metadata(_payload(state), None) is None


def test_context_fingerprint_sensitivity() -> None:
    """Any context code or speaker change → new extra_key; no context →
    None (shared radix subtree) — the voice cross-contamination guard."""
    torch.manual_seed(3)
    codes = torch.randint(0, 2048, (4, 32), dtype=torch.int32)

    def state_for(context_codes, speaker=0):
        return CsmTtsState(
            text="x",
            prompt_ids=[1],
            context=[{"speaker_id": speaker, "text": "a"}],
            context_codes=context_codes,
        )

    base_fp = request_builders._context_fingerprint(state_for([codes]))
    assert base_fp is not None

    changed = codes.clone()
    changed[2, 31] += 1  # a single code in a single frame
    assert request_builders._context_fingerprint(state_for([changed])) != base_fp

    assert (
        request_builders._context_fingerprint(state_for([codes], speaker=1)) != base_fp
    )

    # same inputs → same key (radix reuse), no context → None
    assert request_builders._context_fingerprint(state_for([codes])) == base_fp
    assert (
        request_builders._context_fingerprint(CsmTtsState(text="x", prompt_ids=[1]))
        is None
    )
