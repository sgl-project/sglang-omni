from __future__ import annotations

from sglang_omni.models.fishaudio_s2_pro.runtime.s2pro_sglang_ar import (
    S2ProSGLangRequestData,
    _resolve_sampling_state,
)


def test_resolve_sampling_state_uses_request_defaults_without_repetition() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        temperature=0.7,
        top_p=0.9,
        _previous_semantic_tokens=[11, 12, 13],
    )

    temperature, top_p, previous_tokens = _resolve_sampling_state(
        data,
        ras_window=16,
        ras_temperature=1.5,
        ras_top_p=0.95,
    )

    assert temperature == 0.7
    assert top_p == 0.9
    assert previous_tokens == [11, 12, 13]


def test_resolve_sampling_state_switches_to_ras_on_recent_repetition() -> None:
    data = S2ProSGLangRequestData(
        input_ids=[],
        temperature=0.7,
        top_p=0.9,
        _previous_semantic_tokens=[101, 102, 103, 103],
    )

    temperature, top_p, previous_tokens = _resolve_sampling_state(
        data,
        ras_window=16,
        ras_temperature=1.5,
        ras_top_p=0.95,
    )

    assert temperature == 1.5
    assert top_p == 0.95
    assert previous_tokens == [101, 102, 103, 103]
