# SPDX-License-Identifier: Apache-2.0
"""Contract tests for duplex realtime session negotiation."""

from __future__ import annotations

from dataclasses import asdict

import pytest

from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.schema import JsonObject, SessionConfiguration
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits

MODEL_NAME = "duplex-test"


def build_negotiation(
    capabilities: Capabilities | None = None,
    limits: RuntimeLimits | None = None,
) -> SessionNegotiation:
    return SessionNegotiation(
        model=MODEL_NAME,
        capabilities=capabilities or Capabilities(),
        limits=limits or RuntimeLimits(),
    )


def open_session(negotiation: SessionNegotiation) -> SessionConfiguration:
    config, _ = negotiation.negotiate({}, "CREATED", {"instructions": "be brief"})
    return config


def test_empty_update_grants_deployment_defaults() -> None:
    limits = RuntimeLimits()
    config, granted = build_negotiation(limits=limits).negotiate({}, "CREATED", {})

    assert config == {
        "model": MODEL_NAME,
        "type": "realtime",
        "output_modalities": ["text"],
        "audio": {
            "input": {"format": {"type": "audio/pcm", "rate": 16000}},
            "output": {"format": {"type": "audio/pcm", "rate": 16000}},
        },
    }
    assert granted["native_full_duplex"] is True
    assert granted["client_commit"] is False
    assert granted["native_unit_ms"] == 20
    assert granted["microturn_ms"] is None
    assert granted["limits"] == asdict(limits)
    assert granted["rejections"] == []


def test_unsupported_modality_combination_is_narrowed_and_reported() -> None:
    negotiation = build_negotiation(Capabilities(output_modalities=("text", "audio")))

    config, granted = negotiation.negotiate(
        {}, "CREATED", {"output_modalities": ["audio", "text"]}
    )

    assert config["output_modalities"] == ["audio"]
    assert granted["output_modalities"] == ["audio"]
    assert granted["rejections"] == [
        {
            "field": "output_modalities",
            "requested": ["audio", "text"],
            "reason": "deployment modalities",
            "granted": ["audio"],
        }
    ]


def test_requested_microturn_is_rejected_without_failing_the_update() -> None:
    _, granted = build_negotiation().negotiate(
        {}, "CREATED", {"sglang": {"timebase": {"microturn_ms": 80.0}}}
    )

    assert granted["microturn_ms"] is None
    assert [rejection["field"] for rejection in granted["rejections"]] == [
        "sglang.timebase.microturn_ms"
    ]


@pytest.mark.parametrize(
    ("patch", "code", "param"),
    [
        ({"model": "other-model"}, "invalid_request", None),
        ({"type": "transcription"}, "invalid_request", None),
        ({"unknown": True}, "invalid_request", "session.unknown"),
        ({"instructions": "x" * 9}, "invalid_request", None),
        (
            {"audio": {"input": {"format": {"type": "audio/pcm", "rate": 24000}}}},
            "invalid_request",
            "session.audio.input.format",
        ),
        (
            {"audio": {"input": {"turn_detection": {"type": "server_vad"}}}},
            "not_applicable",
            None,
        ),
        ({"sglang": {"tail_policy": "pad"}}, "invalid_request", None),
        ({"sglang": {"timebase": {"native_unit_ms": 40}}}, "invalid_request", None),
        ({"output_modalities": ["audio"]}, "invalid_request", None),
    ],
)
def test_unsupported_configuration_is_rejected(
    patch: JsonObject, code: str, param: str | None
) -> None:
    negotiation = build_negotiation(limits=RuntimeLimits(max_history_chars=8))

    with pytest.raises(ProtocolError) as exc_info:
        negotiation.negotiate({}, "CREATED", patch)

    assert (exc_info.value.code, exc_info.value.param) == (code, param)


def test_open_session_merges_patch_into_current_configuration() -> None:
    negotiation = build_negotiation(Capabilities(output_modalities=("text", "audio")))
    current = open_session(negotiation)

    config, _ = negotiation.negotiate(current, "OPEN", {"output_modalities": ["audio"]})

    assert config["instructions"] == "be brief"
    assert config["output_modalities"] == ["audio"]


def test_open_session_freezes_admission_fields() -> None:
    negotiation = build_negotiation()
    current = open_session(negotiation)

    with pytest.raises(ProtocolError) as exc_info:
        negotiation.negotiate(current, "OPEN", {"instructions": "be verbose"})

    assert exc_info.value.code == "invalid_state"
