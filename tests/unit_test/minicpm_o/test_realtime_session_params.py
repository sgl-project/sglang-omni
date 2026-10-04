# SPDX-License-Identifier: Apache-2.0
"""Per-session sampling and references reach the model without replacing deployment defaults."""

import base64
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks, SpeechHooks
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.schema import JsonObject
from tests.unit_test.serve.test_realtime_reference_audio import wav_reference

REFERENCE = {"media_type": "audio/wav", "data": wav_reference()}


async def open_session_params(
    config: MiniCPMODuplexPipelineConfig, sglang: JsonObject
) -> JsonObject:
    client = AsyncMock(spec=Client)
    client.open_session.return_value = SessionIdentity("session", 0)
    deployment = build_realtime_deployment(client, config)
    negotiation = SessionNegotiation(
        model="test", capabilities=deployment.capabilities, limits=deployment.limits
    )
    session, _ = negotiation.negotiate(
        {}, "CREATED", {"instructions": "be brief", "sglang": sglang}
    )
    adapter = deployment.adapter_factory()
    await adapter.open("session", session, AsyncMock())
    try:
        client.open_session.assert_awaited_once()
        return client.open_session.call_args.args[0].params
    finally:
        await adapter.close()


@pytest.mark.asyncio
async def test_session_overrides_stay_local_to_the_session() -> None:
    config = MiniCPMODuplexPipelineConfig(
        model_path="unused",
        sampling=MiniCPMODuplexSampling(greedy=True, temperature=0.3),
    )
    override = MiniCPMODuplexSampling(
        greedy=False,
        temperature=0.4,
        top_k=50,
        top_p=0.6,
        repetition_penalty=1.2,
        listen_prob_scale=0.5,
        force_listen_count=0,
        max_new_tokens_per_unit=8,
        repetition_window_size=64,
        talker_temperature=0.6,
        talker_repetition_penalty=1.1,
    ).model_dump()
    overridden = await open_session_params(
        config,
        {
            "sampling": override,
            "reference_audio": REFERENCE,
            "tts_reference_audio": REFERENCE,
        },
    )
    plain = await open_session_params(config, {})
    audio = base64.b64decode(REFERENCE["data"])
    assert overridden == {
        "instructions": "be brief",
        **override,
        "reference_audio": audio,
        "tts_reference_audio": audio,
        "max_slice_nums": 1,
    }
    assert plain == {
        "instructions": "be brief",
        **config.sampling.model_dump(),
        "max_slice_nums": 1,
    }
    assert config.sampling.temperature == 0.3


def test_stage_reference_precedence_is_per_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    state_open = Mock()
    monkeypatch.setattr(
        "sglang_omni.models.minicpm_o.native_stages.MiniCPMOPerceptionState.open",
        state_open,
    )
    perception = PerceptionHooks(
        Mock(), Mock(), Mock(), reference_audio=b"default", image_encoder=Mock()
    )
    runtime = Mock()
    speech = SpeechHooks(runtime, b"default")
    for session_id, params in (
        ("separate", {"reference_audio": b"input", "tts_reference_audio": b"output"}),
        ("shared", {"reference_audio": b"input"}),
        ("default", {}),
    ):
        request = OmniRequest(
            None, params={"instructions": "", "max_slice_nums": 1, **params}
        )
        perception.open(SessionIdentity(session_id), request)
        speech.open(SessionIdentity(session_id), request)
    assert [call.kwargs["reference_audio"] for call in state_open.call_args_list] == [
        b"input",
        b"input",
        b"default",
    ]
    assert [
        call.kwargs["reference_audio"] for call in runtime.open_session.call_args_list
    ] == [b"output", b"input", b"default"]
