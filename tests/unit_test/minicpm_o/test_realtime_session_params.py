# SPDX-License-Identifier: Apache-2.0
"""Per-session sampling and references reach the model without replacing deployment defaults."""

import base64
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    SAMPLE_RATE,
    UNIT_SAMPLES,
)
from sglang_omni.models.minicpm_o.native_config import (
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks, SpeechHooks
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import CONTROL, INTERRUPT, SESSION_CONTROL
from sglang_omni.serve.realtime.negotiation import SessionNegotiation
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    TextFinished,
)
from sglang_omni.serve.realtime.schema import JsonObject, SessionConfiguration
from sglang_omni.serve.realtime.types import InteractionAdapter, Unit
from tests.unit_test.fixtures.session_client import FakeSessionClient
from tests.unit_test.serve.test_realtime_reference_audio import wav_reference

REFERENCE = {"media_type": "audio/wav", "data": wav_reference()}
SESSION_CONFIG: SessionConfiguration = {
    "audio": {"input": {"format": {"type": "audio/pcm", "rate": SAMPLE_RATE}}}
}
FIRST_RESPONSE = "session-response-0"
FIRST_ITEM = "session-response-0-message"
SECOND_RESPONSE = "session-response-1"
SECOND_ITEM = "session-response-1-message"
INTERRUPT_OF_TURN_1 = (
    TimedChunk(CONTROL, 0, 0, 0, {"type": INTERRUPT}, format=SESSION_CONTROL, turn=1),
    None,
    False,
)


def voice(pcm: bytes, *, end_of_turn: bool = False) -> dict[str, object]:
    return dict(
        text="",
        pcm=pcm,
        end_of_turn=end_of_turn,
        is_listen=not pcm or end_of_turn,
        model_end_of_turn=end_of_turn,
    )


class VoiceCoordinator(FakeSessionClient):
    """Answers each unit with the scripted (turn, voice chunk) of its seq, then input_done."""

    def __init__(
        self, voice_chunks: dict[int, tuple[int | None, dict[str, object]]]
    ) -> None:
        super().__init__()
        self.voice_chunks = voice_chunks

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        return await super().append_session(session_identity, chunk)

    def answer(self, chunk: TimedChunk) -> None:
        turn, payload = self.voice_chunks[chunk.seq]
        self.emit("voice", None, payload, input_seq=chunk.seq, turn=turn)


async def open_adapter(
    coordinator: VoiceCoordinator,
) -> tuple[InteractionAdapter, list[OutputEvent]]:
    deployment = build_realtime_deployment(
        Client(coordinator), MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    adapter = deployment.adapter_factory()
    events: list[OutputEvent] = []

    async def record(event: OutputEvent, unit: Unit | None = None) -> None:
        events.append(event)

    await adapter.open("session", SESSION_CONFIG, record)
    return adapter, events


def unit(seq: int) -> Unit:
    return Unit(seq, seq * UNIT_SAMPLES, b"\0\0" * UNIT_SAMPLES, UNIT_SAMPLES)


@pytest.mark.asyncio
async def test_cancel_stops_the_open_reply_and_drops_its_later_audio() -> None:
    """A client stop interrupts the open reply's turn; audio of that turn after the stop is dropped, the reply ends as cancelled when its turn ends, and the next reply completes as usual."""
    heard, unheard, tail, next_reply = (
        b"\1\0" * 4,
        b"\2\0" * 4,
        b"\3\0" * 2,
        b"\4\0" * 4,
    )
    coordinator = VoiceCoordinator(
        {
            0: (1, voice(heard)),
            1: (1, voice(unheard)),
            2: (1, voice(tail, end_of_turn=True)),
            3: (2, voice(next_reply, end_of_turn=True)),
        }
    )
    adapter, events = await open_adapter(coordinator)

    await adapter.process(unit(0))
    await adapter.cancel()
    for seq in (1, 2, 3):
        await adapter.process(unit(seq))
    await adapter.close()

    assert coordinator.controlled == [INTERRUPT_OF_TURN_1]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, heard),
        TextFinished(FIRST_RESPONSE, FIRST_ITEM, ""),
        AudioFinished(FIRST_RESPONSE, FIRST_ITEM),
        ResponseFinished(
            FIRST_RESPONSE, FIRST_ITEM, "", True, "cancelled", "interrupted"
        ),
        ResponseStarted(SECOND_RESPONSE),
        AudioDelta(SECOND_RESPONSE, SECOND_ITEM, next_reply),
        TextFinished(SECOND_RESPONSE, SECOND_ITEM, ""),
        AudioFinished(SECOND_RESPONSE, SECOND_ITEM),
        ResponseFinished(SECOND_RESPONSE, SECOND_ITEM, "", True, "completed", "stop"),
    ]


@pytest.mark.asyncio
async def test_a_second_cancel_of_the_same_reply_sends_no_control() -> None:
    """A reply is interrupted once; a repeated client stop before it ends sends nothing more to the pipeline."""
    heard = b"\1\0" * 4
    coordinator = VoiceCoordinator({0: (1, voice(heard))})
    adapter, events = await open_adapter(coordinator)

    await adapter.process(unit(0))
    await adapter.cancel()
    await adapter.cancel()
    await adapter.close()

    assert coordinator.controlled == [INTERRUPT_OF_TURN_1]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, heard),
    ]


@pytest.mark.asyncio
async def test_cancel_without_an_open_reply_sends_no_control() -> None:
    """A client stop with no open reply sends nothing to the pipeline and emits no event."""
    coordinator = VoiceCoordinator({})
    adapter, events = await open_adapter(coordinator)

    await adapter.cancel()
    await adapter.close()

    assert coordinator.controlled == []
    assert events == []


@pytest.mark.asyncio
async def test_a_reply_without_a_turn_plays_when_nothing_was_cancelled() -> None:
    """Without a client stop, a voice chunk whose turn is unset is played and completes, not fenced as a cancelled turn."""
    pcm = b"\1\0" * 4
    coordinator = VoiceCoordinator({0: (None, voice(pcm, end_of_turn=True))})
    adapter, events = await open_adapter(coordinator)

    await adapter.process(unit(0))
    await adapter.close()

    assert coordinator.controlled == []
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, pcm),
        TextFinished(FIRST_RESPONSE, FIRST_ITEM, ""),
        AudioFinished(FIRST_RESPONSE, FIRST_ITEM),
        ResponseFinished(FIRST_RESPONSE, FIRST_ITEM, "", True, "completed", "stop"),
    ]


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
        Mock(),
        Mock(),
        Mock(),
        reference_audio=b"default",
        image_encoder=Mock(),
        mel_filter_bank=Mock(),
        reference_cache_capacity=1,
    )
    perception.reference_embeds = Mock(
        side_effect=lambda reference_audio: reference_audio
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
    assert [call.kwargs["reference_embeds"] for call in state_open.call_args_list] == [
        b"input",
        b"input",
        b"default",
    ]
    assert [
        call.kwargs["reference_audio"] for call in runtime.open_session.call_args_list
    ] == [b"output", b"input", b"default"]
