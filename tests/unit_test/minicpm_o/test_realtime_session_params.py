# SPDX-License-Identifier: Apache-2.0
"""Per-session sampling and references reach the model without replacing deployment defaults."""

import asyncio
import base64
from typing import Literal
from unittest.mock import AsyncMock, Mock

import pytest

from sglang_omni.client.client import Client
from sglang_omni.models.minicpm_o.components.code2wav import OUTPUT_SAMPLE_RATE
from sglang_omni.models.minicpm_o.components.streaming_perception import (
    SAMPLE_RATE,
    UNIT_SAMPLES,
)
from sglang_omni.models.minicpm_o.native_config import (
    DEFAULT_SPEECH_SETTINGS,
    MiniCPMODuplexPipelineConfig,
    MiniCPMODuplexSampling,
)
from sglang_omni.models.minicpm_o.native_stages import PerceptionHooks, SpeechHooks
from sglang_omni.models.minicpm_o.session_adapters import build_realtime_deployment
from sglang_omni.proto.request import OmniRequest
from sglang_omni.proto.session import SessionIdentity, TimedChunk
from sglang_omni.proto.streams import CONTROL, INTERRUPT, SESSION_CONTROL, TRUNCATE
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
# Each spoken unit's audio lasts 10 ms and the tail unit's 5 ms, so the reply's units end at 10, 20 and 25 ms.
UNIT_AUDIO_MS = 10
UNIT_AUDIO_SAMPLES = OUTPUT_SAMPLE_RATE * UNIT_AUDIO_MS // 1000
HEARD_PCM = b"\1\0" * UNIT_AUDIO_SAMPLES
RUNNING_UNIT_PCM = b"\2\0" * UNIT_AUDIO_SAMPLES
TAIL_PCM = b"\3\0" * (UNIT_AUDIO_SAMPLES // 2)
NEXT_REPLY_PCM = b"\4\0" * UNIT_AUDIO_SAMPLES


def interrupt_of_turn_1(keep_through_unit: int) -> tuple[TimedChunk, None, bool]:
    """The control a stop of turn 1 sends, keeping the reply through unit keep_through_unit, the last one the client received unless a held truncate named an earlier one."""
    return (
        TimedChunk(
            CONTROL,
            0,
            0,
            0,
            {"type": INTERRUPT, "keep_through_unit": keep_through_unit},
            format=SESSION_CONTROL,
            turn=1,
        ),
        None,
        False,
    )


def truncate_of_turn_1(
    audio_end_ms: float, keep_through_unit: int
) -> tuple[TimedChunk, None, bool]:
    """The control a client truncate of turn 1 at audio_end_ms sends, keeping the reply through unit keep_through_unit."""
    return (
        TimedChunk(
            CONTROL,
            0,
            0,
            0,
            {
                "type": TRUNCATE,
                "t_ms": audio_end_ms,
                "keep_through_unit": keep_through_unit,
            },
            format=SESSION_CONTROL,
            turn=1,
        ),
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
        is_speech_cut=False,
    )


class VoiceCoordinator(FakeSessionClient):
    """Records the appends; the test emits each unit's voice chunk and its input_done itself."""

    def __init__(self) -> None:
        super().__init__(auto_done=False)

    async def append_session(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        *,
        ready_timestamp_ns: int | None = None,
        ready_run_id: str | None = None,
    ) -> int:
        return await super().append_session(session_identity, chunk)

    def voice(
        self, input_seq: int, turn: int | None, payload: dict[str, object]
    ) -> None:
        self.emit("voice", None, payload, input_seq=input_seq, turn=turn)


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


def unit(input_seq: int) -> Unit:
    return Unit(
        input_seq, input_seq * UNIT_SAMPLES, b"\0\0" * UNIT_SAMPLES, UNIT_SAMPLES
    )


async def run_unit(
    adapter: InteractionAdapter,
    coordinator: VoiceCoordinator,
    input_seq: int,
    turn: int | None,
    payload: dict[str, object],
    *,
    cancel_at: Literal["before_output", "after_output"] | None = None,
) -> None:
    """Append one unit whose voice chunk arrives while it is in flight; the client stop, when asked for, lands before or after that chunk."""
    processing = asyncio.create_task(adapter.process(unit(input_seq)))
    await asyncio.sleep(0)
    if cancel_at == "before_output":
        await adapter.cancel()
    else:
        pass
    coordinator.voice(input_seq, turn, payload)
    await asyncio.sleep(0)
    if cancel_at == "after_output":
        await adapter.cancel()
    else:
        pass
    coordinator.done(input_seq)
    await processing


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("cancel_at", "last_delivered_unit", "received_pcm"),
    [
        ("before_output", 0, [HEARD_PCM]),
        ("after_output", 1, [HEARD_PCM, RUNNING_UNIT_PCM]),
    ],
    ids=["the_running_unit_arrives_after_the_stop", "the_running_unit_arrived_first"],
)
async def test_cancel_stops_the_open_reply_and_names_the_last_unit_the_client_received(
    cancel_at: Literal["before_output", "after_output"],
    last_delivered_unit: int,
    received_pcm: list[bytes],
) -> None:
    """A client stop interrupts the open reply's turn: the turn's audio that arrives after the stop is dropped, the interrupt names the last unit whose audio the client did receive, the reply ends as cancelled when its turn ends, and the next reply completes as usual."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    await run_unit(
        adapter, coordinator, 1, 1, voice(RUNNING_UNIT_PCM), cancel_at=cancel_at
    )
    await run_unit(adapter, coordinator, 2, 1, voice(TAIL_PCM, end_of_turn=True))
    await run_unit(adapter, coordinator, 3, 2, voice(NEXT_REPLY_PCM, end_of_turn=True))
    await adapter.close()

    assert coordinator.controlled == [interrupt_of_turn_1(last_delivered_unit)]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        *(AudioDelta(FIRST_RESPONSE, FIRST_ITEM, pcm) for pcm in received_pcm),
        TextFinished(FIRST_RESPONSE, FIRST_ITEM, ""),
        AudioFinished(FIRST_RESPONSE, FIRST_ITEM),
        ResponseFinished(
            FIRST_RESPONSE, FIRST_ITEM, "", True, "cancelled", "interrupted"
        ),
        ResponseStarted(SECOND_RESPONSE),
        AudioDelta(SECOND_RESPONSE, SECOND_ITEM, NEXT_REPLY_PCM),
        TextFinished(SECOND_RESPONSE, SECOND_ITEM, ""),
        AudioFinished(SECOND_RESPONSE, SECOND_ITEM),
        ResponseFinished(SECOND_RESPONSE, SECOND_ITEM, "", True, "completed", "stop"),
    ]


@pytest.mark.asyncio
async def test_a_second_cancel_of_the_same_reply_sends_no_control() -> None:
    """A reply is interrupted once; a repeated client stop before it ends sends nothing more to the pipeline."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    await adapter.cancel()
    await adapter.cancel()
    await adapter.close()

    assert coordinator.controlled == [interrupt_of_turn_1(0)]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, HEARD_PCM),
    ]


@pytest.mark.asyncio
async def test_cancel_without_an_open_reply_sends_no_control() -> None:
    """A client stop with no open reply sends nothing to the pipeline and emits no event."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await adapter.cancel()
    await adapter.close()

    assert coordinator.controlled == []
    assert events == []


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("audio_end_ms", "truncate_controls"),
    [
        (25.0, []),
        (30.0, []),
        (20.0, [truncate_of_turn_1(20.0, 1)]),
        (15.0, [truncate_of_turn_1(15.0, 0)]),
        (5.0, [truncate_of_turn_1(5.0, -1)]),
    ],
    ids=[
        "every_received_unit_played",
        "beyond_the_received_audio",
        "through_the_second_unit",
        "inside_the_second_unit",
        "inside_the_first_unit",
    ],
)
async def test_a_truncate_after_the_stop_keeps_the_reply_through_the_last_unit_the_client_played(
    audio_end_ms: float, truncate_controls: list[tuple[TimedChunk, None, bool]]
) -> None:
    """A client truncate after the stop names the last unit whose audio had fully ended at audio_end_ms: a position inside a unit rounds down to the unit before it, one before the first unit keeps none of the reply, and one at or beyond the received audio sends nothing, since the stop already keeps the reply through the last received unit."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    await run_unit(adapter, coordinator, 1, 1, voice(RUNNING_UNIT_PCM))
    await run_unit(
        adapter, coordinator, 2, 1, voice(TAIL_PCM), cancel_at="after_output"
    )
    await adapter.truncate(FIRST_ITEM, audio_end_ms)
    await run_unit(adapter, coordinator, 3, 1, voice(b"", end_of_turn=True))
    await adapter.close()

    assert coordinator.controlled == [interrupt_of_turn_1(2), *truncate_controls]
    assert events[-1] == ResponseFinished(
        FIRST_RESPONSE, FIRST_ITEM, "", True, "cancelled", "interrupted"
    )


@pytest.mark.asyncio
async def test_a_truncate_of_an_open_reply_sends_nothing_and_later_ones_only_move_back() -> (
    None
):
    """A truncate while the reply is open and not stopped sends nothing and does not bend the stop, which keeps the reply through the last unit the client received; after the stop a truncate sends a control only when it keeps less than what is already kept, so a repeated or later position sends nothing."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    await run_unit(adapter, coordinator, 1, 1, voice(RUNNING_UNIT_PCM))
    await adapter.truncate(FIRST_ITEM, 10.0)
    await run_unit(
        adapter, coordinator, 2, 1, voice(TAIL_PCM), cancel_at="before_output"
    )
    await adapter.truncate(FIRST_ITEM, 20.0)
    await adapter.truncate(FIRST_ITEM, 10.0)
    await adapter.truncate(FIRST_ITEM, 5.0)
    await adapter.truncate(FIRST_ITEM, 5.0)
    await adapter.close()

    assert coordinator.controlled == [
        interrupt_of_turn_1(1),
        truncate_of_turn_1(10.0, 0),
        truncate_of_turn_1(5.0, -1),
    ]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, HEARD_PCM),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, RUNNING_UNIT_PCM),
    ]


@pytest.mark.asyncio
async def test_a_late_truncate_of_a_stopped_reply_never_bends_the_stop_of_the_next_one() -> (
    None
):
    """A truncate meant for the stopped reply that arrives once the next reply is open lands nowhere: it sends nothing, and the next reply's stop still keeps that reply through the last unit the client received."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(
        adapter, coordinator, 0, 1, voice(HEARD_PCM), cancel_at="after_output"
    )
    await run_unit(adapter, coordinator, 1, 1, voice(b"", end_of_turn=True))
    await run_unit(adapter, coordinator, 2, 2, voice(NEXT_REPLY_PCM))
    await run_unit(adapter, coordinator, 3, 2, voice(NEXT_REPLY_PCM))
    await adapter.truncate(FIRST_ITEM, 5.0)
    await adapter.cancel()
    await adapter.close()

    assert coordinator.controlled == [
        interrupt_of_turn_1(0),
        (
            TimedChunk(
                CONTROL,
                0,
                0,
                0,
                {"type": INTERRUPT, "keep_through_unit": 3},
                format=SESSION_CONTROL,
                turn=2,
            ),
            None,
            False,
        ),
    ]


@pytest.mark.asyncio
async def test_a_truncate_cuts_a_reply_the_model_ended_until_the_next_reply_starts() -> (
    None
):
    """Once the model ends a reply itself, the usual barge-in of native duplex, a client truncate cuts it like a stopped one: it sends the kept unit, a later position sends nothing, and once the next reply is open a truncate sends nothing."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    await run_unit(
        adapter, coordinator, 1, 1, voice(RUNNING_UNIT_PCM, end_of_turn=True)
    )
    await adapter.truncate(FIRST_ITEM, 10.0)
    await adapter.truncate(FIRST_ITEM, 20.0)
    await run_unit(adapter, coordinator, 2, 2, voice(NEXT_REPLY_PCM))
    await adapter.truncate(SECOND_ITEM, 5.0)
    await adapter.close()

    assert coordinator.controlled == [truncate_of_turn_1(10.0, 0)]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, HEARD_PCM),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, RUNNING_UNIT_PCM),
        TextFinished(FIRST_RESPONSE, FIRST_ITEM, ""),
        AudioFinished(FIRST_RESPONSE, FIRST_ITEM),
        ResponseFinished(FIRST_RESPONSE, FIRST_ITEM, "", True, "completed", "stop"),
        ResponseStarted(SECOND_RESPONSE),
        AudioDelta(SECOND_RESPONSE, SECOND_ITEM, NEXT_REPLY_PCM),
    ]


@pytest.mark.asyncio
async def test_a_truncate_naming_an_earlier_reply_leaves_the_ended_latest_reply_alone() -> (
    None
):
    """Two replies the model ended itself: a truncate naming the first one sends nothing, so its position never cuts the second reply, and a truncate naming the second one cuts that reply."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM, end_of_turn=True))
    await run_unit(adapter, coordinator, 1, 2, voice(NEXT_REPLY_PCM))
    await run_unit(adapter, coordinator, 2, 2, voice(NEXT_REPLY_PCM, end_of_turn=True))
    await adapter.truncate(FIRST_ITEM, 5.0)
    await adapter.truncate(SECOND_ITEM, 10.0)
    await adapter.close()

    assert coordinator.controlled == [
        (
            TimedChunk(
                CONTROL,
                0,
                0,
                0,
                {"type": TRUNCATE, "t_ms": 10.0, "keep_through_unit": 1},
                format=SESSION_CONTROL,
                turn=2,
            ),
            None,
            False,
        )
    ]


@pytest.mark.asyncio
async def test_a_truncate_cuts_the_stopped_reply_until_the_next_reply_starts() -> None:
    """A truncate that arrives after the stopped reply's response.done still cuts it; one with no reply yet, or after the next reply started, sends nothing and raises nothing."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await adapter.truncate(FIRST_ITEM, 10.0)
    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    await run_unit(
        adapter, coordinator, 1, 1, voice(RUNNING_UNIT_PCM), cancel_at="after_output"
    )
    await run_unit(adapter, coordinator, 2, 1, voice(b"", end_of_turn=True))
    await adapter.truncate(FIRST_ITEM, 10.0)
    await run_unit(adapter, coordinator, 3, 2, voice(NEXT_REPLY_PCM))
    await adapter.truncate(SECOND_ITEM, 5.0)
    await adapter.close()

    assert coordinator.controlled == [
        interrupt_of_turn_1(1),
        truncate_of_turn_1(10.0, 0),
    ]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, HEARD_PCM),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, RUNNING_UNIT_PCM),
        TextFinished(FIRST_RESPONSE, FIRST_ITEM, ""),
        AudioFinished(FIRST_RESPONSE, FIRST_ITEM),
        ResponseFinished(
            FIRST_RESPONSE, FIRST_ITEM, "", True, "cancelled", "interrupted"
        ),
        ResponseStarted(SECOND_RESPONSE),
        AudioDelta(SECOND_RESPONSE, SECOND_ITEM, NEXT_REPLY_PCM),
    ]


@pytest.mark.asyncio
async def test_a_unit_delivered_in_several_chunks_is_played_once_its_last_chunk_is() -> (
    None
):
    """A unit whose audio reaches the client in several chunks ends where its last chunk ends, so a truncate between its chunks rounds down to the unit before it."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, 1, voice(HEARD_PCM))
    processing = asyncio.create_task(adapter.process(unit(1)))
    await asyncio.sleep(0)
    coordinator.voice(1, 1, voice(TAIL_PCM))
    coordinator.voice(1, 1, voice(TAIL_PCM))
    await asyncio.sleep(0)
    await adapter.cancel()
    coordinator.done(1)
    await processing
    await adapter.truncate(FIRST_ITEM, 15.0)
    await adapter.truncate(FIRST_ITEM, 20.0)
    await adapter.close()

    assert coordinator.controlled == [
        interrupt_of_turn_1(1),
        truncate_of_turn_1(15.0, 0),
    ]
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, HEARD_PCM),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, TAIL_PCM),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, TAIL_PCM),
    ]


@pytest.mark.asyncio
async def test_a_reply_without_a_turn_plays_when_nothing_was_cancelled() -> None:
    """Without a client stop, a voice chunk whose turn is unset is played and completes, not fenced as a cancelled turn."""
    coordinator = VoiceCoordinator()
    adapter, events = await open_adapter(coordinator)

    await run_unit(adapter, coordinator, 0, None, voice(HEARD_PCM, end_of_turn=True))
    await adapter.close()

    assert coordinator.controlled == []
    assert events == [
        ResponseStarted(FIRST_RESPONSE),
        AudioDelta(FIRST_RESPONSE, FIRST_ITEM, HEARD_PCM),
        TextFinished(FIRST_RESPONSE, FIRST_ITEM, ""),
        AudioFinished(FIRST_RESPONSE, FIRST_ITEM),
        ResponseFinished(FIRST_RESPONSE, FIRST_ITEM, "", True, "completed", "stop"),
    ]


def test_the_deployment_grants_the_client_stop() -> None:
    """The adapter ends the reply on response.cancel and the model listens until the user speaks again, so a page may offer its stop button as a real stop."""
    deployment = build_realtime_deployment(
        Client(VoiceCoordinator()), MiniCPMODuplexPipelineConfig(model_path="unused")
    )
    assert deployment.capabilities.to_granted_capabilities()["supports_cancel"] is True


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
    speech = SpeechHooks(
        runtime,
        b"default",
        gather_window_ms=DEFAULT_SPEECH_SETTINGS.gather_window_ms,
    )
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
