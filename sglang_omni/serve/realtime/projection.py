"""Wire projections; computation emits raw PCM and concrete event values."""

import base64
from typing import Literal

from typing_extensions import TypedDict

from sglang_omni.serve.realtime.control import (
    Accepted,
    Cleared,
    Closed,
    ControlEvent,
    Created,
    Drained,
    Ended,
    Failure,
    UnitCompleted,
    Updated,
)
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    AudioFinished,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    ResponseStatus,
    TextDelta,
    TextFinished,
    TurnFailure,
)
from sglang_omni.serve.realtime.schema import (
    AudioConfig,
    GrantedCapabilities,
    SessionExtension,
    SessionType,
    TailPolicy,
)


class ResponseContent(TypedDict, total=False):
    type: Literal["output_text", "output_audio"]
    text: str
    transcript: str


class ResponseMessage(TypedDict):
    id: str
    object: Literal["realtime.item"]
    type: Literal["message"]
    role: Literal["assistant"]
    content: list[ResponseContent]


class ResponseDetails(TypedDict):
    reason: str


class WireResponse(TypedDict, total=False):
    id: str
    object: Literal["realtime.response"]
    status: ResponseStatus | Literal["in_progress"]
    status_details: ResponseDetails
    output: list[ResponseMessage]
    usage: dict[str, int | float | None] | None


class GrantedSessionExtension(SessionExtension, total=False):
    granted: GrantedCapabilities | None


class WireSession(TypedDict, total=False):
    id: str
    object: Literal["realtime.session"]
    type: SessionType
    model: str
    instructions: str
    output_modalities: list[str]
    audio: AudioConfig
    sglang: GrantedSessionExtension


class WireError(TypedDict, total=False):
    type: str
    code: str
    message: str
    event_id: str | None
    param: str | None


class MediaTime(TypedDict):
    t_start_ms: float
    duration_ms: float


class EventMetadata(TypedDict, total=False):
    fatal: bool
    discarded_ms: float
    unit_id: str
    chunk_seq: int
    media_time: MediaTime


class ServerEvent(TypedDict, total=False):
    type: str
    event_id: str
    response: WireResponse
    response_id: str
    item_id: str
    output_index: int
    content_index: int
    delta: str
    transcript: str
    text: str
    error: WireError
    session: WireSession
    client_event_id: str | None
    reason: str
    sglang: EventMetadata
    unit_id: str
    seq: int
    accepted_end_ms: float
    discarded_ms: float
    consumed_ms: float
    padding_ms: float
    tail_policy: TailPolicy


def project_output(
    event: OutputEvent,
    *,
    output_modalities: tuple[str, ...] | None = None,
) -> ServerEvent:
    if isinstance(event, ResponseStarted):
        return dict(
            type="response.created",
            response=dict(
                id=event.response_id,
                object="realtime.response",
                status="in_progress",
                output=[],
            ),
        )
    elif isinstance(event, ResponseFinished):
        if output_modalities is not None and "text" not in output_modalities:
            content: list[ResponseContent] = []
        else:
            content = [dict(type="output_text", text=event.text)]
        if event.has_audio:
            content.append(dict(type="output_audio", transcript=event.text))
        else:
            pass
        return dict(
            type="response.done",
            response=dict(
                id=event.response_id,
                object="realtime.response",
                status=event.status,
                status_details=dict(reason=event.reason),
                output=[
                    dict(
                        id=event.item_id,
                        object="realtime.item",
                        type="message",
                        role="assistant",
                        content=content,
                    )
                ],
                usage=event.usage,
            ),
        )
    elif isinstance(event, (TextDelta, TextFinished, AudioDelta, AudioFinished)):
        is_audio = isinstance(event, (AudioDelta, AudioFinished))
        is_done = isinstance(event, (TextFinished, AudioFinished))
        if is_audio:
            content_name = "output_audio"
        elif output_modalities == ("audio",):
            content_name = "output_audio_transcript"
        else:
            content_name = "output_text"
        server_event: ServerEvent = dict(
            type=f'response.{content_name}.{"done" if is_done else "delta"}',
            response_id=event.response_id,
            item_id=event.item_id,
            output_index=0,
            content_index=0,
        )
        if isinstance(event, AudioDelta):
            server_event["delta"] = base64.b64encode(event.pcm).decode("ascii")
        elif isinstance(event, TextDelta):
            server_event["delta"] = event.text
        elif isinstance(event, TextFinished) and (
            content_name == "output_audio_transcript"
        ):
            server_event["transcript"] = event.text
        elif isinstance(event, TextFinished):
            server_event["text"] = event.text
        else:
            pass
        return server_event
    elif isinstance(event, TurnFailure):
        return dict(
            type="error",
            error=dict(type=event.error_type, code=event.code, message=event.message),
        )
    else:
        raise TypeError(f"Unsupported typed output: {type(event)}")


def project_control(event: ControlEvent) -> ServerEvent:
    if isinstance(event, UnitCompleted):
        return dict(type="sglang.unit.done", unit_id=event.unit_id)
    elif isinstance(event, Created):
        return dict(
            type="session.created",
            session=dict(
                id=event.session_id,
                object="realtime.session",
                type=event.session_type,
                model=event.model,
                sglang=dict(granted=None),
            ),
        )
    elif isinstance(event, Updated):
        return dict(
            type="session.updated",
            client_event_id=event.client_event_id,
            session={
                **(event.config or {}),
                "id": event.session_id,
                "object": "realtime.session",
                "model": event.model,
                "type": event.session_type,
                "sglang": {
                    **(event.config or {}).get("sglang", {}),
                    "granted": event.granted,
                },
            },
        )
    elif isinstance(event, Failure):
        return dict(
            type="error",
            sglang=dict(fatal=event.is_fatal),
            error=dict(
                type="server_error" if event.is_fatal else "invalid_request_error",
                code=event.code,
                message=event.message,
                event_id=event.client_event_id,
                param=event.param,
            ),
        )
    elif isinstance(event, Closed):
        return dict(
            type="session.closed",
            reason=event.reason,
            client_event_id=event.client_event_id,
        )
    elif isinstance(event, Accepted):
        return dict(
            type="sglang.input_audio.accepted",
            seq=event.sequence,
            accepted_end_ms=event.accepted_end_ms,
            client_event_id=event.client_event_id,
        )
    elif isinstance(event, Cleared):
        return dict(
            type="input_audio_buffer.cleared",
            client_event_id=event.client_event_id,
            sglang=dict(discarded_ms=event.discarded_ms),
        )
    elif isinstance(event, Ended):
        return dict(
            type="sglang.input_audio.ended",
            accepted_end_ms=event.accepted_end_ms,
            tail_policy=event.tail_policy,
            client_event_id=event.client_event_id,
        )
    elif isinstance(event, Drained):
        return dict(
            type="sglang.input_audio.drained",
            accepted_end_ms=event.accepted_end_ms,
            consumed_ms=event.consumed_ms,
            discarded_ms=event.discarded_ms,
            padding_ms=event.padding_ms,
            client_event_id=event.client_event_id,
        )
    else:
        raise TypeError(f"Unsupported control event: {type(event)}")
