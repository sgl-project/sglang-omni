"""Wire projections; computation emits raw PCM and concrete event values."""

import base64
from dataclasses import asdict

from sglang_omni.serve.realtime.control import (
    Accepted,
    Cancelled,
    Cleared,
    Closed,
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
    InputCommitted,
    OutputEvent,
    ResponseFinished,
    ResponseStarted,
    SpeechBoundary,
    TextDelta,
    TextFinished,
    TranscriptionDelta,
    TranscriptionFailure,
    TranscriptionFinished,
    TranscriptionRevision,
    TranscriptionSegment,
    TurnFailure,
)


def project_output(
    event: OutputEvent, *, legacy: bool = False, output_modalities=None
) -> dict:
    if isinstance(event, SpeechBoundary):
        suffix = "started" if event.started else "stopped"
        key = "audio_start_ms" if event.started else "audio_end_ms"
        return dict(
            type=f"input_audio_buffer.speech_{suffix}",
            item_id=event.item_id,
            **{key: event.time_ms},
        )
    if isinstance(event, InputCommitted):
        return dict(type="input_audio_buffer.committed", item_id=event.item_id)
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
    if isinstance(event, ResponseFinished):
        content = [dict(type="text" if legacy else "output_text", text=event.text)]
        if output_modalities is not None and "text" not in output_modalities:
            content = []
        if event.include_audio:
            content.append(
                dict(type="audio" if legacy else "output_audio", transcript=event.text)
            )
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
    if isinstance(event, (TextDelta, TextFinished, AudioDelta, AudioFinished)):
        audio = isinstance(event, (AudioDelta, AudioFinished))
        done = isinstance(event, (TextFinished, AudioFinished))
        name = "audio" if audio else "text"
        if not legacy:
            name = "output_" + name
            if not audio and output_modalities == ["audio"]:
                name = "output_audio_transcript"
        result = dict(
            type=f'response.{name}.{"done" if done else "delta"}',
            response_id=event.response_id,
            item_id=event.item_id,
            output_index=0,
            content_index=int(audio) if legacy else 0,
        )
        if isinstance(event, AudioDelta):
            result["delta"] = base64.b64encode(event.pcm).decode("ascii")
        elif not audio:
            result[
                (
                    ("transcript" if name == "output_audio_transcript" else "text")
                    if done
                    else "delta"
                )
            ] = event.text
        return result
    if isinstance(event, TranscriptionFailure):
        return dict(
            type="conversation.item.input_audio_transcription.failed",
            item_id=event.item_id,
            content_index=0,
            error=dict(
                type="server_error", code=event.code, message=event.message, param=None
            ),
        )
    if isinstance(event, TranscriptionSegment):
        return dict(
            type="conversation.item.input_audio_transcription.segment",
            item_id=event.item_id,
            content_index=0,
            id=event.segment_id,
            start=event.start_ms / 1000,
            end=event.end_ms / 1000,
            text=event.text,
        )
    if isinstance(event, TranscriptionDelta):
        result = dict(item_id=event.item_id, content_index=0)
        if isinstance(event, TranscriptionRevision):
            result.update(
                type="sglang.transcription.revision",
                text=event.text,
                base_revision_id=event.base_revision_id,
                revision_id=event.revision_id,
            )
        elif isinstance(event, TranscriptionFinished):
            result.update(
                type="conversation.item.input_audio_transcription.completed",
                transcript=event.text,
            )
        else:
            result.update(
                type="conversation.item.input_audio_transcription.delta",
                delta=event.text,
            )
        if event.segment_id is not None:
            metadata = dict(
                segment_id=event.segment_id,
                start_ms=event.start_ms,
                end_ms=event.end_ms,
            )
            if isinstance(event, TranscriptionRevision) or legacy:
                result.update(metadata)
            else:
                result["sglang"] = metadata
        return result
    if isinstance(event, TurnFailure):
        return dict(
            type="error",
            error=dict(type=event.type, code=event.code, message=event.message),
        )
    raise TypeError(f"Unsupported typed output: {type(event)}")


def project_control(event):

    if isinstance(event, UnitCompleted):
        return dict(type="sglang.unit.done", unit_id=event.unit_id)
    if isinstance(event, Created):
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
    if isinstance(event, Updated):
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
    if isinstance(event, Failure):
        return dict(
            type="error",
            sglang=dict(fatal=event.fatal),
            error=dict(
                type="server_error" if event.fatal else "invalid_request_error",
                code=event.code,
                message=event.message,
                event_id=event.client_event_id,
                param=event.param,
            ),
        )
    if isinstance(event, Closed):
        return dict(
            type="session.closed",
            reason=event.reason,
            client_event_id=event.client_event_id,
            held=dict(kv_tokens=0, slots={}, bytes=0),
        )
    names = {
        Accepted: "sglang.input_audio.accepted",
        Cleared: "input_audio_buffer.cleared",
        Ended: "sglang.input_audio.ended",
        Drained: "sglang.input_audio.drained",
        Cancelled: "sglang.response.cancelled",
    }
    result = dict(type=names[type(event)], **asdict(event))
    if isinstance(event, Cleared):
        result["sglang"] = dict(discarded_ms=result.pop("discarded_ms"))
    if isinstance(event, Cancelled):
        result["input_policy"] = "preserve"
    return result
