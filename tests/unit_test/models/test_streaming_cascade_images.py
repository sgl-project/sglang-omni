# SPDX-License-Identifier: Apache-2.0
"""Image attachment, history, HTTP conversion and delayed transcript contracts."""

import asyncio
import base64
import io
from dataclasses import replace

import httpx
import pytest
from PIL import Image

from sglang_omni.config.manager import ConfigManager
from sglang_omni.models.streaming_cascade.base import (
    ChatMessage,
    messages_of,
    text_chunk,
    transcript_chunk,
)
from sglang_omni.models.streaming_cascade.dialog import Dialog, DialogConfig
from sglang_omni.models.streaming_cascade.generators import (
    OpenAIChatGenerator,
    OpenAIChatGeneratorConfig,
)
from sglang_omni.models.streaming_cascade.recognizers import (
    OpenAITranscriptionRecognizer,
    OpenAITranscriptionRecognizerConfig,
)
from sglang_omni.proto.session import TimedChunk
from sglang_omni.proto.streams import (
    AUDIO,
    IMAGE,
    IMAGE_FRAME,
    MESSAGES,
    PCM16,
    SPEECH_STARTED,
    SPEECH_STOPPED,
    TURN_DONE,
    TURN_EMPTY,
)
from sglang_omni.serve.realtime.session_bridge import capabilities_of
from tests.unit_test.fixtures.cascade_fakes import drain
from tests.unit_test.fixtures.session_outputs import control_chunk

PROFILE = "examples/streaming_cascade/profiles/qwen3-asr-qwen3-vl-tts-base/launch.yaml"


def image_chunk(color: str, timestamp_ms: float = 0) -> TimedChunk:
    encoded = io.BytesIO()
    Image.new("RGB", (8, 8), color).save(encoded, format="PNG")
    return TimedChunk(IMAGE, timestamp_ms, 0, 0, encoded.getvalue(), format=IMAGE_FRAME)


@pytest.mark.asyncio
async def test_delayed_transcript_keeps_the_question_image_and_next_turn_keeps_its_own() -> (
    None
):
    """Uploading B while ASR is delayed must not attach B to the earlier question."""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def transcription(request: httpx.Request) -> httpx.Response:
        entered.set()
        await release.wait()
        return httpx.Response(200, json={"text": "What color?", "language": "English"})

    recognizer = OpenAITranscriptionRecognizer(
        OpenAITranscriptionRecognizerConfig(base_url="http://asr/v1", model="asr")
    )
    await recognizer.client.aclose()
    recognizer.client = httpx.AsyncClient(transport=httpx.MockTransport(transcription))
    dialog = Dialog(DialogConfig())
    red = image_chunk("red")
    blue = image_chunk("blue", 100)
    await drain(dialog, [red])
    audio = TimedChunk(AUDIO, 0, 100, 0, bytes(3200), format=PCM16)
    transcription_task = asyncio.create_task(drain(recognizer, [audio]))
    await entered.wait()
    await drain(dialog, [blue])
    release.set()
    transcripts = await transcription_task
    assert (transcripts[0].t_start_ms, transcripts[0].duration_ms) == (0, 100)
    first = await drain(dialog, transcripts)
    prompt = messages_of(next(chunk for chunk in first if chunk.format == MESSAGES))
    assert prompt[-1].images[0].image_bytes == red.payload
    next_transcript = replace(
        transcript_chunk("And this one?", True), t_start_ms=100, duration_ms=100
    )
    second = await drain(
        dialog,
        [
            text_chunk("It is red.", 1),
            control_chunk(TURN_DONE, turn=1),
            next_transcript,
        ],
    )
    prompt = messages_of(next(chunk for chunk in second if chunk.format == MESSAGES))
    assert [message.images[0].image_bytes for message in prompt if message.images] == [
        red.payload,
        blue.payload,
    ]


@pytest.mark.asyncio
@pytest.mark.parametrize("add_image", [False, True])
async def test_unheard_reply_merging_keeps_every_question_image(
    add_image: bool,
) -> None:
    """Merging user messages after an unheard reply dropped the earlier image, or put both images after all the text so the model answered the new question from the old image."""
    dialog = Dialog(DialogConfig())
    red = image_chunk("red")
    blue = image_chunk("blue", 100)
    uploads = [blue] if add_image else []
    outputs = await drain(
        dialog,
        [
            red,
            transcript_chunk("What color?", True),
            control_chunk(SPEECH_STARTED, t_ms=100),
            control_chunk(SPEECH_STOPPED, t_ms=200),
            *uploads,
            replace(transcript_chunk("Does it have a handle?", True), t_start_ms=200),
        ],
    )
    prompts = [messages_of(chunk) for chunk in outputs if chunk.format == MESSAGES]
    assert [message.role for message in prompts[-1]] == ["system", "user"]
    assert prompts[-1][-1].content == "What color? Does it have a handle?"
    generator = OpenAIChatGenerator(
        OpenAIChatGeneratorConfig(
            base_url="http://vlm/v1", model="vlm", supports_images=True
        )
    )
    try:
        content_parts = generator.request_body(prompts[-1])["messages"][-1]["content"]
    finally:
        await generator.close()
    expected_parts = (
        [red.payload, "What color?", blue.payload, "Does it have a handle?"]
        if add_image
        else [red.payload, "What color? Does it have a handle?"]
    )
    assert [
        (
            part["text"]
            if part["type"] == "text"
            else base64.b64decode(part["image_url"]["url"].split(",", 1)[1])
        )
        for part in content_parts
    ] == expected_parts


@pytest.mark.asyncio
async def test_images_uploaded_together_reach_one_question_and_newest_stay_in_history() -> (
    None
):
    """Several uploads before one question kept only the last, and eviction dropped a whole message's images."""
    dialog = Dialog(DialogConfig(max_history_images=3))
    first_images = [image_chunk("red"), image_chunk("green")]
    second_images = [image_chunk("blue", 100), image_chunk("yellow", 100)]
    first = await drain(dialog, [*first_images, transcript_chunk("Compare them", True)])
    prompt = messages_of(next(chunk for chunk in first if chunk.format == MESSAGES))
    assert [image.image_bytes for image in prompt[-1].images] == [
        image.payload for image in first_images
    ]
    second = await drain(
        dialog,
        [
            text_chunk("Red and green.", 1),
            control_chunk(TURN_DONE, turn=1),
            *second_images,
            replace(transcript_chunk("And these?", True), t_start_ms=100),
        ],
    )
    prompt = messages_of(next(chunk for chunk in second if chunk.format == MESSAGES))
    assert [[image.image_bytes for image in message.images] for message in prompt] == [
        [],
        [first_images[1].payload],
        [],
        [image.payload for image in second_images],
    ]


@pytest.mark.asyncio
async def test_pending_uploads_over_budget_drop_the_oldest_and_keep_the_session() -> (
    None
):
    """Uploading more images than the history keeps before speaking ended the session."""
    dialog = Dialog(DialogConfig(max_history_images=2))
    uploads = [image_chunk(color) for color in ("red", "green", "blue")]
    outputs = await drain(dialog, [*uploads, transcript_chunk("What are these?", True)])
    prompt = messages_of(next(chunk for chunk in outputs if chunk.format == MESSAGES))
    assert [image.image_bytes for image in prompt[-1].images] == [
        upload.payload for upload in uploads[1:]
    ]


@pytest.mark.asyncio
async def test_vad_question_uses_image_uploaded_after_startup_silence() -> None:
    """Startup silence and the whole onset packet previously left an image uploaded before speech onset off the question."""
    configuration = ConfigManager.from_file(PROFILE).config.component_graph()
    recognizer = OpenAITranscriptionRecognizer(configuration.node("asr").component)
    await recognizer.client.aclose()

    async def transcription(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "What color?", "language": "English"})

    recognizer.client = httpx.AsyncClient(transport=httpx.MockTransport(transcription))
    startup = TimedChunk(AUDIO, 0, 1000, 0, bytes(32000), format=PCM16)
    speech = control_chunk(SPEECH_STARTED, t_ms=1088)
    packet = TimedChunk(AUDIO, 1000, 200, 1, bytes(6400), format=PCM16)
    transcripts = await drain(
        recognizer, [startup, packet, speech, control_chunk(SPEECH_STOPPED, t_ms=1200)]
    )
    assert (transcripts[0].t_start_ms, transcripts[0].duration_ms) == (1088, 112)
    dialog = Dialog(DialogConfig())
    red = image_chunk("red")
    blue = image_chunk("blue", 1040)
    outputs = await drain(dialog, [red, blue, *transcripts])
    messages = messages_of(next(chunk for chunk in outputs if chunk.format == MESSAGES))
    assert [image.image_bytes for image in messages[-1].images] == [
        red.payload,
        blue.payload,
    ]


@pytest.mark.asyncio
async def test_image_profile_manual_override_transcribes_without_speech_events() -> (
    None
):
    configuration = ConfigManager.from_file(PROFILE).config.component_graph()
    manual = configuration.apply({"turn_detection": None})
    recognizer = OpenAITranscriptionRecognizer(manual.node("asr").component)
    await recognizer.client.aclose()

    async def transcription(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"text": "Manual question"})

    recognizer.client = httpx.AsyncClient(transport=httpx.MockTransport(transcription))
    outputs = await drain(
        recognizer,
        [TimedChunk(AUDIO, 0, 100, 0, bytes(3200), format=PCM16)],
    )
    assert outputs[0].payload["text"] == "Manual question"


@pytest.mark.asyncio
async def test_follow_up_and_session_isolation_preserve_structured_images_at_http_boundary() -> (
    None
):
    dialogs = [Dialog(DialogConfig()), Dialog(DialogConfig())]
    uploaded_images = [image_chunk("red"), image_chunk("blue")]
    for dialog, image in zip(dialogs, uploaded_images, strict=True):
        await drain(
            dialog,
            [
                image,
                transcript_chunk("What color?", True),
                text_chunk("A cup.", 1),
                control_chunk(TURN_DONE, turn=1),
            ],
        )
    generator = OpenAIChatGenerator(
        OpenAIChatGeneratorConfig(
            base_url="http://vlm/v1", model="vlm", supports_images=True
        )
    )
    try:
        for dialog, image in zip(dialogs, uploaded_images, strict=True):
            outputs = await drain(
                dialog,
                [
                    replace(
                        transcript_chunk("Describe the details.", True), t_start_ms=100
                    )
                ],
            )
            messages = messages_of(
                next(chunk for chunk in outputs if chunk.format == MESSAGES)
            )
            request_body = generator.request_body(messages)
            content_parts = request_body["messages"][1]["content"]
            assert (
                base64.b64decode(content_parts[0]["image_url"]["url"].split(",", 1)[1])
                == image.payload
            )
            assert content_parts[1] == {"type": "text", "text": "What color?"}
            assert request_body["messages"][-1] == {
                "role": "user",
                "content": "Describe the details.",
            }
            assert all(
                "uncertain" not in message and "images" not in message
                for message in request_body["messages"]
            )
    finally:
        await generator.close()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "max_history_images,max_history_image_bytes", [(1, 512), (8, 80)]
)
async def test_image_history_eviction_preserves_text_and_pending_uploads_on_reconfiguration(
    max_history_images: int, max_history_image_bytes: int
) -> None:
    dialog = Dialog(
        DialogConfig(
            max_history_images=max_history_images,
            max_history_image_bytes=max_history_image_bytes,
        )
    )
    await drain(
        dialog,
        [
            image_chunk("red"),
            transcript_chunk("First", True),
            text_chunk("A red cup.", 1),
            control_chunk(TURN_DONE, turn=1),
            image_chunk("blue", 100),
        ],
    )
    replacement = Dialog(dialog.config)
    replacement.adopt(dialog)
    outputs = await drain(
        replacement, [replace(transcript_chunk("Second", True), t_start_ms=100)]
    )
    prompt = messages_of(next(chunk for chunk in outputs if chunk.format == MESSAGES))
    assert prompt[1].content == "First"
    assert prompt[1].images == ()
    assert prompt[-1].images[0].image_bytes == image_chunk("blue").payload


@pytest.mark.asyncio
async def test_text_only_generator_rejects_images_and_keeps_text_wire_format() -> None:
    dialog = Dialog(DialogConfig())
    await drain(dialog, [image_chunk("red"), transcript_chunk("Question", True)])
    generator = OpenAIChatGenerator(
        OpenAIChatGeneratorConfig(base_url="http://chat/v1", model="chat")
    )
    try:
        with pytest.raises(ValueError, match="does not support images"):
            generator.request_body(dialog.messages)
        assert generator.request_body([ChatMessage(role="user", content="Hello")])[
            "messages"
        ] == [{"role": "user", "content": "Hello"}]
    finally:
        await generator.close()


@pytest.mark.asyncio
async def test_empty_transcript_preserves_uploaded_image_for_the_next_question() -> (
    None
):
    """An empty recognition result discarded the upload before a spoken question."""
    dialog = Dialog(DialogConfig())
    uploaded_image = image_chunk("red")
    outputs = await drain(
        dialog,
        [
            uploaded_image,
            transcript_chunk("   ", True),
            replace(transcript_chunk("What color?", True), t_start_ms=100),
        ],
    )
    assert any(chunk.payload.get("type") == TURN_EMPTY for chunk in outputs)
    prompt = messages_of(next(chunk for chunk in outputs if chunk.format == MESSAGES))
    assert prompt[-1].content == "What color?"
    assert prompt[-1].images[0].image_bytes == uploaded_image.payload


@pytest.mark.asyncio
async def test_reduced_image_budget_preserves_older_images_that_still_fit() -> None:
    """Evicted image bytes were still counted and removed every older image."""
    small_image = image_chunk("red")
    large_image = image_chunk("blue")
    assert len(small_image.payload) < len(large_image.payload)
    dialog = Dialog(DialogConfig())
    await drain(
        dialog,
        [
            small_image,
            transcript_chunk("First question", True),
            text_chunk("First answer", 1),
            control_chunk(TURN_DONE, turn=1),
            replace(large_image, t_start_ms=100),
            replace(transcript_chunk("Second question", True), t_start_ms=100),
            text_chunk("Second answer", 2),
            control_chunk(TURN_DONE, turn=2),
        ],
    )
    replacement = Dialog(DialogConfig(max_history_image_bytes=len(small_image.payload)))
    replacement.adopt(dialog)
    outputs = await drain(
        replacement,
        [replace(transcript_chunk("Follow-up question", True), t_start_ms=200)],
    )
    prompt = messages_of(next(chunk for chunk in outputs if chunk.format == MESSAGES))
    assert prompt[1].images[0].image_bytes == small_image.payload
    assert prompt[3].images == ()
    assert [message.content for message in prompt[1:]] == [
        "First question",
        "First answer",
        "Second question",
        "Second answer",
        "Follow-up question",
    ]


@pytest.mark.asyncio
async def test_image_state_bytes_include_pending_queued_and_retained_uploads() -> None:
    image_dialog = Dialog(DialogConfig(max_history_images=1))
    text_dialog = Dialog(image_dialog.config)
    first_image = image_chunk("red")
    second_image = image_chunk("blue", 100)
    await drain(image_dialog, [first_image])
    assert image_dialog.state_bytes() - text_dialog.state_bytes() == len(
        first_image.payload
    )
    for dialog in (image_dialog, text_dialog):
        await drain(dialog, [transcript_chunk("First question", True)])
    assert image_dialog.state_bytes() - text_dialog.state_bytes() == len(
        first_image.payload
    )
    await drain(image_dialog, [second_image])
    for dialog in (image_dialog, text_dialog):
        await drain(
            dialog,
            [replace(transcript_chunk("Second question", True), t_start_ms=100)],
        )
    assert image_dialog.state_bytes() - text_dialog.state_bytes() == len(
        first_image.payload
    ) + len(second_image.payload)
    replacement = Dialog(image_dialog.config)
    replacement.adopt(image_dialog)
    assert replacement.state_bytes() == image_dialog.state_bytes()
    for dialog in (replacement, text_dialog):
        await drain(
            dialog, [text_chunk("First answer", 1), control_chunk(TURN_DONE, turn=1)]
        )
    assert replacement.state_bytes() - text_dialog.state_bytes() == len(
        second_image.payload
    )


def test_image_profile_requires_explicit_capability_and_routes_images_only_to_dialog() -> (
    None
):
    config = ConfigManager.from_file(PROFILE).config
    assert config.session_ports().input_modalities == ("audio", "image")
    graph = config.component_graph()
    assert IMAGE_FRAME not in graph.node("asr").edges()["input"]
    assert IMAGE_FRAME not in config.session_graph().nodes["vad"].edges()["input"]
    assert IMAGE_FRAME in graph.node("dialog").edges()["input"]
    granted = capabilities_of(config.session_ports()).to_granted_capabilities()
    assert (
        granted["input_image_format"]["max_frames_per_unit"]
        == graph.node("dialog").component.max_history_images
    )
    with pytest.raises(ValueError, match="supports_images"):
        graph.apply({"nodes": {"llm": {"supports_images": False}}})
