# SPDX-License-Identifier: Apache-2.0
"""Gradio UI for the S2-Pro TTS playground."""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import gradio as gr

from playground.tts.api_client import SpeechDemoClient, SpeechDemoClientError
from playground.tts.artifacts import ArtifactStore
from playground.tts.audio_stream import (
    BufferedWavChunkEmitter,
    WavChunkAccumulator,
    wav_duration_seconds,
)
from playground.tts.models import GenerationSettings, SpeechSynthesisRequest

_ARTIFACT_STORE = ArtifactStore()
_LIVE_AUDIO_MIN_CHUNK_DURATION_S = 1.0
_LIVE_AUDIO_MAX_BUFFERED_CHUNKS = 3
_SYNTH_IDLE_LABEL = "Synthesize"
_SYNTH_BUSY_LABEL = "Synthesizing..."
_STREAM_IDLE_LABEL = "Start Streaming"
_STREAM_BUSY_LABEL = "Streaming..."


@dataclass(frozen=True)
class StreamingUiUpdate:
    history: Any
    text_input: Any
    live_audio: Any
    final_audio: Any
    status: Any
    artifact_paths: Any
    pending_stream_result: Any
    synth_button: Any
    stream_button: Any

    def to_gradio_outputs(self) -> tuple[Any, ...]:
        return (
            self.history,
            self.text_input,
            self.live_audio,
            self.final_audio,
            self.status,
            self.artifact_paths,
            self.pending_stream_result,
            self.synth_button,
            self.stream_button,
        )


def _build_request(
    text: str,
    ref_audio: str | None,
    ref_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
) -> tuple[SpeechSynthesisRequest, list[Any]]:
    request = SpeechSynthesisRequest(
        text=text,
        reference_audio_path=ref_audio,
        reference_text=ref_text,
        settings=GenerationSettings(
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            max_new_tokens=max_new_tokens,
        ),
    )
    request.validate()
    return request, request.build_history_user_content()


def _append_history(
    history: list[dict],
    user_content: list[Any],
    assistant_content: Any,
) -> list[dict]:
    return history + [
        {"role": "user", "content": user_content},
        {"role": "assistant", "content": assistant_content},
    ]


def _store_wav_artifact(
    audio_bytes: bytes, artifact_paths: list[str]
) -> tuple[str, list[str]]:
    path = _ARTIFACT_STORE.write_bytes(audio_bytes, suffix=".wav")
    return path, artifact_paths + [path]


def _reset_audio_output() -> dict[str, Any]:
    return gr.update(value=None)


def _keep_audio_output():
    return gr.skip()


def _button_update(
    *,
    value: str | None = None,
    interactive: bool | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {}
    if value is not None:
        kwargs["value"] = value
    if interactive is not None:
        kwargs["interactive"] = interactive
    return gr.update(**kwargs)


def _lock_request_buttons(active_request: str) -> tuple[dict[str, Any], dict[str, Any]]:
    synth_button = _button_update(
        value=(
            _SYNTH_BUSY_LABEL
            if active_request == "non_streaming"
            else _SYNTH_IDLE_LABEL
        ),
        interactive=False,
    )
    stream_button = _button_update(
        value=(
            _STREAM_BUSY_LABEL if active_request == "streaming" else _STREAM_IDLE_LABEL
        ),
        interactive=False,
    )
    return synth_button, stream_button


def _unlock_request_buttons() -> tuple[dict[str, Any], dict[str, Any]]:
    return (
        _button_update(value=_SYNTH_IDLE_LABEL, interactive=True),
        _button_update(value=_STREAM_IDLE_LABEL, interactive=True),
    )


def _prepare_non_stream_request():
    synth_button, stream_button = _lock_request_buttons("non_streaming")
    return "Submitting request...", synth_button, stream_button


def _prepare_stream_request():
    synth_button, stream_button = _lock_request_buttons("streaming")
    return "Connecting to speech stream...", synth_button, stream_button


def _format_audio_duration(audio_duration_s: float) -> str:
    return f"{audio_duration_s:.1f}s audio"


def _format_non_streaming_summary(
    *,
    elapsed_s: float,
    audio_duration_s: float,
    size_bytes: int,
) -> str:
    return (
        f"{_format_audio_duration(audio_duration_s)} | "
        f"{elapsed_s:.1f}s total | {size_bytes / 1024:.0f} KB"
    )


def _format_streaming_summary(
    *,
    elapsed_s: float,
    audio_duration_s: float,
    chunk_count: int,
    first_audio_s: float | None,
) -> str:
    summary = (
        f"{_format_audio_duration(audio_duration_s)} | "
        f"{elapsed_s:.1f}s total | {chunk_count} chunks"
    )
    if first_audio_s is not None:
        summary += f" | first audio {first_audio_s:.2f}s"
    return summary


def _clear_history(artifact_paths: list[str]):
    _ARTIFACT_STORE.cleanup_paths(artifact_paths)
    synth_button, stream_button = _unlock_request_buttons()
    return (
        [],
        _reset_audio_output(),
        "Ready",
        _reset_audio_output(),
        _reset_audio_output(),
        "Ready",
        [],
        None,
        synth_button,
        stream_button,
    )


def _publish_pending_stream_result(pending_result: dict[str, Any] | None):
    synth_button, stream_button = _unlock_request_buttons()
    if not pending_result:
        return (
            gr.skip(),
            gr.skip(),
            gr.skip(),
            None,
            synth_button,
            stream_button,
        )

    return (
        pending_result["history"],
        pending_result["final_audio_path"],
        pending_result["status"],
        None,
        synth_button,
        stream_button,
    )


def make_non_streaming_handler(api_base: str):
    client = SpeechDemoClient(api_base)

    def synthesize(
        text: str,
        ref_audio: str | None,
        ref_text: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        history: list[dict],
        artifact_paths: list[str],
    ) -> tuple[
        list[dict], str, str | None, str, list[str], dict[str, Any], dict[str, Any]
    ]:
        try:
            request, user_content = _build_request(
                text,
                ref_audio,
                ref_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            )
        except ValueError as exc:
            warnings.warn(str(exc))
            synth_button, stream_button = _unlock_request_buttons()
            return (
                history,
                text,
                None,
                str(exc),
                artifact_paths,
                synth_button,
                stream_button,
            )

        try:
            result = client.synthesize(request)
        except SpeechDemoClientError as exc:
            updated_history = _append_history(history, user_content, f"Error: {exc}")
            synth_button, stream_button = _unlock_request_buttons()
            return (
                updated_history,
                "",
                None,
                f"Request failed: {exc}",
                artifact_paths,
                synth_button,
                stream_button,
            )

        audio_path, artifact_paths = _store_wav_artifact(
            result.audio_bytes, artifact_paths
        )
        summary = _format_non_streaming_summary(
            elapsed_s=result.elapsed_s,
            audio_duration_s=wav_duration_seconds(result.audio_bytes),
            size_bytes=result.size_bytes,
        )
        updated_history = _append_history(
            history,
            user_content,
            [
                {"path": audio_path, "mime_type": "audio/wav"},
                summary,
            ],
        )
        synth_button, stream_button = _unlock_request_buttons()
        return (
            updated_history,
            "",
            audio_path,
            summary,
            artifact_paths,
            synth_button,
            stream_button,
        )

    return synthesize


def make_streaming_handler(api_base: str):
    client = SpeechDemoClient(api_base)

    def synthesize_stream(
        text: str,
        ref_audio: str | None,
        ref_text: str,
        temperature: float,
        top_p: float,
        top_k: int,
        max_new_tokens: int,
        history: list[dict],
        artifact_paths: list[str],
    ):
        try:
            request, user_content = _build_request(
                text,
                ref_audio,
                ref_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
            )
        except ValueError as exc:
            warnings.warn(str(exc))
            synth_button, stream_button = _unlock_request_buttons()
            yield StreamingUiUpdate(
                history=history,
                text_input=text,
                live_audio=None,
                final_audio=None,
                status=str(exc),
                artifact_paths=artifact_paths,
                pending_stream_result=None,
                synth_button=synth_button,
                stream_button=stream_button,
            ).to_gradio_outputs()
            return

        in_progress_history = _append_history(
            history, user_content, "Streaming audio..."
        )
        synth_button, stream_button = _lock_request_buttons("streaming")
        yield StreamingUiUpdate(
            history=in_progress_history,
            text_input="",
            live_audio=_reset_audio_output(),
            final_audio=_reset_audio_output(),
            status="Connecting to speech stream...",
            artifact_paths=artifact_paths,
            pending_stream_result=None,
            synth_button=synth_button,
            stream_button=stream_button,
        ).to_gradio_outputs()

        started_at = time.perf_counter()
        accumulator = WavChunkAccumulator()
        live_emitter = BufferedWavChunkEmitter(
            min_emit_duration_s=_LIVE_AUDIO_MIN_CHUNK_DURATION_S,
            max_buffered_chunks=_LIVE_AUDIO_MAX_BUFFERED_CHUNKS,
        )
        chunk_count = 0
        first_audio_s: float | None = None

        try:
            for event in client.stream_synthesize(request):
                if event.audio_bytes is None:
                    continue

                chunk_count += 1
                accumulator.add_wav_chunk(event.audio_bytes)
                live_audio = live_emitter.add_wav_chunk(event.audio_bytes)
                if live_audio is None:
                    yield StreamingUiUpdate(
                        history=in_progress_history,
                        text_input="",
                        live_audio=_keep_audio_output(),
                        final_audio=_keep_audio_output(),
                        status=f"Buffering live playback | chunk {chunk_count}",
                        artifact_paths=artifact_paths,
                        pending_stream_result=None,
                        synth_button=gr.skip(),
                        stream_button=gr.skip(),
                    ).to_gradio_outputs()
                    continue

                if first_audio_s is None:
                    first_audio_s = time.perf_counter() - started_at

                status = (
                    f"Streaming | chunk {chunk_count} | "
                    f"first audio {first_audio_s:.2f}s"
                )
                yield StreamingUiUpdate(
                    history=in_progress_history,
                    text_input="",
                    live_audio=live_audio,
                    final_audio=_keep_audio_output(),
                    status=status,
                    artifact_paths=artifact_paths,
                    pending_stream_result=None,
                    synth_button=gr.skip(),
                    stream_button=gr.skip(),
                ).to_gradio_outputs()
        except SpeechDemoClientError as exc:
            failed_history = _append_history(history, user_content, f"Error: {exc}")
            synth_button, stream_button = _unlock_request_buttons()
            yield StreamingUiUpdate(
                history=failed_history,
                text_input="",
                live_audio=_keep_audio_output(),
                final_audio=_keep_audio_output(),
                status=f"Request failed: {exc}",
                artifact_paths=artifact_paths,
                pending_stream_result=None,
                synth_button=synth_button,
                stream_button=stream_button,
            ).to_gradio_outputs()
            return
        except ValueError as exc:
            failed_history = _append_history(history, user_content, f"Error: {exc}")
            synth_button, stream_button = _unlock_request_buttons()
            yield StreamingUiUpdate(
                history=failed_history,
                text_input="",
                live_audio=_keep_audio_output(),
                final_audio=_keep_audio_output(),
                status=f"Stream parse failed: {exc}",
                artifact_paths=artifact_paths,
                pending_stream_result=None,
                synth_button=synth_button,
                stream_button=stream_button,
            ).to_gradio_outputs()
            return

        final_audio_bytes = accumulator.to_wav_bytes()
        if final_audio_bytes is None:
            failed_history = _append_history(
                history, user_content, "Error: No audio was returned."
            )
            synth_button, stream_button = _unlock_request_buttons()
            yield StreamingUiUpdate(
                history=failed_history,
                text_input="",
                live_audio=_keep_audio_output(),
                final_audio=_keep_audio_output(),
                status="No audio was returned.",
                artifact_paths=artifact_paths,
                pending_stream_result=None,
                synth_button=synth_button,
                stream_button=stream_button,
            ).to_gradio_outputs()
            return

        live_tail = live_emitter.flush()
        if live_tail is not None:
            if first_audio_s is None:
                first_audio_s = time.perf_counter() - started_at
            status = (
                f"Streaming | chunk {chunk_count} | "
                f"first audio {first_audio_s:.2f}s"
            )
            yield StreamingUiUpdate(
                history=in_progress_history,
                text_input="",
                live_audio=live_tail,
                final_audio=_keep_audio_output(),
                status=status,
                artifact_paths=artifact_paths,
                pending_stream_result=None,
                synth_button=gr.skip(),
                stream_button=gr.skip(),
            ).to_gradio_outputs()

        final_audio_path, artifact_paths = _store_wav_artifact(
            final_audio_bytes, artifact_paths
        )

        elapsed_s = time.perf_counter() - started_at
        summary = _format_streaming_summary(
            elapsed_s=elapsed_s,
            audio_duration_s=wav_duration_seconds(final_audio_bytes),
            chunk_count=chunk_count,
            first_audio_s=first_audio_s,
        )
        completed_history = _append_history(
            history,
            user_content,
            [
                {"path": final_audio_path, "mime_type": "audio/wav"},
                summary,
            ],
        )
        pending_result = {
            "history": completed_history,
            "final_audio_path": final_audio_path,
            "status": summary,
        }
        yield StreamingUiUpdate(
            history=gr.skip(),
            text_input=gr.skip(),
            live_audio=_keep_audio_output(),
            final_audio=_keep_audio_output(),
            status=_keep_audio_output(),
            artifact_paths=artifact_paths,
            pending_stream_result=pending_result,
            synth_button=gr.skip(),
            stream_button=gr.skip(),
        ).to_gradio_outputs()

    return synthesize_stream


def create_demo(api_base: str):
    synthesize = make_non_streaming_handler(api_base)
    synthesize_stream = make_streaming_handler(api_base)

    with gr.Blocks(title="S2-Pro TTS Playground") as demo:
        gr.Markdown("## S2-Pro Text-to-Speech")
        gr.Markdown(
            "*First request may take 10-20s due to warmup. Subsequent requests are much faster thanks to KV cache reuse.*",
            elem_classes=["note"],
        )
        gr.Markdown(
            "*Use the mode-specific button to start synthesis. This avoids mixing streaming and non-streaming submit behavior.*"
        )

        artifact_state = gr.State([])
        pending_stream_result = gr.State(None)

        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=4,
                )

                gr.Markdown("#### Voice Cloning (optional)")
                ref_audio = gr.Audio(label="Reference Audio", type="filepath")
                ref_text = gr.Textbox(
                    label="Reference Text",
                    placeholder="Transcript of the reference audio",
                    lines=2,
                )

                with gr.Accordion("Generation Parameters", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.1,
                        maximum=1.5,
                        value=0.8,
                        step=0.05,
                    )
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.1,
                        maximum=1.0,
                        value=0.8,
                        step=0.05,
                    )
                    top_k = gr.Slider(
                        label="Top K",
                        minimum=1,
                        maximum=100,
                        value=30,
                        step=1,
                    )
                    max_new_tokens = gr.Slider(
                        label="Max New Tokens",
                        minimum=128,
                        maximum=4096,
                        value=2048,
                        step=128,
                    )

            with gr.Column(scale=2, min_width=480):
                with gr.Tabs():
                    with gr.Tab("Non-Streaming"):
                        synth_btn = gr.Button("Synthesize", variant="primary")
                        status_text = gr.Textbox(
                            label="Status",
                            value="Ready",
                            interactive=False,
                        )
                        audio_output = gr.Audio(
                            label="Latest Audio",
                            type="filepath",
                            interactive=False,
                        )

                    with gr.Tab("Streaming"):
                        stream_btn = gr.Button("Start Streaming", variant="primary")
                        stream_status = gr.Textbox(
                            label="Stream Status",
                            value="Ready",
                            interactive=False,
                        )
                        stream_audio = gr.Audio(
                            label="Live Audio",
                            streaming=True,
                            autoplay=True,
                            interactive=False,
                        )
                        stream_final_audio = gr.Audio(
                            label="Final Audio",
                            type="filepath",
                            interactive=False,
                        )

                chatbot = gr.Chatbot(label="History", height=420)
                clear_btn = gr.Button("Clear History")

        inputs = [
            text_input,
            ref_audio,
            ref_text,
            temperature,
            top_p,
            top_k,
            max_new_tokens,
            chatbot,
            artifact_state,
        ]
        outputs = [chatbot, text_input, audio_output, status_text, artifact_state]
        outputs = outputs + [synth_btn, stream_btn]

        synth_prepare = synth_btn.click(
            fn=_prepare_non_stream_request,
            inputs=None,
            outputs=[status_text, synth_btn, stream_btn],
            queue=False,
            show_progress="hidden",
        )
        synth_prepare.then(
            fn=synthesize,
            inputs=inputs,
            outputs=outputs,
            trigger_mode="once",
            concurrency_id="tts_request",
            show_progress="minimal",
            show_progress_on=[status_text],
        )
        stream_prepare = stream_btn.click(
            fn=_prepare_stream_request,
            inputs=None,
            outputs=[stream_status, synth_btn, stream_btn],
            queue=False,
            show_progress="hidden",
        )
        stream_prepare.then(
            fn=synthesize_stream,
            inputs=inputs,
            outputs=[
                chatbot,
                text_input,
                stream_audio,
                stream_final_audio,
                stream_status,
                artifact_state,
                pending_stream_result,
                synth_btn,
                stream_btn,
            ],
            trigger_mode="once",
            concurrency_id="tts_request",
            show_progress="minimal",
            show_progress_on=[stream_status],
        )
        stream_audio.stop(
            fn=_publish_pending_stream_result,
            inputs=[pending_stream_result],
            outputs=[
                chatbot,
                stream_final_audio,
                stream_status,
                pending_stream_result,
                synth_btn,
                stream_btn,
            ],
            queue=False,
        )
        clear_btn.click(
            fn=_clear_history,
            inputs=[artifact_state],
            outputs=[
                chatbot,
                audio_output,
                status_text,
                stream_audio,
                stream_final_audio,
                stream_status,
                artifact_state,
                pending_stream_result,
                synth_btn,
                stream_btn,
            ],
        )

    return demo
