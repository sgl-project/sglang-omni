# SPDX-License-Identifier: Apache-2.0
"""Gradio UI for the S2-Pro TTS playground."""

from __future__ import annotations

import tempfile
import time

import gradio as gr

from playground.tts.api_client import SpeechDemoClient, SpeechDemoClientError
from playground.tts.audio_stream import WavChunkAccumulator
from playground.tts.models import GenerationSettings, SpeechSynthesisRequest


def _write_temp_wav(audio_bytes: bytes) -> str:
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


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
    ) -> tuple[list[dict], str, str | None, str]:
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

        try:
            request.validate()
        except ValueError as exc:
            gr.Warning(str(exc))
            return history, text, None, str(exc)

        user_content = request.build_history_user_content()

        try:
            result = client.synthesize(request)
        except SpeechDemoClientError as exc:
            updated_history = history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"Error: {exc}"},
            ]
            return updated_history, "", None, f"Request failed: {exc}"

        audio_path = _write_temp_wav(result.audio_bytes)
        summary = f"{result.elapsed_s:.1f}s | {result.size_bytes / 1024:.0f} KB"
        updated_history = history + [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {"path": audio_path, "mime_type": "audio/wav"},
                    summary,
                ],
            },
        ]
        return updated_history, "", audio_path, summary

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
    ):
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

        try:
            request.validate()
        except ValueError as exc:
            gr.Warning(str(exc))
            yield history, text, None, None, str(exc)
            return

        user_content = request.build_history_user_content()
        in_progress_history = history + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": "Streaming audio..."},
        ]
        yield in_progress_history, "", None, None, "Connecting to speech stream..."

        started_at = time.perf_counter()
        accumulator = WavChunkAccumulator()
        chunk_count = 0
        first_audio_s: float | None = None

        try:
            for event in client.stream_synthesize(request):
                if event.audio_bytes is None:
                    continue

                chunk_count += 1
                if first_audio_s is None:
                    first_audio_s = time.perf_counter() - started_at

                live_audio = accumulator.add_wav_chunk(event.audio_bytes)
                status = (
                    f"Streaming | chunk {chunk_count} | "
                    f"first audio {first_audio_s:.2f}s"
                )
                yield in_progress_history, "", live_audio, None, status
        except SpeechDemoClientError as exc:
            failed_history = history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"Error: {exc}"},
            ]
            yield failed_history, "", None, None, f"Request failed: {exc}"
            return
        except ValueError as exc:
            failed_history = history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": f"Error: {exc}"},
            ]
            yield failed_history, "", None, None, f"Stream parse failed: {exc}"
            return

        final_audio_path = accumulator.write_temp_wav()
        if final_audio_path is None:
            failed_history = history + [
                {"role": "user", "content": user_content},
                {"role": "assistant", "content": "Error: No audio was returned."},
            ]
            yield failed_history, "", None, None, "No audio was returned."
            return

        elapsed_s = time.perf_counter() - started_at
        summary = (
            f"{elapsed_s:.1f}s total | {chunk_count} chunks"
            + (
                f" | first audio {first_audio_s:.2f}s"
                if first_audio_s is not None
                else ""
            )
        )
        completed_history = history + [
            {"role": "user", "content": user_content},
            {
                "role": "assistant",
                "content": [
                    {"path": final_audio_path, "mime_type": "audio/wav"},
                    summary,
                ],
            },
        ]
        yield completed_history, "", None, final_audio_path, summary

    return synthesize_stream


def create_demo(api_base: str) -> gr.Blocks:
    synthesize = make_non_streaming_handler(api_base)
    synthesize_stream = make_streaming_handler(api_base)

    with gr.Blocks(title="S2-Pro TTS Playground") as demo:
        gr.Markdown("## S2-Pro Text-to-Speech")
        gr.Markdown(
            "*First request may take 10-20s due to warmup. Subsequent requests are much faster thanks to KV cache reuse.*",
            elem_classes=["note"],
        )

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
        ]
        outputs = [chatbot, text_input, audio_output, status_text]

        synth_btn.click(
            fn=synthesize,
            inputs=inputs,
            outputs=outputs,
        )
        text_input.submit(
            fn=synthesize,
            inputs=inputs,
            outputs=outputs,
        )
        stream_btn.click(
            fn=synthesize_stream,
            inputs=inputs,
            outputs=[
                chatbot,
                text_input,
                stream_audio,
                stream_final_audio,
                stream_status,
            ],
        )
        clear_btn.click(
            fn=lambda: ([], None, "Ready", None, None, "Ready"),
            outputs=[
                chatbot,
                audio_output,
                status_text,
                stream_audio,
                stream_final_audio,
                stream_status,
            ],
        )

    return demo
