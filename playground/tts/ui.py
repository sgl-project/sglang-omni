# SPDX-License-Identifier: Apache-2.0
"""Gradio UI for the S2-Pro TTS playground."""

from __future__ import annotations

import tempfile

import gradio as gr

from playground.tts.api_client import SpeechDemoClient, SpeechDemoClientError
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


def create_demo(api_base: str) -> gr.Blocks:
    synthesize = make_non_streaming_handler(api_base)

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

                synth_btn = gr.Button("Synthesize", variant="primary")
                status_text = gr.Textbox(
                    label="Status",
                    value="Ready",
                    interactive=False,
                )

            with gr.Column(scale=2, min_width=480):
                chatbot = gr.Chatbot(label="History", height=560)
                audio_output = gr.Audio(
                    label="Latest Audio",
                    type="filepath",
                    interactive=False,
                    visible=False,
                )
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
        clear_btn.click(
            fn=lambda: ([], None, "Ready"),
            outputs=[chatbot, audio_output, status_text],
        )

    return demo
