# SPDX-License-Identifier: Apache-2.0
"""Gradio TTS playground for S2-Pro — text-to-speech with voice cloning."""

from __future__ import annotations

import argparse
import tempfile
import time

import gradio as gr
import httpx

DEFAULT_API_BASE = "http://localhost:8000"


def synthesize(
    text: str,
    ref_audio: str | None,
    ref_text: str,
    temperature: float,
    top_p: float,
    top_k: int,
    max_new_tokens: int,
    history: list[dict],
    api_base: str,
) -> tuple[list[dict], str, str | None]:
    """Call /v1/audio/speech, update history, clear text input."""
    if not text.strip():
        gr.Warning("Please enter some text to synthesize.")
        return history, text, None

    payload: dict = {
        "input": text,
        "voice": "default",
        "response_format": "wav",
        "max_new_tokens": max_new_tokens,
        "temperature": temperature,
        "top_p": top_p,
        "top_k": top_k,
    }

    if ref_audio is not None:
        payload["ref_audio"] = ref_audio
        if ref_text.strip():
            payload["ref_text"] = ref_text.strip()

    # Build user message for history
    user_content = [text]
    if ref_audio is not None:
        user_content.append({"path": ref_audio, "mime_type": "audio/wav"})

    t0 = time.perf_counter()
    try:
        resp = httpx.post(
            f"{api_base}/v1/audio/speech",
            json=payload,
            timeout=120,
        )
        resp.raise_for_status()
    except Exception as exc:
        history = history + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": f"Error: {exc}"},
        ]
        return history, "", None

    elapsed = time.perf_counter() - t0

    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    tmp.write(resp.content)
    tmp.close()

    history = history + [
        {"role": "user", "content": user_content},
        {
            "role": "assistant",
            "content": [
                {"path": tmp.name, "mime_type": "audio/wav"},
                f"{elapsed:.1f}s | {len(resp.content) / 1024:.0f} KB",
            ],
        },
    ]
    return history, "", tmp.name


def create_demo(api_base: str) -> gr.Blocks:
    with gr.Blocks(title="S2-Pro TTS Playground") as demo:
        gr.Markdown("## S2-Pro Text-to-Speech")
        gr.Markdown(
            "*First request may take 10-20s due to warmup. Subsequent requests are much faster thanks to KV cache reuse.*",
            elem_classes=["note"],
        )

        with gr.Row():
            # Left column: input controls
            with gr.Column(scale=1, min_width=320):
                text_input = gr.Textbox(
                    label="Text",
                    placeholder="Enter text to synthesize...",
                    lines=4,
                )

                gr.Markdown("#### Voice Cloning (optional)")
                ref_audio = gr.Audio(
                    label="Reference Audio",
                    type="filepath",
                )
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

            # Right column: chat history
            with gr.Column(scale=2, min_width=480):
                chatbot = gr.Chatbot(
                    label="History",
                    height=560,
                )
                audio_output = gr.Audio(
                    label="Latest Audio",
                    type="filepath",
                    interactive=False,
                    visible=False,
                )
                clear_btn = gr.Button("Clear History")

        synth_btn.click(
            fn=lambda *args: synthesize(*args, api_base=api_base),
            inputs=[
                text_input,
                ref_audio,
                ref_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                chatbot,
            ],
            outputs=[chatbot, text_input, audio_output],
        )
        text_input.submit(
            fn=lambda *args: synthesize(*args, api_base=api_base),
            inputs=[
                text_input,
                ref_audio,
                ref_text,
                temperature,
                top_p,
                top_k,
                max_new_tokens,
                chatbot,
            ],
            outputs=[chatbot, text_input, audio_output],
        )
        clear_btn.click(
            fn=lambda: ([], None),
            outputs=[chatbot, audio_output],
        )

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="S2-Pro TTS Gradio playground")
    parser.add_argument("--api-base", type=str, default=DEFAULT_API_BASE)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.api_base)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
    )


if __name__ == "__main__":
    main()
