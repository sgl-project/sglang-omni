# SPDX-License-Identifier: Apache-2.0
"""Gradio playground for SGLang-Omni — text chat with streaming + media upload."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import tempfile
from collections.abc import Generator
from pathlib import Path

import gradio as gr
import httpx

# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

DEFAULT_API_BASE = "http://localhost:8000"


def fetch_models(api_base: str) -> list[str]:
    """GET /v1/models and return a list of model IDs."""
    try:
        resp = httpx.get(f"{api_base}/v1/models", timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return [m["id"] for m in data.get("data", [])]
    except Exception as exc:
        gr.Warning(f"Could not fetch models: {exc}")
        return []


def file_to_data_uri(file_path: str) -> str:
    """Read a local file and return a data URI (data:<mime>;base64,...)."""
    p = Path(file_path)
    mime, _ = mimetypes.guess_type(p.name)
    if mime is None:
        mime = "application/octet-stream"
    raw = p.read_bytes()
    b64 = base64.b64encode(raw).decode()
    return f"data:{mime};base64,{b64}"


def stream_chat_completion(api_base: str, payload: dict) -> Generator[dict, None, None]:
    """POST /v1/chat/completions with stream=True.

    Yields dicts: ``{"type": "text", "value": "..."}``
    or ``{"type": "audio", "value": "<base64>"}`` (once, at the end).
    """
    payload["stream"] = True
    with httpx.stream(
        "POST",
        f"{api_base}/v1/chat/completions",
        json=payload,
        timeout=None,
    ) as resp:
        resp.raise_for_status()
        audio_b64 = ""
        for line in resp.iter_lines():
            if not line.startswith("data: "):
                continue
            data_str = line[len("data: ") :]
            if data_str.strip() == "[DONE]":
                break
            chunk = json.loads(data_str)
            delta = chunk["choices"][0].get("delta", {})
            text = delta.get("content", "")
            if text:
                yield {"type": "text", "value": text}
            audio = delta.get("audio")
            if audio and audio.get("data"):
                audio_b64 += audio["data"]
        if audio_b64:
            yield {"type": "audio", "value": audio_b64}


# ---------------------------------------------------------------------------
# Chat handler (Gradio generator)
# ---------------------------------------------------------------------------


def make_chat_handler(api_base: str):
    """Return a chat handler closure bound to *api_base*."""

    def chat_handler(
        message: str,
        chat_display: list[dict],
        api_history: list[dict],
        model: str,
        system_prompt: str,
        temperature: float,
        top_p: float,
        top_k: int,
        output_mode: str,
        media_files: list[str] | None,
    ) -> Generator[tuple[list[dict], list[dict], str | None], None, None]:
        # Classify uploaded files by MIME type and encode as data URIs
        images, audios, videos = [], [], []
        for f in media_files or []:
            mime, _ = mimetypes.guess_type(f)
            mime = mime or ""
            if mime.startswith("image/"):
                images.append(file_to_data_uri(f))
            elif mime.startswith("audio/"):
                audios.append(file_to_data_uri(f))
            elif mime.startswith("video/"):
                videos.append(file_to_data_uri(f))

        # Build the user entry for API history (with media, like web playground)
        user_entry: dict = {"role": "user", "content": message or " "}
        if images:
            user_entry["images"] = images
        if audios:
            user_entry["audios"] = audios
        if videos:
            user_entry["videos"] = videos

        # Build API messages from full history (preserves media from prior turns)
        messages: list[dict] = []
        if system_prompt.strip():
            messages.append({"role": "system", "content": system_prompt.strip()})
        messages.extend(api_history)
        messages.append(user_entry)

        payload: dict = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
        }

        # Output modality — matches web playground payload logic
        if output_mode == "text":
            payload["modalities"] = ["text"]
        elif output_mode == "audio":
            payload["modalities"] = ["audio"]
            payload["audio"] = {"format": "wav"}
        elif output_mode == "both":
            payload["modalities"] = ["text", "audio"]
            payload["audio"] = {"format": "wav"}
        # "auto" — omit modalities, let backend decide

        # Current-turn media as top-level fields (matches web playground)
        if images:
            payload["images"] = images
        if audios:
            payload["audios"] = audios
        if videos:
            payload["videos"] = videos

        # Build display content for chatbot (text + inline media previews)
        user_content: list = []
        if message:
            user_content.append(message)
        for f in media_files or []:
            mime, _ = mimetypes.guess_type(f)
            user_content.append({"path": f, "mime_type": mime or ""})
        if not user_content:
            user_content.append(" ")

        # Update API history with the new user entry
        new_api_history = api_history + [user_entry]

        # Streaming: yield (chat_display, api_history, audio_path) progressively
        assistant_text = ""
        audio_path: str | None = None
        display_out = chat_display + [
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": ""},
        ]
        try:
            for chunk in stream_chat_completion(api_base, payload):
                if chunk["type"] == "text":
                    assistant_text += chunk["value"]
                    display_out[-1] = {
                        "role": "assistant",
                        "content": assistant_text,
                    }
                    yield display_out, new_api_history, audio_path
                elif chunk["type"] == "audio":
                    raw = base64.b64decode(chunk["value"])
                    tmp = tempfile.NamedTemporaryFile(suffix=".wav")
                    tmp.write(raw)
                    tmp.close()
                    audio_path = tmp.name
                    yield display_out, new_api_history, audio_path
        except Exception as exc:
            error_msg = f"Error: {exc}"
            display_out[-1] = {"role": "assistant", "content": error_msg}
            yield display_out, new_api_history, audio_path
            return

        # Append assistant response to API history
        new_api_history = new_api_history + [
            {"role": "assistant", "content": assistant_text}
        ]
        yield display_out, new_api_history, audio_path

    return chat_handler


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------


def create_demo(api_base: str) -> gr.Blocks:
    """Build the Gradio Blocks layout."""
    chat_handler = make_chat_handler(api_base)

    with gr.Blocks(title="SGLang-Omni Playground") as demo:
        gr.Markdown("## SGLang-Omni Playground")

        # Hidden state: full conversation history with media data URIs
        api_history = gr.State([])

        with gr.Row():
            # ---- Left column: settings ----
            with gr.Column(scale=1, min_width=260):
                model_dropdown = gr.Dropdown(
                    label="Model",
                    choices=[],
                    value=None,
                    interactive=True,
                    allow_custom_value=True,
                )
                refresh_btn = gr.Button("Refresh models", size="sm")
                system_prompt = gr.Textbox(
                    label="System Prompt",
                    placeholder="e.g., You are a helpful assistant",
                    lines=2,
                )
                output_mode = gr.Dropdown(
                    label="Output",
                    choices=[
                        ("Text only", "text"),
                        # TODO: enable after talker is wired
                        # ("Audio only", "audio"),
                        # ("Text + Audio", "both"),
                        # ("Auto", "auto"),
                    ],
                    value="text",
                    interactive=False,
                )

                media_input = gr.File(
                    label="Images / Audio / Video",
                    file_count="multiple",
                )

                with gr.Accordion("Generation Parameters", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=2.0,
                        value=0.6,
                        step=0.1,
                    )
                    top_p = gr.Slider(
                        label="Top P",
                        minimum=0.0,
                        maximum=1.0,
                        value=0.95,
                        step=0.05,
                    )
                    top_k = gr.Slider(
                        label="Top K",
                        minimum=1,
                        maximum=100,
                        value=20,
                        step=1,
                    )

            # ---- Right column: chat + audio output ----
            with gr.Column(scale=3, min_width=480):
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=520,
                )
                audio_output = gr.Audio(
                    label="Audio Response",
                    type="filepath",
                    interactive=False,
                )
                with gr.Row():
                    user_input = gr.Textbox(
                        label="Message",
                        placeholder="Type your message...",
                        scale=4,
                        show_label=False,
                    )
                    send_btn = gr.Button("Send", variant="primary", scale=1)
                with gr.Row():
                    stop_btn = gr.Button("Stop", variant="stop")
                    clear_btn = gr.Button("Clear")

        # ---- Event wiring ----
        chat_inputs = [
            user_input,
            chatbot,
            api_history,
            model_dropdown,
            system_prompt,
            temperature,
            top_p,
            top_k,
            output_mode,
            media_input,
        ]
        chat_outputs = [chatbot, api_history, audio_output]

        def refresh_models():
            models = fetch_models(api_base)
            value = models[0] if models else None
            return gr.update(choices=models, value=value)

        def clear_all():
            return [], [], None, None

        refresh_btn.click(fn=refresh_models, outputs=model_dropdown)

        # Submit via button or Enter key — clear text + media after send
        submit_event = send_btn.click(
            fn=chat_handler,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(
            fn=lambda: ("", None),
            outputs=[user_input, media_input],
        )

        enter_event = user_input.submit(
            fn=chat_handler,
            inputs=chat_inputs,
            outputs=chat_outputs,
        ).then(
            fn=lambda: ("", None),
            outputs=[user_input, media_input],
        )

        stop_btn.click(fn=None, cancels=[submit_event, enter_event])
        clear_btn.click(
            fn=clear_all,
            outputs=[chatbot, api_history, audio_output, media_input],
        )

        # Auto-fetch models on load
        demo.load(fn=refresh_models, outputs=model_dropdown)

    return demo


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the Gradio playground for SGLang-Omni."
    )
    parser.add_argument(
        "--api-base",
        type=str,
        default=DEFAULT_API_BASE,
        help=f"Base URL of the sglang-omni server (default: {DEFAULT_API_BASE})",
    )
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--share", action="store_true", help="Create a public Gradio link"
    )
    args = parser.parse_args()

    demo = create_demo(args.api_base)
    demo.queue()
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share,
        theme=gr.themes.Soft(),
    )


if __name__ == "__main__":
    main()
