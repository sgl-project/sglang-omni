# SPDX-License-Identifier: Apache-2.0
"""Gradio UI for MLX Qwen3-TTS voice cloning."""

from __future__ import annotations

import gradio as gr

from playground.qwen3_tts_mlx.generator import (
    FRAME_RATE_HZ,
    SAMPLE_RATE,
    LazyGenerator,
    ModelUnsupportedError,
)

_REFERENCE_NOTE = """
**The transcript must cover the whole reference clip.** A partial or mismatched
transcript is a conditioning error, not a tolerance — the official
implementation degrades the same way. Trim the audio and the transcript
together.
"""


def _format_summary(result) -> str:
    cached = (
        "reused cached reference" if result.reference_cached else "encoded reference"
    )
    return "\n".join(
        [
            f"audio          {result.audio_seconds:.2f}s "
            f"({result.frames} frames @ {FRAME_RATE_HZ:g} Hz)",
            f"prompt         {result.prompt_tokens} positions, "
            f"{result.reference_frames} reference frames ({cached})",
            f"prefill        {result.prefill_seconds:.2f}s",
            f"talker         {result.decode_seconds:.2f}s "
            f"({result.decode_seconds / max(result.frames, 1) * 1000:.0f} ms/frame)",
            f"vocoder        {result.vocoder_seconds:.2f}s",
            f"total          {result.total_seconds:.2f}s "
            f"(RTF {result.realtime_factor:.2f})",
        ]
    )


def make_handler(generator: LazyGenerator):
    """Build the Gradio click handler for one synthesis."""

    def synthesize(
        text: str,
        ref_audio: str | None,
        ref_text: str,
        language: str,
        max_seconds: float,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
        seed: int,
        progress=gr.Progress(),
    ):
        if not (text or "").strip():
            raise gr.Error("Enter some text to synthesise.")
        if not ref_audio:
            raise gr.Error("Upload or record reference audio to clone.")
        if not (ref_text or "").strip():
            raise gr.Error(
                "Enter the transcript of the reference audio — cloning is "
                "conditioned on it."
            )

        max_frames = max(1, int(max_seconds * FRAME_RATE_HZ))

        progress(0.0, desc="Loading model" if not generator.loaded else "Preparing")

        def on_frame(count: int) -> None:
            # The talker stops at EOS, so this is an upper bound, not a total.
            progress(
                min(count / max_frames, 1.0),
                desc=f"Generating {count / FRAME_RATE_HZ:.1f}s of audio",
            )

        try:
            result = generator.clone(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
                max_frames=max_frames,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                seed=None if int(seed) < 0 else int(seed),
                on_frame=on_frame,
            )
        except ModelUnsupportedError as exc:
            raise gr.Error(str(exc)) from exc

        if result.frames == 0:
            raise gr.Error(
                "The talker emitted end-of-speech immediately and produced no "
                "audio. Check that the reference transcript matches the "
                "reference audio."
            )

        progress(1.0, desc="Vocoding complete")
        return (SAMPLE_RATE, result.audio), _format_summary(result)

    return synthesize


def create_demo(model_path: str) -> gr.Blocks:
    """Build the playground layout for one checkpoint."""
    generator = LazyGenerator(model_path)
    handler = make_handler(generator)

    with gr.Blocks(title="Qwen3-TTS MLX voice cloning") as demo:
        gr.Markdown(
            "# Qwen3-TTS voice cloning (MLX)\n"
            f"Running `{model_path}` in this process on MLX — no server needed. "
            "Requires a **Base** checkpoint; CustomVoice and VoiceDesign ship no "
            "speech-tokenizer encoder and cannot encode reference audio."
        )

        with gr.Row():
            with gr.Column(scale=3):
                text_input = gr.Textbox(
                    label="Text to speak",
                    placeholder="What the cloned voice should say.",
                    lines=4,
                )
                ref_audio = gr.Audio(
                    label="Reference audio (any sample rate; resampled to 24 kHz)",
                    type="filepath",
                    sources=["upload", "microphone"],
                )
                ref_text = gr.Textbox(
                    label="Reference transcript",
                    placeholder="Exactly what is said in the reference audio.",
                    lines=3,
                )
                gr.Markdown(_REFERENCE_NOTE)

            with gr.Column(scale=2):
                language = gr.Dropdown(
                    label="Language",
                    choices=[
                        "auto",
                        "en",
                        "zh",
                        "ja",
                        "ko",
                        "de",
                        "fr",
                        "es",
                        "it",
                        "pt",
                        "ru",
                    ],
                    value="auto",
                    info="'auto' skips the language control token.",
                )
                max_seconds = gr.Slider(
                    label="Max output seconds",
                    minimum=1,
                    maximum=120,
                    step=1,
                    value=20,
                    info="Upper bound; generation stops early at end-of-speech.",
                )
                with gr.Accordion("Sampling", open=False):
                    temperature = gr.Slider(
                        label="Temperature",
                        minimum=0.0,
                        maximum=1.5,
                        step=0.05,
                        value=0.9,
                        info="0 is greedy.",
                    )
                    top_k = gr.Slider(
                        label="Top-k", minimum=0, maximum=200, step=1, value=50
                    )
                    top_p = gr.Slider(
                        label="Top-p", minimum=0.05, maximum=1.0, step=0.05, value=1.0
                    )
                    repetition_penalty = gr.Slider(
                        label="Repetition penalty",
                        minimum=1.0,
                        maximum=1.5,
                        step=0.01,
                        value=1.05,
                        info="Applied to the talker's recent codec tokens only.",
                    )
                    seed = gr.Number(
                        label="Seed",
                        value=-1,
                        precision=0,
                        info="-1 for a fresh random draw each run.",
                    )
                synthesize_btn = gr.Button("Clone voice", variant="primary")

        audio_output = gr.Audio(label="Generated speech", type="numpy")
        status = gr.Code(label="Run detail", interactive=False)

        synthesize_btn.click(
            fn=handler,
            inputs=[
                text_input,
                ref_audio,
                ref_text,
                language,
                max_seconds,
                temperature,
                top_k,
                top_p,
                repetition_penalty,
                seed,
            ],
            outputs=[audio_output, status],
            api_name="synthesize",
        )

    return demo
