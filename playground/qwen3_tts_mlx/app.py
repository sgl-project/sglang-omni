# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for the MLX Qwen3-TTS voice-cloning Gradio playground."""

from __future__ import annotations

import argparse

from playground.qwen3_tts_mlx.ui import create_demo

DEFAULT_MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="MLX Qwen3-TTS voice-cloning playground (Apple Silicon)"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL,
        help="Local path to a Qwen3-TTS *-Base checkpoint",
    )
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()

    demo = create_demo(args.model_path)
    demo.queue()
    demo.launch(server_name=args.host, server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
