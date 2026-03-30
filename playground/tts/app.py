# SPDX-License-Identifier: Apache-2.0
"""Entrypoint for the S2-Pro TTS Gradio playground."""

from __future__ import annotations

import argparse

from playground.tts.ui import create_demo

DEFAULT_API_BASE = "http://localhost:8000"


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
