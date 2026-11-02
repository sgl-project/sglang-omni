# SPDX-License-Identifier: Apache-2.0
"""Measure whether cascade replies fit their request in length and read well aloud.

replies replays the conversations through a chat server under named system prompts, controls checks a judge setting against hand-written replies with known verdicts, judge scores the replies, live talks to a running cascade, and summary writes the markdown report.
"""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import get_args

from pydantic import JsonValue, TypeAdapter

from benchmarks.cascade_replies.judge import judge_controls, judge_replies
from benchmarks.cascade_replies.live import run_live
from benchmarks.cascade_replies.replies import PromptName, run_replies
from benchmarks.cascade_replies.report import write_summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    replies = commands.add_parser(
        "replies", help="Replay the conversations through one chat server"
    )
    replies.add_argument(
        "--profile",
        type=Path,
        required=True,
        help="launch.yaml whose llm node and dialog prompt are replayed",
    )
    replies.add_argument(
        "--llm", required=True, help="Label of this chat server, such as qwen3.8-27b"
    )
    replies.add_argument(
        "--out", type=Path, required=True, help="Replies as JSON lines; appended to"
    )
    replies.add_argument(
        "--base-url", help="Chat server URL; the profile's llm base_url by default"
    )
    replies.add_argument(
        "--served-model",
        help="Model name the server answers to; the profile's by default",
    )
    replies.add_argument(
        "--prompts", nargs="+", default=["profile"], choices=get_args(PromptName)
    )
    replies.add_argument(
        "--extra-body",
        type=TypeAdapter(dict[str, JsonValue]).validate_json,
        default="{}",
        help='JSON merged into the profile\'s extra_body, such as {"min_tokens": 1}',
    )
    replies.add_argument("--samples", type=int, default=3)
    replies.add_argument("--concurrency", type=int, default=16)
    judge = commands.add_parser("judge", help="Judge the rows of replies")
    judge.add_argument(
        "--replies",
        type=Path,
        nargs="+",
        required=True,
        help="Output files of replies, one per chat server",
    )
    controls = commands.add_parser(
        "controls", help="Judge judge-controls.jsonl against its expected verdicts"
    )
    for command in (judge, controls):
        command.add_argument(
            "--out",
            type=Path,
            required=True,
            help="Judgments as JSON lines; appended to",
        )
        command.add_argument(
            "--base-url",
            required=True,
            help="Judge server URL, such as http://127.0.0.1:30000/v1",
        )
        command.add_argument("--served-model", default="Qwen/Qwen3.8-27B")
        command.add_argument(
            "--enable-thinking", action=argparse.BooleanOptionalAction, default=False
        )
        command.add_argument("--max-tokens", type=int, default=6144)
        command.add_argument("--concurrency", type=int, default=32)
        command.add_argument("--timeout-s", type=float, default=600)
    live = commands.add_parser(
        "live", help="Talk to a running cascade with spoken turns"
    )
    live.add_argument("--url", default="ws://127.0.0.1:8765/v1/realtime")
    live.add_argument(
        "--audio-dir",
        type=Path,
        required=True,
        help="Spoken user turns named <conversation>_<turn>.wav, 16 kHz mono PCM16",
    )
    live.add_argument("--output", type=Path, required=True)
    live.add_argument("--pause-s", type=float, default=1.5)
    live.add_argument(
        "--server-vad-silence-ms",
        type=int,
        help="End turns on this much silence instead of the profile's turn detection",
    )
    summary = commands.add_parser("summary", help="Write the markdown report")
    summary.add_argument("--replies", type=Path, nargs="+", required=True)
    summary.add_argument("--judged", type=Path, required=True)
    summary.add_argument("--out", type=Path, required=True)
    summary.add_argument(
        "--baseline",
        choices=get_args(PromptName),
        help="Prompt the other prompts are compared against",
    )
    summary.add_argument(
        "--live",
        type=Path,
        nargs="*",
        default=[],
        help="Output folders of live, each named by its folder",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "replies":
        asyncio.run(
            run_replies(
                profile_path=args.profile,
                llm_label=args.llm,
                output_path=args.out,
                base_url=args.base_url,
                served_model=args.served_model,
                prompt_names=args.prompts,
                extra_body=args.extra_body,
                samples=args.samples,
                concurrency=args.concurrency,
            )
        )
    elif args.command == "judge":
        asyncio.run(
            judge_replies(
                reply_paths=args.replies,
                output_path=args.out,
                base_url=args.base_url,
                served_model=args.served_model,
                enable_thinking=args.enable_thinking,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                timeout_s=args.timeout_s,
            )
        )
    elif args.command == "controls":
        asyncio.run(
            judge_controls(
                output_path=args.out,
                base_url=args.base_url,
                served_model=args.served_model,
                enable_thinking=args.enable_thinking,
                max_tokens=args.max_tokens,
                concurrency=args.concurrency,
                timeout_s=args.timeout_s,
            )
        )
    elif args.command == "live":
        asyncio.run(
            run_live(
                url=args.url,
                audio_dir=args.audio_dir,
                output_dir=args.output,
                pause_s=args.pause_s,
                server_vad_silence_ms=args.server_vad_silence_ms,
            )
        )
    elif args.command == "summary":
        write_summary(
            reply_paths=args.replies,
            judged_path=args.judged,
            report_path=args.out,
            baseline=args.baseline,
            live_dirs=args.live,
        )
    else:
        raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    main()
else:
    pass
