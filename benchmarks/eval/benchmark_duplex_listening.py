# SPDX-License-Identifier: Apache-2.0
"""Listening quality of the replies behind Full-Duplex-Bench reference exports."""

from __future__ import annotations

import argparse
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="phase", required=True)
    transcribe = commands.add_parser(
        "transcribe",
        help="Cut each reply into a wav and transcribe it with Parakeet (scoring venv)",
    )
    transcribe.add_argument(
        "--manifest",
        type=Path,
        action="append",
        required=True,
        help="A v1.5 reference-manifest.json or v1.0 manifest.json; repeatable",
    )
    transcribe.add_argument("--out", type=Path, required=True)
    transcribe.add_argument("--nemo", type=Path, required=True)
    transcribe.add_argument("--nemo-sha256", required=True)
    transcribe.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    score = commands.add_parser(
        "score",
        help="UTMOS, voice consistency and omitted words per reply (serving venv)",
    )
    score.add_argument("--out", type=Path, required=True)
    score.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    # note (luojiaxuan): the phases run in different venvs; NeMo is only in the
    # scoring venv and the UTMOS and WavLM models only in the serving one, so each
    # imports its own module.
    if args.phase == "transcribe":
        from benchmarks.duplex.listening_replies import transcribe_replies

        transcribe_replies(
            args.manifest, args.out, args.nemo, args.nemo_sha256, args.device
        )
    elif args.phase == "score":
        from benchmarks.duplex.listening_scores import run_score

        run_score(args.out, args.device)
    else:
        raise AssertionError(f"unhandled phase {args.phase}")


if __name__ == "__main__":
    main()
else:
    pass
