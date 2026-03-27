from __future__ import annotations

import argparse
import asyncio
import logging

from benchmarks.core.runner import run_from_config
from benchmarks.core.types import BenchmarkRunConfig

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Model-agnostic performance benchmark runner for sglang-omni."
    )
    parser.add_argument("--base-url", type=str, required=True)
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name for the API request.",
    )
    parser.add_argument(
        "--model-profile",
        type=str,
        required=True,
        help="Registered benchmark profile name or alias.",
    )
    parser.add_argument("--case", type=str, default=None)
    parser.add_argument("--dataset", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="results/benchmark")
    parser.add_argument("--stream", action="store_true")
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-output-tokens", type=int, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--repetition-penalty", type=float, default=None)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument(
        "--request-rate",
        type=float,
        default=float("inf"),
        help="Requests per second (inf = fire all immediately).",
    )
    parser.add_argument(
        "--save-audio", action="store_true", help="Save generated WAV files."
    )
    parser.add_argument("--disable-tqdm", action="store_true")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_config = BenchmarkRunConfig(
        base_url=args.base_url,
        model=args.model,
        model_profile=args.model_profile,
        case_id=args.case,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        stream=args.stream,
        max_samples=args.max_samples,
        max_output_tokens=args.max_output_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        warmup=args.warmup,
        max_concurrency=args.max_concurrency,
        request_rate=args.request_rate,
        save_audio=args.save_audio,
        disable_tqdm=args.disable_tqdm,
    )
    asyncio.run(run_from_config(run_config))


if __name__ == "__main__":
    main()
