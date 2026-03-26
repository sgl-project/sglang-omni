import argparse
import asyncio
import logging

from benchmark_kit.args import add_benchmark_args
from benchmark_kit.benchmarker.omni_benchmarker import OmniBenchmarker
from benchmark_kit.utils import kill_server, launch_server

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark Omni")
    parser = add_benchmark_args(parser)
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"

    server_proc = None
    if args.launch_server:
        server_proc = launch_server(args)
    try:
        benchmarker = OmniBenchmarker(
            dataset_path=args.dataset,
            max_samples=args.max_samples,
            max_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            system_prompt=args.system_prompt,
            base_url=base_url,
            stream=args.stream,
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
            output_dir=args.output_dir,
            save_audio=args.save_audio,
        )

        asyncio.run(benchmarker.run())
    finally:
        if server_proc is not None:
            kill_server(server_proc)


if __name__ == "__main__":
    main()
