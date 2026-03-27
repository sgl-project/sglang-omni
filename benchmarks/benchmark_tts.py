import argparse
import asyncio
import logging

from benchmark_kit.arguments import add_benchmark_args
from benchmark_kit.benchmarker import TTSBenchmarker
from benchmark_kit.utils import kill_server, launch_server

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="TTS Benchmark")
    add_benchmark_args(parser)
    args = parser.parse_args()

    base_url = args.base_url or f"http://{args.host}:{args.port}"

    server_proc = None
    if args.launch_server:
        logger.info("Launching server...")
        server_proc = launch_server(args)

    try:
        benchmarker = TTSBenchmarker(
            dataset=args.dataset,
            no_ref_audio=args.no_ref_audio,
            max_samples=args.max_samples,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            repetition_penalty=args.repetition_penalty,
            base_url=base_url,
            stream=args.stream,
            max_concurrency=args.max_concurrency,
            request_rate=args.request_rate,
            warmup=args.warmup,
        )

        asyncio.run(benchmarker.run(output_dir=args.output_dir))
    finally:
        if server_proc is not None:
            kill_server(server_proc)


if __name__ == "__main__":
    main()
