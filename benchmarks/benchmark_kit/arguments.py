import argparse


def add_benchmark_args(parser: argparse.ArgumentParser):
    # add group - Launching Server
    server_group = parser.add_argument_group("Launching Server")
    server_group.add_argument(
        "--launch-server", action="store_true", help="Launch server as a subprocess"
    )
    server_group.add_argument(
        "--model-path", type=str, required=True, help="Path to the model"
    )
    server_group.add_argument(
        "--host", type=str, default="localhost", help="Host to launch the server"
    )
    server_group.add_argument(
        "--port", type=int, default=8000, help="Port to launch the server"
    )
    server_group.add_argument(
        "--server-timeout", type=int, default=120, help="Timeout to launch the server"
    )
    server_group.add_argument(
        "--relay-backend", type=str, default=None, help="Relay backend to use"
    )
    server_group.add_argument(
        "--config", type=str, default=None, help="The path to the pipeline config file"
    )

    # add group - Independent Server
    independent_server_group = parser.add_argument_group("Independent Server")
    independent_server_group.add_argument(
        "--base-url", type=str, default=None, help="Base URL of the server"
    )

    # add group - Model
    model_group = parser.add_argument_group("Model")
    model_group.add_argument("--stream", action="store_true", help="Use streaming mode")

    # add group - Benchmarking
    benchmark_group = parser.add_argument_group("Benchmark")
    benchmark_group.add_argument(
        "--dataset", type=str, required=True, help="Path to the dataset"
    )
    benchmark_group.add_argument(
        "--system-prompt", type=str, default=None, help="System prompt to prepend"
    )

    # add group - Requests
    requests_group = parser.add_argument_group("Requests")
    requests_group.add_argument(
        "--max-concurrency",
        type=int,
        default=1,
        help="Maximum number of concurrent requests",
    )
    requests_group.add_argument(
        "--request-rate", type=float, default=float("inf"), help="Request rate"
    )
    requests_group.add_argument("--warmup", type=int, default=1, help="Warmup")

    # add group - TTS
    tts_group = parser.add_argument_group("TTS")
    tts_group.add_argument(
        "--no-ref-audio", action="store_true", help="No reference audio"
    )

    # add group - Sampling
    sampling_group = parser.add_argument_group("Sampling")
    sampling_group.add_argument(
        "--max-samples", type=int, default=None, help="Maximum number of samples"
    )
    sampling_group.add_argument(
        "--max-new-tokens", type=int, default=1024, help="Maximum number of new tokens"
    )
    sampling_group.add_argument(
        "--temperature", type=float, default=0.0, help="Temperature"
    )
    sampling_group.add_argument("--top-p", type=float, default=None, help="Top-p")
    sampling_group.add_argument("--top-k", type=int, default=None, help="Top-k")
    sampling_group.add_argument(
        "--repetition-penalty", type=float, default=None, help="Repetition penalty"
    )

    # add group - Saving Results
    saving_group = parser.add_argument_group("Saving Results")
    saving_group.add_argument(
        "--output-dir", type=str, default="results", help="Path to the output directory"
    )
    return parser
