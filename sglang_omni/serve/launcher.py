# SPDX-License-Identifier: Apache-2.0
"""Launch an OpenAI-compatible server from a PipelineConfig.

Usage (programmatic)::

    from sglang_omni.serve.launcher import launch_server
    launch_server(pipeline_config, host="0.0.0.0", port=8000)

Usage (CLI — with config file)::

    sglang-omni-server --config pipeline.json --port 8000

Usage (CLI — built-in pipeline, no JSON needed)::

    sglang-omni-server \\
        --pipeline qwen3-omni \\
        --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct \\
        --port 8000

Export a config to JSON::

    sglang-omni-server --pipeline qwen3-omni --model-id ... --export-config out.json
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import uvicorn

from sglang_omni.client import Client
from sglang_omni.config import PipelineConfig, PipelineRunner, compile_pipeline
from sglang_omni.serve.openai_api import create_app

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Built-in pipeline registry
# ---------------------------------------------------------------------------

_BUILTIN_PIPELINES: dict[str, str] = {
    "qwen3-omni": "sglang_omni.models.qwen3_omni.create_text_first_pipeline_config",
}


def _resolve_builtin(
    pipeline_name: str,
    *,
    model_id: str,
    extra_kwargs: dict[str, Any] | None = None,
) -> PipelineConfig:
    """Instantiate a built-in pipeline config by name."""
    if pipeline_name not in _BUILTIN_PIPELINES:
        available = ", ".join(sorted(_BUILTIN_PIPELINES))
        raise ValueError(
            f"Unknown built-in pipeline: {pipeline_name!r}. " f"Available: {available}"
        )

    from sglang_omni.config.imports import import_string

    factory_path = _BUILTIN_PIPELINES[pipeline_name]
    factory = import_string(factory_path)
    kwargs: dict[str, Any] = {"model_id": model_id}
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    config = factory(**kwargs)
    if not isinstance(config, PipelineConfig):
        raise TypeError(
            f"Pipeline factory {factory_path} returned {type(config)}, "
            "expected PipelineConfig"
        )
    return config


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------


async def _run_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
    client_kwargs: dict[str, Any] | None = None,
) -> None:
    """Compile the pipeline, start stages, and run the OpenAI server.

    This is the async entry point.  For a blocking call use :func:`launch_server`.
    """
    # 1. Compile pipeline config -> Coordinator + Stages
    coordinator, stages = compile_pipeline(pipeline_config)
    runner = PipelineRunner(coordinator, stages)

    # 2. Start the pipeline (coordinator + all stages as async tasks)
    await runner.start()
    logger.info(
        "Pipeline '%s' started (%d stages)",
        pipeline_config.name,
        len(stages),
    )

    try:
        # 3. Build Client -> FastAPI app
        cl_kwargs = client_kwargs or {}
        client = Client(coordinator, **cl_kwargs)
        app = create_app(client, model_name=model_name or pipeline_config.name)

        # 4. Run uvicorn
        config = uvicorn.Config(app, host=host, port=port, log_level=log_level)
        server = uvicorn.Server(config)

        await server.serve()
    finally:
        logger.info("Shutting down pipeline …")
        await runner.stop()
        logger.info("Pipeline stopped.")


def launch_server(
    pipeline_config: PipelineConfig,
    *,
    host: str = "0.0.0.0",
    port: int = 8000,
    model_name: str | None = None,
    log_level: str = "info",
    client_kwargs: dict[str, Any] | None = None,
) -> None:
    """Blocking helper: compile pipeline and start the OpenAI-compatible server.

    Args:
        pipeline_config: Declarative pipeline configuration.
        host: Bind address for the HTTP server.
        port: Bind port for the HTTP server.
        model_name: Model name reported in ``/v1/models`` responses.
            Defaults to the pipeline name.
        log_level: Uvicorn log level.
        client_kwargs: Extra keyword arguments forwarded to
            :class:`~sglang_omni.client.Client`.
    """
    asyncio.run(
        _run_server(
            pipeline_config,
            host=host,
            port=port,
            model_name=model_name,
            log_level=log_level,
            client_kwargs=client_kwargs,
        )
    )


def load_pipeline_config(path: str | Path) -> PipelineConfig:
    """Load a PipelineConfig from a JSON file.

    Args:
        path: Path to a JSON file containing a valid PipelineConfig.

    Returns:
        Parsed PipelineConfig instance.
    """
    text = Path(path).read_text(encoding="utf-8")
    data = json.loads(text)
    return PipelineConfig(**data)


def export_pipeline_config(config: PipelineConfig, path: str | Path) -> None:
    """Export a PipelineConfig to a JSON file.

    Args:
        config: Pipeline configuration to export.
        path: Output JSON file path.
    """
    data = config.model_dump(mode="json")
    Path(path).write_text(
        json.dumps(data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    logger.info("Pipeline config exported to %s", path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI: ``sglang-omni-server`` or ``python -m sglang_omni.serve.launcher``."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Launch an OpenAI-compatible server for sglang-omni.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples:
  # From a JSON config file
  sglang-omni-server --config pipeline.json --port 8000

  # Built-in pipeline (no JSON needed)
  sglang-omni-server --pipeline qwen3-omni --model-id Qwen/Qwen3-Omni-30B-A3B-Instruct

  # Export config to JSON for inspection / editing
  sglang-omni-server --pipeline qwen3-omni --model-id ... --export-config pipeline.json
""",
    )

    # --- Source: pick one ---
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument(
        "--config",
        type=str,
        help="Path to a pipeline config JSON file.",
    )
    source.add_argument(
        "--pipeline",
        type=str,
        choices=sorted(_BUILTIN_PIPELINES),
        help="Name of a built-in pipeline.",
    )

    # --- Pipeline factory args (used with --pipeline) ---
    parser.add_argument(
        "--model-id",
        type=str,
        default=None,
        help="Hugging Face model id (required with --pipeline).",
    )
    parser.add_argument("--dtype", type=str, default=None, help="Model dtype.")
    parser.add_argument("--preprocessing-device", type=str, default=None)
    parser.add_argument("--image-device", type=str, default=None)
    parser.add_argument("--audio-device", type=str, default=None)
    parser.add_argument("--thinker-device", type=str, default=None)
    parser.add_argument("--thinker-max-seq-len", type=int, default=None)
    parser.add_argument(
        "--relay-type",
        type=str,
        default=None,
        choices=["shm", "nccl", "nixl"],
    )

    # --- Export instead of serve ---
    parser.add_argument(
        "--export-config",
        type=str,
        default=None,
        metavar="PATH",
        help="Export the resolved pipeline config to a JSON file and exit.",
    )

    # --- Server options ---
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server bind address (default: 0.0.0.0).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server bind port (default: 8000).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        help="Model name for /v1/models (default: pipeline name).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="info",
        choices=["debug", "info", "warning", "error", "critical"],
        help="Log level (default: info).",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Resolve config ---
    if args.config:
        config = load_pipeline_config(args.config)
    else:
        # --pipeline mode
        if not args.model_id:
            parser.error("--model-id is required when using --pipeline")

        # Collect non-None kwargs to forward to the factory
        factory_kwargs: dict[str, Any] = {}
        for key in (
            "dtype",
            "preprocessing_device",
            "image_device",
            "audio_device",
            "thinker_device",
            "thinker_max_seq_len",
            "relay_type",
        ):
            val = getattr(args, key, None)
            if val is not None:
                factory_kwargs[key] = val

        config = _resolve_builtin(
            args.pipeline,
            model_id=args.model_id,
            extra_kwargs=factory_kwargs,
        )

    # --- Export or serve ---
    if args.export_config:
        export_pipeline_config(config, args.export_config)
        return

    launch_server(
        config,
        host=args.host,
        port=args.port,
        model_name=args.model_name,
        log_level=args.log_level,
    )


if __name__ == "__main__":
    main()
