from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.serve.launcher import launch_server

logger = logging.getLogger(__name__)


def serve(
    ctx: typer.Context,
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
    config: Annotated[
        str, typer.Option(help="Path to a pipeline config JSON file.")
    ] = None,
    host: Annotated[
        str, typer.Option(help="Server bind address (default: 0.0.0.0).")
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Server bind port (default: 8000).")] = 8000,
    model_name: Annotated[
        str, typer.Option(help="Model name for /v1/models (default: pipeline name).")
    ] = None,
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
) -> None:
    """Serve the pipeline."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # --- Resolve config ---
    if config:
        config_manager = ConfigManager.from_file(config)
    else:
        config_manager = ConfigManager.from_model_path(model_path)

    # we use ctx to capture the arguments that are used to modify the configuration on the fly
    # we do expect the extra arguments to be pairs of names and values
    extra_args = config_manager.parse_extra_args(ctx.args)
    merged_config = config_manager.merge_config(extra_args)
    merged_config = merged_config.model_copy(update={"model_path": model_path})

    # print merged configuration
    print("=" * 20, "Merged Configuration", "=" * 20)
    print(
        yaml.dump(
            merged_config.model_dump(mode="json"),
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )
    )
    print("=" * 50)

    launch_server(
        merged_config,
        host=host,
        port=port,
        model_name=model_name,
        log_level=log_level,
    )
