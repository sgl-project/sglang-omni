from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.serve.launcher import launch_server


def _validate_mem_fraction_static_support(
    *,
    mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
    merged_config,
) -> None:
    override_stages = merged_config.mem_fraction_override_stages
    pipeline_name = type(merged_config).__name__

    if (
        mem_fraction_static is not None
        and override_stages.thinker is None
        and override_stages.talker is None
    ):
        raise typer.BadParameter(
            "--mem-fraction-static is not supported by pipeline "
            f"{pipeline_name} (no SGLang AR stage)."
        )
    if thinker_mem_fraction_static is not None and override_stages.thinker is None:
        raise typer.BadParameter(
            "--thinker-mem-fraction-static is not supported by pipeline "
            f"{pipeline_name} (no thinker AR stage)."
        )
    if talker_mem_fraction_static is not None and override_stages.talker is None:
        raise typer.BadParameter(
            "--talker-mem-fraction-static is not supported by pipeline "
            f"{pipeline_name} (no talker AR stage)."
        )


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
    text_only: Annotated[
        bool,
        typer.Option(
            "--text-only",
            help="Use thinker-only pipeline (1 GPU, no talker/speech output).",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option(help="Server bind address (default: 0.0.0.0).")
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Server bind port (default: 8000).")] = 8000,
    model_name: Annotated[
        str, typer.Option(help="Model name for /v1/models (default: pipeline name).")
    ] = None,
    mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            help=(
                "Set SGLang mem_fraction_static for all stages in the selected "
                "pipeline that support this override. This controls SGLang's "
                "weights + KV cache memory budget. If omitted, SGLang chooses "
                "the value automatically."
            )
        ),
    ] = None,
    thinker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            help=(
                "Set SGLang mem_fraction_static only for the pipeline's thinker "
                "stage. Overrides --mem-fraction-static for thinker. Some "
                "pipelines do not expose a thinker AR stage."
            )
        ),
    ] = None,
    talker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            help=(
                "Set SGLang mem_fraction_static only for the pipeline's talker "
                "stage. Overrides --mem-fraction-static for talker. Some "
                "pipelines do not expose a talker AR stage."
            )
        ),
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
    elif text_only:
        config_manager = ConfigManager.from_model_path(model_path, variant="text")
    else:
        config_manager = ConfigManager.from_model_path(model_path)

    # we use ctx to capture the arguments that are used to modify the configuration on the fly
    # we do expect the extra arguments to be pairs of names and values
    extra_args = config_manager.parse_extra_args(ctx.args)
    merged_config = config_manager.merge_config(extra_args)
    merged_config = merged_config.model_copy(
        update={"model_path": model_path},
        deep=True,
    )
    _validate_mem_fraction_static_support(
        mem_fraction_static=mem_fraction_static,
        thinker_mem_fraction_static=thinker_mem_fraction_static,
        talker_mem_fraction_static=talker_mem_fraction_static,
        merged_config=merged_config,
    )
    try:
        merged_config.apply_mem_fraction_static_overrides(
            mem_fraction_static=mem_fraction_static,
            thinker_mem_fraction_static=thinker_mem_fraction_static,
            talker_mem_fraction_static=talker_mem_fraction_static,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

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
