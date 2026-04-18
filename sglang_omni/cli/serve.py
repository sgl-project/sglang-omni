from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.serve.launcher import launch_server

logger = logging.getLogger(__name__)

THINKER_ROLE = "thinker"
TALKER_ROLE = "talker"


def _get_stage_indices(config: PipelineConfig) -> dict[str, int]:
    return {stage.name: idx for idx, stage in enumerate(config.stages)}


def _set_stage_mem_fraction_override(
    extra_args: dict[str, str], stage_index: int, value: float
) -> None:
    extra_args[
        f"stages.{stage_index}.executor.args.server_args_overrides.mem_fraction_static"
    ] = str(value)


def _apply_mem_fraction_overrides(
    extra_args: dict[str, str],
    config: PipelineConfig,
    *,
    mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
) -> dict[str, str]:
    stage_indices = _get_stage_indices(config)
    override_stages = config.mem_fraction_override_stages

    if mem_fraction_static is not None:
        applied_stage_count = 0
        for stage_name in override_stages.values():
            stage_index = stage_indices[stage_name]
            _set_stage_mem_fraction_override(
                extra_args, stage_index, mem_fraction_static
            )
            applied_stage_count += 1
        if applied_stage_count == 0:
            raise ValueError(
                "--mem-fraction-static requires a pipeline with a supported SGLang AR "
                "stage configured in mem_fraction_override_stages"
            )
    if thinker_mem_fraction_static is not None:
        stage_name = override_stages.get(THINKER_ROLE)
        if stage_name is None:
            raise ValueError(
                "--thinker-mem-fraction-static requires a pipeline with a "
                f"{THINKER_ROLE!r} mem-fraction override target"
            )
        stage_index = stage_indices[stage_name]
        _set_stage_mem_fraction_override(
            extra_args, stage_index, thinker_mem_fraction_static
        )
    if talker_mem_fraction_static is not None:
        stage_name = override_stages.get(TALKER_ROLE)
        if stage_name is None:
            raise ValueError(
                "--talker-mem-fraction-static requires a pipeline with a "
                f"{TALKER_ROLE!r} mem-fraction override target"
            )
        stage_index = stage_indices[stage_name]
        _set_stage_mem_fraction_override(
            extra_args, stage_index, talker_mem_fraction_static
        )
    return extra_args


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
                "Override mem_fraction_static for supported SGLang AR stages in the "
                "selected pipeline. If omitted, SGLang hardware-aware auto-sizing is "
                "used."
            )
        ),
    ] = None,
    thinker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            help=(
                "Override mem_fraction_static only for the thinker stage. "
                "Takes precedence over --mem-fraction-static for thinker."
            )
        ),
    ] = None,
    talker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            help=(
                "Override mem_fraction_static only for the talker AR stage. "
                "Takes precedence over --mem-fraction-static for talker."
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
    extra_args = _apply_mem_fraction_overrides(
        extra_args,
        config_manager.config,
        mem_fraction_static=mem_fraction_static,
        thinker_mem_fraction_static=thinker_mem_fraction_static,
        talker_mem_fraction_static=talker_mem_fraction_static,
    )
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
