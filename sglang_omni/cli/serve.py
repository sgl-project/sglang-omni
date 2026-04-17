from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.serve.launcher import launch_server

logger = logging.getLogger(__name__)


_STAGE_TOGGLE_MODE = Literal["default", "on", "off"]


def _normalize_stage_toggle_mode(flag_name: str, value: str) -> _STAGE_TOGGLE_MODE:
    normalized = value.strip().lower()
    if normalized not in {"default", "on", "off"}:
        raise typer.BadParameter(f"{flag_name} must be one of: default, on, off")
    return normalized  # type: ignore[return-value]


def _apply_stage_server_args_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    updates: dict[str, object],
    reason: str,
) -> None:
    matching_stages = [
        stage for stage in pipeline_config.stages if stage.name == stage_name
    ]
    if not matching_stages:
        raise typer.BadParameter(
            f"Stage {stage_name!r} not found in pipeline; cannot set {reason}"
        )

    for stage in matching_stages:
        factory_args = dict(stage.factory_args or {})
        overrides = dict(factory_args.get("server_args_overrides") or {})
        overrides.update(updates)
        factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args


def _apply_stage_cuda_graph_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    mode: _STAGE_TOGGLE_MODE,
) -> None:
    if mode == "default":
        return

    _apply_stage_server_args_override(
        pipeline_config,
        stage_name=stage_name,
        updates={"disable_cuda_graph": mode != "on"},
        reason=f"CUDA graph mode to {mode!r}",
    )


def _apply_stage_torch_compile_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    mode: _STAGE_TOGGLE_MODE,
    max_bs: int | None,
) -> None:
    if mode == "default" and max_bs is None:
        return

    updates: dict[str, object] = {}
    if mode != "default":
        updates["enable_torch_compile"] = mode == "on"
    if max_bs is not None:
        if int(max_bs) < 1:
            raise typer.BadParameter("torch compile max batch size must be >= 1")
        updates["torch_compile_max_bs"] = int(max_bs)

    _apply_stage_server_args_override(
        pipeline_config,
        stage_name=stage_name,
        updates=updates,
        reason=(f"torch compile settings (mode={mode!r}, max_bs={max_bs})"),
    )


def apply_cuda_graph_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    thinker_cuda_graph: str,
    talker_cuda_graph: str,
) -> PipelineConfig:
    thinker_mode = _normalize_stage_toggle_mode(
        "thinker_cuda_graph", thinker_cuda_graph
    )
    talker_mode = _normalize_stage_toggle_mode("talker_cuda_graph", talker_cuda_graph)
    _apply_stage_cuda_graph_override(
        pipeline_config,
        stage_name="thinker",
        mode=thinker_mode,
    )
    _apply_stage_cuda_graph_override(
        pipeline_config,
        stage_name="talker_ar",
        mode=talker_mode,
    )
    return pipeline_config


def apply_torch_compile_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    thinker_torch_compile: str,
    talker_torch_compile: str,
    thinker_torch_compile_max_bs: int | None,
    talker_torch_compile_max_bs: int | None,
) -> PipelineConfig:
    thinker_mode = _normalize_stage_toggle_mode(
        "thinker_torch_compile", thinker_torch_compile
    )
    talker_mode = _normalize_stage_toggle_mode(
        "talker_torch_compile", talker_torch_compile
    )
    _apply_stage_torch_compile_override(
        pipeline_config,
        stage_name="thinker",
        mode=thinker_mode,
        max_bs=thinker_torch_compile_max_bs,
    )
    _apply_stage_torch_compile_override(
        pipeline_config,
        stage_name="talker_ar",
        mode=talker_mode,
        max_bs=talker_torch_compile_max_bs,
    )
    return pipeline_config


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
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
    thinker_cuda_graph: Annotated[
        str,
        typer.Option(
            "--thinker-cuda-graph",
            "--thinker_cuda_graph",
            "--thinker_CUDA_graph",
            help="CUDA graph mode for thinker stage: default|on|off.",
        ),
    ] = "default",
    talker_cuda_graph: Annotated[
        str,
        typer.Option(
            "--talker-cuda-graph",
            "--talker_cuda_graph",
            "--talker_CUDA_graph",
            help="CUDA graph mode for talker_ar stage: default|on|off.",
        ),
    ] = "default",
    thinker_torch_compile: Annotated[
        str,
        typer.Option(
            "--thinker-torch-compile",
            "--thinker_torch_compile",
            help="torch.compile mode for thinker stage: default|on|off.",
        ),
    ] = "default",
    talker_torch_compile: Annotated[
        str,
        typer.Option(
            "--talker-torch-compile",
            "--talker_torch_compile",
            help="torch.compile mode for talker_ar stage: default|on|off.",
        ),
    ] = "default",
    thinker_torch_compile_max_bs: Annotated[
        int | None,
        typer.Option(
            "--thinker-torch-compile-max-bs",
            "--thinker_torch_compile_max_bs",
            help="Override torch_compile_max_bs for thinker stage.",
        ),
    ] = None,
    talker_torch_compile_max_bs: Annotated[
        int | None,
        typer.Option(
            "--talker-torch-compile-max-bs",
            "--talker_torch_compile_max_bs",
            help="Override torch_compile_max_bs for talker_ar stage.",
        ),
    ] = None,
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
    merged_config = merged_config.model_copy(update={"model_path": model_path})
    merged_config = apply_cuda_graph_cli_overrides(
        merged_config,
        thinker_cuda_graph=thinker_cuda_graph,
        talker_cuda_graph=talker_cuda_graph,
    )
    merged_config = apply_torch_compile_cli_overrides(
        merged_config,
        thinker_torch_compile=thinker_torch_compile,
        talker_torch_compile=talker_torch_compile,
        thinker_torch_compile_max_bs=thinker_torch_compile_max_bs,
        talker_torch_compile_max_bs=talker_torch_compile_max_bs,
    )

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
