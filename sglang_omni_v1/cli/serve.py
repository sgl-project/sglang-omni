from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni_v1.config import PipelineConfig
from sglang_omni_v1.config.manager import ConfigManager
from sglang_omni_v1.serve.launcher import launch_server

logger = logging.getLogger(__name__)


_STAGE_TOGGLE_MODE = Literal["default", "on", "off"]


def _normalize_stage_toggle_mode(flag_name: str, value: str) -> _STAGE_TOGGLE_MODE:
    normalized = value.strip().lower()
    if normalized not in {"default", "on", "off"}:
        raise typer.BadParameter(f"{flag_name} must be one of: default, on, off")
    return normalized  # type: ignore[return-value]


def _find_matching_stages(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    reason: str,
):
    matching_stages = [
        stage for stage in pipeline_config.stages if stage.name == stage_name
    ]
    if not matching_stages:
        raise typer.BadParameter(
            f"Stage {stage_name!r} not found in pipeline; cannot set {reason}"
        )
    return matching_stages


def _apply_stage_server_args_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    updates: dict[str, object],
    reason: str,
) -> None:
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=reason,
    )
    for stage in matching_stages:
        factory_args = dict(stage.factory_args or {})
        overrides = dict(factory_args.get("server_args_overrides") or {})
        overrides.update(updates)
        factory_args["server_args_overrides"] = overrides
        stage.factory_args = factory_args

        stage_runtime_overrides = pipeline_config.runtime_overrides.get(stage.name)
        if stage_runtime_overrides is not None:
            runtime_server_args = stage_runtime_overrides.get("server_args_overrides")
            if isinstance(runtime_server_args, dict):
                runtime_server_args.update(updates)


def _stage_has_explicit_mem_fraction_static(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    factory_args: dict[str, object],
) -> bool:
    server_args_overrides = dict(factory_args.get("server_args_overrides") or {})
    if server_args_overrides.get("mem_fraction_static") is not None:
        return True

    runtime_overrides = dict(pipeline_config.runtime_overrides.get(stage_name, {}))
    runtime_server_args_overrides = dict(
        runtime_overrides.get("server_args_overrides") or {}
    )
    return runtime_server_args_overrides.get("mem_fraction_static") is not None


def _validate_mem_fraction_static(flag_name: str, value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 < value < 1.0:
        raise typer.BadParameter(f"{flag_name} must be > 0 and < 1, got {value}")
    return float(value)


def _validate_encoder_mem_reserve(value: float | None) -> float | None:
    if value is None:
        return None
    if not 0.0 <= value < 1.0:
        raise typer.BadParameter("--encoder-mem-reserve must be in [0, 1)")
    return float(value)


def apply_mem_fraction_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
    talker_mem_fraction_static: float | None,
) -> PipelineConfig:
    """Apply CLI mem_fraction_static flags to the pipeline config.

    Precedence (per role): a non-None per-role flag wins over the global flag.
    `--thinker-mem-fraction-static` overrides `--mem-fraction-static` for the
    thinker stage; `--talker-mem-fraction-static` overrides it for the talker
    stage. The global `--mem-fraction-static` is the fallback for any role
    whose per-role flag is omitted.

    Validation: out-of-range values raise typer.BadParameter atomically, before
    any stage mutation, so a partially-applied config cannot leak into the
    launch path.
    """
    mem_fraction_static = _validate_mem_fraction_static(
        "--mem-fraction-static", mem_fraction_static
    )
    thinker_mem_fraction_static = _validate_mem_fraction_static(
        "--thinker-mem-fraction-static", thinker_mem_fraction_static
    )
    talker_mem_fraction_static = _validate_mem_fraction_static(
        "--talker-mem-fraction-static", talker_mem_fraction_static
    )

    role_to_stage = type(pipeline_config).mem_fraction_role_to_stage()
    if mem_fraction_static is not None and not role_to_stage:
        raise typer.BadParameter(
            "--mem-fraction-static requires a pipeline with a supported "
            "SGLang AR mem_fraction_static target"
        )
    if thinker_mem_fraction_static is not None and "thinker" not in role_to_stage:
        raise typer.BadParameter(
            "--thinker-mem-fraction-static is not supported by pipeline "
            f"{type(pipeline_config).__name__}."
        )
    if talker_mem_fraction_static is not None and "talker" not in role_to_stage:
        raise typer.BadParameter(
            "--talker-mem-fraction-static is not supported by pipeline "
            f"{type(pipeline_config).__name__}."
        )

    role_values = {
        "thinker": thinker_mem_fraction_static,
        "talker": talker_mem_fraction_static,
    }
    for role, stage_name in role_to_stage.items():
        role_value = role_values.get(role)
        # Precedence: per-role flag wins over the global flag for this role;
        # the global flag is the fallback when no per-role flag was given.
        final_value = role_value if role_value is not None else mem_fraction_static
        if final_value is not None:
            _apply_stage_server_args_override(
                pipeline_config,
                stage_name=stage_name,
                updates={"mem_fraction_static": final_value},
                reason=f"{role} SGLang mem_fraction_static override",
            )
    return pipeline_config


def apply_encoder_mem_reserve_cli_override(
    pipeline_config: PipelineConfig,
    *,
    encoder_mem_reserve: float | None,
    mem_fraction_static: float | None,
    thinker_mem_fraction_static: float | None,
) -> PipelineConfig:
    if encoder_mem_reserve is None:
        return pipeline_config
    if mem_fraction_static is not None or thinker_mem_fraction_static is not None:
        raise typer.BadParameter(
            "--encoder-mem-reserve is mutually exclusive with "
            "--mem-fraction-static and --thinker-mem-fraction-static"
        )
    encoder_mem_reserve = _validate_encoder_mem_reserve(encoder_mem_reserve)

    role_to_stage = type(pipeline_config).mem_fraction_role_to_stage()
    thinker_stage = role_to_stage.get("thinker")
    if thinker_stage is None:
        raise typer.BadParameter(
            "--encoder-mem-reserve requires a pipeline with a supported "
            "thinker SGLang AR stage"
        )

    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=thinker_stage,
        reason="Qwen thinker encoder memory reserve",
    )
    for stage in matching_stages:
        factory_args = dict(stage.factory_args or {})
        if _stage_has_explicit_mem_fraction_static(
            pipeline_config,
            stage_name=stage.name,
            factory_args=factory_args,
        ):
            raise typer.BadParameter(
                "--encoder-mem-reserve is only valid when thinker "
                "mem_fraction_static is not explicitly pinned"
            )
        factory_args["encoder_mem_reserve"] = encoder_mem_reserve
        stage.factory_args = factory_args

        stage_runtime_overrides = pipeline_config.runtime_overrides.get(stage.name)
        if (
            isinstance(stage_runtime_overrides, dict)
            and "encoder_mem_reserve" in stage_runtime_overrides
        ):
            stage_runtime_overrides["encoder_mem_reserve"] = encoder_mem_reserve
    return pipeline_config


def _parse_gpu_placement(flag_name: str, value: str) -> int | list[int]:
    text = value.strip()
    if not text:
        raise typer.BadParameter(f"{flag_name} must not be empty")

    if text.startswith("["):
        try:
            parsed = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            raise typer.BadParameter(
                f"{flag_name} must be an int or list of ints"
            ) from exc
    elif "," in text:
        parsed = [part.strip() for part in text.split(",")]
    else:
        try:
            gpu = int(text)
        except ValueError as exc:
            raise typer.BadParameter(
                f"{flag_name} must be an int or list of ints"
            ) from exc
        if gpu < 0:
            raise typer.BadParameter(f"{flag_name} GPU ids must be >= 0")
        return gpu

    if not isinstance(parsed, list) or not parsed:
        raise typer.BadParameter(f"{flag_name} must be an int or non-empty list")

    gpus: list[int] = []
    for item in parsed:
        if isinstance(item, int):
            gpu = item
        elif isinstance(item, str):
            try:
                gpu = int(item.strip())
            except ValueError as exc:
                raise typer.BadParameter(
                    f"{flag_name} must contain only integer GPU ids"
                ) from exc
        else:
            raise typer.BadParameter(f"{flag_name} must contain only integer GPU ids")
        if gpu < 0:
            raise typer.BadParameter(f"{flag_name} GPU ids must be >= 0")
        gpus.append(gpu)

    return gpus[0] if len(gpus) == 1 else gpus


def _validate_stage_parallelism_config(stage_name: str, tp_size: int, gpu) -> None:
    if tp_size < 1:
        raise typer.BadParameter(f"{stage_name}_tp_size must be >= 1")
    if tp_size == 1:
        if isinstance(gpu, list) and len(gpu) != 1:
            raise typer.BadParameter(
                f"{stage_name}_gpus must contain exactly 1 GPU id when {stage_name}_tp_size=1"
            )
        return
    if not isinstance(gpu, list):
        raise typer.BadParameter(
            f"{stage_name}_gpus must provide one GPU id per TP rank when {stage_name}_tp_size > 1"
        )
    if len(gpu) != tp_size:
        raise typer.BadParameter(
            f"{stage_name}_gpus must contain exactly {tp_size} GPU ids when {stage_name}_tp_size={tp_size}"
        )
    if len(set(gpu)) != len(gpu):
        raise typer.BadParameter(
            f"{stage_name}_gpus must not contain duplicate GPU ids"
        )


def _apply_stage_gpu_override(
    pipeline_config: PipelineConfig,
    *,
    stage_name: str,
    gpu: int | None,
) -> None:
    if gpu is None:
        return
    if gpu < 0:
        raise typer.BadParameter(f"{stage_name}_gpu must be >= 0")
    matching_stages = _find_matching_stages(
        pipeline_config,
        stage_name=stage_name,
        reason=f"GPU placement to {gpu}",
    )
    for stage in matching_stages:
        stage.gpu = int(gpu)


def apply_parallelism_cli_overrides(
    pipeline_config: PipelineConfig,
    *,
    thinker_tp_size: int | None,
    thinker_gpus: str | None,
    talker_gpu: int | None,
    code2wav_gpu: int | None,
) -> PipelineConfig:
    thinker_gpu_override = (
        _parse_gpu_placement("thinker_gpus", thinker_gpus)
        if thinker_gpus is not None
        else None
    )
    if thinker_tp_size is not None or thinker_gpu_override is not None:
        thinker_stages = _find_matching_stages(
            pipeline_config,
            stage_name="thinker",
            reason="tensor parallel settings",
        )
        for stage in thinker_stages:
            if thinker_tp_size is not None:
                stage.tp_size = int(thinker_tp_size)
            if thinker_gpu_override is not None:
                stage.gpu = thinker_gpu_override
            _validate_stage_parallelism_config("thinker", stage.tp_size, stage.gpu)
            if stage.tp_size == 1 and isinstance(stage.gpu, list):
                stage.gpu = int(stage.gpu[0])

    _apply_stage_gpu_override(
        pipeline_config,
        stage_name="talker_ar",
        gpu=talker_gpu,
    )
    _apply_stage_gpu_override(
        pipeline_config,
        stage_name="code2wav",
        gpu=code2wav_gpu,
    )
    return pipeline_config


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
    mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--mem-fraction-static",
            help=(
                "Set SGLang mem_fraction_static for all supported Qwen AR stages. "
                "If omitted, SGLang chooses the value automatically."
            ),
        ),
    ] = None,
    thinker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--thinker-mem-fraction-static",
            help=(
                "Set SGLang mem_fraction_static for Qwen thinker. Overrides "
                "--mem-fraction-static for thinker."
            ),
        ),
    ] = None,
    talker_mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--talker-mem-fraction-static",
            help=(
                "Set SGLang mem_fraction_static for Qwen talker_ar. Overrides "
                "--mem-fraction-static for talker_ar."
            ),
        ),
    ] = None,
    encoder_mem_reserve: Annotated[
        float | None,
        typer.Option(
            "--encoder-mem-reserve",
            help=(
                "Subtract this fraction from SGLang's auto-picked Qwen thinker "
                "mem_fraction_static for colocated external encoders. Valid only "
                "when thinker mem_fraction_static is not explicitly pinned."
            ),
        ),
    ] = None,
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
    thinker_tp_size: Annotated[
        int | None,
        typer.Option(
            "--thinker-tp-size",
            "--thinker_tp_size",
            help="Set tensor parallel size for thinker stage.",
        ),
    ] = None,
    thinker_gpus: Annotated[
        str | None,
        typer.Option(
            "--thinker-gpus",
            "--thinker_gpus",
            help="GPU ids for thinker TP ranks, e.g. '0,1' or '[0, 1]'.",
        ),
    ] = None,
    talker_gpu: Annotated[
        int | None,
        typer.Option(
            "--talker-gpu",
            "--talker_gpu",
            help="Override GPU id for talker_ar stage.",
        ),
    ] = None,
    code2wav_gpu: Annotated[
        int | None,
        typer.Option(
            "--code2wav-gpu",
            "--code2wav_gpu",
            help="Override GPU id for code2wav stage.",
        ),
    ] = None,
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
    merged_config = apply_mem_fraction_cli_overrides(
        merged_config,
        mem_fraction_static=mem_fraction_static,
        thinker_mem_fraction_static=thinker_mem_fraction_static,
        talker_mem_fraction_static=talker_mem_fraction_static,
    )
    merged_config = apply_encoder_mem_reserve_cli_override(
        merged_config,
        encoder_mem_reserve=encoder_mem_reserve,
        mem_fraction_static=mem_fraction_static,
        thinker_mem_fraction_static=thinker_mem_fraction_static,
    )
    merged_config = apply_parallelism_cli_overrides(
        merged_config,
        thinker_tp_size=thinker_tp_size,
        thinker_gpus=thinker_gpus,
        talker_gpu=talker_gpu,
        code2wav_gpu=code2wav_gpu,
    )
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
