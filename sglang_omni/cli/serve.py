from __future__ import annotations

import logging
import os
import sys
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config.manager import ConfigManager
from sglang_omni.serve.launcher import launch_server
from sglang_omni.utils import print_server_version_banner

_THINKER_STAGE_NAME = "thinker"
# Note (Chenyang): Add for V1.
_V1_CLI_MODULE = "sglang_omni_v1.cli"


def _build_v1_exec_argv(argv: list[str]) -> list[str]:
    filtered_args: list[str] = []
    index = 1  # Skip the executable/script path.
    while index < len(argv):
        arg = argv[index]
        if arg == "--version":
            index += 2
            continue
        if arg.startswith("--version="):
            index += 1
            continue
        filtered_args.append(arg)
        index += 1

    return [sys.executable, "-m", _V1_CLI_MODULE, *filtered_args]


def _dispatch_to_v1_cli() -> None:
    os.execv(sys.executable, _build_v1_exec_argv(sys.argv))


# Note (Chenyang): Add for V1.


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
    thinker_max_seq_len: Annotated[
        int | None,
        typer.Option(
            "--thinker-max-seq-len",
            help="Override thinker max sequence length for Qwen3-style pipelines.",
        ),
    ] = None,
    encoder_mem_reserve: Annotated[
        float | None,
        typer.Option(
            "--encoder-mem-reserve",
            help=(
                "Reserve a fraction of GPU memory for the co-located encoder by "
                "subtracting it from SGLang's auto-picked mem_fraction_static. "
                "Required for long-video / large-encoder workloads where "
                "encoder activation peaks would otherwise OOM. Mutually "
                "exclusive with --mem-fraction-static and "
                "--thinker-mem-fraction-static. Must be in [0, 1)."
            ),
        ),
    ] = None,
    # Note (Chenyang): Add for V1.
    version: Annotated[
        Literal["legacy", "v1"],
        typer.Option(
            help=(
                "Select which server implementation to run. "
                "'legacy' keeps the current mainline pipeline, while 'v1' "
                "dispatches to the copied refactored package."
            )
        ),
    ] = "legacy",
    # Note (Chenyang): Add for V1.
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
) -> None:
    """Serve the pipeline."""
    print_server_version_banner(version, entry="sglang_omni.cli serve")
    # Note (Chenyang): Add for V1.
    if version == "v1":
        unsupported_v1_flags = {
            "--mem-fraction-static": mem_fraction_static,
            "--thinker-mem-fraction-static": thinker_mem_fraction_static,
            "--talker-mem-fraction-static": talker_mem_fraction_static,
            "--thinker-max-seq-len": thinker_max_seq_len,
            "--encoder-mem-reserve": encoder_mem_reserve,
        }
        for flag_name, value in unsupported_v1_flags.items():
            if value is not None:
                raise typer.BadParameter(
                    f"{flag_name} is only supported by the legacy server. "
                    "Remove it when using --version v1."
                )

        _dispatch_to_v1_cli()
        return
    # Note (Chenyang): End for V1.
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
    for flag_name, value in (
        ("--mem-fraction-static", mem_fraction_static),
        ("--thinker-mem-fraction-static", thinker_mem_fraction_static),
        ("--talker-mem-fraction-static", talker_mem_fraction_static),
    ):
        if value is not None and not 0.0 < value < 1.0:
            raise typer.BadParameter(f"{flag_name} must be > 0 and < 1, got {value}")

    role_to_stage = type(merged_config).mem_fraction_role_to_stage()
    if mem_fraction_static is not None and not role_to_stage:
        raise typer.BadParameter(
            "--mem-fraction-static requires a pipeline with a supported "
            "mem_fraction_static override target"
        )
    if thinker_mem_fraction_static is not None and "thinker" not in role_to_stage:
        raise typer.BadParameter(
            "--thinker-mem-fraction-static is not supported by pipeline "
            f"{type(merged_config).__name__}."
        )
    if talker_mem_fraction_static is not None and "talker" not in role_to_stage:
        raise typer.BadParameter(
            "--talker-mem-fraction-static is not supported by pipeline "
            f"{type(merged_config).__name__}."
        )

    role_values = {
        "thinker": thinker_mem_fraction_static,
        "talker": talker_mem_fraction_static,
    }
    for role, stage_name in role_to_stage.items():
        role_value = role_values.get(role)
        final_mem_fraction_static = (
            role_value if role_value is not None else mem_fraction_static
        )
        if final_mem_fraction_static is not None:
            merged_config.apply_server_args_overrides(
                stage_name=stage_name,
                overrides={"mem_fraction_static": final_mem_fraction_static},
            )

    if thinker_max_seq_len is not None:
        if thinker_max_seq_len <= 0:
            raise typer.BadParameter(
                "--thinker-max-seq-len must be a positive integer."
            )
        if _THINKER_STAGE_NAME not in {stage.name for stage in merged_config.stages}:
            raise typer.BadParameter(
                "--thinker-max-seq-len is not supported by pipeline "
                f"{type(merged_config).__name__}."
            )
        merged_config.apply_server_args_overrides(
            stage_name=_THINKER_STAGE_NAME,
            overrides={"thinker_max_seq_len": int(thinker_max_seq_len)},
        )

    if encoder_mem_reserve is not None:
        if not 0.0 <= encoder_mem_reserve < 1.0:
            raise typer.BadParameter("--encoder-mem-reserve must be in [0, 1)")
        if mem_fraction_static is not None or thinker_mem_fraction_static is not None:
            raise typer.BadParameter(
                "--encoder-mem-reserve is mutually exclusive with "
                "--mem-fraction-static and --thinker-mem-fraction-static"
            )
        if _THINKER_STAGE_NAME not in {stage.name for stage in merged_config.stages}:
            raise typer.BadParameter(
                "--encoder-mem-reserve is not supported by pipeline "
                f"{type(merged_config).__name__}."
            )
        merged_config.apply_server_args_overrides(
            stage_name=_THINKER_STAGE_NAME,
            overrides={"encoder_mem_reserve": float(encoder_mem_reserve)},
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
