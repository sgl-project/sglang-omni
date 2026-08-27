from __future__ import annotations

import logging
from typing import Annotated, Literal

import typer
import yaml

from sglang_omni.config import PipelineConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
    Specificity,
)
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.sources import dump_user_config, patches_from_model_path_flag
from sglang_omni.preprocessing.resource_connector import (
    resolve_allowed_local_media_path,
)
from sglang_omni.serve.protocol import DEFAULT_TTS_BATCH_MAX_ITEMS
from sglang_omni.utils.gpu_compat import should_disable_custom_all_reduce_for_gpus

logger = logging.getLogger(__name__)


_QWEN_COLOCATED_CONFIG_CLASS = "Qwen3OmniSpeechColocatedPipelineConfig"


def launch_server(*args: object, **kwargs: object) -> object:
    from sglang_omni.serve.launcher import launch_server as _launch_server

    return _launch_server(*args, **kwargs)


def _validate_colocate_cli_request(
    *,
    colocate: bool,
    config: str | None,
    text_only: bool,
) -> None:
    if not colocate:
        return
    if text_only:
        raise typer.BadParameter("--colocate cannot be combined with --text-only")
    if not config:
        raise typer.BadParameter("--colocate requires --config")


def _validate_colocate_config(pipeline_config: PipelineConfig) -> None:
    if type(pipeline_config).__name__ != _QWEN_COLOCATED_CONFIG_CLASS:
        raise typer.BadParameter(
            f"--colocate requires a {_QWEN_COLOCATED_CONFIG_CLASS} config file"
        )


def _should_print_merged_config(*, colocate: bool, log_level: str) -> bool:
    """Return whether to print the full resolved pipeline config."""

    return colocate or log_level.lower() == "debug"


def _print_merged_config(pipeline_config: PipelineConfig) -> None:
    print("=" * 20, "Merged Configuration", "=" * 20)
    print(
        yaml.dump(
            dump_user_config(pipeline_config),
            sort_keys=False,
            default_flow_style=False,
            indent=2,
        )
    )
    print("=" * 50)


def _validate_allowed_local_media_path(value: str | None) -> str | None:
    if value is None:
        return None
    try:
        return str(resolve_allowed_local_media_path(value))
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc


def _normalize_allowed_media_domains(values: list[str] | None) -> list[str]:
    domains: list[str] = []
    for value in values or []:
        domains.extend(
            part.strip().lower() for part in value.split(",") if part.strip()
        )
    return domains


def _validate_tts_batch_max_items(value: int) -> int:
    if value < 1:
        raise typer.BadParameter("tts batch max items must be greater than 0")
    return value


def _engine_stage_names(pipeline_config: PipelineConfig) -> list[str]:
    config_cls = type(pipeline_config)
    return [
        stage.name
        for stage in pipeline_config.stages
        if config_cls.stage_config_cls(stage.name).engine_stage
    ]


def patches_from_broadcast_flags(
    pipeline_config: PipelineConfig,
    *,
    mem_fraction_static: float | None,
) -> ConfigPatchSet:
    """Translate ``--mem-fraction-static`` into CLI-layer patches.

    The one broadcast flag left: it fans a single value out to every SGLang
    engine stage, which a single dotted path cannot express. As patches the
    fan-out resolves together with the dotted overrides at declared
    precedence -- ``Specificity.BROADCAST``, so an explicit dotted path for
    one stage still outranks it -- and ``sgl-omni config explain`` can name
    the flag that set a value.

    The value's range is the schema's rule (``engine.mem_fraction_static``
    refuses anything outside (0, 1) at resolution); only the broadcast's own
    precondition -- an engine stage to fan out to -- is checked here.
    """
    patches = ConfigPatchSet()

    if mem_fraction_static is None:
        return patches
    engine_stages = _engine_stage_names(pipeline_config)
    if not engine_stages:
        raise typer.BadParameter(
            "--mem-fraction-static requires a pipeline with at least one "
            "SGLang engine stage"
        )
    for stage_name in engine_stages:
        patches.add(
            ConfigPatch.create(
                f"stages.{stage_name}.engine.mem_fraction_static",
                mem_fraction_static,
                ConfigSource(SourceKind.CLI_FLAG, "--mem-fraction-static"),
                root=type(pipeline_config),
                specificity=Specificity.BROADCAST,
            )
        )
    return patches


def _stage_tp_gpu_ids(stage) -> list[int]:
    gpu = stage.gpu
    if gpu is None:
        return []
    if isinstance(gpu, int):
        return [gpu]
    return [int(g) for g in gpu]


def _gate_custom_all_reduce_on_topology(
    stage: object,
    updates: dict[str, object],
    *,
    gpu_ids: tuple[int, ...],
    should_disable: bool,
) -> dict[str, object]:
    """Relax a TP ``disable_custom_all_reduce=True`` override on a P2P-capable topology.

    The config-level overrides disable custom all-reduce for every TP thinker.
    Custom (P2P/NVLink) all-reduce is faster than NCCL and safe when the TP GPUs
    form a P2P mesh, so re-enable it when the caller's topology probe confirms one
    (``should_disable`` is False); otherwise keep it disabled. SGLang still performs
    its own communicator support checks during worker startup before using the
    custom all-reduce path.
    """
    if updates.get("disable_custom_all_reduce") is not True or should_disable:
        return updates
    refined = dict(updates)
    refined["disable_custom_all_reduce"] = False
    logger.info(
        "Enabling custom all-reduce for stage '%s': GPUs %s form a P2P mesh",
        stage.name,
        list(gpu_ids),
    )
    return refined


def tensor_parallel_engine_writes(
    pipeline_config: PipelineConfig,
) -> dict[str, object]:
    """The engine keys TP derivation would fill, as dotted-path writes.

    Only gaps are listed: a key the merged configuration already sets in the
    stage's engine block is not the derivation's to touch.
    """
    config_cls = type(pipeline_config)
    topology_gated_custom_ar_stages = (
        config_cls.topology_gated_custom_all_reduce_stages()
    )
    topology_gated_custom_ar_cache: dict[tuple[int, ...], bool] = {}
    writes: dict[str, object] = {}
    for stage in pipeline_config.stages:
        updates = config_cls.tensor_parallel_server_args_overrides(
            stage_name=stage.name,
            tp_size=stage.tp_size,
        )
        if not updates:
            continue
        if stage.name in topology_gated_custom_ar_stages:
            gpu_ids = tuple(_stage_tp_gpu_ids(stage))
            if gpu_ids not in topology_gated_custom_ar_cache:
                topology_gated_custom_ar_cache[gpu_ids] = (
                    should_disable_custom_all_reduce_for_gpus(gpu_ids)
                )
            updates = _gate_custom_all_reduce_on_topology(
                stage,
                updates,
                gpu_ids=gpu_ids,
                should_disable=topology_gated_custom_ar_cache[gpu_ids],
            )
        already_set = stage.engine.overrides() if stage.engine is not None else {}
        for key, value in updates.items():
            if key in already_set:
                continue
            writes[f"stages.{stage.name}.engine.{key}"] = value
    return writes


def apply_tensor_parallel_engine_overrides(
    pipeline_config: PipelineConfig,
) -> PipelineConfig:
    """Write engine overrides the resolved TP settings imply.

    Derived from the *resolved* configuration (TP size, GPU topology), so it
    cannot be a patch: the value does not exist until every source has been
    merged. Written through ``ConfigPath`` into a dump and re-validated, so
    the derivation goes through the same door as every other write.

    A derivation only fills gaps: a key the merged configuration already sets
    in the stage's engine block -- from the model default, the file, or a
    dotted flag -- is left exactly as it was set.
    """
    config_cls = type(pipeline_config)
    writes = tensor_parallel_engine_writes(pipeline_config)
    if not writes:
        return pipeline_config
    data = pipeline_config.model_dump()
    for path_text, value in writes.items():
        ConfigPath.parse(path_text, config_cls).write(data, value)
    return config_cls(**data)


def serve(
    ctx: typer.Context,
    model_path: Annotated[
        str | None,
        typer.Option(
            help=(
                "The Hugging Face model ID or the path to the model directory. "
                "Required unless --config provides model_path."
            )
        ),
    ] = None,
    config: Annotated[
        str | None, typer.Option(help="Path to a pipeline config file.")
    ] = None,
    text_only: Annotated[
        bool,
        typer.Option(
            "--text-only",
            help="Use thinker-only pipeline (1 GPU, no talker/speech output).",
        ),
    ] = False,
    colocate: Annotated[
        bool,
        typer.Option(
            "--colocate",
            help="Run Qwen speech with GPU stages colocated on one GPU.",
        ),
    ] = False,
    host: Annotated[
        str, typer.Option(help="Server bind address (default: 0.0.0.0).")
    ] = "0.0.0.0",
    port: Annotated[int, typer.Option(help="Server bind port (default: 8000).")] = 8000,
    model_name: Annotated[
        str, typer.Option(help="Model name for /v1/models (default: pipeline name).")
    ] = None,
    allowed_local_media_path: Annotated[
        str | None,
        typer.Option(
            "--allowed-local-media-path",
            "--allowed_local_media_path",
            help=(
                "Directory allowed for file:// media references in TTS requests. "
                "Local file references are disabled when this is omitted."
            ),
        ),
    ] = None,
    allowed_media_domain: Annotated[
        list[str] | None,
        typer.Option(
            "--allowed-media-domain",
            "--allowed_media_domain",
            help=(
                "Restrict remote media references to this domain. Repeat the "
                "flag to allow multiple domains. When omitted, remote HTTP(S) "
                "references from any public host are allowed."
            ),
        ),
    ] = None,
    tts_batch_max_items: Annotated[
        int,
        typer.Option(
            "--tts-batch-max-items",
            help="Maximum number of items accepted by /v1/audio/speech/batch.",
        ),
    ] = DEFAULT_TTS_BATCH_MAX_ITEMS,
    mem_fraction_static: Annotated[
        float | None,
        typer.Option(
            "--mem-fraction-static",
            help=(
                "Set engine.mem_fraction_static on every SGLang engine stage. "
                "A dotted per-stage flag (--<stage>.engine.mem_fraction_static) "
                "overrides this for that stage. If omitted, SGLang chooses "
                "the value automatically."
            ),
        ),
    ] = None,
    log_level: Annotated[
        Literal["debug", "info", "warning", "error", "critical"],
        typer.Option(help="Log level (default: info)."),
    ] = "info",
    enable_realtime: Annotated[
        bool,
        typer.Option(
            "--enable-realtime",
            "--enable_realtime",
            help="Mount the OpenAI Realtime WebSocket endpoint at /v1/realtime.",
        ),
    ] = False,
) -> None:
    """Serve the pipeline.

    Per-stage settings are dotted flags mirroring the YAML ``stages:``
    mapping, with the ``stages.`` prefix implied:
    ``--thinker.engine.mem_fraction_static 0.6``,
    ``--vocoder.factory.max_concurrency 8``, ``--talker_ar.gpu 1``,
    ``--thinker.process thinker``.
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    _validate_colocate_cli_request(
        colocate=colocate,
        config=config,
        text_only=text_only,
    )

    # --- Resolve config ---
    if config:
        try:
            config_manager = ConfigManager.from_file(config)
        except (ConfigPathError, ValueError) as exc:
            # A file that does not parse, an unknown path in a stage entry,
            # two entries disagreeing about one path: all of these carry a
            # message written to be read, not a traceback.
            raise typer.BadParameter(str(exc)) from exc
    elif text_only:
        if model_path is None:
            raise typer.BadParameter("--model-path is required unless --config is set")
        config_manager = ConfigManager.from_model_path(model_path, variant="text")
    else:
        if model_path is None:
            raise typer.BadParameter("--model-path is required unless --config is set")
        config_manager = ConfigManager.from_model_path(model_path)

    # we use ctx to capture the arguments that are used to modify the configuration on the fly
    # we do expect the extra arguments to be pairs of names and values
    try:
        extra_args = config_manager.parse_extra_args(ctx.args)
    except ValueError as exc:
        # A dotted flag with no value, or a stray token that pairs with
        # nothing: the message names the argument, a traceback would bury it.
        raise typer.BadParameter(str(exc)) from exc
    # The broadcast flags become patches so they resolve together with the
    # dotted overrides at declared precedence, instead of being written over
    # the merged config afterwards.
    flag_patches = patches_from_broadcast_flags(
        config_manager.config,
        mem_fraction_static=mem_fraction_static,
    )
    if config and model_path is not None:
        # Given both, the flag is a command line source and outranks the
        # file's model_path. Merged as a patch rather than assigned onto the
        # result, so `sgl-omni config resolve/explain` previews exactly what
        # launches here.
        flag_patches = flag_patches.merge(
            patches_from_model_path_flag(model_path, config_manager.config)
        )
    try:
        merged_config = config_manager.merge_config(
            extra_args,
            extra_patches=flag_patches,
        )
    except (ConfigPathError, ValueError) as exc:
        # Unknown path, unwritable path, two sources disagreeing at one
        # precedence, or a value the schema refuses. Every one of these
        # carries a message written to be read; a traceback would bury it
        # under the merge internals.
        raise typer.BadParameter(str(exc)) from exc
    if colocate:
        _validate_colocate_config(merged_config)
    merged_config = apply_tensor_parallel_engine_overrides(merged_config)

    if _should_print_merged_config(colocate=colocate, log_level=log_level):
        _print_merged_config(merged_config)

    launch_server(
        merged_config,
        host=host,
        port=port,
        model_name=model_name,
        log_level=log_level,
        enable_realtime=enable_realtime,
        allowed_local_media_path=_validate_allowed_local_media_path(
            allowed_local_media_path
        ),
        allowed_media_domains=_normalize_allowed_media_domains(allowed_media_domain),
        tts_batch_max_items=_validate_tts_batch_max_items(tts_batch_max_items),
    )
