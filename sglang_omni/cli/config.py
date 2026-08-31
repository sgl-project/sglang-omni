from __future__ import annotations

import logging
from dataclasses import replace
from enum import Enum
from typing import Annotated, Any, NamedTuple, Optional

import typer
import yaml

from sglang_omni.config.compat import canonicalize_dotted_key
from sglang_omni.config.manager import ConfigManager, resolve_config_cls_for_model_path
from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
)
from sglang_omni.config.path import ConfigPath, ConfigPathError
from sglang_omni.config.resolver import ConfigResolver, ResolvedConfig, diff_configs
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import (
    dump_user_config,
    patches_from_dotted_cli,
    patches_from_model_path_flag,
    sources_from_config_file,
)

logger = logging.getLogger(__name__)

config_app = typer.Typer(help="Inspect, resolve and export the pipeline configuration")

_MODEL_PATH_HELP = "The Hugging Face model ID or the path to the model directory."
_CONFIG_HELP = "Path to a pipeline config file, as accepted by `sgl-omni serve`."
_TEXT_ONLY_HELP = "Use the thinker-only pipeline, as `sgl-omni serve --text-only` does."
_MEM_FRACTION_HELP = (
    "Set engine.mem_fraction_static on every SGLang engine stage, as "
    "`sgl-omni serve --mem-fraction-static` does."
)


def _dump_yaml(data: Any) -> str:
    return yaml.dump(
        data,
        sort_keys=False,  # preserve order
        default_flow_style=False,  # use block style (not inline)
        indent=2,  # control indentation
        allow_unicode=True,
    )


@config_app.command()
def view(
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
) -> None:
    """View the model's pipeline configuration."""
    config_cls = resolve_config_cls_for_model_path(model_path)
    config = config_cls(model_path=model_path)
    print(_dump_yaml(dump_user_config(config)))


@config_app.command()
def export(
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Path to the output JSON file.")
    ] = None,
) -> None:
    """Export the default pipeline configuration to a YAML file."""
    # get the default pipeline config for the model

    config_cls = resolve_config_cls_for_model_path(model_path)
    config = config_cls(model_path=model_path)

    # export config in a yaml file
    if output_path is None:
        output_path = f"./config_{config.name}.yaml"

    with open(output_path, "w") as f:
        f.write(_dump_yaml(dump_user_config(config)))
    print(f"Pipeline config exported to {output_path}")


# ----------------------------------------------------------------------
# resolve / explain
# ----------------------------------------------------------------------


class ResolveOutput(str, Enum):
    """What ``config resolve`` writes to stdout."""

    config = "config"
    diff = "diff"
    provenance = "provenance"


class Resolution(NamedTuple):
    """Everything these commands need to answer a question about a launch."""

    baseline: PipelineConfig
    """The config before any source was applied, for `--show diff`."""

    resolved: ResolvedConfig


def _resolve_sources(
    *,
    model_path: str | None,
    config_file: str | None,
    text_only: bool,
    mem_fraction_static: float | None,
    argv: list[str],
) -> Resolution:
    """Build the configuration ``sgl-omni serve`` would build from this input.

    Not a re-implementation of the launch path: ``ConfigManager.merge_config``
    resolves the very same patch set through the very same resolver. The only
    difference is that a launch throws the provenance away and this keeps it,
    which is what lets these commands answer *which source set this value*.
    """
    # Local import: serve owns the broadcast flag's fan-out and the TP
    # derivation; importing them here keeps the two commands building
    # identical configurations.
    from sglang_omni.cli.serve import (
        apply_tensor_parallel_engine_overrides,
        patches_from_broadcast_flags,
        tensor_parallel_engine_writes,
    )

    if config_file is None and model_path is None:
        raise typer.BadParameter("--model-path is required unless --config is set")

    try:
        if config_file:
            baseline, patches = sources_from_config_file(config_file)
            if model_path is not None:
                # `serve` semantics: given both, the flag is a command line
                # source and outranks the file's own model_path.
                patches = patches.merge(
                    patches_from_model_path_flag(model_path, baseline)
                )
        else:
            manager = ConfigManager.from_model_path(
                str(model_path), variant="text" if text_only else None
            )
            baseline, patches = manager.config, ConfigPatchSet()

        patches = patches.merge(
            patches_from_broadcast_flags(
                baseline, mem_fraction_static=mem_fraction_static
            )
        )
        extra_args = ConfigManager(baseline).parse_extra_args(list(argv))
        patches = patches.merge(
            patches_from_dotted_cli(extra_args, baseline, origin="command line")
        )
        resolved = ConfigResolver(baseline).resolve(patches)
        # The same post-merge derivation `serve` runs; it only fills engine
        # keys no source set, so the previewed config is the launched one.
        derivation_writes = tensor_parallel_engine_writes(resolved.config)
        derived = apply_tensor_parallel_engine_overrides(resolved.config)
        # The derivation may fill a key a source explicitly cleared (an
        # engine key set to none is unset again, and unset keys are the
        # derivation's to fill). Re-record what the launched config holds so
        # explain's headline reports it, with the winner's own input kept on
        # its line and the rewrite noted.
        for patch in resolved.patches.ordered():
            resolved.provenance.record_resolved(patch.key, patch.path.read(derived))
        # A purely derived key has no patch; give it an entry of its own so
        # explain names the derivation instead of claiming a model default.
        derivation_source = ConfigSource(
            SourceKind.MODEL_DEFAULT,
            detail="derived from the resolved TP settings at launch",
        )
        for path_text, value in derivation_writes.items():
            if resolved.provenance.touched(path_text):
                continue
            derivation_patch = ConfigPatch.create(
                path_text, value, derivation_source, root=type(baseline)
            )
            resolved.provenance.record(derivation_patch, winning=True)
            resolved.provenance.record_resolved(
                path_text, derivation_patch.path.read(derived)
            )
        return Resolution(baseline, replace(resolved, config=derived))
    except (ConfigPathError, ValueError) as exc:
        # An unknown path, a path the schema will not let a user write, two
        # sources disagreeing at one precedence, a malformed dotted argument,
        # a config file that does not parse: all of these are things the user
        # typed, and all of them carry a message written to be read. A
        # traceback would bury it.
        raise typer.BadParameter(str(exc)) from exc


@config_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def resolve(
    ctx: typer.Context,
    model_path: Annotated[str | None, typer.Option(help=_MODEL_PATH_HELP)] = None,
    config: Annotated[str | None, typer.Option(help=_CONFIG_HELP)] = None,
    text_only: Annotated[
        bool, typer.Option("--text-only", help=_TEXT_ONLY_HELP)
    ] = False,
    mem_fraction_static: Annotated[
        float | None, typer.Option("--mem-fraction-static", help=_MEM_FRACTION_HELP)
    ] = None,
    show: Annotated[
        ResolveOutput,
        typer.Option(
            help=(
                "config: the whole resolved pipeline. "
                "diff: only the fields the sources changed. "
                "provenance: every source that contributed, winner last."
            )
        ),
    ] = ResolveOutput.config,
) -> None:
    """Show the configuration a `serve` command with these arguments would use.

    Takes the same arguments as `sgl-omni serve`, including dotted overrides
    (the stages. prefix is implied, exactly as on serve):

        sgl-omni config resolve --model-path Qwen/Qwen3-Omni \\
            --thinker.tp_size 4 --show diff
    """
    resolution = _resolve_sources(
        model_path=model_path,
        config_file=config,
        text_only=text_only,
        mem_fraction_static=mem_fraction_static,
        argv=ctx.args,
    )
    provenance = resolution.resolved.provenance

    if show is ResolveOutput.config:
        print(_dump_yaml(dump_user_config(resolution.resolved.config)))
        return

    if show is ResolveOutput.diff:
        # Against the baseline config rather than against the recorded patches,
        # so that a field an alias carried along is shown as changed too.
        changes = diff_configs(resolution.baseline, resolution.resolved.config)
        if not changes:
            print("No configuration source changed the pipeline's defaults.")
            return
        for change in changes:
            print(f"{change.path}: {change.expected!r} -> {change.actual!r}")
        return

    if not provenance.paths():
        print("No configuration source touched the pipeline's defaults.")
        return
    print("\n\n".join(provenance.explain(path) for path in provenance.paths()))


@config_app.command(
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)
def explain(
    ctx: typer.Context,
    # Declared in the pre-``Annotated`` style, and spelled ``Optional[str]``
    # rather than ``str | None``, on purpose. This file is compiled with
    # ``from __future__ import annotations``, and typer 0.9 -- the floor this
    # project supports -- resolves those string annotations without
    # ``include_extras``, so an ``Annotated[..., typer.Argument()]`` marker is
    # dropped and the parameter silently becomes ``--path``; the positional then
    # falls through to ``ctx.args`` and is parsed as half a dotted override.
    # The options below survive that only because a default of ``None`` makes
    # ``get_type_hints`` rewrite them into ``Optional[str]`` anyway, which this
    # parameter -- defaulting to an ``ArgumentInfo`` -- does not get.
    path: Optional[str] = typer.Argument(
        None,
        help=(
            "Config path, canonical (stages.thinker.factory.max_seq_len) or "
            "CLI-spelled (thinker.factory.max_seq_len). Omit to list every "
            "path a source touched. Pass it before any dotted override so "
            "it is not mistaken for one."
        ),
    ),
    model_path: Annotated[str | None, typer.Option(help=_MODEL_PATH_HELP)] = None,
    config: Annotated[str | None, typer.Option(help=_CONFIG_HELP)] = None,
    text_only: Annotated[
        bool, typer.Option("--text-only", help=_TEXT_ONLY_HELP)
    ] = False,
    mem_fraction_static: Annotated[
        float | None, typer.Option("--mem-fraction-static", help=_MEM_FRACTION_HELP)
    ] = None,
) -> None:
    """Say where one configuration value came from, and what it overrode.

        sgl-omni config explain stages.thinker.factory.max_seq_len \\
            --config omni.yaml --thinker.factory.max_seq_len 8192
    """
    resolution = _resolve_sources(
        model_path=model_path,
        config_file=config,
        text_only=text_only,
        mem_fraction_static=mem_fraction_static,
        argv=ctx.args,
    )
    provenance = resolution.resolved.provenance

    if path is None:
        if not provenance.paths():
            print("No configuration source touched the pipeline's defaults.")
            return
        for touched in provenance.paths():
            winner = provenance.winner(touched)
            assert winner is not None  # a touched path always has a winner
            value = provenance.resolved_value(touched, winner.value)
            print(f"{touched} = {value!r}  <- {winner.source.describe()}")
        return

    try:
        canonical = canonicalize_dotted_key(path, resolution.baseline)
        compiled = ConfigPath.parse(canonical, type(resolution.baseline))
    except ConfigPathError as exc:
        # Carries `did you mean:` suggestions, which a traceback would bury.
        raise typer.BadParameter(str(exc)) from exc

    if provenance.touched(compiled.raw):
        print(provenance.explain(compiled.raw))
        return
    try:
        value = compiled.read(resolution.resolved.config)
    except ConfigPathError:
        # The path parses (a free mapping key or extra factory key can) but
        # nothing wrote it, so there is no value to show.
        print(f"{compiled.raw} is not set")
        print("  no source touched this path and the model declares no default")
        return
    print(f"{compiled.raw} = {value!r}")
    print("  no source touched this path; the value is the model's own default")
