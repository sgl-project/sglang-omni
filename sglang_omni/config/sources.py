# SPDX-License-Identifier: Apache-2.0
"""The canonical user-facing surfaces for setting a configuration path.

Two spellings, one path language. In YAML, per-stage settings live under the
``stages:`` mapping, keyed by stage name::

    config_cls: MossTTSPipelineConfig
    model_path: OpenMOSS-Team/MOSS-TTS

    stages:
      tts_engine:
        tp_size: 2
        engine:
          mem_fraction_static: 0.7
      vocoder:
        model:
          dtype: bfloat16

On the command line the same paths appear as dotted flags, with the
``stages.`` prefix implied -- the flag starts from the stage name, exactly
as the mapping does::

    sgl-omni serve --config omni.yaml \\
        --tts_engine.tp_size 2 \\
        --tts_engine.engine.mem_fraction_static 0.7

Every stage name must be one the config class defines: stage topology --
which stages exist, how they route, where requests enter -- lives in the
model's ``config.py`` and is not user configuration. The file overrides
settings on those stages; every leaf it writes becomes one patch, so
provenance and conflict detection stay per value.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import yaml
from pydantic import ValidationError

from sglang_omni.config.patch import (
    ConfigPatch,
    ConfigPatchSet,
    ConfigSource,
    SourceKind,
    Specificity,
)
from sglang_omni.config.path import ConfigPath, ConfigPathError, SegmentKind
from sglang_omni.config.schema import PipelineConfig

__all__ = [
    "dump_user_config",
    "patches_from_dotted_cli",
    "patches_from_model_path_flag",
    "patches_from_shared_block",
    "patches_from_stages_mapping",
    "sources_from_config_file",
]


class _DuplicateKeyRefusingLoader(yaml.SafeLoader):
    """SafeLoader that refuses duplicate mapping keys instead of keeping the last.

    yaml.safe_load silently collapses ``model_path`` written twice before any
    patch exists, so same-precedence duplicate detection would never see the
    conflict. Refusing at parse time keeps "one path, one writer per source"
    true for the file surface too.
    """

    def construct_mapping(self, node, deep=False):  # type: ignore[override]
        seen: set[Any] = set()
        for key_node, _value_node in node.value:
            key = self.construct_object(key_node, deep=deep)
            if isinstance(key, (dict, list, set)):
                continue  # unhashable; let SafeLoader produce its own error
            if key in seen:
                raise yaml.constructor.ConstructorError(
                    None,
                    None,
                    f"duplicate mapping key {key!r}",
                    key_node.start_mark,
                )
            seen.add(key)
        return super().construct_mapping(node, deep=deep)


# Top-level blocks earlier config surfaces accepted. They are refused with
# directions rather than falling through to a generic unknown-field error,
# because every one of them has an exact spelling in the current surface.
_REMOVED_TOP_LEVEL_BLOCKS: dict[str, str] = {
    "stage_overrides": (
        "the stage_overrides block was removed; write the same settings "
        "under the stages: mapping, e.g. stages.<name>.gpu_memory_fraction"
    ),
    "runtime_overrides": (
        "runtime_overrides was removed; write stages.<name>.engine.* or "
        "stages.<name>.factory.* instead"
    ),
    "entry_stage": (
        "entry_stage is not user configuration; the entry stage is declared "
        "by the model's config class"
    ),
}

_STAGES_LIST_GUIDANCE = (
    "stages must be a mapping keyed by stage name; the list form is no "
    "longer accepted. Move each entry under its name and drop the name: "
    "key:\n\n    stages:\n      thinker:\n        tp_size: 2"
)


def patches_from_dotted_cli(
    extra_args: Mapping[str, Any] | Iterable[tuple[str, Any]],
    config: PipelineConfig,
    *,
    origin: str = "extra CLI args",
) -> ConfigPatchSet:
    """Normalize ``--thinker.engine.mem_fraction_static 0.6`` style arguments.

    The ``stages.`` prefix is implied on the command line -- the flag starts
    from the stage name, exactly as the YAML ``stages:`` mapping starts from
    it -- and the explicit prefix is refused so each path has one spelling.
    Accepts a mapping or ordered ``(key, value)`` pairs; the pairs keep a
    repeated flag visible, so the resolver's duplicate check rules on it
    instead of the parser keeping whichever spelling came last.
    """
    # Local import: the spelling rewrites are compat's job, and compat imports
    # this module, so importing it at module level would be a cycle.
    from sglang_omni.config.compat import canonicalize_dotted_key

    items = extra_args.items() if isinstance(extra_args, Mapping) else extra_args
    patchset = ConfigPatchSet()
    for key, value in items:
        head, _, rest = str(key).partition(".")
        if head == "stages" and rest:
            raise ConfigPathError(
                f"--{key}: the stages. prefix is implied on the command "
                f"line; write --{rest}",
                raw=str(key),
            )
        canonical = canonicalize_dotted_key(str(key), config)
        compiled = ConfigPath.parse(canonical, type(config))
        if (
            not compiled.is_leaf
            and compiled.segments[-1].kind is not SegmentKind.FREEFORM
        ):
            # The CLI writes one leaf at a time. A whole group as one flag
            # value would either replace the container (resetting siblings
            # other sources set) or need its own merge semantics; the YAML
            # ``stages:`` mapping is the surface for writing blocks. A
            # free-form key is exempt: its type is open, so it *is* the leaf.
            raise ConfigPathError(
                f"--{key} addresses a settings group, not a single value; "
                f"write one field below it, e.g. --{key}.<field> <value>",
                raw=str(key),
            )
        patchset.add(
            ConfigPatch.create(
                compiled,
                value,
                ConfigSource(SourceKind.CLI_DOTTED, origin),
                root=type(config),
            )
        )
    return patchset


def patches_from_model_path_flag(
    model_path: str,
    config: PipelineConfig,
    *,
    origin: str = "--model-path",
) -> ConfigPatchSet:
    """Translate the typed ``--model-path`` flag into a CLI-layer patch.

    Only needed alongside ``--config``: without a file the flag *selects* the
    baseline and there is nothing to override. With one, the flag is a command
    line source like any other and must outrank the file's ``model_path`` --
    as a patch, so that a launch and ``config explain`` agree on the value and
    on where it came from.
    """
    source = ConfigSource(SourceKind.CLI_FLAG, origin)
    return ConfigPatchSet().add(
        ConfigPatch.create("model_path", model_path, source, root=type(config))
    )


def patches_from_shared_block(
    shared_block: Any,
    config_cls: type[PipelineConfig],
    stage_names: Iterable[str],
    *,
    origin: str = "",
) -> ConfigPatchSet:
    """Expand the ``shared:`` selector list into per-stage patches.

    Each entry pairs a ``select:`` block with a stage-shaped body; the body is
    applied to every stage the selector matches, one leaf patch per stage, at
    ``Specificity.ROLE`` -- so a path written explicitly for one stage (under
    ``stages:`` or as a dotted flag) always outranks the broadcast without
    being a conflict. Selectors:

    * ``stages`` -- explicit stage names (each must exist);
    * ``engine`` -- ``true`` matches every SGLang engine stage;
    * ``exclude`` -- removes stages after the positive selectors matched.

    The expansion happens before duplicate checking and validation; resolved
    values are stored per stage, and provenance shows each expanded patch with
    the entry's index so ``config explain`` can name the selector that wrote
    a value. Two entries expanding onto one leaf are a conflict, exactly as
    two explicit writes at one precedence are.
    """
    if not isinstance(shared_block, list):
        raise ValueError(
            "shared must be a list of {select: ..., <settings>} "
            f"entries, got {type(shared_block).__name__}"
        )

    names = list(stage_names)
    patches = ConfigPatchSet()
    for index, entry in enumerate(shared_block):
        label = f"shared[{index}]"
        if not isinstance(entry, Mapping) or "select" not in entry:
            raise ValueError(f"{label} must be a mapping with a select: block")
        body = {key: value for key, value in entry.items() if key != "select"}
        if not body:
            raise ValueError(f"{label} selects stages but writes nothing")
        matched = _select_stages(entry["select"], config_cls, names, label=label)
        source = ConfigSource(SourceKind.YAML_FILE, origin, detail=label)
        for stage_name in matched:
            for path, value in _flatten(f"stages.{stage_name}", dict(body), config_cls):
                patches.add(
                    ConfigPatch.create(
                        path,
                        value,
                        source,
                        root=config_cls,
                        specificity=Specificity.ROLE,
                    )
                )
    return patches


def _select_stages(
    select: Any,
    config_cls: type[PipelineConfig],
    stage_names: list[str],
    *,
    label: str,
) -> list[str]:
    """Resolve one ``select:`` block to the stage names it matches."""
    if not isinstance(select, Mapping) or not select:
        raise ValueError(f"{label}.select must be a non-empty mapping")
    unknown_keys = set(select) - {"stages", "engine", "exclude"}
    if unknown_keys:
        raise ValueError(
            f"{label}.select has unknown selector(s) {sorted(unknown_keys)}; "
            "supported: stages, engine, exclude"
        )

    matched = list(stage_names)
    if "stages" in select:
        wanted = select["stages"]
        if not isinstance(wanted, list) or not all(
            isinstance(name, str) for name in wanted
        ):
            raise ValueError(f"{label}.select.stages must be a list of names")
        missing = [name for name in wanted if name not in stage_names]
        if missing:
            raise ValueError(
                f"{label}.select.stages names unknown stage(s) {missing}; "
                f"this pipeline has: {', '.join(stage_names)}"
            )
        matched = [name for name in matched if name in wanted]
    if "engine" in select:
        engine = select["engine"]
        if not isinstance(engine, bool):
            raise ValueError(
                f"{label}.select.engine must be true or false, got {engine!r}"
            )
        if engine:
            matched = [
                name
                for name in matched
                if config_cls.stage_config_cls(name).engine_stage
            ]
    excluded = select.get("exclude") or []
    if not isinstance(excluded, list) or not all(
        isinstance(name, str) for name in excluded
    ):
        raise ValueError(f"{label}.select.exclude must be a list of names")
    missing_excluded = [name for name in excluded if name not in stage_names]
    if missing_excluded:
        raise ValueError(
            f"{label}.select.exclude names unknown stage(s) {missing_excluded}; "
            f"this pipeline has: {', '.join(stage_names)}"
        )
    matched = [name for name in matched if name not in excluded]
    if not matched:
        raise ValueError(f"{label}.select matches no stage of this pipeline")
    return matched


def patches_from_stages_mapping(
    stages_block: Any,
    config_cls: type[PipelineConfig],
    known_names: Iterable[str],
    *,
    origin: str = "",
) -> ConfigPatchSet:
    """Split the ``stages:`` mapping into per-leaf patches.

    Every name must be a stage the config class defines: stage topology --
    which stages exist, how they route, where requests enter -- lives in the
    model's ``config.py``, and a config file only overrides settings on it.
    Each leaf the body writes becomes one patch at the user-file layer.
    """
    if isinstance(stages_block, list):
        raise ValueError(_STAGES_LIST_GUIDANCE)
    if not isinstance(stages_block, Mapping):
        raise ValueError(
            "stages must be a mapping from stage name to stage settings, "
            f"got {type(stages_block).__name__}"
        )

    known = list(known_names)
    source = ConfigSource(SourceKind.YAML_FILE, origin)
    patches = ConfigPatchSet()

    for stage_name, body in stages_block.items():
        if not isinstance(stage_name, str) or not stage_name:
            raise ValueError(
                f"stages keys must be stage names written as strings, "
                f"got {stage_name!r}"
            )
        if not isinstance(body, Mapping):
            raise ValueError(f"stages.{stage_name} must be a mapping")
        if "name" in body:
            raise ValueError(
                f"stages.{stage_name} must not set name: the mapping key is "
                "the stage's name"
            )
        if stage_name not in known:
            raise ValueError(
                f"stages.{stage_name}: no stage named {stage_name!r} in "
                f"{config_cls.__name__}; stage topology lives in the model's "
                f"config class. This pipeline has: {', '.join(known)}"
            )
        prefix = f"stages.{stage_name}"
        for path, value in _flatten(prefix, dict(body), config_cls):
            # coerce applies the same lossless-scalar contract as dotted CLI
            # flags, so ``tp_size: 32.0`` is refused rather than lax-converted.
            patches.add(ConfigPatch.create(path, value, source, root=config_cls))

    return patches


def sources_from_config_file(
    file_path: str,
) -> tuple[PipelineConfig, ConfigPatchSet]:
    """Split a config file into the config it declares and its overrides.

    ``ConfigManager.from_file`` folds the returned patches into the config,
    which is what launching needs and the opposite of what explaining needs:
    by the time the caller holds the config, the stage entry that set a value
    has already been absorbed into it. Keeping the two apart is what lets
    ``sgl-omni config explain`` name the file as a source.
    """
    # Local import: the registry pulls in model packages, several of which
    # import this module's callers.
    from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

    try:
        with open(file_path, "r", encoding="utf-8") as f:
            try:
                data = yaml.load(f, Loader=_DuplicateKeyRefusingLoader)
            except yaml.YAMLError as exc:
                raise ValueError(
                    f"Config file {file_path!r} is not valid YAML: {exc}"
                ) from exc
    except FileNotFoundError as exc:
        raise ValueError(f"Config file {file_path!r} does not exist") from exc
    if not isinstance(data, dict):
        raise ValueError(f"Config file {file_path!r} must contain a mapping")

    data = dict(data)
    for block, guidance in _REMOVED_TOP_LEVEL_BLOCKS.items():
        if block in data:
            raise ValueError(f"Config file {file_path!r}: {guidance}")

    if "config_cls" not in data:
        raise ValueError(
            f"Config file {file_path!r} must name its pipeline class in " "config_cls"
        )
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config_cls_by_name(data["config_cls"])
    stages_block = data.pop("stages", None)
    shared_block = data.pop("shared", None)
    overrides = {key: value for key, value in data.items() if key != "config_cls"}

    # The baseline is built from the class's own defaults plus only the keys
    # construction requires (model_path, typically): every other file value
    # is applied exactly once, as a patch. Baking them all would apply the
    # file twice -- the baseline would already carry the value, so a diff
    # against it shows nothing and provenance calls the same value both the
    # model default and the file's write.
    construction: dict[str, Any] = {}
    while True:
        try:
            config = config_cls(**construction)
            break
        except ValidationError as exc:
            missing = [
                str(error["loc"][0])
                for error in exc.errors()
                if error["type"] == "missing" and str(error["loc"][0]) in overrides
            ]
            if not missing:
                raise
            for key in missing:
                construction[key] = overrides[key]

    patches = ConfigPatchSet()
    source = ConfigSource(SourceKind.YAML_FILE, str(file_path))
    for key, value in overrides.items():
        if isinstance(value, dict) and not ConfigPath.parse(key, config_cls).is_leaf:
            for leaf_path, leaf_value in _flatten(key, value, config_cls):
                patches.add(
                    ConfigPatch.create(leaf_path, leaf_value, source, root=config_cls)
                )
        else:
            patches.add(ConfigPatch.create(key, value, source, root=config_cls))
    if stages_block is not None:
        patches = patches.merge(
            patches_from_stages_mapping(
                stages_block,
                config_cls,
                (stage.name for stage in config.stages),
                origin=str(file_path),
            )
        )
    if shared_block is not None:
        # Expanded against the settled stage list, so a selector can reach a
        # stage this same file added.
        patches = patches.merge(
            patches_from_shared_block(
                shared_block,
                config_cls,
                (stage.name for stage in config.stages),
                origin=str(file_path),
            )
        )
    return config, patches


def dump_user_config(config: PipelineConfig) -> dict[str, Any]:
    """Dump a config in the shape a config file is written in.

    The internal stage list becomes the user-facing ``stages:`` mapping: the
    key carries the name (so ``name`` leaves the body), and a non-engine
    stage's ``engine: null`` placeholder is dropped because writing below it
    is a path error. ``entry_stage`` is dropped too -- it belongs to the
    model's config class, not to a config file. The result round-trips
    through :func:`sources_from_config_file` back to an equal config.
    """
    data = config.model_dump(mode="json")
    data.pop("entry_stage", None)
    stages: dict[str, Any] = {}
    for stage in data.get("stages", []):
        body = dict(stage)
        body.pop("name", None)
        if body.get("engine") is None:
            body.pop("engine", None)
        stages[stage["name"]] = body
    data["stages"] = stages
    return data


def _flatten(
    prefix: str,
    value: dict[str, Any],
    root: type[PipelineConfig],
) -> list[tuple[str, Any]]:
    """Split a nested stage entry into one patch per leaf.

    Stopping at schema leaves keeps the by-name merge a merge (an omitted
    sibling keeps its default) while keeping provenance per value rather
    than per block.
    """
    out: list[tuple[str, Any]] = []
    for key, child in value.items():
        path = f"{prefix}.{key}"
        if isinstance(child, dict) and not ConfigPath.parse(path, root).is_leaf:
            out.extend(_flatten(path, child, root))
        else:
            out.append((path, child))
    return out
