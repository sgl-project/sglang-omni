# SPDX-License-Identifier: Apache-2.0
"""Canonical configuration paths compiled from the typed pipeline schema.

A :class:`ConfigPath` is *compiled against the typed schema*, not evaluated
against a dict. That buys three things a dict walker cannot offer:

* an illegal path fails at parse time with nearby legal paths, instead of
  raising ``KeyError`` deep inside a ``model_dump()`` copy;
* every path knows its declared type, so a CLI string can be coerced by
  pydantic rather than by ``int()``/``float()`` guessing;
* every path knows whether it is part of the public surface, so derived
  values such as ``config_cls`` can be refused up front.

Stages are addressed **by name** (``stages.thinker.factory.max_seq_len``) and
compile against their *own* stage type: the root config class declares a
``StageConfig`` subclass per stage name, so a model-specific ``factory.*``
field exists only on the stage that declares it, and ``engine.*`` exists
only on stages whose type drives an SGLang engine. Positional indices are
not accepted anywhere.
"""

from __future__ import annotations

import difflib
import types
import typing
from dataclasses import dataclass
from enum import Enum
from typing import Any, get_args, get_origin

from pydantic import BaseModel, TypeAdapter

from sglang_omni.config.schema import PipelineConfig, StageConfig

__all__ = [
    "ConfigPath",
    "ConfigPathError",
    "PathVisibility",
    "Segment",
    "SegmentKind",
    "coerce_scalar_text",
    "iter_schema_paths",
]

# Guard against a pathological schema; the real tree is only a few levels deep.
_MAX_SCHEMA_DEPTH = 12


class SegmentKind(str, Enum):
    """What one dotted segment addresses."""

    FIELD = "field"
    """A declared field of a pydantic model."""

    NAMED_ITEM = "named_item"
    """An element of a name-keyed collection, e.g. ``stages.thinker``."""

    MAPPING_KEY = "mapping_key"
    """A key of a typed mapping, e.g. ``env.CUDA_VISIBLE_DEVICES``."""

    FREEFORM = "freeform"
    """A key below an untyped ``Any``; no schema information remains."""


class PathVisibility(str, Enum):
    """Whether a path may be written by a user-facing configuration source."""

    PUBLIC = "public"
    INTERNAL = "internal"


_NONE_SENTINEL = "none"
"""Textual spelling that clears a field, inherited from the V1 dotted CLI."""


# Ordered specific -> general; the first matching pattern wins. ``*`` matches
# exactly one segment, ``**`` matches zero or more trailing segments.
# Paths that no longer exist are not listed here: an unknown field fails at
# parse time with nearby legal paths.
_VISIBILITY_RULES: tuple[tuple[str, PathVisibility, str], ...] = (
    (
        "config_cls",
        PathVisibility.INTERNAL,
        "derived from the config class name in PipelineConfig.model_post_init; "
        "setting it by hand has no effect on which class is used",
    ),
    (
        "stages",
        PathVisibility.INTERNAL,
        "stages are configured per name (stages.<name>.<field>); the "
        "collection cannot be created or replaced wholesale from a "
        "user-facing source",
    ),
    (
        "entry_stage",
        PathVisibility.INTERNAL,
        "the entry stage is part of the stage topology and is declared by "
        "the model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.name",
        PathVisibility.INTERNAL,
        "a stage name is its address; the config class declares it and every "
        "user-facing source addresses the stage by that name",
    ),
    (
        "stages.*.factory_path",
        PathVisibility.INTERNAL,
        "the stage factory is part of the stage topology and is declared by "
        "the model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.next",
        PathVisibility.INTERNAL,
        "stage routing is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.terminal",
        PathVisibility.INTERNAL,
        "stage routing is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.route_fn",
        PathVisibility.INTERNAL,
        "stage routing is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.stream_to",
        PathVisibility.INTERNAL,
        "stream wiring is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.stream_done_to_fn",
        PathVisibility.INTERNAL,
        "stream wiring is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.wait_for",
        PathVisibility.INTERNAL,
        "fan-in wiring is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.wait_for_fn",
        PathVisibility.INTERNAL,
        "fan-in wiring is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.merge_fn",
        PathVisibility.INTERNAL,
        "fan-in wiring is part of the stage topology and is declared by the "
        "model's config class, not set from a user-facing source",
    ),
    (
        "stages.*.project_payload",
        PathVisibility.INTERNAL,
        "payload projection is part of the stage topology and is declared by "
        "the model's config class, not set from a user-facing source",
    ),
)


# Removed configuration surfaces, kept only as error guidance: writing one of
# these names fails like any other unknown field, but the message says where
# the setting lives now instead of offering a fuzzy-match suggestion.
_REMOVED_FIELD_GUIDANCE: dict[str, str] = {
    "factory_args": (
        "factory_args was removed: user-tunable knobs live in the stage's "
        "factory./engine. groups, and constructor wiring in "
        "PipelineConfig.stage_factory_kwargs()"
    ),
    "runtime": (
        "the runtime group was removed: resources.total_gpu_memory_fraction "
        "is now the stage-level gpu_memory_fraction, sglang_server_args is "
        "now engine.*, memory.kv_cache_bytes is now engine.kv_cache_bytes, "
        "memory.total_reserve_bytes and enforce_total_reserve are stage-level "
        "fields, and the remaining fields moved to factory.*"
    ),
    "runtime_arg_map": (
        "runtime_arg_map was removed: stage factories take canonical "
        "parameter names, so no per-stage translation table exists"
    ),
    "runtime_overrides": (
        "runtime_overrides was removed: write stages.<name>.engine.* or "
        "stages.<name>.factory.* instead"
    ),
    "stage_overrides": (
        "stage_overrides was removed: per-stage settings are written under "
        "the stages: mapping (stages.<name>.<field>)"
    ),
}


class ConfigPathError(ValueError):
    """A dotted path could not be compiled against the schema."""

    def __init__(
        self,
        message: str,
        *,
        raw: str,
        resolved_prefix: str = "",
        suggestions: tuple[str, ...] = (),
    ) -> None:
        self.raw = raw
        self.resolved_prefix = resolved_prefix
        self.suggestions = tuple(suggestions)
        if self.suggestions:
            message = f"{message}\n  did you mean: " + ", ".join(self.suggestions)
        super().__init__(message)


@dataclass(frozen=True)
class Segment:
    """One compiled segment of a canonical path."""

    raw: str
    kind: SegmentKind
    annotation: Any
    """Declared type of the value *at* this segment."""

    container: Any
    """Declared type the segment was resolved against."""


@dataclass(frozen=True)
class ConfigPath:
    """A dotted path compiled against a :class:`PipelineConfig` schema."""

    raw: str
    root: type[BaseModel]
    segments: tuple[Segment, ...]

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------

    @classmethod
    def parse(
        cls,
        raw: str,
        root: type[BaseModel] = PipelineConfig,
    ) -> "ConfigPath":
        """Compile ``raw`` against ``root``, or raise :class:`ConfigPathError`."""
        if not isinstance(raw, str) or not raw.strip():
            raise ConfigPathError(
                "Config path must be a non-empty string", raw=str(raw)
            )

        parts = raw.split(".")
        if any(not part for part in parts):
            raise ConfigPathError(f"Config path {raw!r} has an empty segment", raw=raw)

        segments: list[Segment] = []
        current: Any = root
        for index, part in enumerate(parts):
            prefix = ".".join(parts[:index])
            segment = _descend(current, part, raw=raw, prefix=prefix)
            if (
                segment.kind is SegmentKind.NAMED_ITEM
                and index == 1
                and parts[0] == "stages"
                and isinstance(root, type)
                and issubclass(root, PipelineConfig)
            ):
                # Per-stage-type compilation: the root config class declares
                # which StageConfig subclass each stage name uses, so the
                # remaining segments resolve against that stage's own fields
                # (engine marker, model-specific factory.* fields) rather than
                # the generic base type.
                segment = Segment(
                    raw=segment.raw,
                    kind=segment.kind,
                    annotation=root.stage_config_cls(part),
                    container=segment.container,
                )
            segments.append(segment)
            current = segments[-1].annotation

        return cls(raw=raw, root=root, segments=tuple(segments))

    # ------------------------------------------------------------------
    # shape
    # ------------------------------------------------------------------

    @property
    def parts(self) -> tuple[str, ...]:
        return tuple(segment.raw for segment in self.segments)

    @property
    def value_type(self) -> Any:
        return self.segments[-1].annotation

    @property
    def is_leaf(self) -> bool:
        """True when nothing can be addressed below this path.

        Lists of scalars are leaves: they are replaced as a whole, never
        indexed into.
        """
        return not _is_traversable(self.value_type)

    @property
    def stage_name(self) -> str | None:
        """Name of the stage this path lives in, if any."""
        for index, segment in enumerate(self.segments):
            if segment.kind is SegmentKind.NAMED_ITEM and index > 0:
                if self.segments[index - 1].raw == "stages":
                    return segment.raw
        return None

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.raw

    # ------------------------------------------------------------------
    # visibility
    # ------------------------------------------------------------------

    @property
    def visibility(self) -> PathVisibility:
        return self._visibility_rule[0]

    @property
    def visibility_reason(self) -> str:
        return self._visibility_rule[1]

    @property
    def _visibility_rule(self) -> tuple[PathVisibility, str]:
        generic = _generic_form(self.segments)
        for pattern, visibility, reason in _VISIBILITY_RULES:
            if _pattern_matches(pattern, generic):
                return visibility, reason
        return PathVisibility.PUBLIC, ""

    def is_public(self) -> bool:
        """True when a user-facing source may write this path."""
        return self.visibility is PathVisibility.PUBLIC

    def require_writable(self) -> None:
        """Raise unless this path may be written by a user-facing source."""
        if self.is_public():
            return
        raise ConfigPathError(
            f"Config path {self.raw!r} is {self.visibility.value} and cannot be "
            f"set directly: {self.visibility_reason}",
            raw=self.raw,
        )

    # ------------------------------------------------------------------
    # values
    # ------------------------------------------------------------------

    def coerce(self, value: Any) -> Any:
        """Convert a raw (usually textual) value into this path's declared type.

        Numeric conversions only go the lossless way: an int fits a float
        field and 0/1 fit a bool field, but a bool never fits a numeric field
        (``true`` would silently become 1) and a float never fits an int
        field (32.0 would be truncated into shape). A YAML scalar arrives
        here natively typed; a CLI flag arrives as text and is parsed to a
        scalar first, so both spellings answer to the same rule.
        """
        annotation = self.value_type
        if not isinstance(value, str):
            if annotation is not Any and annotation is not None:
                self._refuse_lossy_scalar(value, annotation)
            return value
        # ``none`` is a sentinel, not a string, and has been one since the first
        # dotted-CLI implementation (``ConfigManager._convert_scalar``). It has
        # to be honoured before the type adapter runs: pydantic's smart union
        # mode is happy to satisfy ``str | None`` with the *string* ``"none"``,
        # which would silently store the word where the user meant to clear the
        # field. Applied whatever the declared type is, exactly as V1 did -- on
        # a field that cannot be None the assignment then fails validation,
        # which is also what V1 produced.
        if value.lower() == _NONE_SENTINEL:
            return None
        if annotation is Any or annotation is None:
            return coerce_scalar_text(value)
        scalar = coerce_scalar_text(value)
        if not isinstance(scalar, str) and _annotation_scalars(annotation) & {
            int,
            float,
            bool,
        }:
            # The text reads as a number or boolean and the field is numeric
            # or boolean: judge the parsed scalar, so "true" against an int
            # field is a boolean (refused) rather than JSON the adapter would
            # lax-coerce to 1.
            self._refuse_lossy_scalar(scalar, annotation)
            try:
                return TypeAdapter(annotation).validate_python(scalar)
            except Exception:
                # Out-of-type in a way the schema explains better (rebuild
                # validation names the path and the rule).
                return scalar
        try:
            return TypeAdapter(annotation).validate_python(value)
        except Exception:
            pass
        try:
            return TypeAdapter(annotation).validate_json(value)
        except Exception:
            return coerce_scalar_text(value)

    def _refuse_lossy_scalar(self, value: Any, annotation: Any) -> None:
        allowed = _annotation_scalars(annotation)
        if bool in allowed:
            return
        if isinstance(value, bool):
            raise ConfigPathError(
                f"{self.raw} expects {_type_name(annotation)}, got a boolean",
                raw=self.raw,
            )
        if isinstance(value, float) and int in allowed and float not in allowed:
            raise ConfigPathError(
                f"{self.raw} expects an integer, got a float ({value!r})",
                raw=self.raw,
            )

    def read(self, source: BaseModel | dict[str, Any]) -> Any:
        """Read the value at this path from a config instance or a dumped dict."""
        current: Any = source
        if isinstance(current, BaseModel):
            current = current.model_dump()
        for segment in self.segments:
            current = _read_segment(current, segment, path=self.raw)
        return current

    def write(self, data: dict[str, Any], value: Any) -> None:
        """Assign ``value`` at this path inside a dumped config dict, in place.

        ``data`` is expected to be the output of ``PipelineConfig.model_dump()``.
        Missing containers are created only where the schema allows a free
        mapping key; a missing stage is an error, never a silent insert.
        """
        current: Any = data
        for segment in self.segments[:-1]:
            current = _read_segment(
                current, segment, path=self.raw, create_missing=True
            )
        last = self.segments[-1]
        if last.kind is SegmentKind.NAMED_ITEM:
            index = _named_index(current, last.raw, path=self.raw)
            current[index] = value
            return
        if not isinstance(current, dict):
            raise ConfigPathError(
                f"Cannot set {self.raw!r}: {_join(self.parts[:-1])} is "
                f"{type(current).__name__}, not a mapping",
                raw=self.raw,
            )
        current[last.raw] = value


# ----------------------------------------------------------------------
# schema walking
# ----------------------------------------------------------------------


def _descend(container: Any, part: str, *, raw: str, prefix: str) -> Segment:
    """Resolve one segment against ``container``'s declared type."""
    core = _unwrap_optional(container)

    if _is_model(core):
        fields = core.model_fields
        if part == "engine" and _is_non_engine_stage(core):
            raise ConfigPathError(
                f"{_join_prefix(prefix)} is not an engine stage: the engine "
                "block only exists on stages whose factory drives an SGLang "
                "engine, so there is nothing for engine settings to reach here",
                raw=raw,
                resolved_prefix=prefix,
            )
        if part == "audio_chunking" and _is_chunkless_pipeline(core):
            raise ConfigPathError(
                f"{core.__name__} does not support audio chunking: the "
                "audio_chunking policy only exists on pipelines whose model "
                "declares allow_audio_chunking, so there is nothing for "
                "these settings to reach here",
                raw=raw,
                resolved_prefix=prefix,
            )
        if part in fields:
            annotation = fields[part].annotation
            if fields[part].metadata:
                # Field constraints (ge/gt/Literal/min_length) are declared
                # statically on the schema; carrying them into the segment
                # lets coerce's type adapter enforce them at conversion, the
                # same rule the rebuild enforces at resolution.
                params = (annotation, *fields[part].metadata)
                annotation = typing.Annotated[params]
            return Segment(
                raw=part,
                kind=SegmentKind.FIELD,
                annotation=annotation,
                container=core,
            )
        if core.model_config.get("extra") == "allow":
            # An extra-allow model accepts keys beyond its declared fields
            # (the free ServerArgs tail of ``engine``). No schema information
            # remains below such a key.
            return Segment(
                raw=part,
                kind=SegmentKind.FREEFORM,
                annotation=Any,
                container=core,
            )
        guidance = _REMOVED_FIELD_GUIDANCE.get(part)
        if guidance is not None and issubclass(core, (PipelineConfig, StageConfig)):
            raise ConfigPathError(
                f"{core.__name__} has no field {part!r}"
                + (f" at {prefix!r}" if prefix else "")
                + f": {guidance}",
                raw=raw,
                resolved_prefix=prefix,
            )
        raise ConfigPathError(
            f"{core.__name__} has no field {part!r}"
            + (f" at {prefix!r}" if prefix else ""),
            raw=raw,
            resolved_prefix=prefix,
            suggestions=_suggest(part, sorted(fields), prefix),
        )

    item_type = _named_collection_item(core)
    if item_type is not None:
        if part.isdigit():
            raise ConfigPathError(
                f"{_join_prefix(prefix)} is addressed by name, not by index; "
                f"use e.g. {_join_prefix(prefix)}.thinker instead of {part!r}",
                raw=raw,
                resolved_prefix=prefix,
            )
        # Stage names come from the document, not the schema, so any identifier
        # is structurally valid here; existence is checked when the path is
        # bound to a concrete config.
        return Segment(
            raw=part,
            kind=SegmentKind.NAMED_ITEM,
            annotation=item_type,
            container=core,
        )

    value_type = _mapping_value_type(core)
    if value_type is not None:
        return Segment(
            raw=part,
            kind=SegmentKind.MAPPING_KEY,
            annotation=value_type,
            container=core,
        )

    if core is Any:
        return Segment(
            raw=part, kind=SegmentKind.FREEFORM, annotation=Any, container=Any
        )

    if _is_plain_sequence(core):
        raise ConfigPathError(
            f"{_join_prefix(prefix)} is a list value and is replaced as a whole; "
            f"it cannot be indexed by {part!r}",
            raw=raw,
            resolved_prefix=prefix,
        )

    raise ConfigPathError(
        f"{_join_prefix(prefix)} is a leaf of type {_type_name(core)}; "
        f"nothing can be addressed below it (got {part!r})",
        raw=raw,
        resolved_prefix=prefix,
    )


def _annotation_scalars(annotation: Any) -> set[type]:
    """The scalar base types a declared annotation admits, unions flattened."""
    out: set[type] = set()
    stack = [annotation]
    while stack:
        current = stack.pop()
        origin = get_origin(current)
        if origin is typing.Annotated:
            stack.append(get_args(current)[0])
            continue
        if origin is typing.Union or origin is types.UnionType:
            stack.extend(get_args(current))
            continue
        if current is type(None):
            continue
        target = origin or current
        if isinstance(target, type):
            out.add(target)
    return out


def _unwrap_optional(annotation: Any) -> Any:
    """Strip ``Annotated`` metadata and ``| None`` from a declared type.

    Unions of real types are left untouched. ``Annotated`` shows up through
    ``SerializeAsAny[StageConfig]`` on the stages list; the wrapper only
    changes serialization, not what the path addresses.
    """
    if get_origin(annotation) is typing.Annotated:
        annotation = get_args(annotation)[0]
    origin = get_origin(annotation)
    if origin is typing.Union or origin is types.UnionType:
        args = [arg for arg in get_args(annotation) if arg is not type(None)]
        if len(args) == 1:
            return args[0]
        return annotation
    return annotation


def _is_model(annotation: Any) -> bool:
    # ``get_origin`` guards the ``issubclass`` call: on Python 3.10 a subscripted
    # generic such as ``list[StageConfig]`` *is* an instance of ``type``, so
    # ``issubclass`` receives a non-class and raises ``TypeError``. 3.11 changed
    # that, which is why the omission is invisible on a newer interpreter.
    return (
        isinstance(annotation, type)
        and get_origin(annotation) is None
        and issubclass(annotation, BaseModel)
    )


def _named_collection_item(annotation: Any) -> type[BaseModel] | None:
    """Return the item type when ``annotation`` is a name-keyed model list."""
    if get_origin(annotation) not in (list, tuple):
        return None
    args = get_args(annotation)
    if not args:
        return None
    item = _unwrap_optional(args[0])
    if _is_model(item) and "name" in item.model_fields:
        return item
    return None


def _mapping_value_type(annotation: Any) -> Any | None:
    if get_origin(annotation) is not dict:
        return None
    args = get_args(annotation)
    return args[1] if len(args) == 2 else Any


def _is_plain_sequence(annotation: Any) -> bool:
    return get_origin(annotation) in (list, tuple, set, frozenset)


def _is_non_engine_stage(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and get_origin(annotation) is None
        and issubclass(annotation, StageConfig)
        and not annotation.engine_stage
    )


def _is_chunkless_pipeline(annotation: Any) -> bool:
    return (
        isinstance(annotation, type)
        and get_origin(annotation) is None
        and issubclass(annotation, PipelineConfig)
        and not annotation.allow_audio_chunking
    )


def _is_traversable(annotation: Any) -> bool:
    core = _unwrap_optional(annotation)
    if _is_model(core):
        return True
    if _named_collection_item(core) is not None:
        return True
    if _mapping_value_type(core) is not None:
        return True
    return core is Any


def _type_name(annotation: Any) -> str:
    # Constraint metadata is not part of the name a user reads.
    if get_origin(annotation) is typing.Annotated:
        return _type_name(get_args(annotation)[0])
    # Same 3.10 caveat as _is_model: ``list[int]`` passes ``isinstance(_, type)``
    # there and would render as a bare ``list``, losing its parameter.
    if isinstance(annotation, type) and get_origin(annotation) is None:
        return annotation.__name__
    return str(annotation).replace("typing.", "")


# ----------------------------------------------------------------------
# value access on dumped dicts
# ----------------------------------------------------------------------


def _read_segment(
    current: Any,
    segment: Segment,
    *,
    path: str,
    create_missing: bool = False,
) -> Any:
    if segment.kind is SegmentKind.NAMED_ITEM:
        if not isinstance(current, list):
            raise ConfigPathError(
                f"Cannot resolve {path!r}: expected a list of named entries, "
                f"got {type(current).__name__}",
                raw=path,
            )
        return current[_named_index(current, segment.raw, path=path)]

    if not isinstance(current, dict):
        raise ConfigPathError(
            f"Cannot resolve {path!r}: expected a mapping at {segment.raw!r}, "
            f"got {type(current).__name__}",
            raw=path,
        )

    if segment.raw not in current or current[segment.raw] is None:
        if not create_missing:
            if segment.raw in current:
                return current[segment.raw]
            raise ConfigPathError(
                f"Cannot resolve {path!r}: {segment.raw!r} is not present",
                raw=path,
                suggestions=_suggest(segment.raw, sorted(map(str, current)), ""),
            )
        current[segment.raw] = {}
    return current[segment.raw]


def _named_index(items: list[Any], name: str, *, path: str) -> int:
    for index, item in enumerate(items):
        if isinstance(item, dict) and item.get("name") == name:
            return index
    available = [
        str(item["name"]) for item in items if isinstance(item, dict) and "name" in item
    ]
    raise ConfigPathError(
        f"Cannot resolve {path!r}: no entry named {name!r}",
        raw=path,
        suggestions=_suggest(name, available, ""),
    )


# ----------------------------------------------------------------------
# patterns, suggestions, schema enumeration
# ----------------------------------------------------------------------


def _generic_form(segments: tuple[Segment, ...]) -> tuple[str, ...]:
    """Replace document-defined names with ``*`` so rules stay model-agnostic."""
    return tuple(
        "*" if segment.kind is SegmentKind.NAMED_ITEM else segment.raw
        for segment in segments
    )


def _pattern_matches(pattern: str, parts: tuple[str, ...]) -> bool:
    pattern_parts = pattern.split(".")
    if pattern_parts and pattern_parts[-1] == "**":
        head = pattern_parts[:-1]
        if len(parts) < len(head):
            return False
        return all(p in ("*", q) for p, q in zip(head, parts))
    if len(pattern_parts) != len(parts):
        return False
    return all(p in ("*", q) for p, q in zip(pattern_parts, parts))


def _suggest(word: str, candidates: list[str], prefix: str) -> tuple[str, ...]:
    if not candidates:
        return ()
    close = difflib.get_close_matches(word, candidates, n=3, cutoff=0.5)
    chosen = close or candidates[:5]
    return tuple(f"{prefix}.{name}" if prefix else name for name in chosen)


def _join(parts: tuple[str, ...]) -> str:
    return ".".join(parts)


def _join_prefix(prefix: str) -> str:
    return prefix or "<root>"


def coerce_scalar_text(value: Any) -> Any:
    """Best-effort scalar parsing for untyped (``Any``) positions.

    Mirrors the historical behaviour of ``ConfigManager._convert_scalar`` so
    free-form escape hatches keep parsing the same way.
    """
    if not isinstance(value, str):
        return value

    lowered = value.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered == _NONE_SENTINEL:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def iter_schema_paths(
    root: type[BaseModel] = PipelineConfig,
    *,
    include_non_public: bool = False,
) -> list[str]:
    """Enumerate every addressable path in the schema.

    Name-keyed collections and free mappings are rendered with a ``*``
    placeholder because their keys come from the document, not the schema.
    """
    out: list[str] = []

    def walk(annotation: Any, prefix: tuple[str, ...], depth: int) -> None:
        if depth > _MAX_SCHEMA_DEPTH:
            return
        core = _unwrap_optional(annotation)

        if _is_model(core):
            for name, field in core.model_fields.items():
                if name == "audio_chunking" and _is_chunkless_pipeline(core):
                    continue
                child = prefix + (name,)
                _emit(child)
                walk(field.annotation, child, depth + 1)
            return

        item = _named_collection_item(core)
        if item is not None:
            child = prefix + ("*",)
            walk(item, child, depth + 1)
            return

        value_type = _mapping_value_type(core)
        if value_type is not None and value_type is not Any:
            child = prefix + ("*",)
            _emit(child)
            walk(value_type, child, depth + 1)
            return

    def _emit(parts: tuple[str, ...]) -> None:
        if not include_non_public:
            for pattern, visibility, _ in _VISIBILITY_RULES:
                if visibility is not PathVisibility.PUBLIC and _pattern_matches(
                    pattern, parts
                ):
                    return
        out.append(".".join(parts))

    walk(root, (), 0)
    return out
