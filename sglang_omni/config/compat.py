# SPDX-License-Identifier: Apache-2.0
"""Translation of dotted CLI keys into canonical configuration paths.

On the command line the ``stages.`` prefix is implied: a per-stage flag
starts from the stage name (``--thinker.engine.mem_fraction_static 0.6``),
mirroring the YAML ``stages:`` mapping where the stage name is also the
outermost key the user writes. This module owns that rewrite. Everything downstream sees canonical paths
(``stages.thinker.engine.mem_fraction_static``) and nothing else.

The canonical surfaces themselves -- dotted CLI flags and the YAML
``stages:`` mapping -- live in :mod:`sglang_omni.config.sources`.
"""

from __future__ import annotations

import difflib

from sglang_omni.config.path import ConfigPathError
from sglang_omni.config.schema import PipelineConfig

__all__ = ["canonicalize_dotted_key"]


def canonicalize_dotted_key(key: str, config: PipelineConfig) -> str:
    """Rewrite a dotted CLI key into canonical form against ``config``.

    The first segment decides the rewrite: a stage name of this pipeline
    gets the implied ``stages.``
    prefix, a top-level field passes through, and anything else is refused
    with both kinds of candidates. A path that already starts with
    ``stages.`` is accepted as canonical, so pasted canonical paths (e.g.
    into ``config explain``) keep working; the CLI flag surface refuses that
    prefix separately.
    """
    parts = key.split(".")
    head = parts[0]
    stage_names = [stage.name for stage in config.stages]
    top_fields = list(type(config).model_fields)

    if head in stage_names and len(parts) > 1:
        if head in top_fields:
            raise ConfigPathError(
                f"{key!r} is ambiguous: {head!r} is both a stage name and a "
                "top-level configuration field. Rename the stage; the CLI "
                "spelling cannot distinguish the two",
                raw=key,
            )
        parts = ["stages", *parts]
    elif head in stage_names:
        raise ConfigPathError(
            f"{key!r} names a whole stage; write one field below it, e.g. "
            f"--{head}.tp_size",
            raw=key,
        )
    elif head not in top_fields and head != "stages":
        candidates = sorted(set(stage_names) | set(top_fields))
        close = difflib.get_close_matches(head, candidates, n=3, cutoff=0.5)
        raise ConfigPathError(
            f"{key!r} does not start from a stage name or a top-level "
            f"configuration field of {type(config).__name__}\n"
            f"  stages of this pipeline: {', '.join(stage_names)}",
            raw=key,
            suggestions=tuple(close or candidates[:5]),
        )

    return ".".join(parts)
