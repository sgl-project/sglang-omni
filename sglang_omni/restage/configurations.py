"""Typed configuration dimensions for Restage placement searches."""

from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from itertools import product
from typing import Any

from sglang_omni.config.patch import ConfigPatchSet
from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig
from sglang_omni.config.sources import patches_from_dotted_cli


@dataclass(frozen=True)
class ConfigurationChoice:
    selections: dict[str, Mapping[str, Any]]
    config: PipelineConfig | None
    rejection: str | None = None


def enumerate_configurations(
    config: PipelineConfig,
    dimensions: Mapping[str, Sequence[Mapping[str, Any]]],
) -> Iterator[ConfigurationChoice]:
    """Combine explicit choices, each a group of serving CLI overrides.

    Related changes such as TP degree and its initial device list belong in
    one choice. Dimensions can vary process membership, memory budgets, SM
    environment settings or model knobs supported by the config schema.
    Schema acceptance does not establish model or hardware support.
    """
    if any(not choices for choices in dimensions.values()):
        raise ValueError("Each search dimension needs at least one choice")
    names = tuple(dimensions)
    for values in product(*dimensions.values()):
        selections = dict(zip(names, values))
        patches = ConfigPatchSet()
        try:
            for name, overrides in selections.items():
                patches = patches.merge(
                    patches_from_dotted_cli(
                        overrides, config, origin=f"Restage dimension {name}"
                    )
                )
            resolved = ConfigResolver(config).resolve(patches).config
        except ValueError as exc:
            yield ConfigurationChoice(selections, None, str(exc))
        else:
            yield ConfigurationChoice(selections, resolved)
