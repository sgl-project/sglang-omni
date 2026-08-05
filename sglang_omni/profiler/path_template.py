# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from string import Formatter

_SUPPORTED_FIELDS = frozenset({"run_id", "stage"})


def validate_trace_path_template(template: str) -> None:
    """Validate the supported trace path replacement-field syntax."""
    for _, field_name, format_spec, conversion in Formatter().parse(template):
        if field_name is None:
            continue
        if field_name not in _SUPPORTED_FIELDS:
            raise ValueError(
                f"unsupported replacement field {field_name!r}; "
                "only {run_id} and {stage} are supported"
            )
        if conversion is not None:
            raise ValueError(
                f"conversion !{conversion} is not supported for {field_name!r}"
            )
        if format_spec:
            raise ValueError(
                f"format specifier {format_spec!r} is not supported for "
                f"{field_name!r}"
            )


def format_trace_path_template(template: str, *, run_id: str, stage: str) -> str:
    """Validate and render a trace path template."""
    validate_trace_path_template(template)
    return template.format(run_id=run_id, stage=stage)
