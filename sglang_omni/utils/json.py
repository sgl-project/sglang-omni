# SPDX-License-Identifier: Apache-2.0
"""Types for values produced by JSON decoding and serialization."""

from typing import TypeAlias

JsonValue: TypeAlias = (
    str | int | float | bool | None | list["JsonValue"] | dict[str, "JsonValue"]
)
