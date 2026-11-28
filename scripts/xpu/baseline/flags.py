# SPDX-License-Identifier: Apache-2.0
"""Serve flags from several sources merged into one command line, a later source replacing an earlier one."""

from __future__ import annotations


def flag_groups(arguments: tuple[str, ...] | list[str]) -> dict[str, list[str]]:
    """Each --flag with the values that follow it, keyed by its name, in order."""
    groups: dict[str, list[str]] = {}
    name = ""
    for token in arguments:
        if token.startswith("--"):
            name = token.partition("=")[0]
            groups[name] = [token]
        elif name:
            groups[name].append(token)
        else:
            raise ValueError(f"{token!r} comes before any --flag in {list(arguments)}")
    return groups


def merged_flags(*sources: tuple[str, ...] | list[str]) -> list[str]:
    """One command line from the sources in order of precedence, so the last to name a flag sets it."""
    merged: dict[str, list[str]] = {}
    for source in sources:
        merged.update(flag_groups(source))
    return [token for group in merged.values() for token in group]
