"""Small compatibility helpers around SGLang ``ServerArgs``."""

from __future__ import annotations

from typing import Any


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Apply an audited ServerArgs mutation at the right lifecycle stage.

    A record that is not published yet takes the change as a declaration
    through declare_resolution: the field keeps the caller's input, and the
    declaration is what resolution_result, the resolved view and the bags
    projected at publish answer with. The published record is read-only and its
    values live on the config bags, so the mutation goes to get_context().override.
    """
    from sglang.srt.runtime_context import get_context

    context = get_context()
    try:
        published_server_args = context.server_args
    except ValueError:
        published_server_args = None

    if published_server_args is server_args:
        context.override(source, **fields)
        return

    from sglang.srt.arg_groups.overrides import declare_resolution

    declare_resolution(server_args, source, **fields)


__all__ = ["override_server_args"]
