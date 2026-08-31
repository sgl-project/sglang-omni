"""Small compatibility helpers around SGLang ``ServerArgs``."""

from __future__ import annotations

from typing import Any


def get_global_server_args():
    """Return SGLang's process-global server args through a lazy import."""
    from sglang.srt.server_args import get_global_server_args as _get_global_server_args

    return _get_global_server_args()


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Apply an audited ServerArgs mutation at the right lifecycle stage.

    A record that is not published yet is resolved in place through
    declare_late_resolution, so every holder of the instance sees the value.
    The published record is read-only and its resolved values live on the
    config bags, so the mutation goes to get_context().override.
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

    from sglang.srt.arg_groups.overrides import declare_late_resolution

    declare_late_resolution(server_args, source, **fields)


__all__ = ["get_global_server_args", "override_server_args"]
