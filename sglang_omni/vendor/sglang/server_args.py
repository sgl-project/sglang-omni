"""Small compatibility helpers around SGLang ``ServerArgs``."""

from __future__ import annotations

from typing import Any

from sglang.srt.server_args import get_global_server_args


def override_server_args(server_args: Any, source: str, **fields: Any) -> None:
    """Apply an audited post-resolution ServerArgs mutation.

    Route every Omni mutation through override(), the single 0.5.16 mutation
    entry point, with an Omni source label; strict mutation checking stays
    usable.
    """
    server_args.override(source, **fields)


__all__ = ["get_global_server_args", "override_server_args"]
