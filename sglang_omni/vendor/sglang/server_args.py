"""Vendor wrapper for sglang.srt.server_args.

Centralize third-party imports and apply optional monkey patches here.
"""

from __future__ import annotations

from sglang.srt.server_args import get_global_server_args

# Optional monkey patches (example):
# def get_global_server_args():
#     return None

__all__ = ["get_global_server_args"]
