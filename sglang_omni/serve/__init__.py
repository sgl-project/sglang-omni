# SPDX-License-Identifier: Apache-2.0
"""HTTP serving utilities."""

from sglang_omni.serve.openai_api import create_app

__all__ = ["create_app", "launch_server"]


def __getattr__(name: str):
    if name == "launch_server":
        from sglang_omni.serve.launcher import launch_server

        return launch_server
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
