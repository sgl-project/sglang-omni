# SPDX-License-Identifier: Apache-2.0

"""SGLang ``serve`` backend adapter for SGLang-Omni.

The entry-point factory is intentionally lightweight: discovering installed
backends must not import the Omni runtime or initialize CUDA. The existing
Typer command remains the single owner of Omni's command-line interface.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang.cli.serve_backends import ServeRequest


def run(request: ServeRequest) -> None:
    """Forward an SGLang serve request through the existing Omni CLI."""
    import typer

    from sglang_omni.cli import app

    command = typer.main.get_command(app)
    command.main(
        args=["serve", *request.argv],
        prog_name="sglang",
        standalone_mode=True,
    )


def create_backend():
    """Create the versioned backend descriptor consumed by SGLang core."""
    from sglang.cli.serve_backends import SERVE_BACKEND_API_VERSION, ServeBackend

    return ServeBackend(
        api_version=SERVE_BACKEND_API_VERSION,
        run=run,
        # Omni supports config-only launches where model_path lives in YAML.
        requires_model_path=False,
    )


__all__ = ["create_backend", "run"]
