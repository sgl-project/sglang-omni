from __future__ import annotations

import typer

from .config import config_app
from .serve import serve as _serve

app = typer.Typer()


def _router(ctx: typer.Context) -> None:
    from sglang_omni_router.serve import main

    main(list(ctx.args))


# Register the subcommands.
app.add_typer(config_app, name="config")
app.command(
    "serve", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)(_serve)
app.command(
    "router",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(_router)

__all__ = ["app"]
