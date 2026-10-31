"""Command-line entry point for SGLang-Omni."""

from typing import Annotated

from typer import Exit, Option, Typer, echo

from sglang_omni import __version__

from .check_gpu import check_gpu
from .config import config_app
from .serve import serve as _serve

app = Typer()


def print_version(is_version_requested: bool) -> bool:
    """Print the package version and stop command dispatch when requested."""
    if is_version_requested:
        echo(f"sgl-omni {__version__}")
        raise Exit()
    else:
        return is_version_requested


@app.callback()
def main(
    is_version_requested: Annotated[
        bool,
        Option(
            "--version",
            callback=print_version,
            is_eager=True,
            help="Show the version and exit.",
        ),
    ] = False,
) -> None:
    """Serve and inspect SGLang-Omni pipelines."""


# Register the subcommands.
app.add_typer(config_app, name="config")
app.command("check-gpu")(check_gpu)
app.command(
    "serve", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)(_serve)

__all__ = ["app"]
