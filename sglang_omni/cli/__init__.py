from typer import Typer

from .check_gpu import check_gpu
from .config import config_app
from .serve import serve as _serve

app = Typer()

# Register the subcommands.
app.add_typer(config_app, name="config")
app.command("check-gpu")(check_gpu)
app.command(
    "serve", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)(_serve)

__all__ = ["app"]
