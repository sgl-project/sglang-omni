import sys
from contextlib import redirect_stdout

from typer import Typer

with redirect_stdout(sys.stderr):
    from sglang_omni.cli.check_gpu import check_gpu
    from sglang_omni.cli.config import config_app
    from sglang_omni.cli.serve import ServeCommand
    from sglang_omni.cli.serve import serve as _serve

app = Typer()

# Register the subcommands.
app.add_typer(config_app, name="config")
app.command("check-gpu")(check_gpu)
app.command(
    "serve",
    cls=ServeCommand,
    context_settings={
        "allow_extra_args": True,
        "help_option_names": ["-h", "--help"],
        "ignore_unknown_options": True,
    },
)(_serve)

__all__ = ["app"]
