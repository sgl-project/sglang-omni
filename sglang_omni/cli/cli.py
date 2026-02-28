from typer import Typer

from .config import config_app
from .serve import serve

app = Typer()

# register the subcommands
app.add_typer(config_app, name="config")
app.command(
    "serve", context_settings={"allow_extra_args": True, "ignore_unknown_options": True}
)(serve)


if __name__ == "__main__":
    app()
