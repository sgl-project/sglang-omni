# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
from typing import Annotated

import typer

from sglang_omni.diagnostics.gpu import collect_gpu_diagnostics, render_gpu_diagnostics


def check_gpu(
    json_output: Annotated[
        bool,
        typer.Option(
            "--json",
            help="Emit the diagnostic report as machine-readable JSON.",
        ),
    ] = False,
    strict: Annotated[
        bool,
        typer.Option(
            "--strict",
            help="Exit nonzero when warnings are present or no accelerator is visible.",
        ),
    ] = False,
) -> None:
    """Report GPU mapping, runtime versions, and installed backends."""

    report = collect_gpu_diagnostics()
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(render_gpu_diagnostics(report))

    if strict:
        environment = report["environment"]
        accelerator = environment.get("accelerator")
        no_accelerator = (
            accelerator == "cpu"
            if accelerator is not None
            else not environment["cuda_available"]
        )
        if report["warnings"] or no_accelerator or not report["gpus"]:
            raise typer.Exit(code=1)
