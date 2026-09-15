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
    """Report accelerator mapping, runtime versions, and installed backends."""

    report = collect_gpu_diagnostics()
    if json_output:
        typer.echo(json.dumps(report, indent=2, sort_keys=True))
    else:
        typer.echo(render_gpu_diagnostics(report))

    if strict:
        environment = report["environment"]
        accelerator_available = environment.get(
            "accelerator_available", environment.get("cuda_available", False)
        )
        if (
            report["warnings"]
            or not accelerator_available
            or not environment.get("logical_device_count", 0)
        ):
            raise typer.Exit(code=1)
