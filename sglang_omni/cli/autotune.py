"""Restage: calibrate, rank residency shapes, and measure the exported ones."""

import asyncio
import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

import typer

from sglang_omni.cli.config import _resolve_sources
from sglang_omni.restage.calibration import Constants
from sglang_omni.restage.plan import SearchSpace, write_plan

autotune_app = typer.Typer(
    help="Calibrate a pipeline, rank its residency shapes, and measure the top ones."
)


def _load_config(config: Path):
    return _resolve_sources(
        model_path=None,
        config_file=str(config),
        text_only=False,
        mem_fraction_static=None,
        argv=[],
    ).resolved.config


@autotune_app.command()
def calibrate(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Serving pipeline YAML.")
    ],
    spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Workload JSON: task, samples, stage geometry and probe settings.",
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="New directory for probes and constants.")
    ],
):
    """Run the two probes on the caller's GPU and write constants.json."""
    from benchmarks.benchmarker.restage_probe import calibrate as run_calibration
    from benchmarks.benchmarker.restage_probe import load_calibration_spec

    try:
        options = load_calibration_spec(spec)
        constants = asyncio.run(
            run_calibration(config_path=config, destination=output, **options)
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(constants.model_dump_json(indent=2))


@autotune_app.command()
def plan(
    config: Annotated[
        Path, typer.Option(exists=True, dir_okay=False, help="Serving pipeline YAML.")
    ],
    constants: Annotated[
        Path,
        typer.Option(
            exists=True, dir_okay=False, help="constants.json from calibrate."
        ),
    ],
    search_space: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="JSON device budget and optional configuration dimensions.",
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="New directory for the ranking and YAML.")
    ],
    top: Annotated[
        int, typer.Option(min=1, help="Candidates exported for measurement.")
    ] = 3,
):
    """Rank every residency shape by predicted throughput; export the top ones."""
    try:
        space = SearchSpace.model_validate_json(
            search_space.read_text(encoding="utf-8")
        )
        summary = write_plan(
            _load_config(config), space, Constants.load(constants), output, top=top
        )
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc)) from exc
    typer.echo(
        f"Ranked {summary['candidates']} shapes, rejected {summary['rejected']}; "
        f"exported {len(summary['exported'])} plus the baseline to {output}. "
        "Predicted only; run measures them."
    )


@autotune_app.command()
def run(
    spec: Annotated[
        Path,
        typer.Option(
            exists=True,
            dir_okay=False,
            help="Campaign JSON naming a plan_directory (or configs), workload and SLO.",
        ),
    ],
    output: Annotated[
        Path, typer.Option(help="Campaign results directory; new unless resuming.")
    ],
    resume: Annotated[
        bool,
        typer.Option(
            help="Resume using caller-verified run_identity and matching inputs."
        ),
    ] = False,
):
    """Measure the exported candidates on caller-allocated GPUs and select a winner."""
    from benchmarks.benchmarker.restage_campaign import (
        execute_campaign,
        load_campaign_spec,
    )

    try:
        options = load_campaign_spec(spec)
    except (ValueError, OSError) as exc:
        raise typer.BadParameter(str(exc), param_hint="--spec") from exc
    try:
        campaign = execute_campaign(destination=output, resume=resume, **options)
    except TypeError as exc:
        raise typer.BadParameter(str(exc), param_hint="--spec") from exc
    selection = asyncio.run(campaign)
    typer.echo(json.dumps(asdict(selection), indent=2))
