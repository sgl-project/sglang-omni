from __future__ import annotations

import logging
from typing import Annotated

import typer
import yaml
from transformers import AutoConfig

from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY

logger = logging.getLogger(__name__)

config_app = typer.Typer(
    help="View and export the pipeline configuration for local editing"
)


@config_app.command()
def view(
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
) -> None:
    """View the model's pipeline configuration."""
    hf_config = AutoConfig.from_pretrained(model_path)
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config(hf_config.architectures[0])
    config = config_cls(model_path=model_path)
    config_json = config.model_dump(mode="json")
    print(
        yaml.dump(
            config_json,
            sort_keys=False,  # preserve order
            default_flow_style=False,  # use block style (not inline)
            indent=2,  # control indentation
            allow_unicode=True,
        )
    )


@config_app.command()
def export(
    model_path: Annotated[
        str,
        typer.Option(
            help="The Hugging Face model ID or the path to the model directory."
        ),
    ],
    output_path: Annotated[
        str, typer.Option(help="Path to the output JSON file.")
    ] = None,
) -> None:
    """Export the default pipeline configuration to a YAML file."""
    # get the default pipeline config for the model

    hf_config = AutoConfig.from_pretrained(model_path)
    config_cls = PIPELINE_CONFIG_REGISTRY.get_config(hf_config.architectures[0])
    config = config_cls(model_path=model_path)

    # export config in a yaml file
    if output_path is None:
        output_path = f"./config_{config.name}.yaml"

    with open(output_path, "w") as f:
        yaml.dump(
            config.model_dump(mode="json"),
            f,
            sort_keys=False,
            default_flow_style=False,
            indent=2,
            allow_unicode=True,
        )
    print(f"Pipeline config exported to {output_path}")
