# SPDX-License-Identifier: Apache-2.0
"""Metrics startup switch tests."""

from types import SimpleNamespace
from unittest.mock import patch

from sglang_omni.cli.serve import serve
from sglang_omni.config import PipelineConfig, StageConfig
from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.runtime import resolve_factory_signature_args


def metrics_factory(
    server_args_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    return dict(server_args_overrides or {})


def serve_kwargs(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "ctx": SimpleNamespace(args=[]),
        "model_path": "dummy",
        "config": None,
        "text_only": False,
        "colocate": False,
        "host": "0.0.0.0",
        "port": 8000,
        "model_name": None,
        "mem_fraction_static": None,
        "log_level": "info",
    }
    values.update(overrides)
    return values


def pipeline_manager() -> ConfigManager:
    return ConfigManager(
        PipelineConfig(
            model_path="dummy",
            stages=[
                StageConfig(
                    name="stage",
                    process="pipeline",
                    factory_path=(
                        "tests.unit_test.fixtures.pipeline_fakes.dummy_factory"
                    ),
                    terminal=True,
                )
            ],
        )
    )


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_disables_metrics_by_default(from_model_path, launch_server) -> None:
    from_model_path.return_value = pipeline_manager()

    serve(**serve_kwargs())

    assert launch_server.call_args.kwargs["enable_metrics"] is False


@patch("sglang_omni.cli.serve.launch_server")
@patch("sglang_omni.cli.serve.ConfigManager.from_model_path")
def test_cli_enables_metrics_explicitly(from_model_path, launch_server) -> None:
    from_model_path.return_value = pipeline_manager()

    serve(**serve_kwargs(enable_metrics=True))

    assert launch_server.call_args.kwargs["enable_metrics"] is True


def test_stage_runtime_enables_upstream_sglang_metrics() -> None:
    overrides = resolve_factory_signature_args(
        metrics_factory,
        {
            "server_args_overrides": {
                "extra_metric_labels": {"deployment": "test"},
                "max_running_requests": 4,
            }
        },
        defaults={},
        runtime_server_args_overrides={
            "enable_metrics": True,
            "extra_metric_labels": {"omni_stage": "engine", "replica": "0"},
        },
    )

    assert overrides["server_args_overrides"] == {
        "enable_metrics": True,
        "extra_metric_labels": {
            "deployment": "test",
            "omni_stage": "engine",
            "replica": "0",
        },
        "max_running_requests": 4,
    }
