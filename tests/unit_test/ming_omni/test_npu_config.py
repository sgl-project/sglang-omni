# SPDX-License-Identifier: Apache-2.0
"""Device resolution policy tests for Ming-Omni pipelines."""

from __future__ import annotations

import pytest

from sglang_omni.models.ming_omni import config as ming_config


@pytest.mark.parametrize(
    "config_type",
    [
        ming_config.MingOmniPipelineConfig,
        ming_config.MingOmniSpeechPipelineConfig,
        ming_config.MingOmniStreamingSpeechPipelineConfig,
    ],
)
def test_ming_pipelines_defer_device_resolution_to_factories(config_type) -> None:
    config = config_type(model_path="dummy")
    for stage in config.stages:
        assert stage.factory.device is None
