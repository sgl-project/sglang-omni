# SPDX-License-Identifier: Apache-2.0
"""Stream delivery caps: defaults, dotted CLI overrides and validation."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sglang_omni.config.resolver import ConfigResolver
from sglang_omni.config.schema import PipelineConfig, StreamDeliveryConfig
from sglang_omni.config.sources import patches_from_dotted_cli


def test_stream_delivery_caps_are_on_by_default(
    pipeline_config: PipelineConfig,
) -> None:
    assert pipeline_config.stream_delivery.max_request_backlog_bytes == 256 * 1024**2
    assert pipeline_config.stream_delivery.max_total_backlog_bytes == 1024**3


def test_dotted_overrides_set_and_clear_stream_delivery_caps(
    pipeline_config: PipelineConfig,
) -> None:
    patches = patches_from_dotted_cli(
        {
            "stream_delivery.max_total_backlog_bytes": "2147483648",
            "stream_delivery.max_request_backlog_bytes": "none",
        },
        pipeline_config,
    )
    resolved = ConfigResolver(pipeline_config).resolve(patches).config
    assert resolved.stream_delivery.max_total_backlog_bytes == 2147483648
    assert resolved.stream_delivery.max_request_backlog_bytes is None


def test_stream_delivery_rejects_zero_and_a_total_below_the_request_cap() -> None:
    with pytest.raises(ValidationError):
        StreamDeliveryConfig(max_total_backlog_bytes=0)
    with pytest.raises(ValueError, match="must be at least max_request_backlog_bytes"):
        StreamDeliveryConfig(max_request_backlog_bytes=100, max_total_backlog_bytes=10)
