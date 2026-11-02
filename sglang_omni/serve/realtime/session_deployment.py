# SPDX-License-Identifier: Apache-2.0
"""Serve an audio Session graph through its declared realtime ports."""

from sglang_omni.client.client import Client
from sglang_omni.config.session_audio import AudioSessionPipelineConfig
from sglang_omni.serve.realtime.manager import (
    RealtimeDeployment,
    runtime_limits_for_pipeline,
)
from sglang_omni.serve.realtime.session_bridge import SessionBridge, capabilities_of


def build_realtime_deployment(
    client: Client, pipeline_config: AudioSessionPipelineConfig
) -> RealtimeDeployment:
    ports = pipeline_config.session_ports()
    limits = pipeline_config.session_limits()
    return RealtimeDeployment(
        capabilities=capabilities_of(ports),
        adapter_factory=lambda: SessionBridge(client, ports, limits=limits),
        limits=runtime_limits_for_pipeline(pipeline_config),
        session_limits=limits,
    )
