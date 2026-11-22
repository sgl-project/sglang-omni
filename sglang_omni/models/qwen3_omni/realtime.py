# SPDX-License-Identifier: Apache-2.0
"""Serve the Qwen3-Omni Session graph with one realtime connection per session slot."""

from dataclasses import replace

from sglang_omni.client.client import Client
from sglang_omni.models.qwen3_omni.config import Qwen3OmniSessionPipelineConfig
from sglang_omni.serve.realtime import session_deployment
from sglang_omni.serve.realtime.manager import RealtimeDeployment


def build_realtime_deployment(
    client: Client, pipeline_config: Qwen3OmniSessionPipelineConfig
) -> RealtimeDeployment:
    """The shared Session graph deployment, admitting as many connections as max_sessions."""
    return replace(
        session_deployment.build_realtime_deployment(client, pipeline_config),
        max_connections=pipeline_config.max_sessions,
    )
