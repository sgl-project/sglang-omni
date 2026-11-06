# SPDX-License-Identifier: Apache-2.0
"""Serves MIBURI over /v1/realtime: the PersonaPlex session deployment, with motion as a plugin output."""

from __future__ import annotations

from sglang_omni.client.client import Client
from sglang_omni.models.miburi.config import MiburiPipelineConfig
from sglang_omni.models.personaplex.realtime import (
    create_realtime_deployment as create_speech_deployment,
)
from sglang_omni.serve.realtime.manager import RealtimeDeployment


def create_realtime_deployment(
    client: Client, pipeline_config: MiburiPipelineConfig
) -> RealtimeDeployment:
    """The speech deployment reads its voices from model_path, which for it is the speech backbone's checkpoint."""
    return create_speech_deployment(
        client,
        pipeline_config.model_copy(
            update={"model_path": pipeline_config.speech_model_path}
        ),
    )


__all__ = ["create_realtime_deployment"]
