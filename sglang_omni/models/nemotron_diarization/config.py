# SPDX-License-Identifier: Apache-2.0
"""Uploaded and live audio diarization using native Sortformer inference."""

from typing import ClassVar

from sglang_omni.config import FactoryArgs, PipelineConfig, StageConfig


class NemotronDiarizationPipelineConfig(PipelineConfig):
    architecture: ClassVar[str] = "Nemotron3DiarizationForAudioFrameClassification"

    stages: list[StageConfig] = [
        StageConfig(
            name="diarization",
            process="diarization",
            factory_path="sglang_omni.models.nemotron_diarization.stages.create_diarization_executor",
            factory=FactoryArgs(
                profile="offline",
                max_concurrency=1,
                max_live_sessions=8,
                max_running_requests=16,
                max_queued_requests=64,
            ),
            gpu=0,
            terminal=True,
        )
    ]


EntryClass = NemotronDiarizationPipelineConfig
