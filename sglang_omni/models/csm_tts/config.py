# SPDX-License-Identifier: Apache-2.0
# Pipeline config follows the sglang-omni TTS pipeline-config pattern (cf. Higgs / Qwen3-TTS).
"""Pipeline configuration for CSM TTS (V1).

MUST stay sglang/CUDA-free: ``import_pipeline_configs`` pkgutil-scans
``sglang_omni/models/*`` and SILENTLY skips packages whose import fails — so
this module never imports ``stages.py`` (the way higgs config.py:9 does); the
concurrency constant is inlined instead.


"""

from __future__ import annotations

from typing import ClassVar

from sglang_omni.config import PipelineConfig, StageConfig

_PKG = "sglang_omni.models.csm_tts"

# Inlined (do NOT import stages.py — registry-silent-skip gotcha). Keep in
# sync with stages.DEFAULT_MAX_CONCURRENCY.
_DEFAULT_MAX_CONCURRENCY = 8


class CsmTtsPipelineConfig(PipelineConfig):
    """4-stage CSM TTS pipeline:
    preprocessing → audio_encoder → tts_engine → vocoder.

    Scheduler-visible AR step = one 80 ms frame (paged backbone forward →
    cb0 sample → 31-step depth inner AR → 32-code frame → streamed to the
    vocoder). Every non-terminal stage sets ``next=`` — schema validation
    requires exactly one of ``next``/``terminal`` per stage
    (config/schema.py:298-302).
    """

    architecture: ClassVar[str] = "CsmForConditionalGeneration"

    model_path: str
    stages: list[StageConfig] = [
        StageConfig(
            name="preprocessing",
            process="pipeline",
            factory=f"{_PKG}.stages.create_preprocessing_executor",
            factory_args={"max_concurrency": _DEFAULT_MAX_CONCURRENCY},
            next="audio_encoder",
        ),
        StageConfig(
            name="audio_encoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_audio_encoder_executor",
            factory_args={
                "device": "cuda",
                "max_batch_size": _DEFAULT_MAX_CONCURRENCY,
            },
            gpu=0,
            next="tts_engine",
        ),
        # Engine-tuning defaults live in the FACTORY, not here (contract):
        # stages.create_sglang_tts_engine_executor pins the 3060 budget from
        # the VRAM table — context_length=2048, mem_fraction_static
        # 0.5, cuda_graph_max_bs 8, dtype bfloat16, and disable_cuda_graph=
        # True (eager is the correctness baseline; set False to enable CUDA graphs). The
        # YAML's server_args_overrides deep-merge over those defaults.
        StageConfig(
            name="tts_engine",
            process="pipeline",
            factory=f"{_PKG}.stages.create_sglang_tts_engine_executor",
            factory_args={
                "device": "cuda",
                "max_new_tokens": 125,  # frames (DEFAULT_MAX_FRAMES)
                "enable_async_decode": False,  # set True to enable async decode
                "server_args_overrides": {
                    "max_running_requests": _DEFAULT_MAX_CONCURRENCY,
                },
            },
            gpu=0,
            next="vocoder",
            stream_to=["vocoder"],
        ),
        StageConfig(
            name="vocoder",
            process="pipeline",
            factory=f"{_PKG}.stages.create_vocoder_executor",
            factory_args={
                "device": "cuda",
                "dtype": "float32",  # conv-transpose decode stability
                "max_batch_size": _DEFAULT_MAX_CONCURRENCY,
            },
            gpu=0,
            terminal=True,
            can_accept_stream_before_payload=True,
        ),
    ]


EntryClass = CsmTtsPipelineConfig
