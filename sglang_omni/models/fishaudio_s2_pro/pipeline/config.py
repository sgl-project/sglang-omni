# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration factory for the FishAudio S2-Pro TTS pipeline.

Same 3-stage architecture as S1, but defaults to the S2-Pro checkpoint
and uses ``fishaudio_s1`` stage executors (the DualAR runtime is shared).
"""

from __future__ import annotations

from sglang_omni.config import ExecutorConfig, PipelineConfig, RelayConfig, StageConfig
from sglang_omni.models.fishaudio_s2_pro.pipeline.next_stage import (
    PREPROCESSING_STAGE,
    TTS_ENGINE_STAGE,
    VOCODER_STAGE,
)

DEFAULT_MODEL_ID = "fishaudio/openaudio-s2-pro"


def create_tts_pipeline_config(
    *,
    model_id: str = DEFAULT_MODEL_ID,
    tts_device: str = "cuda:0",
    vocoder_device: str = "cuda:0",
    max_new_tokens: int = 2048,
    max_seq_len: int = 4096,
    use_compile: bool = True,
    use_radix_cache: bool = True,
    relay_type: str = "shm",
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    """Create a 3-stage TTS pipeline config for FishAudio S2-Pro.

    Stages::

        preprocessing (CPU)  →  tts_engine (GPU)  →  vocoder (GPU)
           tokenize text         DualAR decode        DAC codec decode
           encode ref audio      VQ code generation    VQ codes → audio

    Uses the same stage executors as S1 (``fishaudio_s1.pipeline.stages``),
    since the model architecture (DualAR) is identical.

    Args:
        model_id: HF model ID or local checkpoint path.
        tts_device: Device for the TTS engine stage.
        vocoder_device: Device for the vocoder stage.
        max_new_tokens: Maximum decode steps for the TTS engine.
        max_seq_len: Maximum sequence length for KV cache allocation.
        use_compile: Enable torch.compile for decode steps.
        use_radix_cache: Enable radix-tree prefix cache for voice ref reuse.
        relay_type: Tensor relay backend (``"shm"``, ``"nccl"``, ``"nixl"``).
        fused_stages: Optional stage fusion groups.

    Returns:
        A :class:`PipelineConfig` ready for ``compile_pipeline()``.
    """

    _s1_pkg = "sglang_omni.models.fishaudio_s1.pipeline"
    _s2_pkg = "sglang_omni.models.fishaudio_s2_pro.pipeline"

    def _relay(device: str) -> RelayConfig:
        return RelayConfig(type=relay_type, device=device)

    return PipelineConfig(
        name="fishaudio_s2_pro_tts",
        model_path=model_id,
        entry_stage=PREPROCESSING_STAGE,
        fused_stages=fused_stages or [],
        stages=[
            StageConfig(
                name=PREPROCESSING_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_s1_pkg}.stages.create_preprocessing_executor",
                    args={"model_id": model_id},
                ),
                get_next=f"{_s2_pkg}.next_stage.preprocessing_next",
                relay=_relay("cpu"),
            ),
            StageConfig(
                name=TTS_ENGINE_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_s1_pkg}.stages.create_tts_engine_executor",
                    args={
                        "model_id": model_id,
                        "device": tts_device,
                        "max_new_tokens": max_new_tokens,
                        "max_seq_len": max_seq_len,
                        "use_compile": use_compile,
                        "use_radix_cache": use_radix_cache,
                    },
                ),
                get_next=f"{_s2_pkg}.next_stage.tts_engine_next",
                relay=_relay(tts_device),
            ),
            StageConfig(
                name=VOCODER_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_s1_pkg}.stages.create_vocoder_executor",
                    args={
                        "model_id": model_id,
                        "device": vocoder_device,
                    },
                ),
                get_next=f"{_s2_pkg}.next_stage.vocoder_next",
                relay=_relay(vocoder_device),
            ),
        ],
    )
