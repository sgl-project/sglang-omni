# SPDX-License-Identifier: Apache-2.0
"""Pipeline configuration factory for the FishAudio S2-Pro TTS pipeline.

Uses ``FishQwen3OmniForCausalLM`` with the ``qwen3.py`` inference module,
which is a completely different architecture from S1's ``DualARTransformer``.
All three stages use S2-Pro-specific executors.
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
    use_radix_cache: bool = False,
    relay_type: str = "shm",
    fused_stages: list[list[str]] | None = None,
) -> PipelineConfig:
    """Create a 3-stage TTS pipeline config for FishAudio S2-Pro.

    Stages::

        preprocessing (CPU)  →  tts_engine (GPU)  →  vocoder (GPU)
           build Qwen3 prompt    FishQwen3Omni gen    VQGAN decode
           encode ref audio      semantic + codebook   VQ codes → audio

    Uses ``FishQwen3OmniForCausalLM`` (loaded via ``AutoModel.from_pretrained``)
    with the ``qwen3.generate`` function for the TTS engine stage.

    Args:
        model_id: HF model ID or local checkpoint path.
        tts_device: Device for the TTS engine stage.
        vocoder_device: Device for the vocoder stage.
        max_new_tokens: Maximum decode steps for the TTS engine.
        max_seq_len: Maximum sequence length for KV cache allocation.
        use_compile: Enable torch.compile for decode steps.
        use_radix_cache: Not yet supported for S2-Pro.
        relay_type: Tensor relay backend (``"shm"``, ``"nccl"``, ``"nixl"``).
        fused_stages: Optional stage fusion groups.

    Returns:
        A :class:`PipelineConfig` ready for ``compile_pipeline()``.
    """

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
                    factory=f"{_s2_pkg}.stages.create_preprocessing_executor",
                    args={"model_id": model_id, "vqgan_device": tts_device},
                ),
                get_next=f"{_s2_pkg}.next_stage.preprocessing_next",
                relay=_relay("cpu"),
            ),
            StageConfig(
                name=TTS_ENGINE_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_s2_pkg}.stages.create_tts_engine_executor",
                    args={
                        "model_id": model_id,
                        "device": tts_device,
                        "max_new_tokens": max_new_tokens,
                        "max_seq_len": max_seq_len,
                        "use_compile": use_compile,
                    },
                ),
                get_next=f"{_s2_pkg}.next_stage.tts_engine_next",
                relay=_relay(tts_device),
            ),
            StageConfig(
                name=VOCODER_STAGE,
                executor=ExecutorConfig(
                    factory=f"{_s2_pkg}.stages.create_vocoder_executor",
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
