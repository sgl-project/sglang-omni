# SPDX-License-Identifier: Apache-2.0
"""Stage executors for Qwen3-Omni pipelines.

Each factory instantiates a Qwen3-Omni-specific component and delegates the
common orchestration logic to :mod:`sglang_omni.models.omni_base.stages`.
"""

from __future__ import annotations

from transformers import AutoTokenizer

from sglang_omni.executors import EngineExecutor, PreprocessingExecutor
from sglang_omni.models.omni_base.stages import (
    create_aggregate_executor as create_aggregate_executor,  # re-export
)
from sglang_omni.models.omni_base.stages import (
    create_decode_executor as _create_generic_decode_executor,
)
from sglang_omni.models.omni_base.stages import create_encoder_executor
from sglang_omni.models.omni_base.stages import (
    create_thinker_executor as _create_generic_thinker_executor,
)
from sglang_omni.models.qwen3_omni.components.audio_encoder import Qwen3OmniAudioEncoder
from sglang_omni.models.qwen3_omni.components.image_encoder import Qwen3OmniImageEncoder
from sglang_omni.models.qwen3_omni.components.preprocessor import Qwen3OmniPreprocessor
from sglang_omni.models.qwen3_omni.components.thinker import Qwen3OmniSplitThinker
from sglang_omni.models.qwen3_omni.pipeline.merge import decode_events
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    AUDIO_STAGE,
    IMAGE_STAGE,
    THINKER_STAGE,
)
from sglang_omni.proto import StagePayload


def create_preprocessing_executor(model_id: str) -> PreprocessingExecutor:
    """Create a Qwen3-Omni preprocessing executor."""
    preprocessor = Qwen3OmniPreprocessor(model_id=model_id)

    async def _preprocess(payload: StagePayload) -> StagePayload:
        return await preprocessor(payload)

    return PreprocessingExecutor(_preprocess)


def create_image_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    """Create a Qwen3-Omni image encoder executor."""
    model = Qwen3OmniImageEncoder(model_id=model_id, device=device, dtype=dtype)
    return create_encoder_executor(stage_name=IMAGE_STAGE, model=model, device=device)


def create_audio_encoder_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
) -> EngineExecutor:
    """Create a Qwen3-Omni audio encoder executor."""
    model = Qwen3OmniAudioEncoder(model_id=model_id, device=device, dtype=dtype)
    return create_encoder_executor(stage_name=AUDIO_STAGE, model=model, device=device)


def create_thinker_executor(
    model_id: str,
    *,
    device: str = "cuda",
    dtype: str | None = None,
    max_seq_len: int = 8192,
) -> EngineExecutor:
    """Create a Qwen3-Omni thinker executor."""
    model = Qwen3OmniSplitThinker(model_id=model_id, device=device, dtype=dtype)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _create_generic_thinker_executor(
        model=model,
        tokenizer=tokenizer,
        decode_events_fn=decode_events,
        stage_name=THINKER_STAGE,
        max_seq_len=max_seq_len,
        device=device,
    )


def create_decode_executor(model_id: str) -> PreprocessingExecutor:
    """Create a Qwen3-Omni decode executor."""
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    return _create_generic_decode_executor(
        tokenizer=tokenizer,
        decode_events_fn=decode_events,
        thinker_stage_name=THINKER_STAGE,
    )
