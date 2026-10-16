# SPDX-License-Identifier: Apache-2.0
"""Native SGLang Omni inference stage for Irodori-TTS."""

from __future__ import annotations

import logging

import torch

from sglang_omni.models.irodori_tts.batching import synthesize_batch
from sglang_omni.models.irodori_tts.model_loader import checkpoint_file, load_model
from sglang_omni.models.irodori_tts.synthesis import synthesize
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler
from sglang_omni.utils.device import resolve_concrete_device

logger = logging.getLogger(__name__)


def create_irodori_executor(
    model_path: str,
    *,
    device: str | None = None,
    gpu_id: int | None = None,
    model_precision: str,
    codec_precision: str,
    codec_repo: str,
    max_seconds: float,
    max_batch_size: int,
    max_batch_wait_ms: int,
    max_batch_tokens: int,
) -> SimpleScheduler[StagePayload, StagePayload]:
    resolved_device = torch.device(str(resolve_concrete_device(device, gpu_id)))
    resolved_checkpoint = checkpoint_file(model_path)
    (
        model_configuration,
        model,
        tokenizer,
        caption_tokenizer,
        codec,
        max_text_len,
        max_caption_len,
        max_ref_seconds,
    ) = load_model(
        resolved_checkpoint,
        resolved_device,
        model_precision,
        codec_repo,
        codec_precision,
    )

    def run_synthesis(payload: StagePayload) -> StagePayload:
        return synthesize(
            payload,
            model_configuration=model_configuration,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            max_ref_seconds=max_ref_seconds,
            max_seconds=max_seconds,
        )

    def run_synthesis_batch(
        payloads: list[StagePayload],
    ) -> list[StagePayload | BaseException]:
        return synthesize_batch(
            payloads,
            model_configuration=model_configuration,
            model=model,
            tokenizer=tokenizer,
            caption_tokenizer=caption_tokenizer,
            codec=codec,
            max_text_len=max_text_len,
            max_caption_len=max_caption_len,
            max_ref_seconds=max_ref_seconds,
            max_seconds=max_seconds,
            max_batch_size=max_batch_size,
            max_batch_tokens=max_batch_tokens,
        )

    logger.info(
        "Loaded native Irodori-TTS model from %s on %s with model precision %s "
        "and codec precision %s; request batching is enabled (max batch size %d)",
        resolved_checkpoint,
        resolved_device,
        model_precision,
        codec_precision,
        max_batch_size,
    )
    return SimpleScheduler(
        run_synthesis,
        batch_compute_fn=run_synthesis_batch,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
        max_concurrency=1,
    )
