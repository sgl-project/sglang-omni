# SPDX-License-Identifier: Apache-2.0
"""Compile plans for MOSS-Transcribe-Diarize."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from sglang_omni.compilation import CompilePlan, CompileTarget, CompileWarmupCase


def create_moss_td_encoder_compile_plan(
    model: Any,
    *,
    chunk_buckets: Sequence[int],
    input_feature_len: int,
) -> CompilePlan:
    p = next(model.whisper_encoder.parameters())
    frames = int(input_feature_len)
    num_mel_bins = int(model.config.audio_config.num_mel_bins)
    pos = torch.arange((frames - 1) // 2 + 1, device=p.device, dtype=torch.long)

    return CompilePlan(
        name="moss_transcribe_diarize.whisper_encoder",
        targets=(
            CompileTarget(
                name="moss_transcribe_diarize.whisper_encoder",
                eager=model.whisper_encoder,
                install=lambda compiled: setattr(model, "_compiled_encoder", compiled),
                compile_kwargs={"dynamic": False},
                bucket_fn=lambda feats, *args, **kwargs: (
                    int(feats.shape[0]),
                    int(feats.shape[-1]),
                ),
                warmup_cases=tuple(
                    CompileWarmupCase(
                        f"chunks={n}",
                        args=(
                            torch.zeros(
                                n,
                                num_mel_bins,
                                frames,
                                device=p.device,
                                dtype=p.dtype,
                            ),
                            pos,
                            None,
                        ),
                        bucket=(n, frames),
                        repeat=3,
                    )
                    for n in chunk_buckets
                ),
                restrict_to_warmed_buckets=True,
            ),
        ),
    )
