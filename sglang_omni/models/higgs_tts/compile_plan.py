# SPDX-License-Identifier: Apache-2.0
"""Compile plans for Higgs TTS standalone codec stages."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.compilation import CompilePlan, CompileTarget, CompileWarmupCase


def create_higgs_audio_encoder_compile_plan(codec: Any) -> CompilePlan:
    acoustic_encoder = codec.model.acoustic_encoder
    parameter = next(acoustic_encoder.parameters(), None)
    warmup_device = parameter.device if parameter is not None else torch.device("cpu")
    warmup_dtype = parameter.dtype if parameter is not None else torch.float32

    return CompilePlan(
        name="higgs_tts.audio_encoder",
        targets=(
            CompileTarget(
                name="higgs_tts.audio_encoder",
                eager=acoustic_encoder.forward,
                install=lambda compiled: setattr(acoustic_encoder, "forward", compiled),
                compile_kwargs={"mode": "default", "dynamic": True},
                bucket_fn=lambda waveform, *args, **kwargs: tuple(waveform.shape),
                warmup_cases=(
                    CompileWarmupCase(
                        "one-second-reference",
                        args=(
                            torch.zeros(
                                1,
                                1,
                                codec.SAMPLE_RATE,
                                device=warmup_device,
                                dtype=warmup_dtype,
                            ),
                        ),
                        bucket=(1, 1, codec.SAMPLE_RATE),
                    ),
                ),
            ),
        ),
    )


def create_higgs_vocoder_compile_plan(
    codec: Any,
    *,
    warmup_frame_counts: tuple[int, ...],
) -> CompilePlan:
    eager_decode = codec.model.decode
    num_quantizers = int(codec.model.config.num_quantizers)

    return CompilePlan(
        name="higgs_tts.vocoder_decode",
        targets=(
            CompileTarget(
                name="higgs_tts.vocoder_decode",
                eager=eager_decode,
                install=lambda compiled: setattr(codec.model, "decode", compiled),
                compile_kwargs={"dynamic": True},
                bucket_fn=lambda codes, *args, **kwargs: (
                    int(codes.shape[0]),
                    int(codes.shape[-1]),
                ),
                warmup_cases=tuple(
                    CompileWarmupCase(
                        f"batch={batch_size},frames={frame_count}",
                        args=(
                            torch.zeros(
                                batch_size,
                                num_quantizers,
                                frame_count,
                                dtype=torch.long,
                                device=codec.device,
                            ),
                        ),
                        bucket=(batch_size, frame_count),
                    )
                    for frame_count in warmup_frame_counts
                    for batch_size in (1, 2)
                ),
            ),
        ),
    )
