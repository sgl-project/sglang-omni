# SPDX-License-Identifier: Apache-2.0
"""Cross-request batching for the MiniCPM-o audio encoder stage."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import TypedDict

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    feature_lens_after_pooling,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import (
    EncoderRequestData,
    build_encoder_request,
)
from sglang_omni.profiler.event_recorder import emit as emit_event
from sglang_omni.profiler.event_recorder import get_recorder
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache


@dataclass(kw_only=True)
class AudioBatchItem:
    index: int
    payload: StagePayload
    state: MiniCPMOPipelineState
    request: EncoderRequestData
    features: torch.Tensor
    lengths: torch.Tensor


class AudioEncoderOutput(TypedDict, total=False):
    audio_embeds: torch.Tensor


def apply_audio_encoder_result(
    payload: StagePayload,
    state: MiniCPMOPipelineState,
    encoder_out: AudioEncoderOutput,
) -> StagePayload:
    state.encoder_outs["audio_encoder"] = encoder_out
    payload.data = state.to_dict()
    return payload


def encode_audio_payload(
    payload: StagePayload,
    *,
    encoder: MiniCPMOAudioEncoder,
    cache: StageOutputCache,
) -> StagePayload:
    metadata = audio_batch_metadata([payload])
    emit_event(
        request_id=payload.request_id,
        stage="audio_encoder",
        event_name="encoder_start",
        metadata=metadata,
    )
    try:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name="audio_encoder")
        cached = (
            None if request.skip_result is not None else cache.get(request.cache_key)
        )
        if request.skip_result is not None:
            emit_audio_encoder_event(payload, "encoder_skipped")
            encoder_out = request.skip_result
        elif cached is not None:
            emit_audio_encoder_event(payload, "encoder_cache_hit")
            encoder_out = cached
        else:
            emit_audio_encoder_event(payload, "encoder_cache_miss")
            emit_audio_encoder_event(payload, "encoder_forward_start")
            with torch.no_grad():
                encoder_out = encoder(**request.model_inputs)
            emit_audio_encoder_event(payload, "encoder_forward_end")
            cache.put(request.cache_key, encoder_out)
            emit_audio_encoder_event(payload, "encoder_cache_put_end")
        return apply_audio_encoder_result(payload, state, encoder_out)
    finally:
        emit_event(
            request_id=payload.request_id,
            stage="audio_encoder",
            event_name="encoder_end",
            metadata=metadata,
        )


def batch_audio_encoder_payloads(
    payloads: list[StagePayload],
    *,
    encoder: MiniCPMOAudioEncoder,
    cache: StageOutputCache,
) -> list[StagePayload]:
    metadata = audio_batch_metadata(payloads)
    for payload in payloads:
        emit_event(
            request_id=payload.request_id,
            stage="audio_encoder",
            event_name="encoder_start",
            metadata=metadata,
        )
    try:
        return compute_audio_encoder_batch(payloads, encoder=encoder, cache=cache)
    finally:
        for payload in payloads:
            emit_event(
                request_id=payload.request_id,
                stage="audio_encoder",
                event_name="encoder_end",
                metadata=metadata,
            )


def audio_batch_metadata(payloads: list[StagePayload]) -> dict[str, int | str]:
    metadata: dict[str, int | str] = {
        "request_batch_size": len(payloads),
        "batch_leader": payloads[0].request_id,
    }
    if not get_recorder().is_active():
        return metadata

    audio_rows = 0
    valid_mel_frames = 0
    max_mel_frames = 0
    for payload in payloads:
        state = MiniCPMOPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name="audio_encoder")
        features = request.model_inputs.get("audio_features")
        lengths = request.model_inputs.get("audio_feature_lens")
        if not isinstance(features, torch.Tensor) or not isinstance(
            lengths, torch.Tensor
        ):
            continue
        audio_rows += int(lengths.numel())
        valid_mel_frames += int(lengths.sum())
        max_mel_frames = max(max_mel_frames, int(features.shape[-1]))
    metadata.update(
        {
            "audio_rows": audio_rows,
            "valid_mel_frames": valid_mel_frames,
            "max_mel_frames": max_mel_frames,
            "padded_mel_frames": audio_rows * max_mel_frames - valid_mel_frames,
        }
    )
    return metadata


def emit_audio_encoder_event(payload: StagePayload, event_name: str) -> None:
    emit_event(
        request_id=payload.request_id,
        stage="audio_encoder",
        event_name=event_name,
    )


def compute_audio_encoder_batch(
    payloads: list[StagePayload],
    *,
    encoder: MiniCPMOAudioEncoder,
    cache: StageOutputCache,
) -> list[StagePayload]:
    results: list[StagePayload | None] = [None] * len(payloads)
    active: list[AudioBatchItem] = []
    duplicate_waiters: dict[
        str, list[tuple[int, StagePayload, MiniCPMOPipelineState]]
    ] = defaultdict(list)
    active_cache_keys: set[str] = set()

    for index, payload in enumerate(payloads):
        state = MiniCPMOPipelineState.from_dict(payload.data)
        request = build_encoder_request(state, stage_name="audio_encoder")
        if request.skip_result is not None:
            emit_audio_encoder_event(payload, "encoder_skipped")
            results[index] = apply_audio_encoder_result(
                payload, state, request.skip_result
            )
            continue

        cached = cache.get(request.cache_key)
        if cached is not None:
            emit_audio_encoder_event(payload, "encoder_cache_hit")
            results[index] = apply_audio_encoder_result(payload, state, cached)
            continue

        if request.cache_key is not None and request.cache_key in active_cache_keys:
            duplicate_waiters[request.cache_key].append((index, payload, state))
            continue

        emit_audio_encoder_event(payload, "encoder_cache_miss")
        features = request.model_inputs["audio_features"]
        lengths = request.model_inputs["audio_feature_lens"]
        assert isinstance(features, torch.Tensor)
        assert isinstance(lengths, torch.Tensor)
        active.append(
            AudioBatchItem(
                index=index,
                payload=payload,
                state=state,
                request=request,
                features=features,
                lengths=lengths.reshape(-1),
            )
        )
        if request.cache_key is not None:
            active_cache_keys.add(request.cache_key)

    if not active:
        pass
    else:
        max_frames = max(int(item.features.shape[-1]) for item in active)
        features = torch.cat(
            [
                F.pad(
                    item.features,
                    (0, max_frames - int(item.features.shape[-1])),
                )
                for item in active
            ],
            dim=0,
        )
        lengths = torch.cat([item.lengths for item in active])
        for item in active:
            emit_audio_encoder_event(item.payload, "encoder_forward_start")
        try:
            with torch.no_grad():
                combined = encoder(
                    audio_features=features,
                    audio_feature_lens=lengths,
                )["audio_embeds"]
        finally:
            for item in active:
                emit_audio_encoder_event(item.payload, "encoder_forward_end")

        cursor = 0
        computed_by_cache_key: dict[str, AudioEncoderOutput] = {}
        for item in active:
            output_size = int(
                feature_lens_after_pooling(
                    item.lengths.to("cpu"), encoder.audio_pool_step
                ).sum()
            )
            encoder_out: AudioEncoderOutput = {
                "audio_embeds": combined[cursor : cursor + output_size]
            }
            cursor += output_size
            cache.put(item.request.cache_key, encoder_out)
            emit_audio_encoder_event(item.payload, "encoder_cache_put_end")
            if item.request.cache_key is not None:
                computed_by_cache_key[item.request.cache_key] = encoder_out
            results[item.index] = apply_audio_encoder_result(
                item.payload, item.state, encoder_out
            )

        assert (
            cursor == combined.shape[0]
        ), f"Batched audio output split consumed {cursor} of {combined.shape[0]} rows"

        for cache_key, waiters in duplicate_waiters.items():
            encoder_out = computed_by_cache_key[cache_key]
            for index, payload, state in waiters:
                results[index] = apply_audio_encoder_result(payload, state, encoder_out)

    assert all(
        result is not None for result in results
    ), "Batched audio encoder did not produce every request result"
    return [result for result in results if result is not None]
