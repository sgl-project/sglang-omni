# SPDX-License-Identifier: Apache-2.0
"""Cross-request batching for the MiniCPM-o audio encoder stage."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from sglang_omni.models.minicpm_o.components.audio_encoder import (
    MiniCPMOAudioEncoder,
    feature_lens_after_pooling,
)
from sglang_omni.models.minicpm_o.payload_types import MiniCPMOPipelineState
from sglang_omni.models.minicpm_o.request_builders import build_encoder_request
from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.stage_cache import StageOutputCache


@dataclass(kw_only=True)
class AudioBatchRequest:
    audio_features: torch.Tensor
    audio_feature_lengths: torch.Tensor
    cache_key: str | None
    pipeline_states: list[MiniCPMOPipelineState]


def batch_audio_encoder_payloads(
    payloads: list[StagePayload],
    *,
    encoder: MiniCPMOAudioEncoder,
    cache: StageOutputCache,
) -> list[StagePayload]:
    pipeline_states = [
        MiniCPMOPipelineState.from_dict(payload.data) for payload in payloads
    ]
    uncached_requests: list[AudioBatchRequest] = []
    requests_by_cache_key: dict[str, AudioBatchRequest] = {}
    for state in pipeline_states:
        request = build_encoder_request(state, stage_name="audio_encoder")
        cached = cache.get(request.cache_key)
        if request.skip_result is not None:
            state.encoder_outs["audio_encoder"] = request.skip_result
        elif cached is not None:
            state.encoder_outs["audio_encoder"] = cached
        elif (
            request.cache_key is not None and request.cache_key in requests_by_cache_key
        ):
            requests_by_cache_key[request.cache_key].pipeline_states.append(state)
        else:
            batch_request = AudioBatchRequest(
                audio_features=request.model_inputs["audio_features"],
                audio_feature_lengths=request.model_inputs[
                    "audio_feature_lens"
                ].reshape(-1),
                cache_key=request.cache_key,
                pipeline_states=[state],
            )
            uncached_requests.append(batch_request)
            if request.cache_key is not None:
                requests_by_cache_key[request.cache_key] = batch_request
            else:
                pass

    if uncached_requests:
        if len(uncached_requests) == 1:
            outputs = [
                encoder(
                    audio_features=uncached_requests[0].audio_features,
                    audio_feature_lens=uncached_requests[0].audio_feature_lengths,
                )
            ]
        else:
            max_mel_frames = max(
                request.audio_features.shape[-1] for request in uncached_requests
            )
            audio_features = torch.cat(
                [
                    F.pad(
                        request.audio_features,
                        (0, max_mel_frames - request.audio_features.shape[-1]),
                    )
                    for request in uncached_requests
                ],
                dim=0,
            )
            audio_feature_lengths = torch.cat(
                [request.audio_feature_lengths for request in uncached_requests]
            )
            original_mel_frame_counts = torch.tensor(
                [
                    request.audio_features.shape[-1]
                    for request in uncached_requests
                    for _ in range(request.audio_features.shape[0])
                ]
            )
            audio_embeddings = encoder(
                audio_features=audio_features,
                audio_feature_lens=audio_feature_lengths,
                original_mel_frame_counts=original_mel_frame_counts,
            )["audio_embeds"]
            request_embedding_lengths = [
                int(
                    feature_lens_after_pooling(
                        request.audio_feature_lengths.to("cpu"), encoder.audio_pool_step
                    ).sum()
                )
                for request in uncached_requests
            ]
            # note (wirybeaver): a cached request must not retain its siblings' storage.
            outputs = [
                {"audio_embeds": request_embeddings.clone()}
                for request_embeddings in audio_embeddings.split(
                    request_embedding_lengths
                )
            ]

        for request, output in zip(uncached_requests, outputs, strict=True):
            cache.put(request.cache_key, output)
            for state in request.pipeline_states:
                state.encoder_outs["audio_encoder"] = output
    else:
        pass

    for payload, state in zip(payloads, pipeline_states, strict=True):
        payload.data = state.to_dict()
    return payloads
