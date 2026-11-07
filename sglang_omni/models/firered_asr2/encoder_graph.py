# SPDX-License-Identifier: Apache-2.0
"""Bucketed graph capture of the FP32 acoustic encoder and projector."""

import logging
from dataclasses import dataclass

import torch

from sglang_omni.models.firered_asr2.audio import FireRedAdapter
from sglang_omni.models.firered_asr2.conformer import ConformerEncoder
from sglang_omni.platforms.device_graph import (
    DeviceGraphBackend,
    DeviceGraphPool,
    ReplayableGraph,
)

logger = logging.getLogger(__name__)


@dataclass(kw_only=True)
class EncoderGraph:
    graph: ReplayableGraph
    features: torch.Tensor
    lengths: torch.Tensor
    embeddings: torch.Tensor


class FireRedEncoderGraph:
    def __init__(
        self,
        encoder: ConformerEncoder,
        projector: FireRedAdapter,
        backend: DeviceGraphBackend,
    ) -> None:
        self.encoder: ConformerEncoder = encoder
        self.projector: FireRedAdapter = projector
        self.backend: DeviceGraphBackend = backend
        self.pool: DeviceGraphPool = backend.graph_pool_handle()
        self.graphs: dict[tuple[int, int], EncoderGraph] = {}
        self.replay_count: int = 0

    @torch.no_grad()
    def __call__(self, features: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        batch_size, frames, feature_size = features.shape
        frame_bucket = 1 << (frames - 1).bit_length()
        graph_key = (batch_size, frame_bucket)
        captured = self.graphs.get(graph_key)
        if captured is None:
            static_features = features.new_zeros(batch_size, frame_bucket, feature_size)
            static_features[:, :frames].copy_(features)
            static_lengths = lengths.clone()
            for _ in range(2):
                encoded, encoder_lengths, _ = self.encoder(
                    static_features, static_lengths
                )
                self.projector(encoded, encoder_lengths)
            torch.get_device_module(features.device).synchronize()
            with self.backend.capture(pool=self.pool) as graph:
                encoded, encoder_lengths, _ = self.encoder(
                    static_features, static_lengths
                )
                embeddings, _ = self.projector(encoded, encoder_lengths)
            captured = EncoderGraph(
                graph=graph,
                features=static_features,
                lengths=static_lengths,
                embeddings=embeddings,
            )
            self.graphs[graph_key] = captured
            logger.info(
                f"Captured FireRed encoder graph: batch={batch_size}, frames={frame_bucket}"
            )
        else:
            captured.features.zero_()
            captured.features[:, :frames].copy_(features)
            captured.lengths.copy_(lengths)
        captured.graph.replay()
        self.replay_count += 1
        logger.debug(f"Replayed FireRed encoder graph: count={self.replay_count}")
        # note (cocoa00): Microbatches share graph storage; retain each result before replay.
        return captured.embeddings.clone()
