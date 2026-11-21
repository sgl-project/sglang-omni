# SPDX-License-Identifier: Apache-2.0
"""Fakes for MoE blocks that build a TopK and the experts it must be bound to."""

from sglang.srt.layers.moe.utils import RoutingMethodType
from torch import nn


class RecordingTopK(nn.Module):
    def __init__(self, **kwargs: object) -> None:
        super().__init__()
        self.bound: tuple[str, RoutingMethodType | None] | None = None

    def bind_runner_backend(
        self,
        runner_backend: str,
        *,
        routing_method_type: RoutingMethodType | None = None,
    ) -> None:
        self.bound = (runner_backend, routing_method_type)


class FakeExperts(nn.Module):
    runner_backend = "triton"
    routing_method_type = RoutingMethodType.Renormalize

    def __init__(self, **kwargs: object) -> None:
        super().__init__()
