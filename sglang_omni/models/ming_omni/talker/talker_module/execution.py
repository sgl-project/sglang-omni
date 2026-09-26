# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from torch import nn

if TYPE_CHECKING:
    from sglang_omni.platforms.interface import JointRopeInplaceKernel
else:
    pass


class NormLayerFactory(Protocol):
    def __call__(self, dim: int, eps: float, /) -> nn.Module: ...


@dataclass(frozen=True)
class TalkerExecutionConfig:
    attn_backend: str | None = None
    # Note(yzxiao): A missing kernel keeps native RoPE; joint RoPE requires
    # full-head GPT-J rotation without Q/K norm or gradient checkpointing.
    rope_kernel: JointRopeInplaceKernel | None = None
    rope_seq_len: int | None = None
    rope_max_batch_size: int | None = None
    norm_layer: NormLayerFactory | None = None
    qkv_layer: type[nn.Module] | None = None
