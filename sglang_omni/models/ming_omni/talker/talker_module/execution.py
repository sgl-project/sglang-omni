# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.platforms.interface import JointRopeInplaceKernel


@dataclass(frozen=True)
class TalkerExecutionConfig:
    attn_backend: str | None = None
    # Note(yzxiao): None keeps the shared Ming-Omni native path. Binding a
    # kernel selects full-head GPT-J rotation without Q/K norm or gradient
    # checkpointing in these acoustic components.
    rope_kernel: JointRopeInplaceKernel | None = None
    rope_seq_len: int | None = None
    rope_max_batch_size: int | None = None
