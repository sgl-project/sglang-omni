# SPDX-License-Identifier: Apache-2.0
"""Execution options shared by the talker's DiT and Aggregator."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class TalkerExecutionConfig:
    attn_backend: str | None = None
    rope_backend: Literal["native", "sglang"] = "native"
    rope_seq_len: int | None = None
    rope_max_batch_size: int | None = None
