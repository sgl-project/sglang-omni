# SPDX-License-Identifier: Apache-2.0
"""State handed from the YuE2 AR stage to the acoustic stage."""

from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class Yue2State(DeclarativeStateBase):
    semantic_prefix: list[int] = wire(default_factory=list, codec="list")
    semantic_tokens: list[int] = wire(default_factory=list, codec="list")
    seed: int = wire(0, codec="int")
    ode_steps: int = wire(0, codec="int")
    abc_truncated: bool = wire(False, codec="bool")
    semantic_truncated: bool = wire(False, codec="bool")
