# SPDX-License-Identifier: Apache-2.0
"""State handed from the Stable Audio 3 preprocessing stage to the generation stage."""

from dataclasses import dataclass

from sglang_omni.scheduling.pipeline_state import DeclarativeStateBase, wire


@dataclass
class StableAudio3State(DeclarativeStateBase):
    prompt: str = wire("", codec="str")
    duration_seconds: float = wire(0.0, codec="float")
    seed: int = wire(0, codec="int")
