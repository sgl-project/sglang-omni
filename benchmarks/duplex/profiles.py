# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for the shared native session benchmark."""

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal["nemotron-voicechat-pr2188", "minicpmo-native-pr2377"]
DEFAULT_PROFILE: ProfileName = "nemotron-voicechat-pr2188"


@dataclass(frozen=True, kw_only=True)
class DuplexProfile:
    native_unit_ms: int
    output_sample_rate: int
    output_modalities: tuple[str, ...]
    stop_requires_eos: bool
    continuous_output: bool


PROFILES: dict[ProfileName, DuplexProfile] = {
    "nemotron-voicechat-pr2188": DuplexProfile(
        native_unit_ms=80,
        output_sample_rate=22050,
        output_modalities=("audio",),
        stop_requires_eos=True,
        continuous_output=True,
    ),
    "minicpmo-native-pr2377": DuplexProfile(
        native_unit_ms=1000,
        output_sample_rate=24000,
        output_modalities=("audio", "text"),
        stop_requires_eos=False,
        continuous_output=False,
    ),
}
