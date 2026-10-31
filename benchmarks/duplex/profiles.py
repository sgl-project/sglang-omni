# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for the shared duplex session benchmark."""

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal[
    "nemotron-voicechat-pr2188",
    "minicpmo-native-pr2377",
    "qwen3-omni-half-duplex",
    "turn-based",
    "personaplex",
]
DEFAULT_PROFILE: ProfileName = "nemotron-voicechat-pr2188"
# note (luojiaxuan): "native" is the full-duplex session protocol with sglang.*
# receipts; "legacy" is the turn-based conversation protocol, with `modalities` and a
# top-level `turn_detection`; "turn_based" is the session protocol with sglang.*
# receipts whose server takes turns by server VAD and answers each one.
Protocol = Literal["native", "legacy", "turn_based"]


@dataclass(frozen=True, kw_only=True)
class DuplexProfile:
    native_unit_ms: int
    output_sample_rate: int
    stop_requires_eos: bool
    continuous_output: bool
    protocol: Protocol
    input_sample_rate: int = 16000
    initial_extra_output_frames: int = 0


PROFILES: dict[ProfileName, DuplexProfile] = {
    "personaplex": DuplexProfile(
        native_unit_ms=80,
        input_sample_rate=24000,
        output_sample_rate=24000,
        stop_requires_eos=True,
        continuous_output=True,
        protocol="native",
        initial_extra_output_frames=1,
    ),
    "nemotron-voicechat-pr2188": DuplexProfile(
        native_unit_ms=80,
        output_sample_rate=22050,
        stop_requires_eos=True,
        continuous_output=True,
        protocol="native",
    ),
    "minicpmo-native-pr2377": DuplexProfile(
        native_unit_ms=1000,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="native",
    ),
    "qwen3-omni-half-duplex": DuplexProfile(
        native_unit_ms=20,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="legacy",
    ),
    "turn-based": DuplexProfile(
        native_unit_ms=200,
        output_sample_rate=24000,
        stop_requires_eos=False,
        continuous_output=False,
        protocol="turn_based",
    ),
}
