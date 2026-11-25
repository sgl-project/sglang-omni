# SPDX-License-Identifier: Apache-2.0
"""Model-specific contracts for the shared duplex session benchmark."""

from dataclasses import dataclass
from typing import Literal

ProfileName = Literal[
    "nemotron-voicechat-pr2188",
    "minicpmo-native-pr2377",
    "qwen3-omni-d2-80ms",
    "qwen3-omni-d2-160ms",
    "qwen3-omni-d2-320ms",
    "qwen3-omni-d2-640ms",
    "qwen3-omni-d2-1040ms",
    "qwen3-omni-half-duplex",
    "turn-based",
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
    supports_server_interrupt: bool = False


QWEN_D2_INTERVALS_MS: dict[ProfileName, int] = {
    "qwen3-omni-d2-80ms": 80,
    "qwen3-omni-d2-160ms": 160,
    "qwen3-omni-d2-320ms": 320,
    "qwen3-omni-d2-640ms": 640,
    "qwen3-omni-d2-1040ms": 1040,
}

PROFILES: dict[ProfileName, DuplexProfile] = {
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
    **{
        profile: DuplexProfile(
            native_unit_ms=interval_ms,
            output_sample_rate=24000,
            stop_requires_eos=True,
            continuous_output=True,
            protocol="native",
            supports_server_interrupt=True,
        )
        for profile, interval_ms in QWEN_D2_INTERVALS_MS.items()
    },
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
