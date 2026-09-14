# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Any, Mapping, Protocol

from .semantic_vad import SemanticEOUModel, SemanticTurnDetector, SemanticVADConfig
from .vad import Emit, StreamingVAD, VADConfig


class TurnDetector(Protocol):
    def process(self, pcm_bytes: bytes) -> list[Emit]: ...

    def reset(self) -> None: ...


@dataclass(frozen=True)
class TurnDetectorBuild:
    detector: TurnDetector
    effective_config: dict[str, Any]


def build_turn_detector(
    config: Mapping[str, Any],
    smart_turn_model: SemanticEOUModel | None,
) -> TurnDetectorBuild:
    raw_type = config.get("type")
    requested_type = str(getattr(raw_type, "value", raw_type) or "server_vad")
    if requested_type == "semantic_vad" and smart_turn_model is not None:
        eagerness = str(config.get("eagerness") or "medium")
        if eagerness == "auto":
            eagerness = "medium"
        semantic_config = replace(
            SemanticVADConfig.from_eagerness(eagerness),
            speech_threshold=_optional_float(
                config.get("threshold"),
                SemanticVADConfig.speech_threshold,
                minimum=0.0,
                maximum=1.0,
            ),
            prefix_padding_ms=_optional_int(
                config.get("prefix_padding_ms"),
                SemanticVADConfig.prefix_padding_ms,
                minimum=0,
            ),
        )
        detector = SemanticTurnDetector(
            smart_turn_model,
            semantic_config,
        )
        effective = dict(config)
        effective["type"] = "semantic_vad"
        effective["eagerness"] = eagerness
        effective.pop("silence_duration_ms", None)
        return TurnDetectorBuild(detector, effective)

    server_config = VADConfig(
        threshold=_optional_float(
            config.get("threshold"), VADConfig.threshold, minimum=0.0, maximum=1.0
        ),
        prefix_padding_ms=_optional_int(
            config.get("prefix_padding_ms"), VADConfig.prefix_padding_ms, minimum=0
        ),
        silence_duration_ms=_optional_int(
            config.get("silence_duration_ms"),
            VADConfig.silence_duration_ms,
            minimum=0,
        ),
    )
    effective = dict(config)
    effective["type"] = "server_vad"
    effective["eagerness"] = None
    return TurnDetectorBuild(StreamingVAD(server_config), effective)


def _optional_float(
    value: Any,
    default: float,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
) -> float:
    if value is None:
        return default
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Value must be finite, got {value!r}")
    if minimum is not None and result < minimum:
        raise ValueError(f"Value must be >= {minimum}, got {result}")
    if maximum is not None and result > maximum:
        raise ValueError(f"Value must be <= {maximum}, got {result}")
    return result


def _optional_int(value: Any, default: int, *, minimum: int | None = None) -> int:
    if value is None:
        return default
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"Value must be >= {minimum}, got {result}")
    return result
