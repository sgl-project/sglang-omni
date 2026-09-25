"""Validate and negotiate realtime session configuration."""

import copy
from dataclasses import asdict, dataclass

from pydantic import ValidationError

from sglang_omni.serve.realtime.schema import (
    GrantedCapabilities,
    SessionConfiguration,
    SessionState,
    SessionType,
    SessionUpdateRequest,
)
from sglang_omni.serve.realtime.types import Capabilities, ProtocolError, RuntimeLimits


def merge_config(
    current: dict[str, object], patch: dict[str, object]
) -> dict[str, object]:
    merged_config = copy.deepcopy(current)
    for key, patch_value in patch.items():
        current_value = merged_config.get(key)
        if isinstance(patch_value, dict) and isinstance(current_value, dict):
            merged_config[key] = merge_config(current_value, patch_value)
        else:
            merged_config[key] = copy.deepcopy(patch_value)
    return merged_config


def admission_fields(config: SessionConfiguration) -> SessionConfiguration:
    """Fields fixed once the session is OPEN; everything else may be hot-updated."""
    admission_config = copy.deepcopy(config)
    admission_config.pop("output_modalities", None)
    admission_config.get("audio", {}).get("input", {}).pop("turn_detection", None)
    return admission_config


@dataclass(kw_only=True)
class SessionNegotiation:
    model: str
    capabilities: Capabilities
    limits: RuntimeLimits

    def negotiate(
        self, current: SessionConfiguration, state: SessionState, patch: object
    ) -> tuple[SessionConfiguration, GrantedCapabilities]:
        try:
            candidate = SessionUpdateRequest.model_validate(
                {
                    "session": (
                        merge_config(dict(current), patch)
                        if isinstance(patch, dict)
                        else patch
                    )
                }
            ).session
        except ValidationError as exc:
            location = ".".join(str(part) for part in exc.errors()[0]["loc"])
            raise ProtocolError(
                "invalid_request", "invalid session configuration", location
            ) from exc
        if state == "OPEN" and admission_fields(candidate) != admission_fields(current):
            raise ProtocolError("invalid_state", "session field is frozen")
        elif (
            "instructions" in candidate
            and len(candidate["instructions"]) > self.limits.max_history_chars
        ):
            raise ProtocolError(
                "invalid_request", "instructions exceed context or have invalid type"
            )
        elif candidate.get("model", self.model) != self.model:
            raise ProtocolError("invalid_request", "model differs from deployment")
        elif candidate.get("type", "realtime") != "realtime":
            raise ProtocolError("invalid_request", "session type is unavailable")
        else:
            pass
        self.validate_audio(candidate)
        requested_modalities = candidate.get(
            "output_modalities", list(self.capabilities.output_modalities)
        )
        granted_modalities = [
            modality
            for modality in requested_modalities
            if modality in self.capabilities.output_modalities
        ][:1]
        if not granted_modalities:
            raise ProtocolError("invalid_request", "no supported output combination")
        else:
            pass
        self.validate_extension(candidate)
        return self.grant(candidate, requested_modalities, granted_modalities)

    def validate_audio(self, candidate: SessionConfiguration) -> None:
        audio = candidate.get("audio", {})
        for direction, audio_format, sample_rate_hz in (
            (
                "input",
                audio.get("input", {}).get("format"),
                self.capabilities.input_sample_rate_hz,
            ),
            (
                "output",
                audio.get("output", {}).get("format"),
                self.capabilities.output_sample_rate_hz,
            ),
        ):
            if audio_format is not None and audio_format != dict(
                type="audio/pcm", rate=sample_rate_hz
            ):
                raise ProtocolError(
                    "invalid_request",
                    "unsupported PCM format or sample rate",
                    f"session.audio.{direction}.format",
                )
            else:
                pass
        if audio.get("input", {}).get("turn_detection") is not None:
            raise ProtocolError(
                "not_applicable", "VAD is only available for turn-based sessions"
            )
        else:
            pass

    def validate_extension(self, candidate: SessionConfiguration) -> None:
        extension = candidate.get("sglang", {})
        native_unit_ms = extension.get("timebase", {}).get(
            "native_unit_ms", self.capabilities.native_unit_ms
        )
        if (
            extension.get("tail_policy", self.capabilities.tail_policy)
            != self.capabilities.tail_policy
        ):
            raise ProtocolError("invalid_request", "tail policy is unavailable")
        elif native_unit_ms != self.capabilities.native_unit_ms:
            raise ProtocolError("invalid_request", "native cadence is fixed")
        else:
            pass

    def grant(
        self,
        candidate: SessionConfiguration,
        requested_modalities: list[str],
        granted_modalities: list[str],
    ) -> tuple[SessionConfiguration, GrantedCapabilities]:
        granted = self.capabilities.to_granted_capabilities()
        granted.update(
            {
                "output_modalities": granted_modalities,
                "limits": asdict(self.limits),
                "rejections": [],
            }
        )
        requested_microturn_ms = (
            candidate.get("sglang", {}).get("timebase", {}).get("microturn_ms")
        )
        if requested_microturn_ms is not None:
            granted["rejections"].append(
                dict(
                    field="sglang.timebase.microturn_ms",
                    requested=requested_microturn_ms,
                    reason="external chunks are variable length; no fixed external cadence",
                    granted=None,
                )
            )
        else:
            pass
        granted["microturn_ms"] = None
        if granted_modalities != requested_modalities:
            granted["rejections"].append(
                dict(
                    field="output_modalities",
                    requested=requested_modalities,
                    reason="deployment modalities",
                    granted=granted_modalities,
                )
            )
        else:
            pass
        audio = candidate.setdefault("audio", {})
        audio.setdefault("input", {}).setdefault(
            "format",
            dict(type="audio/pcm", rate=self.capabilities.input_sample_rate_hz),
        )
        audio.setdefault("output", {}).setdefault(
            "format",
            dict(type="audio/pcm", rate=self.capabilities.output_sample_rate_hz),
        )
        session_type: SessionType = "realtime"
        candidate.update(
            {
                "model": self.model,
                "type": session_type,
                "output_modalities": granted_modalities,
            }
        )
        return candidate, granted
