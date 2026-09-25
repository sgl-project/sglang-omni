"""Realtime unit identity, adapter contracts and resource configuration."""

from __future__ import annotations

import math
from collections.abc import Awaitable
from dataclasses import asdict, dataclass
from typing import Protocol, get_args

from sglang_omni.serve.realtime.control import ControlEvent
from sglang_omni.serve.realtime.output import OutputEvent
from sglang_omni.serve.realtime.schema import (
    GrantedCapabilities,
    Interaction,
    PartialStyle,
    SessionConfiguration,
    TailPolicy,
)

DEFAULT_SAMPLE_RATE_HZ = 16000
PCM16_BYTES_PER_SAMPLE = 2
MS_PER_SECOND = 1000
MAX_BUFFERED_INPUT_S = 60
SUPPORTED_OUTPUT_MODALITIES = frozenset({"text", "audio"})


class ProtocolError(ValueError):
    def __init__(self, code: str, message: str, param: str | None = None) -> None:
        self.code = code
        self.param = param
        super().__init__(message)


def samples_to_ms(sample_count: int, sample_rate_hz: int) -> float:
    return sample_count * MS_PER_SECOND / sample_rate_hz


@dataclass(frozen=True)
class RuntimeLimits:
    max_input_bytes: int = (
        MAX_BUFFERED_INPUT_S * DEFAULT_SAMPLE_RATE_HZ * PCM16_BYTES_PER_SAMPLE
    )
    max_output_bytes: int = 4 * 1024 * 1024
    max_output_events: int = 256
    max_history_chars: int = 64 * 1024
    cleanup_timeout_s: float = 30

    def __post_init__(self) -> None:
        if any(
            not math.isfinite(limit) or limit <= 0 for limit in asdict(self).values()
        ):
            raise ValueError("runtime limits must be finite and positive")
        else:
            pass


@dataclass(frozen=True)
class Capabilities:
    interaction: Interaction = "native"
    input_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    output_sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ
    output_modalities: tuple[str, ...] = ("text",)
    native_unit_ms: int = 20
    tail_policy: TailPolicy = "flush"
    partial_style: PartialStyle = "append_only"

    def __post_init__(self) -> None:
        if self.interaction != "native":
            raise ValueError("unsupported interaction")
        elif (
            self.input_sample_rate_hz <= 0
            or self.output_sample_rate_hz <= 0
            or self.native_unit_ms <= 0
        ):
            raise ValueError("positive rates and cadence required")
        elif self.input_sample_rate_hz * self.native_unit_ms % MS_PER_SECOND:
            raise ValueError("native cadence must contain whole samples")
        elif self.tail_policy not in get_args(TailPolicy):
            raise ValueError("unsupported tail policy")
        elif (
            not self.output_modalities
            or set(self.output_modalities) - SUPPORTED_OUTPUT_MODALITIES
        ):
            raise ValueError("unsupported output modalities")
        else:
            pass

    @property
    def native_unit_bytes(self) -> int:
        native_unit_samples = (
            self.input_sample_rate_hz * self.native_unit_ms // MS_PER_SECOND
        )
        return native_unit_samples * PCM16_BYTES_PER_SAMPLE

    def input_duration_ms(self, sample_count: int) -> float:
        return samples_to_ms(sample_count, self.input_sample_rate_hz)

    def to_granted_capabilities(self) -> GrantedCapabilities:
        return dict(
            interaction=self.interaction,
            native_full_duplex=self.interaction == "native",
            proactive_output=False,
            turn_control=[None],
            client_commit=False,
            input_modalities=["audio"],
            output_modalities=list(self.output_modalities),
            input_audio_format=dict(type="audio/pcm", rate=self.input_sample_rate_hz),
            output_audio_format=dict(type="audio/pcm", rate=self.output_sample_rate_hz),
            native_unit_ms=self.native_unit_ms,
            first_unit_ms=self.native_unit_ms,
            microturn_ms="variable",
            tail_policy=self.tail_policy,
            supports_server_interrupt=False,
            supports_truncate=False,
            supports_resume=False,
            partial_style=self.partial_style,
            pressure_policy="reject",
            strict_order=True,
        )


@dataclass(frozen=True)
class Unit:
    index: int
    start_sample: int
    pcm: bytes
    real_samples: int
    eos: bool = False
    output_modalities: tuple[str, ...] | None = None

    @property
    def unit_id(self) -> str:
        return f"unit_{self.index}"


@dataclass(frozen=True, kw_only=True)
class Envelope:
    event: OutputEvent | ControlEvent
    # Note (Junnan Li): Control acknowledgements and response terminals survive the purge at close.
    is_control: bool = False
    unit: Unit | None = None
    chunk_index: int = 0
    output_modalities: tuple[str, ...] | None = None


class OutputSink(Protocol):
    def __call__(
        self, event: OutputEvent, unit: Unit | None = None
    ) -> Awaitable[None]: ...


class InteractionAdapter:
    """Adapters override what they support; the runtime never probes for methods."""

    def set_limits(self, limits: RuntimeLimits) -> None:
        pass

    async def open(
        self, session_id: str, config: SessionConfiguration, emit: OutputSink
    ) -> None:
        raise NotImplementedError

    async def process(self, unit: Unit) -> int | tuple[int, int]:
        raise NotImplementedError

    async def clear(self) -> int:
        return 0

    async def close(self) -> None:
        raise NotImplementedError


class AdapterFactory(Protocol):
    def __call__(self) -> InteractionAdapter: ...
