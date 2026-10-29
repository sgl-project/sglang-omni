# SPDX-License-Identifier: Apache-2.0
"""Time-operation tables and numerical qualification for optimized Flow."""

from __future__ import annotations

import math
import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal, cast

import torch
from torch._ops import OpOverload
from torch.utils._python_dispatch import TorchDispatchMode

from sglang_omni.models.firered_tts3.compiled_dit import CompiledDiTExecutor
from sglang_omni.models.firered_tts3.dit import DiT
from sglang_omni.models.firered_tts3.weights import TimeParameter

# note (Codex): these are numerical screening budgets, not audio-quality guarantees.
DEFAULT_FLOW_ATOL = 1e-3
DEFAULT_FLOW_RTOL = 1e-2
NRMSE_REFERENCE_FLOOR = 1e-12


class TimeCacheUnavailable(RuntimeError):
    """The caller must retry the complete flow using the original compiled DiT."""


def finite_byte_equal(actual: torch.Tensor, expected: torch.Tensor) -> bool:
    if actual.shape != expected.shape or actual.dtype != expected.dtype:
        return False
    else:
        return bool(torch.isfinite(actual).all().item()) and torch.equal(
            actual.contiguous().view(torch.uint8),
            expected.contiguous().view(torch.uint8),
        )


@dataclass(frozen=True, kw_only=True)
class GemmSignature:
    input_shape: tuple[int, ...]
    input_stride: tuple[int, ...]
    weight_shape: tuple[int, ...]
    weight_stride: tuple[int, ...]
    bias_shape: tuple[int, ...]
    bias_stride: tuple[int, ...]
    output_shape: tuple[int, ...]
    output_stride: tuple[int, ...]
    dtype: torch.dtype
    device: torch.device


@dataclass(frozen=True, kw_only=True)
class TimeGemmValue:
    signature: GemmSignature
    output: torch.Tensor


@dataclass(frozen=True, kw_only=True)
class TimeGemmTable:
    """Graph entries must retain this table until their graphs are destroyed."""

    model_id: int
    n_timesteps: int
    steps: tuple[Mapping[str, TimeGemmValue], ...]

    @property
    def nbytes(self) -> int:
        return sum(
            value.output.numel() * value.output.element_size()
            for step in self.steps
            for value in step.values()
        )


@dataclass(kw_only=True)
class TimeGemmCall:
    index: int
    seen: set[str] = field(default_factory=set)


class TimeGemmMode(TorchDispatchMode):
    """Copy exact original addmm outputs into their original destination buffers."""

    def __init__(
        self,
        weights: Mapping[int, str],
        *,
        phase: Literal["collect", "cached"],
        n_timesteps: int,
        table: TimeGemmTable | None = None,
    ) -> None:
        super().__init__()
        if phase == "cached" and table is None:
            raise TimeCacheUnavailable(
                "Cached time execution requires a prepared table"
            )
        else:
            pass
        self.weights = weights
        self.phase = phase
        self.n_timesteps = n_timesteps
        self.table = table
        self.collected: list[dict[str, TimeGemmValue]] = [
            {} for _ in range(n_timesteps)
        ]
        self.active_call: ContextVar[TimeGemmCall | None] = ContextVar(
            "firered_time_gemm_call", default=None
        )

    @contextmanager
    def step(self, index: int) -> Iterator[None]:
        if self.active_call.get() is not None or not 0 <= index < self.n_timesteps:
            raise TimeCacheUnavailable(f"Invalid or nested time step {index}")
        else:
            pass
        call = TimeGemmCall(index=index)
        token = self.active_call.set(call)
        try:
            yield
            missing = set(self.weights.values()) - call.seen
            if missing:
                raise TimeCacheUnavailable(
                    f"Time step {index} missed compiled addmm operations: {sorted(missing)}"
                )
            else:
                pass
        finally:
            self.active_call.reset(token)

    def __torch_dispatch__(
        self,
        func: OpOverload,
        types: tuple[type, ...],
        args: tuple[object, ...] = (),
        kwargs: dict[str, object] | None = None,
    ) -> object:
        kwargs = kwargs or {}
        if func is not torch.ops.aten.addmm.out:
            return func(*args, **kwargs)
        else:
            pass
        bias, inputs, weight = cast(
            tuple[torch.Tensor, torch.Tensor, torch.Tensor], args
        )
        name = self.weights.get(weight.data_ptr())
        if name is None:
            return func(*args, **kwargs)
        else:
            pass
        call = self.active_call.get()
        if call is None or name in call.seen:
            raise TimeCacheUnavailable(f"Unscoped or repeated time addmm: {name}")
        else:
            pass
        if kwargs.get("alpha", 1) != 1 or kwargs.get("beta", 1) != 1:
            raise TimeCacheUnavailable(f"Unsupported time addmm scaling: {name}")
        else:
            pass
        call.seen.add(name)
        out = cast(torch.Tensor, kwargs["out"])
        signature = GemmSignature(
            input_shape=tuple(inputs.shape),
            input_stride=tuple(inputs.stride()),
            weight_shape=tuple(weight.shape),
            weight_stride=tuple(weight.stride()),
            bias_shape=tuple(bias.shape),
            bias_stride=tuple(bias.stride()),
            output_shape=tuple(out.shape),
            output_stride=tuple(out.stride()),
            dtype=out.dtype,
            device=out.device,
        )
        if self.phase == "collect":
            if name in self.collected[call.index]:
                raise TimeCacheUnavailable(f"Repeated time table entry: {name}")
            else:
                pass
            result = func(*args, **kwargs)
            self.collected[call.index][name] = TimeGemmValue(
                signature=signature,
                output=out.clone(memory_format=torch.preserve_format),
            )
            return result
        else:
            assert self.table is not None
            value = self.table.steps[call.index].get(name)
            if value is None or value.signature != signature:
                raise TimeCacheUnavailable(
                    f"Time table signature changed: step={call.index}, operation={name}"
                )
            else:
                pass
            out.copy_(value.output)
            return out


class TimeConditionCache:
    """Prepare tables owned by their active graph entries and dispatch modes."""

    def __init__(self, dit: DiT) -> None:
        self.source = dit
        weights = {
            module.weight.data_ptr(): name
            for name, module in dit.named_modules()
            if isinstance(module, torch.nn.Linear)
            and (name.startswith("t_embedder.") or ".adaLN_modulation." in name)
        }
        expected = 2 + len(dit.blocks) + 1
        if len(weights) != expected:
            raise TimeCacheUnavailable(
                f"Expected {expected} independent time Linear weights, found {len(weights)}"
            )
        else:
            pass
        self.weights = MappingProxyType(weights)
        self.parameters = {
            name: TimeParameter(
                pointer=parameter.data_ptr(),
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                dtype=parameter.dtype,
                device=parameter.device,
                version=(
                    None if torch.is_inference(parameter) else parameter._version
                ),  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            )
            for name, parameter in dit.named_parameters()
            if name.startswith("t_embedder.") or ".adaLN_modulation." in name
        }
        self.version_checks_available = all(
            parameter.version is not None for parameter in self.parameters.values()
        )

    def validate_weights(self) -> None:
        """Recreate the runner after weight changes, including unversioned writes."""
        current = dict(self.source.named_parameters())
        for name, expected in self.parameters.items():
            parameter = current.get(name)
            if parameter is None:
                raise TimeCacheUnavailable(f"Time parameter was removed: {name}")
            else:
                pass
            actual = TimeParameter(
                pointer=parameter.data_ptr(),
                shape=tuple(parameter.shape),
                stride=tuple(parameter.stride()),
                dtype=parameter.dtype,
                device=parameter.device,
                version=(
                    None if torch.is_inference(parameter) else parameter._version
                ),  # noqa: SLF001  # leading-underscore: PyTorch tensor version API
            )
            if actual != expected:
                raise TimeCacheUnavailable(
                    f"Time parameter changed; rebuild cache: {name}"
                )
            else:
                pass

    def prepare(
        self,
        executor: CompiledDiTExecutor,
        example_x: torch.Tensor,
        t_span: torch.Tensor,
        n_timesteps: int,
    ) -> TimeGemmTable:
        self.validate_weights()
        if executor.source is not self.source:
            raise TimeCacheUnavailable(
                "Time cache and executor have different model weights"
            )
        else:
            pass
        if (
            example_x.device.type != "cuda"
            or example_x.ndim != 3
            or example_x.shape[0] <= 0
            or example_x.shape[0] % 2
            or example_x.dtype not in (torch.float32, torch.bfloat16)
            or t_span.dtype is not torch.float32
            or t_span.ndim != 1
            or n_timesteps <= 0
            or t_span.numel() != n_timesteps + 1
            or t_span.device != example_x.device
        ):
            raise TimeCacheUnavailable(
                "Unsupported time-table input or schedule schema"
            )
        else:
            pass
        if torch.cuda.is_current_stream_capturing():
            raise TimeCacheUnavailable(
                "Time tables must be prepared before graph capture"
            )
        else:
            pass
        collector = TimeGemmMode(self.weights, phase="collect", n_timesteps=n_timesteps)
        batch_size = example_x.shape[0] // 2
        with torch.inference_mode():
            for index in range(n_timesteps):
                t = t_span[index].view(-1, 1, 1).expand(batch_size, -1, -1)
                t = torch.cat([t, t], dim=0).to(example_x.dtype)
                reference = executor.original(x=example_x, t=t)
                with collector.step(index), executor.dispatch((collector,)):
                    observed = executor.run(x=example_x, t=t)
                if not finite_byte_equal(observed, reference):
                    raise TimeCacheUnavailable(
                        f"Observed original graph changed output at time step {index}"
                    )
                else:
                    pass
        table = TimeGemmTable(
            model_id=id(self.source),
            n_timesteps=n_timesteps,
            steps=tuple(MappingProxyType(dict(row)) for row in collector.collected),
        )
        mode = self.mode(table)
        with torch.inference_mode():
            for index in range(n_timesteps):
                t = t_span[index].view(-1, 1, 1).expand(batch_size, -1, -1)
                t = torch.cat([t, t], dim=0).to(example_x.dtype)
                reference = executor.original(x=example_x, t=t)
                with mode.step(index), executor.dispatch((mode,)):
                    cached = executor.run(x=example_x, t=t)
                if not finite_byte_equal(cached, reference):
                    raise TimeCacheUnavailable(
                        f"Cached original graph changed output at time step {index}"
                    )
                else:
                    pass
        return table

    def mode(self, table: TimeGemmTable) -> TimeGemmMode:
        if table.model_id != id(self.source):
            raise TimeCacheUnavailable("Time table belongs to different model weights")
        else:
            return TimeGemmMode(
                self.weights, phase="cached", n_timesteps=table.n_timesteps, table=table
            )


@dataclass(frozen=True, kw_only=True)
class FlowPrecisionResult:
    accepted: bool
    byte_exact: bool
    schema_valid: bool
    finite: bool
    max_abs: float | None = None
    rmse: float | None = None
    worst_row_nrmse: float | None = None
    out_of_bounds: int | None = None


@dataclass(frozen=True, kw_only=True)
class FlowPrecisionGate:
    """Use exact checks by default; load the convolution budget explicitly from env."""

    atol: float = 0.0
    rtol: float = 0.0

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value) and value >= 0 for value in (self.atol, self.rtol)
        ):
            raise ValueError("Flow tolerances must be finite non-negative numbers")
        else:
            pass

    @classmethod
    def from_environment(cls) -> FlowPrecisionGate:
        try:
            return cls(
                atol=float(
                    os.environ.get("FIRERED_TTS3_FLOW_ATOL", str(DEFAULT_FLOW_ATOL))
                ),
                rtol=float(
                    os.environ.get("FIRERED_TTS3_FLOW_RTOL", str(DEFAULT_FLOW_RTOL))
                ),
            )
        except ValueError as error:
            raise ValueError(
                "FIRERED_TTS3_FLOW_ATOL and FIRERED_TTS3_FLOW_RTOL "
                "must be finite non-negative numbers"
            ) from error

    def compare(
        self, actual: torch.Tensor, expected: torch.Tensor
    ) -> FlowPrecisionResult:
        schema_valid = (
            actual.shape == expected.shape
            and actual.dtype == expected.dtype
            and actual.device == expected.device
            and actual.is_floating_point()
            and actual.ndim >= 1
            and actual.numel() > 0
        )
        if not schema_valid:
            return FlowPrecisionResult(
                accepted=False, byte_exact=False, schema_valid=False, finite=False
            )
        elif not bool(torch.isfinite(actual).all() & torch.isfinite(expected).all()):
            return FlowPrecisionResult(
                accepted=False, byte_exact=False, schema_valid=True, finite=False
            )
        else:
            byte_exact = finite_byte_equal(actual, expected)
            reference = expected.double()
            delta = actual.double() - reference
            error = delta.abs()
            out_of_bounds = int(
                (error > self.atol + self.rtol * reference.abs()).sum().item()
            )
            row_error = delta.reshape(delta.shape[0], -1).square().mean(dim=1).sqrt()
            row_reference = (
                reference.reshape(reference.shape[0], -1).square().mean(dim=1).sqrt()
            )
            # note (Codex): zero reference rows need a finite diagnostic denominator.
            row_nrmse = row_error / row_reference.clamp_min(NRMSE_REFERENCE_FLOOR)
            return FlowPrecisionResult(
                accepted=(
                    byte_exact if self.atol == self.rtol == 0 else out_of_bounds == 0
                ),
                byte_exact=byte_exact,
                schema_valid=True,
                finite=True,
                max_abs=float(error.max().item()),
                rmse=float(delta.square().mean().sqrt().item()),
                worst_row_nrmse=float(row_nrmse.max().item()),
                out_of_bounds=out_of_bounds,
            )
