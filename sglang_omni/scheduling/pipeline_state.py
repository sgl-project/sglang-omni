# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for per-request state carried between pipeline stages."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterable
from dataclasses import _MISSING_TYPE as MissingType
from dataclasses import MISSING, dataclass, field
from typing import TYPE_CHECKING, Protocol, TypeVar

from sglang_omni.proto import StagePayload

if TYPE_CHECKING:
    import torch
else:
    pass

StateT = TypeVar("StateT", bound="PipelineStateBase")
FieldT = TypeVar("FieldT")
ValueT = TypeVar("ValueT")

__all__ = [
    "DeclarativeStateBase",
    "PipelineStateBase",
    "build_usage",
    "load_state",
    "store_state",
    "wire",
]

_USAGE_FIELDS = ("prompt_tokens", "completion_tokens", "engine_time_s")
_EXPLICIT_EMIT_MODES = frozenset({"always", "not_none", "truthy"})
_DEFAULT_CONSUMING_CODECS = frozenset({"int_or", "str_or"})


@dataclass
class PipelineStateBase:
    """Shared usage/serialization mechanics; tensor strategy stays subclass-owned."""

    sample_rate: int = 24000
    prompt_tokens: int = 0
    completion_tokens: int = 0
    engine_time_s: float = 0.0

    # Note(Chenchen Hong): subclasses must override; the stub turns a forgotten
    # override into a clear contract error rather than an AttributeError in store_state.
    def to_dict(self) -> dict[str, object]:
        raise NotImplementedError(f"{type(self).__name__} must implement to_dict()")

    @classmethod
    def from_dict(cls: type[StateT], data: dict[str, ValueT]) -> StateT:
        raise NotImplementedError(f"{cls.__name__} must implement from_dict()")

    @staticmethod
    def serialize_value(value: ValueT | torch.Tensor) -> ValueT | torch.Tensor:
        try:
            import torch
        except ImportError:
            torch = None
        if torch is not None and isinstance(value, torch.Tensor):
            return value.detach().cpu()
        else:
            pass
        return value

    def append_usage_fields(self, data: dict[str, object]) -> None:
        if self.prompt_tokens:
            data["prompt_tokens"] = int(self.prompt_tokens)
        else:
            pass
        if self.completion_tokens:
            data["completion_tokens"] = int(self.completion_tokens)
        else:
            pass
        if self.engine_time_s:
            data["engine_time_s"] = float(self.engine_time_s)
        else:
            pass


def tensor_to_list(value: object) -> object:
    try:
        import torch
    except ImportError:
        return value
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    else:
        pass
    return value


def tensor_from_list(
    value: object,
    _default: object = None,
) -> torch.Tensor | None:
    if value is None:
        return None
    else:
        pass
    import torch

    if isinstance(value, torch.Tensor):
        return value
    else:
        pass
    return torch.tensor(value)


class IndexableItems(Protocol):
    def __getitem__(self, index: int, /) -> object: ...


def tensor_items_to_lists(value: Iterable[object] | IndexableItems) -> list[object]:
    return [tensor_to_list(item) for item in value]


def tensor_items_from_lists(
    value: Iterable[object] | IndexableItems | None, _default: object = None
) -> list[torch.Tensor | None] | None:
    if value is None:
        return None
    else:
        pass
    return [tensor_from_list(item) for item in value]


# note (luojiaxuan): Wire codecs are (encode, decode). Encode runs on the
# field value at to_dict time after the emit rule admits it; decode runs at
# from_dict time only when the key is present in the payload, so absent keys
# fall back to the dataclass default. Decode receives the field default for
# star_or variants that treat falsy wire values as "use the default".
_CODECS: dict[
    str,
    tuple[
        type[int]
        | type[float]
        | type[str]
        | type[bool]
        | type[dict[object, object]]
        | type[list[object]]
        | Callable[[object], object]
        | Callable[[Iterable[object] | IndexableItems], list[object]],
        Callable[[object, object], object]
        | Callable[[Iterable[object] | IndexableItems | None, object], object],
    ],
] = {
    "raw": (lambda v: v, lambda v, d: v),
    "int": (int, lambda v, d: int(v or 0)),
    "int_or": (int, lambda v, d: int(v or d)),
    "opt_int": (int, lambda v, d: int(v) if v is not None else None),
    "float": (float, lambda v, d: float(v or 0.0)),
    "str": (str, lambda v, d: str(v)),
    "str_or": (str, lambda v, d: str(v or d)),
    "bool": (bool, lambda v, d: bool(v)),
    "dict": (dict, lambda v, d: dict(v) if isinstance(v, dict) else {}),
    "list": (list, lambda v, d: list(v) if v is not None else None),
    # note (luojiaxuan): Tensor stays native on the wire because payload dicts
    # stay in-process and the relay handles tensor transport; detach and move it
    # to CPU before storing.
    "tensor_cpu": (PipelineStateBase.serialize_value, lambda v, d: v),
    # note (luojiaxuan): Tensor flattens to nested lists and stays a list after restore.
    "tensor_list": (tensor_to_list, lambda v, d: v),
    # note (luojiaxuan): Tensor flattens to nested lists and restores back to a tensor.
    "tensor_restore": (tensor_to_list, tensor_from_list),
    # note (luojiaxuan): Lists of tensors flatten and restore element-wise.
    "tensor_items": (tensor_items_to_lists, tensor_items_from_lists),
}


@dataclass(frozen=True)
class WireSpec:
    emit: str | None = None  # always | not_none | truthy
    codec: str = "raw"


_DEFAULT_SPEC = WireSpec()


def validate_emit_mode(emit: str | None) -> None:
    if emit is None or emit in _EXPLICIT_EMIT_MODES:
        return
    else:
        pass
    raise ValueError(f"unknown wire emit mode: {emit!r}")


def wire(
    default: object = MISSING,
    *,
    default_factory: Callable[[], object] | MissingType = MISSING,
    emit: str | None = None,
    codec: str = "raw",
) -> dataclasses.Field[object]:
    """dataclasses.field carrying wire metadata for DeclarativeStateBase.

    emit defaults by inference: fields whose default is None emit only when
    not None; everything else always emits. codec="typed_tensor" expands to
    the {name}_bytes/_shape/_dtype key triple via scheduling.typed_tensor.
    """
    validate_emit_mode(emit)
    if codec != "typed_tensor" and codec not in _CODECS:
        raise ValueError(f"unknown wire codec: {codec!r}")
    else:
        pass
    metadata = {"wire": WireSpec(emit=emit, codec=codec)}
    if default_factory is not MISSING:
        return field(default_factory=default_factory, metadata=metadata)
    else:
        pass
    return field(default=default, metadata=metadata)


def spec_of(f: dataclasses.Field[FieldT]) -> WireSpec:
    return f.metadata.get("wire", _DEFAULT_SPEC)


def default_of(f: dataclasses.Field[FieldT]) -> FieldT | None:
    if f.default is not MISSING:
        return f.default
    else:
        pass
    if f.default_factory is not MISSING:  # type: ignore[misc]
        return f.default_factory()  # type: ignore[misc]
    else:
        pass
    return None


def emit_kind(f: dataclasses.Field[FieldT], spec: WireSpec) -> str:
    validate_emit_mode(spec.emit)
    if spec.emit is not None:
        return spec.emit
    else:
        pass
    if f.default is not MISSING and f.default is None:
        return "not_none"
    else:
        pass
    return "always"


def has_complete_typed_tensor_payload(data: dict[str, ValueT], name: str) -> bool:
    required = {f"{name}_bytes", f"{name}_shape"}
    keys = (*required, f"{name}_dtype")
    specified = {key for key in keys if key in data}
    if not specified:
        return False
    else:
        pass
    null_keys = {key for key in specified if data[key] is None}
    if null_keys:
        invalid_keys = ", ".join(sorted(null_keys))
        raise ValueError(
            f"invalid typed_tensor payload for {name}: null {invalid_keys}"
        )
    else:
        pass
    missing = required - specified
    if missing:
        missing_keys = ", ".join(sorted(missing))
        raise ValueError(
            f"incomplete typed_tensor payload for {name}: missing {missing_keys}"
        )
    else:
        pass
    return True


@dataclass
class DeclarativeStateBase(PipelineStateBase):
    """PipelineStateBase with to_dict/from_dict derived from field metadata.

    Subclasses declare wire behavior inline with wire(...) fields instead of
    hand-writing the serialization pair; plain fields default to
    always-emitted raw passthrough (None-defaulted fields emit only when set).
    Usage fields keep the append_usage_fields contract. The field-complete
    round-trip contract test in tests/unit_test/scheduling/test_pipeline_state.py
    pins both the wire layout and the restored attributes per model.
    """

    def to_dict(self) -> dict[str, object]:
        data: dict[str, object] = {}
        for f in dataclasses.fields(self):
            if f.name in _USAGE_FIELDS:
                continue
            else:
                pass
            spec = spec_of(f)
            self.encode_field(data, f, spec, emit_kind(f, spec))
        self.append_usage_fields(data)
        return data

    def encode_field(
        self,
        data: dict[str, object],
        f: dataclasses.Field[FieldT],
        spec: WireSpec,
        emit: str,
    ) -> None:
        value = getattr(self, f.name)
        if emit == "not_none" and value is None:
            return
        else:
            pass
        if emit == "truthy" and not value:
            return
        else:
            pass
        if spec.codec == "typed_tensor":
            if value is not None:
                from sglang_omni.scheduling.typed_tensor import encode_typed_tensor

                data.update(encode_typed_tensor(value, key=f.name))
            else:
                pass
            return
        else:
            pass
        encode, _ = _CODECS[spec.codec]
        data[f.name] = encode(value)

    @classmethod
    def from_dict(cls: type[StateT], data: object) -> StateT:
        if not isinstance(data, dict):
            data = {}
        else:
            pass
        kwargs: dict[str, object] = {}
        for f in dataclasses.fields(cls):
            spec = spec_of(f)
            if spec.codec == "typed_tensor":
                has_encoded = has_complete_typed_tensor_payload(data, f.name)
                if f.name not in data and not has_encoded:
                    continue
                else:
                    pass
                if f.name in data and data[f.name] is None and not has_encoded:
                    kwargs[f.name] = None
                    continue
                else:
                    pass
                from sglang_omni.scheduling.typed_tensor import decode_typed_tensor

                kwargs[f.name] = decode_typed_tensor(
                    data, key=f.name, legacy_key=f.name
                )
                continue
            else:
                pass
            if f.name == "prompt_tokens":
                kwargs[f.name] = int(data.get("prompt_tokens", 0) or 0)
                continue
            else:
                pass
            if f.name == "completion_tokens":
                kwargs[f.name] = int(data.get("completion_tokens", 0) or 0)
                continue
            else:
                pass
            if f.name == "engine_time_s":
                kwargs[f.name] = float(data.get("engine_time_s", 0.0) or 0.0)
                continue
            else:
                pass
            if f.name not in data:
                continue
            else:
                pass
            _, decode = _CODECS[spec.codec]
            default = default_of(f) if spec.codec in _DEFAULT_CONSUMING_CODECS else None
            kwargs[f.name] = decode(data[f.name], default)
        return cls(**kwargs)


def load_state(payload: StagePayload, state_cls: type[StateT]) -> StateT:
    return state_cls.from_dict(payload.data)


def store_state(payload: StagePayload, state: PipelineStateBase) -> StagePayload:
    payload.data = state.to_dict()
    return payload


def build_usage(state: PipelineStateBase) -> dict[str, int | float] | None:
    if not (state.prompt_tokens or state.completion_tokens or state.engine_time_s):
        return None
    else:
        pass
    usage: dict[str, int | float] = {
        "prompt_tokens": int(state.prompt_tokens),
        "completion_tokens": int(state.completion_tokens),
        "total_tokens": int(state.prompt_tokens + state.completion_tokens),
    }
    if state.engine_time_s:
        usage["engine_time_s"] = round(float(state.engine_time_s), 6)
    else:
        pass
    return usage
