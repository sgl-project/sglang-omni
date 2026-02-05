from __future__ import annotations

import os
import warnings
from typing import Any, Generic, Optional, TypeVar

T = TypeVar("T")


class EnvVar(Generic[T]):
    _allow_set_name = True

    def __init__(self, default: T):
        self.default = default
        # NOTE: environ can only accept str values, so we need a flag to indicate
        # whether the env var is explicitly set to None.
        self._set_to_none = False

    def __set_name__(self, owner, name):
        assert EnvVar._allow_set_name, "Usage like `a = envs.A` is not allowed"
        self.name = name

    def parse(self, value: str) -> Any:
        raise NotImplementedError()

    def get(self) -> Optional[T]:
        value = os.getenv(self.name)

        # Explicitly set to None
        if self._set_to_none:
            assert value == str(None)
            return None

        # Not set, return default
        if value is None:
            return self.default

        try:
            return self.parse(value)
        except ValueError as e:
            warnings.warn(
                f'Invalid value for {self.name}: {e}, using default "{self.default}"'
            )
            return self.default

    def set(self, value: Any):
        self._set_to_none = value is None
        os.environ[self.name] = str(value)

    def is_set(self):
        return self.name in os.environ

    def __str__(self):
        return str(self.get())


class EnvTuple(EnvVar[tuple]):
    def parse(self, value: str) -> tuple[str, ...]:
        return tuple(s.strip() for s in value.split(",") if s.strip())


class EnvStr(EnvVar[str]):
    def parse(self, value: str) -> str:
        return value


class EnvBool(EnvVar[bool]):
    def parse(self, value: str) -> bool:
        value = value.lower()
        if value in ["true", "1", "yes", "y"]:
            return True
        if value in ["false", "0", "no", "n"]:
            return False
        raise ValueError(f'"{value}" is not a valid boolean value')


class EnvInt(EnvVar[int]):
    def parse(self, value: str) -> int:
        try:
            return int(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid integer value')


class EnvFloat(EnvVar[float]):
    def parse(self, value: str) -> float:
        try:
            return float(value)
        except ValueError:
            raise ValueError(f'"{value}" is not a valid float value')


class Environ:
    # singleton instance
    _instance: Environ | None = None

    # logging
    SGLOMNI_LOG_LEVEL = EnvStr("INFO")

    def __new__(cls):
        # single instance
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance


OMNIENV = Environ()
