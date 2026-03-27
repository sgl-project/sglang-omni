from __future__ import annotations

import importlib
import logging
import pkgutil
from dataclasses import dataclass, field
from typing import Type

from .base import BenchmarkProfile

logger = logging.getLogger(__name__)

_PROFILE_CLASSES: dict[str, Type[BenchmarkProfile]] = {}


def register_profile(cls: Type[BenchmarkProfile]) -> Type[BenchmarkProfile]:
    """Decorator that registers a BenchmarkProfile subclass by its profile_name."""
    if cls.profile_name in _PROFILE_CLASSES:
        raise ValueError(f"Duplicate profile registration: '{cls.profile_name}'")
    _PROFILE_CLASSES[cls.profile_name] = cls
    return cls


def _import_profile_subpackages(
    package_name: str,
    strict: bool = False,
) -> None:
    """Import all subpackages under *package_name*, triggering @register_profile decorators."""
    package = importlib.import_module(package_name)

    for _, name, ispkg in pkgutil.iter_modules(package.__path__, package_name + "."):
        if not ispkg:
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            if strict:
                raise
            logger.warning("Ignore import error when loading %s: %s", name, exc)


@dataclass
class BenchmarkProfileRegistry:
    profiles: dict[str, Type[BenchmarkProfile]] = field(default_factory=dict)

    def register_profiles(
        self,
        package_name: str,
        strict: bool = False,
    ) -> None:
        _import_profile_subpackages(package_name, strict=strict)
        for profile_name, profile_cls in _PROFILE_CLASSES.items():
            if profile_name in self.profiles:
                raise ValueError(f"Benchmark profile '{profile_name}' already registered")
            self.profiles[profile_name] = profile_cls

    def get_by_name_or_alias(self, value: str) -> BenchmarkProfile:
        matches = [
            profile_cls
            for profile_cls in self.profiles.values()
            if profile_cls.matches_alias(value)
        ]
        if not matches:
            raise ValueError(
                f"Unknown benchmark profile '{value}'. Supported profiles: {sorted(self.profiles)}"
            )
        if len(matches) > 1:
            raise ValueError(
                f"Ambiguous benchmark profile '{value}'. Matches: "
                f"{[profile_cls.profile_name for profile_cls in matches]}"
            )
        return matches[0]()

    def list_profiles(self) -> list[str]:
        return sorted(self.profiles)


PROFILE_REGISTRY = BenchmarkProfileRegistry()
PROFILE_REGISTRY.register_profiles("benchmarks.profiles")
