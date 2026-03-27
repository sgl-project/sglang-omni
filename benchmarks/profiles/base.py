from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar

from benchmarks.adapters.model.base import ModelAdapter
from benchmarks.core.types import BenchmarkCaseSpec
from benchmarks.datasets.base import DatasetLoader


class BenchmarkProfile(ABC):
    profile_name: ClassVar[str]
    aliases: ClassVar[tuple[str, ...]] = ()
    request_family: ClassVar[str]
    default_case: ClassVar[str]

    def __init__(self) -> None:
        self.model_adapter = self.build_model_adapter()

    @property
    @abstractmethod
    def cases(self) -> dict[str, BenchmarkCaseSpec]:
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset_loader(self) -> DatasetLoader:
        raise NotImplementedError

    @abstractmethod
    def build_model_adapter(self) -> ModelAdapter:
        raise NotImplementedError

    def get_case(self, case_id: str | None) -> BenchmarkCaseSpec:
        if case_id is None and len(self.cases) > 1:
            raise ValueError(
                f"Profile '{self.profile_name}' supports multiple cases "
                f"{sorted(self.cases)}; --case is required"
            )
        resolved_case_id = case_id or self.default_case
        if resolved_case_id not in self.cases:
            raise ValueError(
                f"Unsupported case '{resolved_case_id}' for profile '{self.profile_name}'. "
                f"Supported cases: {sorted(self.cases)}"
            )
        return self.cases[resolved_case_id]

    @classmethod
    def matches_alias(cls, value: str) -> bool:
        if value == cls.profile_name:
            return True
        return value in cls.aliases
