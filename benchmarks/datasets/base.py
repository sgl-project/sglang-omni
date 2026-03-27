from __future__ import annotations

from abc import ABC, abstractmethod

from benchmarks.core.types import NormalizedSample


class DatasetLoader(ABC):
    name: str

    @abstractmethod
    def load(
        self,
        *,
        dataset_path: str | None,
        max_samples: int | None,
    ) -> list[NormalizedSample]:
        raise NotImplementedError
