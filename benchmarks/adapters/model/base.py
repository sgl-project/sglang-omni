from __future__ import annotations

from abc import ABC, abstractmethod

from benchmarks.core.types import BenchmarkCaseSpec, BenchmarkRunConfig, NormalizedSample, PreparedRequest


class ModelAdapter(ABC):
    name: str

    @abstractmethod
    def build_request(
        self,
        *,
        sample: NormalizedSample,
        case_spec: BenchmarkCaseSpec,
        run_config: BenchmarkRunConfig,
        served_model_name: str,
    ) -> PreparedRequest:
        raise NotImplementedError

