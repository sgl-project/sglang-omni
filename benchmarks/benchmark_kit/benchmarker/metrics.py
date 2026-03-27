from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from .data import Data


@dataclass
class MetricsResult:
    value: float
    unit: str
    description: str


class Metrics(ABC):

    def __init__(self, name: str):
        self.name = name
        self.results = []

    @abstractmethod
    def compute(self, input_data: Data, response: Any) -> MetricsResult:
        pass

    @abstractmethod
    def append_result(self, metrics_result: MetricsResult) -> str:
        pass

    @abstractmethod
    def compute_summary(self) -> MetricsResult:
        pass
