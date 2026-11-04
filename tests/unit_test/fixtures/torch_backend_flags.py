# SPDX-License-Identifier: Apache-2.0
"""Set cuBLAS flags for one test and restore them afterwards.

The cuBLAS module routes its flags through __getattr__ and __setattr__ with no instance dict, so
monkeypatch records no previous value and cannot restore one at teardown.
"""

from collections.abc import Iterator
from typing import Protocol

import pytest
import torch


class CublasFlagSetter(Protocol):
    def __call__(self, name: str, value: bool) -> None: ...


@pytest.fixture
def cublas_flags() -> Iterator[CublasFlagSetter]:
    module = torch.backends.cuda.matmul
    saved: dict[str, bool] = {}

    def set_flag(name: str, value: bool) -> None:
        saved.setdefault(name, getattr(module, name))
        setattr(module, name, value)

    yield set_flag
    for name, value in saved.items():
        setattr(module, name, value)
