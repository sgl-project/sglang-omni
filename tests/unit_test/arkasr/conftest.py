# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterator

import pytest
import sglang.srt.layers.linear as sglang_linear
from sglang.srt.runtime_context import get_parallel


@pytest.fixture(autouse=True)
def _tp1_parallel_context(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Run CPU ARK model tests with the production stage's TP=1 topology."""
    # RowParallelLinear enters the symmetric-memory context even at TP=1.
    # CPU tests have no process group; the object is unused when symmetric
    # allocation is disabled.
    monkeypatch.setattr(sglang_linear, "get_tp_group", lambda: object())
    with get_parallel().override(tp_rank=0, tp_size=1):
        yield
