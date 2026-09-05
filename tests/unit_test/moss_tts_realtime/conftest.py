# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni.scheduling import omni_scheduler as omni_scheduler_module


@pytest.fixture(autouse=True)
def _runtime_context_bags(monkeypatch: pytest.MonkeyPatch):
    """Publish the sglang runtime-context namespaces unit tests touch.

    OmniScheduler.stream_output reads ``get_serving().weight_version`` and
    ``_admin_model_info`` reads ``get_model()/get_serving()``; those config
    bags only exist inside a real serving process, so unit tests stub them
    like the core scheduler suite does.
    """
    monkeypatch.setattr(
        omni_scheduler_module,
        "get_serving",
        lambda: SimpleNamespace(weight_version=None),
    )
    monkeypatch.setattr(
        omni_scheduler_module,
        "get_model",
        lambda: SimpleNamespace(model_path="fake-model", load_format="auto"),
    )
