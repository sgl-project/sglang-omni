# SPDX-License-Identifier: Apache-2.0
"""Patches for running build() with SGLang's real argument parsing as a given SM version, on a CPU host."""

from __future__ import annotations

import pytest

from sglang_omni.platforms import current_platform
from sglang_omni.scheduling import bootstrap, engine_factory
from sglang_omni.utils import cuda_graph_batch_validator

# note (Xuanfei): override_platform flags under which SGLang picks trtllm_mha, as on a B200.
SM100_PLATFORM: dict[str, bool] = {"is_sm100": True, "is_hopper_with_cuda_12_3": False}


def patch_visible_gpu(
    monkeypatch: pytest.MonkeyPatch, *, sm_version: int | None, is_cuda: bool = True
) -> None:
    """Report this SM version for the visible GPU and skip graph capture and attestation, which need a real model runner."""
    monkeypatch.setattr(
        engine_factory, "get_visible_gpu_sm_version", lambda gpu_id: sm_version
    )
    monkeypatch.setattr(current_platform, "is_cuda", lambda: is_cuda)
    monkeypatch.setattr(bootstrap, "init_sglang_cuda_graphs", lambda model_worker: None)
    monkeypatch.setattr(
        cuda_graph_batch_validator,
        "attest_prefill_cuda_graphs",
        lambda model_runner, *, operator_selected: None,
    )
