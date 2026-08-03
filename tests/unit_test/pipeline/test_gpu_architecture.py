# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import pytest

from sglang_omni.utils import gpu_compat
from sglang_omni.utils.gpu_compat import (
    describe_sglang_runtime_configuration,
    gpu_architecture_for_sm,
)


@pytest.mark.parametrize(
    ("sm_version", "architecture"),
    [
        (89, "ada"),
        (90, "hopper"),
        (100, "blackwell-datacenter"),
        (120, "blackwell-consumer"),
        (103, "sm103"),
        (None, "unknown"),
    ],
)
def test_gpu_architecture_identity_is_explicit(
    sm_version: int | None,
    architecture: str,
) -> None:
    assert gpu_architecture_for_sm(sm_version) == architecture


def test_runtime_configuration_reports_global_backend_for_each_phase(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_compat,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 89,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend=None,
        prefill_attention_backend=None,
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("flashinfer", "flashinfer"),
    )

    description = describe_sglang_runtime_configuration(server_args, gpu_id=0)

    assert description == (
        "SGLang runtime configuration: gpu_id=0, sm=89, architecture=ada, "
        "attention_backend=flashinfer, decode_attention_backend=flashinfer, "
        "prefill_attention_backend=flashinfer, sampling_backend=pytorch"
    )


def test_runtime_configuration_reports_explicit_phase_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_compat,
        "get_visible_gpu_sm_version",
        lambda _gpu_id: 90,
    )
    server_args = SimpleNamespace(
        attention_backend="flashinfer",
        decode_attention_backend="triton",
        prefill_attention_backend="fa3",
        sampling_backend="pytorch",
        get_attention_backends=lambda: ("fa3", "triton"),
    )

    description = describe_sglang_runtime_configuration(server_args, gpu_id=1)

    assert description == (
        "SGLang runtime configuration: gpu_id=1, sm=90, architecture=hopper, "
        "attention_backend=flashinfer, decode_attention_backend=triton, "
        "prefill_attention_backend=fa3, sampling_backend=pytorch"
    )
