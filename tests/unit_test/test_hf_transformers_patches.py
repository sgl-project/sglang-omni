# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from sglang_omni.utils.hf_transformers_patches import (
    patch_transformers_stream_capture_detection,
)


def test_patch_detects_capture_through_tensor_device_module(monkeypatch) -> None:
    from transformers import masking_utils
    from transformers.utils import import_utils

    def original(tensor=None):
        return False

    monkeypatch.setattr(import_utils, "is_tracing", original)
    monkeypatch.setattr(masking_utils, "is_tracing", original)
    device_module = SimpleNamespace(is_current_stream_capturing=lambda: True)
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)

    patch_transformers_stream_capture_detection()

    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert import_utils.is_tracing(tensor) is True
    assert masking_utils.is_tracing(tensor) is True


def test_patch_preserves_upstream_tracing_and_is_idempotent(monkeypatch) -> None:
    from transformers.utils import import_utils

    upstream_calls = []

    def original(tensor=None):
        upstream_calls.append(tensor)
        return True

    monkeypatch.setattr(import_utils, "is_tracing", original)
    patch_transformers_stream_capture_detection()
    first_wrapper = import_utils.is_tracing
    patch_transformers_stream_capture_detection()

    assert import_utils.is_tracing is first_wrapper
    assert first_wrapper(None) is True
    assert upstream_calls == [None]


def test_patch_uses_default_accelerator_when_tensor_is_absent(monkeypatch) -> None:
    from transformers.utils import import_utils

    monkeypatch.setattr(import_utils, "is_tracing", lambda tensor=None: False)
    device_module = SimpleNamespace(is_current_stream_capturing=lambda: True)
    monkeypatch.setattr(torch, "get_device_module", lambda: device_module)
    patch_transformers_stream_capture_detection()

    assert import_utils.is_tracing() is True


def test_patch_leaves_cpu_tensor_on_upstream_path(monkeypatch) -> None:
    from transformers.utils import import_utils

    monkeypatch.setattr(import_utils, "is_tracing", lambda tensor=None: False)
    monkeypatch.setattr(
        torch,
        "get_device_module",
        lambda device: (_ for _ in ()).throw(AssertionError("unexpected query")),
    )
    patch_transformers_stream_capture_detection()

    tensor = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    assert import_utils.is_tracing(tensor) is False


def test_patch_treats_missing_capture_api_as_not_capturing(monkeypatch) -> None:
    from transformers.utils import import_utils

    monkeypatch.setattr(import_utils, "is_tracing", lambda tensor=None: False)
    monkeypatch.setattr(torch, "get_device_module", lambda device: SimpleNamespace())
    patch_transformers_stream_capture_detection()

    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    assert import_utils.is_tracing(tensor) is False


def test_patch_does_not_hide_unexpected_capture_errors(monkeypatch) -> None:
    from transformers.utils import import_utils

    monkeypatch.setattr(import_utils, "is_tracing", lambda tensor=None: False)

    def fail_capture_query() -> bool:
        raise AssertionError("unexpected capture failure")

    device_module = SimpleNamespace(
        is_current_stream_capturing=fail_capture_query,
    )
    monkeypatch.setattr(torch, "get_device_module", lambda device: device_module)
    patch_transformers_stream_capture_detection()

    tensor = SimpleNamespace(device=SimpleNamespace(type="npu"))
    with pytest.raises(AssertionError, match="unexpected capture failure"):
        import_utils.is_tracing(tensor)
