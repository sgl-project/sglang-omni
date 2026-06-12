# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Callable

import pytest

import sglang_omni.utils.gpu_compat as gpu_compat


def _returns(value):
    def _stub(*_args, **_kwargs):
        return value

    return _stub


def _compute_capability_by_gpu(
    capabilities: dict[int, tuple[int, int] | None],
) -> Callable[[int, dict[str, str] | None], tuple[int, int] | None]:
    def _stub(gpu_id: int, _env=None) -> tuple[int, int] | None:
        return capabilities[gpu_id]

    return _stub


def test_get_gpu_compat_env_defaults_respects_existing_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_compat,
        "visible_gpus_need_flashinfer_cuda_norm",
        _returns(True),
    )

    assert (
        gpu_compat.get_gpu_compat_env_defaults(
            {gpu_compat._FLASHINFER_USE_CUDA_NORM: "0"}
        )
        == {}
    )


def test_get_gpu_compat_env_defaults_for_architecture_workaround(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_compat,
        "visible_gpus_need_flashinfer_cuda_norm",
        _returns(True),
    )

    assert gpu_compat.get_gpu_compat_env_defaults({}) == {
        gpu_compat._FLASHINFER_USE_CUDA_NORM: "1",
    }


@pytest.mark.parametrize(
    ("capability", "expected"),
    [
        ((gpu_compat._BLACKWELL_MIN_MAJOR_COMPUTE_CAPABILITY, 0), True),
        ((gpu_compat._BLACKWELL_MIN_MAJOR_COMPUTE_CAPABILITY - 1, 9), False),
        (None, False),
    ],
)
def test_visible_gpus_need_flashinfer_cuda_norm_uses_compute_capability(
    capability: tuple[int, int] | None,
    expected: bool,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(gpu_compat, "_visible_gpu_ids", _returns([0]))
    monkeypatch.setattr(
        gpu_compat,
        "_get_compute_capability",
        _compute_capability_by_gpu({0: capability}),
    )

    assert gpu_compat.visible_gpus_need_flashinfer_cuda_norm({}) is expected


def test_visible_gpus_need_flashinfer_cuda_norm_checks_all_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = {"CUDA_VISIBLE_DEVICES": "2,4"}
    monkeypatch.setattr(gpu_compat, "_visible_gpu_ids", _returns([0, 1]))
    monkeypatch.setattr(
        gpu_compat,
        "_get_compute_capability",
        _compute_capability_by_gpu(
            {
                0: (gpu_compat._BLACKWELL_MIN_MAJOR_COMPUTE_CAPABILITY - 1, 9),
                1: (gpu_compat._BLACKWELL_MIN_MAJOR_COMPUTE_CAPABILITY, 0),
            }
        ),
    )

    assert gpu_compat.visible_gpus_need_flashinfer_cuda_norm(env) is True


def test_visible_gpu_ids_follow_cuda_visible_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3,5")

    assert gpu_compat._visible_gpu_ids() == [0, 1]


def test_visible_gpu_ids_without_cuda_visible_devices_uses_device_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeNvml:
        shutdown_called = False

        def nvmlInit(self) -> None:
            pass

        def nvmlDeviceGetCount(self) -> int:
            return 3

        def nvmlShutdown(self) -> None:
            self.shutdown_called = True

    fake_nvml = FakeNvml()
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(gpu_compat, "_try_import_pynvml", _returns(fake_nvml))

    assert gpu_compat._visible_gpu_ids() == [0, 1, 2]
    assert fake_nvml.shutdown_called is True


def test_compute_capability_preview_env_does_not_use_torch_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch_imported = False

    def _fake_import_module(_name: str):
        nonlocal torch_imported
        torch_imported = True
        raise AssertionError("torch fallback should not run for a preview env")

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(gpu_compat, "_try_import_pynvml", _returns(None))
    monkeypatch.setattr(gpu_compat.importlib, "import_module", _fake_import_module)

    assert gpu_compat._get_compute_capability(0, {"CUDA_VISIBLE_DEVICES": "1"}) is None
    assert torch_imported is False


def test_apply_gpu_compat_env_defaults_sets_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        gpu_compat,
        "get_gpu_compat_env_defaults",
        _returns({gpu_compat._FLASHINFER_USE_CUDA_NORM: "1"}),
    )
    env: dict[str, str] = {}

    applied = gpu_compat.apply_gpu_compat_env_defaults(env)

    assert applied == {gpu_compat._FLASHINFER_USE_CUDA_NORM: "1"}
    assert env[gpu_compat._FLASHINFER_USE_CUDA_NORM] == "1"
