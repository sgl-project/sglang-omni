# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest

from sglang_omni.models.cosmos3.reasoner import create_reasoner_scheduler
from sglang_omni.models.cosmos3.stages import create_generation_scheduler
from sglang_omni.utils import device as device_utils


class NativeStartupReached(Exception):
    pass


@pytest.mark.parametrize(
    "factory", [create_generation_scheduler, create_reasoner_scheduler]
)
@pytest.mark.parametrize("devices", [None, [3, 5]])
def test_native_factory_uses_resolved_device_index(
    monkeypatch, tmp_path, factory, devices
):
    calls = []

    def resolve(device, index):
        calls.append((device, index))
        return SimpleNamespace(type="cuda", index=3)

    monkeypatch.setattr(device_utils, "resolve_concrete_device", resolve)
    captured = []

    def startup(**kwargs):
        captured.append(kwargs)
        raise NativeStartupReached()

    if factory is create_generation_scheduler:
        from sglang.multimodal_gen.runtime.server_args import ServerArgs

        monkeypatch.setattr(ServerArgs, "from_kwargs", startup)
        options = {"output_dir": str(tmp_path)}
    else:
        import sglang

        monkeypatch.setattr(sglang, "Engine", startup)
        options = {}
    with pytest.raises(NativeStartupReached):
        factory(
            "checkpoint", device=None, gpu_id=None, runtime_gpu_ids=devices, **options
        )
    assert calls == [(None, None)]
    kwargs = captured[0]
    if factory is create_generation_scheduler and devices is not None:
        assert kwargs["gpu_ids"] == [3, 5]
    else:
        assert kwargs["base_gpu_id"] == 3
    if factory is create_reasoner_scheduler and devices is not None:
        assert kwargs["gpu_id_step"] == 2
        assert kwargs["tp_size"] == 2


@pytest.mark.parametrize(
    "factory", [create_generation_scheduler, create_reasoner_scheduler]
)
def test_native_factory_rejects_cpu_before_native_startup(monkeypatch, factory):
    monkeypatch.setattr(
        device_utils,
        "resolve_concrete_device",
        lambda device, index: SimpleNamespace(type="cpu", index=None),
    )
    with pytest.raises(ValueError, match="indexed accelerator"):
        factory("checkpoint", device="cpu")
