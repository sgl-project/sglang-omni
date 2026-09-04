# SPDX-License-Identifier: Apache-2.0
"""A stage may only pass device=None to a factory that resolves it.

An unresolved None reaches torch.device(), which raises, or nn.Module.to(), which
silently leaves the model on the CPU. Neither surfaces in the model tests, since
those need real checkpoints, so each qualified factory is driven here with the heavy
dependencies patched out and the resolution observed directly.
"""

from __future__ import annotations

import importlib
import pathlib
from types import SimpleNamespace

import pytest

import sglang_omni.platforms as platforms

_MODELS = sorted(
    p.parent.name for p in pathlib.Path("sglang_omni/models").glob("*/config.py")
)

# Every stage that relies on device=None. Adding one means adding a test below that
# proves the factory resolves it.
_NONE_DEVICE_STAGES = {
    ("qwen3_asr", "asr"),
    ("qwen3_omni", "audio_encoder"),
    ("qwen3_omni", "code2wav"),
    ("qwen3_omni", "image_encoder"),
}


def test_the_qualified_stages_leave_device_to_the_factory() -> None:
    """Each qualified stage defers device to its factory's own resolution.

    The factory group only passes a set value; an unset ``factory.device``
    means the factory's signature default governs. Each stage listed above
    must actually defer (no config-pinned device), and its factory's
    resolution of the absent device is proven by a test below."""
    for model, stage_name in sorted(_NONE_DEVICE_STAGES):
        module = importlib.import_module(f"sglang_omni.models.{model}.config")
        config = module.EntryClass(model_path="unused")
        assert config.stage_named(stage_name).factory.device is None, (
            model,
            stage_name,
        )


@pytest.mark.parametrize("factory_name", ["image_encoder", "audio_encoder"])
def test_qwen3_omni_encoder_stages_resolve_none_to_the_platform(
    monkeypatch: pytest.MonkeyPatch, factory_name: str
) -> None:
    from sglang_omni.models.qwen3_omni import stages
    from sglang_omni.scheduling import simple_scheduler

    built: dict[str, object] = {}

    class _Encoder:
        def __init__(
            self,
            *,
            model_path,
            device,
            dtype,
            enable_layer_cuda_graph: bool | None = None,
        ):
            del model_path, dtype
            built["device"] = device
            built["enable_layer_cuda_graph"] = enable_layer_cuda_graph

        def __getattr__(self, name):
            del name
            return 2

    encoder_attr = {
        "image_encoder": "Qwen3OmniImageEncoder",
        "audio_encoder": "Qwen3OmniAudioEncoder",
    }[factory_name]
    monkeypatch.setattr(stages, encoder_attr, _Encoder)
    monkeypatch.setattr(
        simple_scheduler, "SimpleScheduler", lambda *a, **k: SimpleNamespace()
    )

    getattr(stages, f"create_{factory_name}_executor")("unused", device=None)

    assert built["device"] == platforms.current_platform.device_type
    assert built["enable_layer_cuda_graph"] is (
        False if factory_name == "audio_encoder" else None
    )


def test_qwen3_omni_code2wav_resolves_none_to_a_concrete_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.qwen3_omni.components import code2wav_scheduler

    model = SimpleNamespace(
        total_upsample=1, config=SimpleNamespace(num_quantizers=4), eval=lambda: None
    )
    model.eval = lambda: model
    monkeypatch.setattr(
        code2wav_scheduler, "load_code2wav_model", lambda *a, **k: model
    )

    scheduler = code2wav_scheduler.create_code2wav_scheduler("unused", device=None)

    assert scheduler._device.type == platforms.current_platform.device_type
    if platforms.current_platform.device_type != "cpu":
        # Placement was not requested, so the backend's current card is bound.
        # A cpu device correctly carries no index.
        assert scheduler._device.index is not None


def test_qwen3_asr_stage_forwards_none_to_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The factory must hand None down rather than substitute a literal.

    Patching the base builder's build() also proves it is the builder in play: a
    factory using an unrelated builder would leave this spy untouched. What build()
    then does with None is covered in test_server_args_builder_device.py.
    """
    from sglang_omni.models.qwen3_asr import stages
    from sglang_omni.scheduling import engine_factory

    seen: dict[str, object] = {}

    def spy_build(self, model_path, **kwargs):
        del self, model_path
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        engine_factory.SGLangGenerationEngineBuilder, "build", spy_build
    )

    stages.create_sglang_qwen3_asr_executor("unused", device=None, gpu_id=1)

    assert "device" in seen, "the factory did not route through the shared builder"
    assert seen["device"] is None
    # Placement injects gpu_id only when the signature declares it. Without it the
    # builder resolved a bare accelerator and told SGLang card 0.
    assert seen["gpu_id"] == 1
