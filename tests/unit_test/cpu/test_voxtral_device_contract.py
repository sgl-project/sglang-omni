# SPDX-License-Identifier: Apache-2.0
"""Voxtral-TTS device and compile contracts on CPU."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni import platforms


def test_voxtral_generation_stage_forwards_none_to_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The generation stage used to pin ``device="cuda:0"`` in its signature
    default, which the config never overrode, so the literal followed the stage
    onto a CPU host and died at torch.cuda.set_device.

    Patching the base builder's build() also proves it is the builder in play: a
    factory using an unrelated builder would leave this spy untouched.
    """
    from sglang_omni.models.voxtral_tts.pipeline import stages
    from sglang_omni.scheduling import engine_factory

    seen: dict[str, object] = {}

    def spy_build(self, model_path, **kwargs):
        del self, model_path
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        engine_factory.SGLangGenerationEngineBuilder, "build", spy_build
    )

    stages.create_generation_executor("unused", gpu_id=1)

    assert "device" in seen, "the factory did not route through the shared builder"
    assert seen["device"] is None
    assert seen["gpu_id"] == 1


@pytest.mark.parametrize(
    ("device", "gpu_id", "expected"),
    [
        ("cpu", 2, "cpu"),
        (None, 0, "cpu"),
        (None, None, "cpu"),
    ],
)
def test_voxtral_vocoder_resolves_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    device: str | None,
    gpu_id: int | None,
    expected: str,
) -> None:
    from sglang_omni.models.voxtral_tts.pipeline import stages

    seen: dict[str, object] = {}

    monkeypatch.setattr(platforms.current_platform, "device_type", "cpu", raising=False)
    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(
        stages,
        "_load_audio_tokenizer",
        lambda checkpoint_dir, audio_config, dev: seen.setdefault("device", dev),
    )
    monkeypatch.setattr(
        stages,
        "_VoxtralTTSVocoder",
        lambda tokenizer: SimpleNamespace(
            build_scheduler=lambda **kwargs: SimpleNamespace()
        ),
    )

    stages.create_vocoder_executor("unused", device=device, gpu_id=gpu_id)

    assert seen["device"] == expected


def test_voxtral_disables_torch_compile_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Inductor plans strides from the meta kernel of
    sgl_kernel.rotary_embedding_cpu, which disagrees with the real kernel
    (expected 4096, got 6144), so the compiled graph trips assert_size_stride
    during warmup. Narrow to Voxtral on purpose — Qwen3-ASR compiles fine on
    CPU — so this must not become a platform-wide claim that CPU cannot compile.
    """
    from sglang_omni.models.voxtral_tts.pipeline.engine_builder import (
        VoxtralTtsEngineBuilder,
    )

    monkeypatch.setattr(platforms.current_platform, "is_cpu", lambda: True)
    defaults = VoxtralTtsEngineBuilder().generation_defaults(dtype="bfloat16")

    assert defaults["enable_torch_compile"] is False
