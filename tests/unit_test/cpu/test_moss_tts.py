# SPDX-License-Identifier: Apache-2.0
"""CPU placement contracts for MOSS-TTS."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sglang_omni import platforms


def test_moss_tts_codec_device_resolves_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts.stages import _resolve_codec_device

    monkeypatch.setattr(platforms.current_platform, "device_type", "cpu", raising=False)

    assert _resolve_codec_device(None, 0) == "cpu"
    assert _resolve_codec_device(None, None) == "cpu"
    assert _resolve_codec_device("cpu", 3) == "cpu"


def test_moss_tts_engine_defers_device_to_the_shared_builder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang_omni.models.moss_tts import stages
    from sglang_omni.scheduling import engine_factory

    seen: dict[str, object] = {}

    def spy_build(self, model_path, **kwargs):
        del self, model_path
        seen.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(
        engine_factory.SGLangGenerationEngineBuilder, "build", spy_build
    )

    stages.create_sglang_tts_engine_executor("unused")

    assert seen["device"] is None
