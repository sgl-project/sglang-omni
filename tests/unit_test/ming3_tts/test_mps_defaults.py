# SPDX-License-Identifier: Apache-2.0
"""Ming checkpoints inherit shared MPS defaults and preserve explicit opt-ins."""

import pytest

from sglang_omni.models.ming3_tts import config as config_module


@pytest.mark.parametrize("dense", [False, True])
@pytest.mark.parametrize("mode", [None, "off", "on", "auto"])
def test_checkpoint_mps_defaults_and_explicit_modes(
    monkeypatch: pytest.MonkeyPatch, dense: bool, mode: str | None
) -> None:
    monkeypatch.setattr(
        config_module,
        "try_resolve_ming3_dense_tts",
        lambda *_args, **_kwargs: config_module.DENSE_ARCHITECTURE if dense else None,
    )
    values = {"model_path": "test/Ming3-TTS"}
    if mode is not None:
        values["mps"] = mode
    else:
        pass
    config = config_module.Ming3TTSPipelineConfig(**values)
    assert config.mps == (mode if mode is not None else "off")
    assert config.stages[0].gpu == (0 if dense else None)
    rebuilt = config_module.Ming3TTSPipelineConfig.model_validate(config.model_dump())
    assert rebuilt.model_dump() == config.model_dump()


@pytest.mark.parametrize("mode", [None, "off", "on", "auto"])
def test_explicit_stages_keep_shared_mps_default_and_opt_ins(
    monkeypatch: pytest.MonkeyPatch, mode: str | None
) -> None:
    def unexpected_resolution(*_args, **_kwargs):
        raise AssertionError("explicit stages must not resolve checkpoint defaults")

    monkeypatch.setattr(
        config_module, "try_resolve_ming3_dense_tts", unexpected_resolution
    )
    values = {
        "model_path": "test/Ming3-TTS",
        "stages": config_module.default_stages(True),
    }
    if mode is not None:
        values["mps"] = mode
    else:
        pass
    config = config_module.Ming3TTSPipelineConfig(**values)
    assert config.mps == (mode if mode is not None else "off")
    assert config.stages[0].gpu == 0
