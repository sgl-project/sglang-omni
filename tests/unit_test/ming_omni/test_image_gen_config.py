# SPDX-License-Identifier: Apache-2.0
"""Ming-Omni image generation pipeline config tests."""

from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest

from sglang_omni.models.ming_omni.pipeline.next_stage import (
    AGGREGATE_STAGE,
    AUDIO_STAGE,
    DECODE_STAGE,
    IMAGE_GEN_STAGE,
    IMAGE_STAGE,
    PREPROCESSING_STAGE,
    TALKER_STAGE,
    THINKER_STAGE,
)


def _stages_by_name(config):
    return {stage.name: stage for stage in config.stages}


def _with_thinker_tp(stages, *, gpu, tp_size: int = 2):
    copied = [stage.model_copy(deep=True) for stage in stages]
    for stage in copied:
        if stage.name == THINKER_STAGE:
            stage.tp_size = tp_size
            stage.parallelism = stage.parallelism.model_copy(update={"tp": tp_size})
            stage.gpu = gpu
    return copied


def _set_gpu(stages, stage_name: str, gpu, *, tp_size: int | None = None):
    copied = [stage.model_copy(deep=True) for stage in stages]
    for stage in copied:
        if stage.name == stage_name:
            stage.gpu = gpu
            if tp_size is not None:
                stage.tp_size = tp_size
                stage.parallelism = stage.parallelism.model_copy(update={"tp": tp_size})
    return copied


def test_ming_image_config_builds_expected_graph_and_image_stage_wiring() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    config = MingOmniImagePipelineConfig(
        model_path="dummy",
        dit_type="zimage",
        dit_model_path="/models/zimage",
    )

    assert [stage.name for stage in config.stages] == [
        PREPROCESSING_STAGE,
        AUDIO_STAGE,
        IMAGE_STAGE,
        AGGREGATE_STAGE,
        THINKER_STAGE,
        DECODE_STAGE,
        IMAGE_GEN_STAGE,
    ]
    stages = _stages_by_name(config)

    assert stages[PREPROCESSING_STAGE].factory_args["enable_image_gen"] is True
    assert stages[THINKER_STAGE].factory_args["capture_hidden"] is True
    assert stages[THINKER_STAGE].next == [DECODE_STAGE, IMAGE_GEN_STAGE]
    assert stages[IMAGE_GEN_STAGE].terminal is True
    assert stages[IMAGE_GEN_STAGE].factory_args == {
        "device": "cuda",
        "dit_type": "zimage",
        "dit_model_path": "/models/zimage",
    }
    assert config.terminal_stages == [DECODE_STAGE, IMAGE_GEN_STAGE]


def test_ming_image_config_advanced_flags_are_explicit() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    default_config = MingOmniImagePipelineConfig(model_path="dummy")
    default_image_gen = _stages_by_name(default_config)[IMAGE_GEN_STAGE]

    assert "enable_standalone_semantic_encoder" not in default_image_gen.factory_args
    assert "enable_byt5_text_rendering" not in default_image_gen.factory_args

    advanced_config = MingOmniImagePipelineConfig(
        model_path="dummy",
        enable_standalone_semantic_encoder=True,
        enable_byt5_text_rendering=True,
    )
    advanced_image_gen = _stages_by_name(advanced_config)[IMAGE_GEN_STAGE]

    assert advanced_image_gen.factory_args["enable_standalone_semantic_encoder"] is True
    assert advanced_image_gen.factory_args["enable_byt5_text_rendering"] is True


def test_ming_variants_include_image_generation_configs() -> None:
    from sglang_omni.models.ming_omni.config import Variants

    assert set(Variants) == {
        "text",
        "speech",
        "streaming_speech",
        "image",
    }


def test_ming_image_launcher_places_thinker_tp_and_image_gen(monkeypatch) -> None:
    from examples.run_ming_omni_image_server import _launch_image_server

    captured: dict[str, object] = {}
    serve_module = ModuleType("sglang_omni.serve")

    def fake_launch_server(config, **kwargs):
        captured["config"] = config
        captured["kwargs"] = kwargs

    serve_module.launch_server = fake_launch_server
    monkeypatch.setitem(sys.modules, "sglang_omni.serve", serve_module)

    args = SimpleNamespace(
        model_path="dummy",
        relay_backend="shm",
        tp_size=2,
        gpu_thinker=0,
        gpu_img_gen=2,
        dit_model_path="/models/zimage",
        dit_type="zimage",
        enable_standalone_semantic_encoder=True,
        enable_byt5_text_rendering=True,
        mem_fraction_static=0.8,
        host="127.0.0.1",
        port=8000,
        model_name="ming-omni-image",
    )

    _launch_image_server(args)

    config = captured["config"]
    stages = _stages_by_name(config)
    thinker = stages[THINKER_STAGE]
    image_gen = stages[IMAGE_GEN_STAGE]
    overrides = thinker.factory_args["server_args_overrides"]

    assert thinker.gpu == [0, 1]
    assert thinker.tp_size == 2
    assert thinker.parallelism.tp == 2
    assert image_gen.gpu == 2
    assert image_gen.factory_args["dit_type"] == "zimage"
    assert image_gen.factory_args["dit_model_path"] == "/models/zimage"
    assert image_gen.factory_args["enable_standalone_semantic_encoder"] is True
    assert image_gen.factory_args["enable_byt5_text_rendering"] is True
    assert captured["kwargs"]["model_name"] == "ming-omni-image"
    assert overrides["mem_fraction_static"] == 0.8
    assert overrides["disable_custom_all_reduce"] is True


def test_ming_image_rejects_image_gen_overlap_with_explicit_thinker_tp_gpus() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    base = MingOmniImagePipelineConfig(model_path="dummy")
    stages = _with_thinker_tp(base.stages, gpu=[0, 2])
    stages = _set_gpu(stages, IMAGE_GEN_STAGE, 2)

    with pytest.raises(ValueError, match="image_gen.*collides"):
        MingOmniImagePipelineConfig(model_path="dummy", stages=stages)


def test_ming_image_rejects_image_gen_overlap_with_scalar_thinker_tp_range() -> None:
    from sglang_omni.models.ming_omni.config import MingOmniImagePipelineConfig

    base = MingOmniImagePipelineConfig(model_path="dummy")
    stages = _with_thinker_tp(base.stages, gpu=0)
    stages = _set_gpu(stages, IMAGE_GEN_STAGE, 1)

    with pytest.raises(ValueError, match="image_gen.*collides"):
        MingOmniImagePipelineConfig(model_path="dummy", stages=stages)


def test_text_and_speech_configs_do_not_enable_image_gen_or_hidden_capture() -> None:
    from sglang_omni.models.ming_omni.config import (
        MingOmniPipelineConfig,
        MingOmniSpeechPipelineConfig,
    )

    text = _stages_by_name(MingOmniPipelineConfig(model_path="dummy"))
    speech = _stages_by_name(MingOmniSpeechPipelineConfig(model_path="dummy"))

    assert "enable_image_gen" not in text[PREPROCESSING_STAGE].factory_args
    assert "capture_hidden" not in text[THINKER_STAGE].factory_args
    assert "enable_image_gen" not in speech[PREPROCESSING_STAGE].factory_args
    assert "capture_hidden" not in speech[THINKER_STAGE].factory_args
    assert speech[THINKER_STAGE].next == [DECODE_STAGE, TALKER_STAGE]
