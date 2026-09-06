# SPDX-License-Identifier: Apache-2.0
"""Contracts for the native MiniMax Music 3 MLX backend."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")


def test_tiny_ar_and_acoustic_pipeline_is_finite() -> None:
    from sglang_omni.models.minimax_music3.mlx.ar import generate_frame_hiddens
    from sglang_omni.models.minimax_music3.mlx.config import ModelConfig
    from sglang_omni.models.minimax_music3.mlx.euler import denoise_chunk
    from sglang_omni.models.minimax_music3.mlx.loader import (
        MiniMaxMusic3MlxAcousticModel,
        MiniMaxMusic3MlxARModel,
    )

    config = ModelConfig.tiny()
    ar_model = MiniMaxMusic3MlxARModel(config)
    acoustic_model = MiniMaxMusic3MlxAcousticModel(config)
    mx.eval(ar_model.parameters(), acoustic_model.parameters())
    text_ids = mx.array(
        [[1, 2, 3], [1, config.audio_cfg_token_id, 3]],
        dtype=mx.int32,
    )

    hidden = generate_frame_hiddens(
        ar_model.language_model,
        ar_model.rvq_depth_decoder,
        config,
        text_ids,
        max_frames=2,
        seed=7,
    )
    condition = acoustic_model.condition_encoder(hidden)
    noise = mx.random.normal((1, config.dit_in_channels, condition.shape[1])).astype(
        condition.dtype
    )
    latent, _ = denoise_chunk(
        acoustic_model.transformer,
        noise,
        condition,
        num_inference_steps=1,
    )
    waveform = acoustic_model.vocoder(latent)
    mx.eval(waveform)

    assert hidden.shape == (1, 2, config.num_codebooks * config.hidden_size)
    assert waveform.shape[0:2] == (1, 2)
    assert np.isfinite(np.asarray(waveform)).all()


def test_split_loader_reads_one_converted_artifact(tmp_path: Path) -> None:
    from mlx.utils import tree_flatten

    from sglang_omni.models.minimax_music3.mlx.config import ModelConfig
    from sglang_omni.models.minimax_music3.mlx.loader import (
        MiniMaxMusic3MlxAcousticModel,
        MiniMaxMusic3MlxARModel,
        load_mlx_acoustic_model,
        load_mlx_ar_model,
    )

    config = ModelConfig.tiny()
    ar_model = MiniMaxMusic3MlxARModel(config)
    acoustic_model = MiniMaxMusic3MlxAcousticModel(config)
    for model in (ar_model, acoustic_model):
        nn.quantize(
            model,
            group_size=32,
            bits=8,
            mode="mxfp8",
            class_predicate=model.model_quant_predicate,
        )
    mx.eval(ar_model.parameters(), acoustic_model.parameters())
    raw_config = config.to_dict()
    raw_config["architectures"] = ["MiniMaxMusic3ForConditionalGeneration"]
    raw_config["quantization"] = {
        "group_size": 32,
        "bits": 8,
        "mode": "mxfp8",
    }
    (tmp_path / "config.json").write_text(json.dumps(raw_config))
    weights = dict(
        tree_flatten(ar_model.parameters()) + tree_flatten(acoustic_model.parameters())
    )
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)

    loaded_ar = load_mlx_ar_model(str(tmp_path))
    loaded_acoustic = load_mlx_acoustic_model(str(tmp_path))

    assert loaded_ar.config.hidden_size == config.hidden_size
    assert loaded_acoustic.config.dit_num_layers == config.dit_num_layers
    assert isinstance(
        loaded_ar.language_model.model.layers[0].self_attn.q_proj,
        nn.QuantizedLinear,
    )
    assert isinstance(
        loaded_acoustic.transformer.transformer_blocks[0].ff_in,
        nn.QuantizedLinear,
    )


@pytest.mark.parametrize(
    ("model_cls", "selected_path", "untouched_path"),
    [
        (
            "ar",
            "language_model.model.layers.0.self_attn.q_proj",
            "language_model.lm_head",
        ),
        (
            "acoustic",
            "transformer.transformer_blocks.0.ff_in",
            "vocoder.conv_in",
        ),
    ],
)
def test_quantization_policy_keeps_heads_and_vocoder_dense(
    model_cls: str,
    selected_path: str,
    untouched_path: str,
) -> None:
    from sglang_omni.models.minimax_music3.mlx.config import ModelConfig
    from sglang_omni.models.minimax_music3.mlx.loader import (
        MiniMaxMusic3MlxAcousticModel,
        MiniMaxMusic3MlxARModel,
    )

    cls = (
        MiniMaxMusic3MlxARModel if model_cls == "ar" else MiniMaxMusic3MlxAcousticModel
    )
    model = cls(ModelConfig.tiny())
    modules = dict(model.named_modules())

    assert model.model_quant_predicate(selected_path, modules[selected_path])
    assert not model.model_quant_predicate(untouched_path, modules[untouched_path])


def test_euler_abort_is_observed_between_steps() -> None:
    from sglang_omni.models.minimax_music3.mlx.euler import denoise_chunk

    calls = 0

    class Transformer:
        def __call__(self, hidden, timestep, condition):
            nonlocal calls
            del timestep, condition
            calls += 1
            return mx.zeros_like(hidden)

    with pytest.raises(InterruptedError, match="aborted"):
        denoise_chunk(
            Transformer(),
            mx.zeros((1, 2, 3)),
            mx.zeros((1, 3, 4)),
            num_inference_steps=3,
            should_abort=lambda: calls >= 1,
        )

    assert calls == 2
