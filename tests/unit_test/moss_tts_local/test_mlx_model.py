# SPDX-License-Identifier: Apache-2.0
"""Focused native-MLX coverage for MOSS-TTS Local."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

mx = pytest.importorskip("mlx.core")

from mlx.utils import tree_flatten  # noqa: E402

from sglang_omni.models.moss_tts_local.mlx import runner as runner_module  # noqa: E402
from sglang_omni.models.moss_tts_local.mlx.config import ModelConfig  # noqa: E402
from sglang_omni.models.moss_tts_local.mlx.model import MossTTSLocalModel  # noqa: E402
from sglang_omni.models.moss_tts_local.mlx.runner import sample  # noqa: E402
from sglang_omni.models.moss_tts_local.request_builders import (  # noqa: E402
    MossTTSLocalSGLangRequestData,
)


def tiny_config() -> ModelConfig:
    return ModelConfig.from_dict(
        {
            "model_type": "moss_tts_local",
            "n_vq": 2,
            "audio_vocab_size": 16,
            "audio_pad_code": 16,
            "audio_assistant_slot_token_id": 60,
            "audio_end_token_id": 61,
            "qwen3_config": {
                "model_type": "qwen3",
                "hidden_size": 32,
                "num_hidden_layers": 2,
                "intermediate_size": 64,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "head_dim": 8,
                "vocab_size": 64,
                "max_position_embeddings": 64,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10_000,
            },
            "gpt2_config": {
                "n_embd": 32,
                "n_layer": 1,
                "n_head": 4,
                "n_inner": 64,
                "activation_function": "silu",
                "position_embedding_type": "rope",
                "rope_base": 10_000,
            },
        }
    )


def test_mlx_model_generates_one_complete_codec_row() -> None:
    model = MossTTSLocalModel(tiny_config())
    prompt = mx.array([[[1, 16, 16], [2, 3, 4]]], dtype=mx.int32)
    hidden = model.backbone(prompt, model.make_cache())
    row = model.decode_frame(
        hidden[:, -1, :],
        sample_text=lambda logits: mx.argmax(logits, axis=-1),
        sample_audio=lambda logits, _channel: mx.argmax(logits, axis=-1),
    )
    mx.eval(row)

    assert hidden.shape == (1, 2, 32)
    assert row.shape == (1, 3)
    assert row[0, 0].item() in {60, 61}
    assert all(0 <= code < 16 for code in row[0, 1:].tolist())


@pytest.mark.accelerator
@pytest.mark.skipif(not mx.metal.is_available(), reason="requires Apple Metal")
def test_runner_loads_local_model_and_decodes_request(tmp_path: Path) -> None:
    model = MossTTSLocalModel(tiny_config())
    (tmp_path / "config.json").write_text(json.dumps(asdict(model.config)))
    mx.save_safetensors(
        str(tmp_path / "model.safetensors"), dict(tree_flatten(model.parameters()))
    )
    runner_class = runner_module.make_moss_tts_local_mlx_runner_class()
    runner = runner_class(str(tmp_path), disable_radix_cache=True, pool_size=64)
    request_state = MossTTSLocalSGLangRequestData(
        prompt_rows=torch.tensor([[1, 16, 16], [2, 3, 4]]),
        sampling_seed=1234,
    )
    request = SimpleNamespace(omni_data=request_state)
    prefill = runner.prefill_start("request", [1, 2], [1, 2], [], [], 0, req=request)
    first_token = runner.prefill_finalize(prefill)
    first_rows = runner.pop_completed_rows("request")
    assert first_rows[0][0] == first_token
    assert len(first_rows[0]) == model.config.channels

    decode = runner.decode_batch_start(["request"])
    chained_decode = runner.decode_batch_start_chained(decode)
    for pending in (decode, chained_decode):
        tokens = runner.decode_batch_finalize(pending)
        rows = runner.pop_completed_rows("request")
        assert len(rows) == 1
        assert rows[0][0] == tokens[0]
        assert all(0 <= code < 16 for code in rows[0][1:])
    runner.remove_request("request")
    assert not runner.has_request("request")


def test_mlx_config_exposes_scheduler_vocabulary_layout() -> None:
    config = tiny_config()

    assert config.vocab_size_list == [64, 17, 17]


def test_seeded_sampling_is_position_stable() -> None:
    logits = mx.array([[0.1, 0.2, 0.3, 0.4]], dtype=mx.float32)
    kwargs = {
        "temperature": 1.7,
        "top_p": 0.8,
        "top_k": 3,
        "seed": 1234,
        "position": 7,
    }

    first = sample(logits, **kwargs)
    second = sample(logits, **kwargs)
    mx.eval(first, second)

    assert first.item() == second.item()


def test_model_rejects_non_local_checkpoint() -> None:
    config = tiny_config()
    config.model_type = "moss_tts_delay"
    with pytest.raises(ValueError, match="moss_tts_local"):
        MossTTSLocalModel(config)
