# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from pathlib import Path

from sglang_omni.models.zonos2_tts.hf_config import (
    build_zonos2_hf_config_dict,
    ensure_zonos2_hf_layout,
    load_zonos2_params,
)


def test_load_zonos2_params_supports_nested_model_and_data_sidecar(tmp_path) -> None:
    ckpt = tmp_path / "run" / "checkpoint"
    ckpt.mkdir(parents=True)
    (ckpt / "params.json").write_text(
        json.dumps(
            {
                "model": {
                    "model_type": "zonos2",
                    "dim": 128,
                    "n_layers": 2,
                    "special_topk_layers": {"1": 2},
                    "moe_balancing_strategy": "current",
                }
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "run" / "config.yaml").write_text(
        "\n".join(
            [
                "data:",
                "  speaking_rate_enabled: true",
                "  speaking_rate_buckets: ['0-8', '8+']",
                "  quality_enabled: true",
                "  quality_features:",
                "    lufs: true",
                "    snr: false",
                "  quality_buckets:",
                "    lufs: ['-10-0', '0+']",
                "  speaker_embedding_origin_token_enabled: true",
                "  speaker_embedding_cartesia_clone_source_token_enabled: true",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    params = load_zonos2_params(ckpt)

    assert params["speaking_rate_num_buckets"] == 2
    assert params["quality_features"] == ["lufs"]
    assert params["quality_buckets"] == {"lufs": ["-10-0", "0+"]}
    assert params["speaker_background_token_enabled"] is True
    assert params["accurate_mode_token_enabled"] is True
    assert params["text_vocab"] == 455
    assert params["special_topk_layers"] == {1: 2}
    assert params["moe_balancing_strategy"] == "quantile"


def test_hf_config_enables_decode_state_pool() -> None:
    config = build_zonos2_hf_config_dict(
        {"model_type": "zonos2", "dim": 128, "n_layers": 2}
    )

    assert config["enable_decode_state_pool"] is True


def test_hf_config_defaults_match_zonos2_checkpoint_shape() -> None:
    config = build_zonos2_hf_config_dict({"model_type": "zonos2"})

    assert config["text_vocab"] == 519
    assert config["max_position_embeddings"] == 6144
    assert config["intermediate_size"] == 3072
    assert config["speaker_enabled"] is True
    assert config["moe_n_experts"] == 16
    assert config["moe_router_topk"] == 1


def test_ensure_zonos2_hf_layout_accepts_model_pt(tmp_path) -> None:
    ckpt = tmp_path / "checkpoint"
    ckpt.mkdir()
    (ckpt / "params.json").write_text(
        json.dumps({"model_type": "zonos2", "dim": 128, "n_layers": 2}),
        encoding="utf-8",
    )
    (ckpt / "model.pt").write_bytes(b"not a real checkpoint")

    layout = ensure_zonos2_hf_layout(ckpt)

    layout_path = Path(layout)
    assert (layout_path / "config.json").is_file()
    assert (layout_path / "params.json").exists()
    assert (layout_path / "pytorch_model.bin").exists()
