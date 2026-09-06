# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import os
import tempfile

from sglang_omni.models.voxtral_asr.model_config import VoxtralRealtimeConfig

_REALTIME_PARAMS = {
    "dim": 2560,
    "n_layers": 32,
    "head_dim": 80,
    "hidden_dim": 8192,
    "n_heads": 32,
    "n_kv_heads": 8,
    "vocab_size": 131072,
    "rope_theta": 1000000.0,
    "norm_eps": 1e-5,
    "max_seq_len": 131072,
    "tied_embeddings": True,
    "model_type": "voxtral",
    "multimodal": {
        "whisper_model_args": {
            "encoder_args": {
                "dim": 1280,
                "n_layers": 32,
                "hidden_dim": 5120,
                "n_heads": 20,
                "head_dim": 64,
                "vocab_size": 51864,
                "max_source_positions": 1500,
                "causal": True,
                "sliding_window": 750,
                "pos_embed": "sinusoidal",
                "audio_encoding_args": {
                    "num_mel_bins": 128,
                    "window_size": 400,
                    "hop_length": 160,
                    "sampling_rate": 16000,
                    "global_log_mel_max": 4.0,
                },
            },
            "downsample_args": {"downsample_factor": 2},
        }
    },
}


def test_parse_voxtral_realtime_config() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        params_path = os.path.join(tmpdir, "params.json")
        with open(params_path, "w") as f:
            json.dump(_REALTIME_PARAMS, f)

        cfg = VoxtralRealtimeConfig.from_model_path(tmpdir)

        assert cfg.text_config.dim == 2560
        assert cfg.text_config.n_layers == 32
        assert cfg.text_config.n_kv_heads == 8

        assert cfg.audio_config.dim == 1280
        assert cfg.audio_config.n_layers == 32
        assert cfg.audio_config.is_causal is True
        assert cfg.audio_config.sliding_window == 750
        assert cfg.audio_config.downsample_factor == 2
        assert cfg.audio_config.block_pool_size == 2
        assert cfg.audio_config.audio_encoding_args.num_mel_bins == 128
        assert cfg.audio_config.audio_encoding_args.sampling_rate == 16000
