# SPDX-License-Identifier: Apache-2.0
"""Real native KDA, MLA and MoE decoding must survive speculative rollback."""

import json
import os
import subprocess
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    generator = np.random.default_rng(26)
    text = dict(
        num_hidden_layers=2,
        num_attention_heads=16,
        head_dim=32,
        layer_group_size=2,
        qk_rope_head_dim=16,
        qk_nope_head_dim=16,
        v_head_dim=32,
        kv_lora_rank=16,
        short_conv_kernel_size=4,
        first_k_dense_replace=1,
        num_experts=2,
        num_experts_per_tok=1,
        n_group=1,
        topk_group=1,
        max_position_embeddings=1024,
        rms_norm_eps=1e-6,
        rope_theta=10000,
        kda_lower_bound=-5,
        routed_scaling_factor=1,
        no_kda_lora=True,
        kda_safe_gate=True,
        rope_scaling=None,
        rope_interleave=True,
        hidden_act="silu",
        num_shared_experts=1,
        q_lora_rank=16,
        gated_attention_proj_granularity_type="head_wise",
        scoring_func="sigmoid",
        num_nextn_predict_layers=1,
        mtp_use_kda=False,
    )
    audio = dict(
        n_mels=128,
        n_audio_state=32,
        n_audio_head=1,
        n_audio_layer=0,
        downsampler_kernel_size_list=[],
        downsampler_stride_list=[],
    )
    (tmp_path / "config.json").write_text(
        json.dumps(
            dict(
                model_type="bailing_moe_v3_audio", text_config=text, audio_config=audio
            )
        )
    )
    (tmp_path / "generation_config.json").write_text(json.dumps({"eos_token_id": [31]}))
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {}})
    )
    (tmp_path / "vocab.json").write_text(
        json.dumps({chr(65 + i): i for i in range(32)})
    )
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")
    weights: dict[str, mx.array] = {}

    def random_weight(name: str, shape: tuple[int, ...]) -> None:
        weights[name] = mx.array(generator.normal(0, 0.12, shape).astype(np.float32))

    random_weight("model.word_embeddings.weight", (32, 32))
    random_weight("lm_head.weight", (32, 32))
    weights["model.norm.weight"] = mx.ones((32,))
    for name, inputs in (("conv1", 128), ("conv2", 32)):
        random_weight(f"model.audio.{name}.weight", (32, inputs, 3))
        weights[f"model.audio.{name}.bias"] = mx.zeros((32,))
    weights["model.audio.positional_embedding"] = mx.zeros((1500, 32))
    weights["model.audio.ln_post.weight"] = mx.ones((32,))
    weights["model.audio.ln_post.bias"] = mx.zeros((32,))
    random_weight("linear_proj_audio.1.weight", (32, 32))
    for layer in range(3):
        prefix = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            weights[f"{prefix}.{norm}.weight"] = mx.ones((32,))
        if layer == 2:
            for norm in ("enorm", "hnorm", "final_layernorm"):
                weights[f"{prefix}.{norm}.weight"] = mx.ones((32,))
            random_weight(f"{prefix}.eh_proj.weight", (32, 64))
        else:
            pass
        if layer == 0:
            for projection in ("q", "k", "v", "f", "g", "o"):
                shape = (32, 512) if projection == "o" else (512, 32)
                random_weight(f"{prefix}.attention.{projection}_proj.weight", shape)
            for projection in ("q", "k", "v"):
                random_weight(
                    f"{prefix}.attention.{projection}_conv1d.weight", (512, 1, 4)
                )
            random_weight(f"{prefix}.attention.b_proj.weight", (16, 32))
            weights[f"{prefix}.attention.A_log"] = mx.zeros((16,))
            weights[f"{prefix}.attention.dt_bias"] = mx.zeros((512,))
            weights[f"{prefix}.attention.o_norm.weight"] = mx.ones((32,))
        else:
            for projection, shape in (
                ("q_a_proj", (16, 32)),
                ("q_b_proj", (512, 16)),
                ("kv_a_proj_with_mqa", (32, 32)),
                ("kv_b_proj", (768, 16)),
                ("g_proj", (16, 32)),
                ("dense", (32, 512)),
            ):
                random_weight(f"{prefix}.attention.{projection}.weight", shape)
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                weights[f"{prefix}.attention.{norm}.weight"] = mx.ones((16,))
        mlps = (
            [f"{prefix}.mlp"]
            if layer == 0
            else [
                f"{prefix}.mlp.shared_experts",
                f"{prefix}.mlp.experts.0",
                f"{prefix}.mlp.experts.1",
            ]
        )
        for mlp in mlps:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                random_weight(f"{mlp}.{projection}.weight", (32, 32))
        if layer > 0:
            random_weight(f"{prefix}.mlp.gate.weight", (2, 32))
            weights[f"{prefix}.mlp.gate.expert_bias"] = mx.zeros((2,))
        else:
            pass
    mx.save_safetensors(str(tmp_path / "model.safetensors"), weights)
    return tmp_path


def test_cache_rejection_budget_and_repeated_requests(checkpoint: Path) -> None:
    probe = Path(os.environ["NATIVE_RUNTIME_BIN"]) / "ming3_mtp_probe"
    rows = json.loads(subprocess.check_output([str(probe), str(checkpoint)], text=True))
    assert sum(row["proposed"] for row in rows) > sum(row["accepted"] for row in rows)
    assert any(row["accepted"] > 0 for row in rows)
    assert all(row["tokens"] <= row["budget"] for row in rows)
