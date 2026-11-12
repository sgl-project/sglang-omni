# SPDX-License-Identifier: Apache-2.0
"""NEXTN decoding of a small random Ming3 checkpoint matches plain decoding.

The checkpoint has one KDA layer, one latent-attention MoE layer and the
prediction layer, whose drafts are accepted often enough that both the commit
and the rollback paths run.
"""

from __future__ import annotations

import json
import math
import random
import struct
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest

VOCABULARY = 32
HIDDEN = 32
# Layer weights stay small next to the embeddings, and the prediction layer
# starts from the identity on the next token's embedding, so the draft is
# right often enough to be accepted while every layer still changes outputs.
LAYER_DEVIATION = 0.12


class TensorHeader(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class ProbeRow(TypedDict):
    budget: int
    tokens: int
    finish_reason: str
    draft_tokens: int
    accepted_draft_tokens: int


def write_safetensors(
    path: Path, tensors: dict[str, tuple[list[int], list[float]]]
) -> None:
    header: dict[str, TensorHeader] = {}
    data = bytearray()
    for name, (shape, values) in tensors.items():
        header[name] = {
            "dtype": "F32",
            "shape": shape,
            "data_offsets": [len(data), len(data) + 4 * len(values)],
        }
        data += struct.pack(f"<{len(values)}f", *values)
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))


@pytest.fixture
def checkpoint(tmp_path: Path) -> Path:
    text = {
        "num_hidden_layers": 2,
        "num_attention_heads": 16,
        "head_dim": 32,
        "layer_group_size": 2,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": 32,
        "kv_lora_rank": 16,
        "short_conv_kernel_size": 4,
        "first_k_dense_replace": 1,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "max_position_embeddings": 1024,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "kda_lower_bound": -5,
        "routed_scaling_factor": 1,
        "no_kda_lora": True,
        "kda_safe_gate": True,
        "rope_scaling": None,
        "rope_interleave": True,
        "hidden_act": "silu",
        "num_shared_experts": 1,
        "q_lora_rank": 16,
        "gated_attention_proj_granularity_type": "head_wise",
        "scoring_func": "sigmoid",
        "num_nextn_predict_layers": 1,
        "mtp_use_kda": False,
    }
    audio = {
        "n_mels": 128,
        "n_audio_state": HIDDEN,
        "n_audio_head": 1,
        "n_audio_layer": 0,
        "downsampler_kernel_size_list": [],
        "downsampler_stride_list": [],
    }
    (tmp_path / "config.json").write_text(
        json.dumps(
            {
                "model_type": "bailing_moe_v3_audio",
                "text_config": text,
                "audio_config": audio,
            }
        )
    )
    (tmp_path / "generation_config.json").write_text(
        json.dumps({"eos_token_id": [VOCABULARY - 1]})
    )
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps({"added_tokens_decoder": {}})
    )
    (tmp_path / "vocab.json").write_text(
        json.dumps({chr(65 + i): i for i in range(VOCABULARY)})
    )
    (tmp_path / "merges.txt").write_text("#version: 0.2\n")

    generator = random.Random(26)
    tensors: dict[str, tuple[list[int], list[float]]] = {}

    def normal(name: str, *shape: int, deviation: float = LAYER_DEVIATION) -> None:
        tensors[name] = (
            list(shape),
            [generator.gauss(0, deviation) for _ in range(math.prod(shape))],
        )

    def constant(name: str, value: float, *shape: int) -> None:
        tensors[name] = (list(shape), [value] * math.prod(shape))

    normal("model.word_embeddings.weight", VOCABULARY, HIDDEN, deviation=1)
    normal("lm_head.weight", VOCABULARY, HIDDEN, deviation=1)
    constant("model.norm.weight", 1, HIDDEN)
    for name, inputs in (("conv1", 128), ("conv2", HIDDEN)):
        normal(f"model.audio.{name}.weight", HIDDEN, inputs, 3)
        constant(f"model.audio.{name}.bias", 0, HIDDEN)
    constant("model.audio.positional_embedding", 0, 1500, HIDDEN)
    constant("model.audio.ln_post.weight", 1, HIDDEN)
    constant("model.audio.ln_post.bias", 0, HIDDEN)
    normal("linear_proj_audio.1.weight", HIDDEN, HIDDEN)
    # Layer 0 is KDA, layer 1 latent attention with MoE, and layer 2 the
    # prediction layer.
    for layer in range(3):
        prefix = f"model.layers.{layer}"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            constant(f"{prefix}.{norm}.weight", 1, HIDDEN)
        if layer == 0:
            for projection in ("q", "k", "v", "f", "g"):
                normal(f"{prefix}.attention.{projection}_proj.weight", 512, HIDDEN)
            normal(f"{prefix}.attention.o_proj.weight", HIDDEN, 512)
            for projection in ("q", "k", "v"):
                normal(f"{prefix}.attention.{projection}_conv1d.weight", 512, 1, 4)
            normal(f"{prefix}.attention.b_proj.weight", 16, HIDDEN)
            constant(f"{prefix}.attention.A_log", 0, 16)
            constant(f"{prefix}.attention.dt_bias", 0, 512)
            constant(f"{prefix}.attention.o_norm.weight", 1, HIDDEN)
            mlps = [f"{prefix}.mlp"]
        else:
            for projection, shape in (
                ("q_a_proj", (16, HIDDEN)),
                ("q_b_proj", (512, 16)),
                ("kv_a_proj_with_mqa", (HIDDEN, HIDDEN)),
                ("kv_b_proj", (768, 16)),
                ("g_proj", (16, HIDDEN)),
                ("dense", (HIDDEN, 512)),
            ):
                normal(f"{prefix}.attention.{projection}.weight", *shape)
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                constant(f"{prefix}.attention.{norm}.weight", 1, 16)
            normal(f"{prefix}.mlp.gate.weight", 2, HIDDEN)
            constant(f"{prefix}.mlp.gate.expert_bias", 0, 2)
            mlps = [
                f"{prefix}.mlp.shared_experts",
                f"{prefix}.mlp.experts.0",
                f"{prefix}.mlp.experts.1",
            ]
        if layer == 2:
            for norm in ("enorm", "hnorm", "final_layernorm"):
                constant(f"{prefix}.{norm}.weight", 1, HIDDEN)
            normal(f"{prefix}.eh_proj.weight", HIDDEN, 2 * HIDDEN)
            values = tensors[f"{prefix}.eh_proj.weight"][1]
            for index in range(HIDDEN):
                values[index * 2 * HIDDEN + index] += 1
        else:
            pass
        for mlp in mlps:
            for projection in ("gate_proj", "up_proj", "down_proj"):
                normal(f"{mlp}.{projection}.weight", HIDDEN, HIDDEN)
    write_safetensors(tmp_path / "model.safetensors", tensors)
    return tmp_path


def probe(probe_binary: Path, checkpoint: Path) -> list[ProbeRow]:
    return json.loads(
        subprocess.check_output([str(probe_binary), str(checkpoint)], text=True)
    )


def test_nextn_matches_plain_across_budgets(
    ming3_mtp_probe_binary: Path, checkpoint: Path
) -> None:
    rows = probe(ming3_mtp_probe_binary, checkpoint)
    assert all(row["tokens"] <= row["budget"] for row in rows)
    assert any(row["tokens"] > 5 for row in rows)
    accepted_draft_tokens = sum(row["accepted_draft_tokens"] for row in rows)
    assert 0 < accepted_draft_tokens < sum(row["draft_tokens"] for row in rows)


def test_nextn_stops_at_an_immediate_stop_token(
    ming3_mtp_probe_binary: Path, checkpoint: Path
) -> None:
    (checkpoint / "generation_config.json").write_text(
        json.dumps({"eos_token_id": list(range(VOCABULARY))})
    )
    rows = probe(ming3_mtp_probe_binary, checkpoint)
    assert all(row["tokens"] == 1 and row["finish_reason"] == "stop" for row in rows)
