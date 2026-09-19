# SPDX-License-Identifier: Apache-2.0
"""Contracts for the MiniMax Music 3 Torch-MPS backend."""

from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from sglang_omni.models.minimax_music3.rvq_decoder import RVQDepthDecoder
from sglang_omni.models.minimax_music3.torch_mps import (
    _build_text_pair,
    _load_rvq_depth_decoder,
)


def test_modular_rvq_checkpoint_maps_to_torch_decoder(tmp_path: Path) -> None:
    component_dir = tmp_path / "rvq_depth_decoder"
    component_dir.mkdir()
    config = {
        "hidden_size": 32,
        "num_layers": 2,
        "num_attention_heads": 4,
        "intermediate_size": 48,
        "audio_vocab_size": 16,
        "num_codebooks": 3,
        "max_position_embeddings": 8,
    }
    (component_dir / "config.json").write_text(json.dumps(config))
    decoder = RVQDepthDecoder(
        hidden_size=32,
        num_layers=2,
        num_heads=4,
        intermediate_size=48,
        audio_vocab_size=16,
        num_codebooks=3,
        max_seq_len=8,
    )
    expected = decoder.state_dict()
    state = {
        "projection.weight": expected["projection.weight"],
        "pos_embedding.weight": expected["pos_embedding.weight"],
        "norm.weight": expected["norm.weight"],
        "audio_embeddings.weight": torch.randn(32, 32),
    }
    for index in range(2):
        state[f"audio_heads.{index}.weight"] = expected[f"audio_heads.{index}.weight"]
    for index in range(2):
        source = f"layers.{index}"
        state[f"{source}.input_layernorm.weight"] = expected[
            f"{source}.input_layernorm.weight"
        ]
        state[f"{source}.post_attention_layernorm.weight"] = expected[
            f"{source}.post_attention_layernorm.weight"
        ]
        q, k, v = expected[f"{source}.self_attn.in_proj_weight"].chunk(3)
        state[f"{source}.attn.to_q.weight"] = q
        state[f"{source}.attn.to_k.weight"] = k
        state[f"{source}.attn.to_v.weight"] = v
        state[f"{source}.attn.to_out.weight"] = expected[
            f"{source}.self_attn.out_proj.weight"
        ]
        for projection in ("gate_proj", "up_proj", "down_proj"):
            state[f"{source}.{projection}.weight"] = expected[
                f"{source}.{projection}.weight"
            ]
    save_file(state, component_dir / "diffusion_pytorch_model.safetensors")

    loaded, embeddings = _load_rvq_depth_decoder(
        tmp_path,
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    for name, value in expected.items():
        torch.testing.assert_close(loaded.state_dict()[name], value)
    torch.testing.assert_close(embeddings.weight, state["audio_embeddings.weight"])


def test_text_pair_replaces_only_unconditional_prompt_body() -> None:
    class Tokenizer:
        def __call__(self, prompt, return_tensors):
            assert prompt == "prompt"
            assert return_tensors == "pt"
            return {"input_ids": torch.tensor([[1, 2, 3, 4, 5]])}

    paired = _build_text_pair(
        "prompt",
        Tokenizer(),
        device=torch.device("cpu"),
    )

    assert paired[0].tolist() == [1, 2, 3, 4, 5]
    assert paired[1].tolist() == [1, 151654, 151654, 4, 5]
