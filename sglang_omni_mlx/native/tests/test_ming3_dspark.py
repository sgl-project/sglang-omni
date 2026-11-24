# SPDX-License-Identifier: Apache-2.0
"""DSpark decoding of a small random Ming ASR checkpoint matches plain decoding.

The target has one KDA layer and one latent-attention MoE layer whose weights
stay small next to the embeddings, so each token mostly predicts a fixed
successor. The draft's Markov head carries that successor table, so its blocks
are accepted often enough that full acceptance, rollback and the ngram ban all
run in verification.
"""

from __future__ import annotations

import json
import math
import random
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest
from test_ming3_mtp import LAYER_DEVIATION, write_safetensors

VOCABULARY = 32
HIDDEN = 64
HEADS = 2
HEAD_DIMENSION = 32
AUDIO_HIDDEN = 32
MARKOV_RANK = HIDDEN
# The draft cannot propose the last two target ids.
DRAFT_VOCABULARY = VOCABULARY - 2
BLOCK_SIZE = 8


class ProbeRow(TypedDict):
    budget: int
    no_repeat_ngram_size: int
    token_ids: list[int]
    finish_reason: str
    verify_round_count: int
    proposed_token_count: int
    accepted_token_count: int


Tensors = dict[str, tuple[list[int], list[float]]]


def write_target(
    directory: Path, generator: random.Random, layer_deviation: float
) -> Tensors:
    directory.mkdir()
    text = {
        "hidden_size": HIDDEN,
        "vocab_size": VOCABULARY,
        "intermediate_size": HIDDEN,
        "moe_intermediate_size": HIDDEN,
        "moe_shared_expert_intermediate_size": HIDDEN,
        "num_hidden_layers": 2,
        "num_attention_heads": HEADS,
        "head_dim": HEAD_DIMENSION,
        "layer_group_size": 2,
        "kv_lora_rank": 16,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 16,
        "qk_nope_head_dim": 16,
        "v_head_dim": HEAD_DIMENSION,
        "short_conv_kernel_size": 4,
        "num_experts": 2,
        "num_experts_per_tok": 1,
        "n_group": 1,
        "topk_group": 1,
        "first_k_dense_replace": 1,
        "max_position_embeddings": 1024,
        "eos_token_id": VOCABULARY - 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000,
        "routed_scaling_factor": 1,
        "kda_lower_bound": -5,
        "no_kda_lora": True,
        "kda_safe_gate": True,
        "rope_interleave": True,
        "hidden_act": "silu",
        "use_qkv_bias": False,
        "use_bias": False,
        "tie_word_embeddings": False,
        "gated_attention_proj_granularity_type": "head_wise",
        "num_shared_experts": 1,
        "rope_scaling": None,
        "scoring_func": "sigmoid",
    }
    audio = {
        "n_mels": 128,
        "n_audio_state": AUDIO_HIDDEN,
        "n_audio_head": 1,
        "n_audio_layer": 1,
        "n_audio_ctx": 8,
        "whisper_chunk_size": 15,
        "downsampler_kernel_size_list": [3],
        "downsampler_stride_list": [2],
    }
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "ming_v3_asr",
                "audio_token_id": VOCABULARY - 2,
                "quantization": {"group_size": 64, "bits": 4},
                "text_config": text,
                "audio_config": audio,
            }
        )
    )
    tensors: Tensors = {}

    def normal(name: str, *shape: int, deviation: float = layer_deviation) -> None:
        tensors[name] = (
            list(shape),
            [generator.gauss(0, deviation) for _ in range(math.prod(shape))],
        )

    def constant(name: str, value: float, *shape: int) -> None:
        tensors[name] = (list(shape), [value] * math.prod(shape))

    projection = HEADS * HEAD_DIMENSION
    normal("model.word_embeddings.weight", VOCABULARY, HIDDEN, deviation=1)
    normal("lm_head.weight", VOCABULARY, HIDDEN, deviation=1)
    # Zero rows leave the stop token unlikely, so requests run long, and the
    # mask token too, which the draft cannot propose.
    tensors["lm_head.weight"][1][-2 * HIDDEN :] = [0.0] * 2 * HIDDEN
    constant("model.norm.weight", 1, HIDDEN)
    normal("model.audio.conv1.weight", AUDIO_HIDDEN, 3, 128)
    normal("model.audio.conv2.weight", AUDIO_HIDDEN, 3, AUDIO_HIDDEN)
    for name in ("conv1", "conv2", "ln_post"):
        constant(f"model.audio.{name}.bias", 0, AUDIO_HIDDEN)
    constant("model.audio.ln_post.weight", 1, AUDIO_HIDDEN)
    constant("model.audio.positional_embedding", 0, 8, AUDIO_HIDDEN)
    block = "model.audio.blocks.0"
    for norm in ("attn_ln", "mlp_ln"):
        constant(f"{block}.{norm}.weight", 1, AUDIO_HIDDEN)
        constant(f"{block}.{norm}.bias", 0, AUDIO_HIDDEN)
    for name in ("query", "key", "value", "out"):
        normal(f"{block}.attn.{name}.weight", AUDIO_HIDDEN, AUDIO_HIDDEN)
        if name != "key":
            constant(f"{block}.attn.{name}.bias", 0, AUDIO_HIDDEN)
        else:
            pass
    normal(f"{block}.mlp.0.weight", 4 * AUDIO_HIDDEN, AUDIO_HIDDEN)
    constant(f"{block}.mlp.0.bias", 0, 4 * AUDIO_HIDDEN)
    normal(f"{block}.mlp.2.weight", AUDIO_HIDDEN, 4 * AUDIO_HIDDEN)
    constant(f"{block}.mlp.2.bias", 0, AUDIO_HIDDEN)
    normal("linear_proj_audio.0.weight", HIDDEN, 3, AUDIO_HIDDEN)
    constant("linear_proj_audio.0.bias", 0, HIDDEN)
    normal("linear_proj_audio.3.weight", HIDDEN, HIDDEN)
    constant("linear_proj_audio.3.bias", 0, HIDDEN)
    # Layer 0 is KDA with a dense MLP, layer 1 latent attention with MoE.
    for layer in range(2):
        prefix = f"model.layers.{layer}"
        attention = f"{prefix}.attention"
        for norm in ("input_layernorm", "post_attention_layernorm"):
            constant(f"{prefix}.{norm}.weight", 1, HIDDEN)
        if layer == 0:
            for name in ("q", "k", "v"):
                normal(f"{attention}.{name}_proj.weight", projection, HIDDEN)
                normal(f"{attention}.{name}_conv1d.weight", projection, 4, 1)
            for name in ("f", "g"):
                normal(f"{attention}.{name}_proj.weight", projection, HIDDEN)
            normal(f"{attention}.b_proj.weight", HEADS, HIDDEN)
            normal(f"{attention}.o_proj.weight", HIDDEN, projection)
            constant(f"{attention}.A_log", 0, HEADS)
            constant(f"{attention}.dt_bias", 0, projection)
            constant(f"{attention}.o_norm.weight", 1, HEAD_DIMENSION)
            for name in ("gate_proj", "up_proj"):
                normal(f"{prefix}.mlp.{name}.weight", HIDDEN, HIDDEN)
            normal(f"{prefix}.mlp.down_proj.weight", HIDDEN, HIDDEN)
        else:
            for name, shape in (
                ("q_a_proj", (16, HIDDEN)),
                ("q_b_proj", (HEADS * 32, 16)),
                ("kv_a_proj_with_mqa", (16 + 16, HIDDEN)),
                ("kv_b_proj", (HEADS * (16 + HEAD_DIMENSION), 16)),
                ("g_proj", (HEADS, HIDDEN)),
                ("dense", (HIDDEN, HEADS * HEAD_DIMENSION)),
            ):
                normal(f"{attention}.{name}.weight", *shape)
            for norm in ("q_a_layernorm", "kv_a_layernorm"):
                constant(f"{attention}.{norm}.weight", 1, 16)
            normal(f"{prefix}.mlp.gate.weight", 2, HIDDEN)
            constant(f"{prefix}.mlp.gate.expert_bias", 0, 2)
            for name in ("gate_proj", "up_proj"):
                normal(f"{prefix}.mlp.switch_mlp.{name}.weight", 2, HIDDEN, HIDDEN)
                normal(f"{prefix}.mlp.shared_experts.{name}.weight", HIDDEN, HIDDEN)
            normal(f"{prefix}.mlp.switch_mlp.down_proj.weight", 2, HIDDEN, HIDDEN)
            normal(f"{prefix}.mlp.shared_experts.down_proj.weight", HIDDEN, HIDDEN)
    write_safetensors(directory / "model.safetensors", tensors)
    return tensors


def write_draft(
    directory: Path,
    generator: random.Random,
    target: Tensors,
    layer_deviation: float,
    markov_sign: float,
) -> None:
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "speculators_model_type": "dspark",
                "aux_hidden_state_layer_ids": [0, 1],
                "block_size": BLOCK_SIZE,
                "draft_vocab_size": DRAFT_VOCABULARY,
                "mask_token_id": VOCABULARY - 2,
                "sample_from_anchor": True,
                "sliding_window_non_causal": False,
                "markov_head_type": "vanilla",
                "markov_rank": MARKOV_RANK,
                "transformer_layer_config": {
                    "hidden_size": HIDDEN,
                    "intermediate_size": HIDDEN,
                    "num_hidden_layers": 1,
                    "num_attention_heads": HEADS,
                    "num_key_value_heads": HEADS,
                    "head_dim": HEAD_DIMENSION,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {"rope_theta": 10000},
                    "vocab_size": VOCABULARY,
                },
            }
        )
    )
    tensors: Tensors = {}

    def normal(name: str, *shape: int) -> None:
        tensors[name] = (
            list(shape),
            [generator.gauss(0, layer_deviation) for _ in range(math.prod(shape))],
        )

    def constant(name: str, value: float, *shape: int) -> None:
        tensors[name] = (list(shape), [value] * math.prod(shape))

    projection = HEADS * HEAD_DIMENSION
    normal("fc.weight", HIDDEN, 2 * HIDDEN)
    constant("hidden_norm.weight", 1, HIDDEN)
    constant("norm.weight", 1, HIDDEN)
    prefix = "layers.0"
    for norm in ("input_layernorm", "post_attention_layernorm"):
        constant(f"{prefix}.{norm}.weight", 1, HIDDEN)
    for name in ("q_proj", "k_proj", "v_proj"):
        normal(f"{prefix}.self_attn.{name}.weight", projection, HIDDEN)
    normal(f"{prefix}.self_attn.o_proj.weight", HIDDEN, projection)
    for norm in ("q_norm", "k_norm"):
        constant(f"{prefix}.self_attn.{norm}.weight", 1, HEAD_DIMENSION)
    for name in ("gate_proj", "up_proj"):
        normal(f"{prefix}.mlp.{name}.weight", HIDDEN, HIDDEN)
    normal(f"{prefix}.mlp.down_proj.weight", HIDDEN, HIDDEN)
    normal("lm_head.weight", DRAFT_VOCABULARY, HIDDEN)
    # Draft id j is target id j, so d2t holds zero offsets.
    constant("d2t", 0, DRAFT_VOCABULARY)
    # w2(w1[previous]) is markov_sign times the target's logits of the previous
    # token's embedding, the successor the small target layers mostly keep: a
    # sign of 1 drafts that successor, -1 the target's least likely token.
    tensors["markov_head.markov_w1.weight"] = target["model.word_embeddings.weight"]
    shape, values = target["lm_head.weight"]
    tensors["markov_head.markov_w2.weight"] = (
        [DRAFT_VOCABULARY, shape[1]],
        [markov_sign * value for value in values[: DRAFT_VOCABULARY * shape[1]]],
    )
    write_safetensors(directory / "model.safetensors", tensors)


def probe(
    probe_binary: Path,
    directory: Path,
    *,
    layer_deviation: float,
    markov_sign: float,
    draft_policy: str,
) -> list[ProbeRow]:
    """Rows of a probe run, which fails if drafted decoding leaves plain's."""
    generator = random.Random(3)
    target = write_target(directory / "target", generator, layer_deviation)
    write_draft(directory / "draft", generator, target, layer_deviation, markov_sign)
    return json.loads(
        subprocess.check_output(
            [
                str(probe_binary),
                str(directory / "target"),
                str(directory / "draft"),
                draft_policy,
            ],
            text=True,
        )
    )


@pytest.mark.parametrize("draft_policy", ["fixed", "adaptive"])
def test_dspark_matches_plain_across_budgets_and_ngram_bans(
    ming3_dspark_probe_binary: Path, tmp_path: Path, draft_policy: str
) -> None:
    rows = probe(
        ming3_dspark_probe_binary,
        tmp_path,
        layer_deviation=LAYER_DEVIATION,
        markov_sign=1,
        draft_policy=draft_policy,
    )
    assert all(len(row["token_ids"]) <= row["budget"] for row in rows)
    accepted = sum(row["accepted_token_count"] for row in rows)
    assert 0 < accepted < sum(row["proposed_token_count"] for row in rows)
    longest = {
        row["no_repeat_ngram_size"]: row["token_ids"]
        for row in rows
        if row["budget"] == 64
    }
    assert all(len(token_ids) == 64 for token_ids in longest.values())
    # The ban changed what the target verified, and no bigram repeats.
    assert longest[2] != longest[0]
    bigrams = list(zip(longest[2], longest[2][1:]))
    assert len(set(bigrams)) == len(bigrams)


def test_adaptive_backoff_resumes_a_draft_faster_than_plain(
    ming3_dspark_probe_binary: Path, tmp_path: Path
) -> None:
    """Without layers the target and the draft are the same successor table.

    Every block is accepted, so the plain probe after round 9 resumes
    speculation, whose context then needs the probe's positions; the decoder
    refuses to draft from a context that does not cover the target's.
    """
    rows = probe(
        ming3_dspark_probe_binary,
        tmp_path,
        layer_deviation=0,
        markov_sign=1,
        draft_policy="adaptive",
    )
    # The ngram bans redirect the target where the draft cannot follow.
    [longest] = [
        row for row in rows if row["budget"] == 256 and row["no_repeat_ngram_size"] == 0
    ]
    assert len(longest["token_ids"]) == 256
    assert longest["accepted_token_count"] == longest["proposed_token_count"]
    assert longest["verify_round_count"] > 9


def test_adaptive_backoff_stops_a_draft_slower_than_plain(
    ming3_dspark_probe_binary: Path, tmp_path: Path
) -> None:
    """A draft of the target's least likely tokens is never accepted, so a
    plain probe ends speculation and plain decoding finishes the request."""
    rows = probe(
        ming3_dspark_probe_binary,
        tmp_path,
        layer_deviation=LAYER_DEVIATION,
        markov_sign=-1,
        draft_policy="adaptive",
    )
    [longest] = [
        row for row in rows if row["budget"] == 256 and row["no_repeat_ngram_size"] == 0
    ]
    assert len(longest["token_ids"]) == 256
    assert longest["accepted_token_count"] == 0
    # Each round commits one token, so drafting to the end would take 255.
    assert longest["verify_round_count"] < 64
