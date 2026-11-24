# SPDX-License-Identifier: Apache-2.0
"""DFlash decoding keeps the target's and the draft's caches exact.

The target is one Qwen3 layer whose attention is uniform (zero queries and
keys) and reads only a decoy token's value, which it adds to a poison
direction. Without the decoy in its cache the target is a successor table, so
plain decoding never meets it; one decoy row left behind after a rejection
turns every later prediction into the poison token. Each draft proposes one
constant token: the decoy, rejected every round; a token the target accepts
every other position; or the fixed point the target accepts every time.
"""

from __future__ import annotations

import json
import struct
import subprocess
from pathlib import Path
from typing import TypedDict

import pytest

HIDDEN = 64
VOCABULARY = 48
INTERMEDIATE = 8
BLOCK_SIZE = 4
# Embedding dimensions past the token dimensions: the decoy's value and the
# direction it is added to.
DECOY_DIMENSION = 62
POISON_DIMENSION = 63
# Large enough that one decoy row among the longest request's positions still
# outweighs the current token in the final norm.
POISON_GAIN = 1000.0
DECOY = 45
POISON = 46
MASK = 47
# Rejection chain: 12 -> 13 ... 20 -> 13; alternation: 29 -> 30 <-> 31; fixed
# point: 39 -> 40 -> 40.
SUCCESSORS = {
    **{token: token + 1 for token in range(10, 20)},
    20: 13,
    27: 28,
    28: 29,
    29: 30,
    30: 31,
    31: 30,
    38: 39,
    39: 40,
    40: 40,
}
BUDGETS = [1, 2, 3, 4, 5, 7, 16, 64]


class TensorHeader(TypedDict):
    dtype: str
    shape: list[int]
    data_offsets: list[int]


class Request(TypedDict):
    prompt_ids: list[int]
    guessed_ids: list[int]
    budget: int


class ProbeRow(TypedDict):
    plain_ids: list[int]
    drafted_ids: list[int]
    target_cache_length: int
    draft_cache_length: int
    verify_round_count: int
    proposed_token_count: int
    accepted_token_count: int


def write_safetensors(
    path: Path, tensors: dict[str, tuple[str, list[int], list[float]]]
) -> None:
    header: dict[str, TensorHeader] = {}
    data = bytearray()
    for name, (dtype, shape, values) in tensors.items():
        if dtype == "F32":
            encoded_values = struct.pack(f"<{len(values)}f", *values)
        else:
            encoded_values = struct.pack(f"<{len(values)}q", *map(int, values))
        header[name] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [len(data), len(data) + len(encoded_values)],
        }
        data += encoded_values
    encoded = json.dumps(header).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + bytes(data))


def matrix(
    rows: int, columns: int, entries: dict[tuple[int, int], float]
) -> list[float]:
    values = [0.0] * (rows * columns)
    for (row, column), value in entries.items():
        values[row * columns + column] = value
    return values


def layer_config() -> dict[str, int | float]:
    return {
        "num_hidden_layers": 1,
        "num_attention_heads": 1,
        "num_key_value_heads": 1,
        "head_dim": HIDDEN,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10000.0,
        "hidden_size": HIDDEN,
        "vocab_size": VOCABULARY,
    }


def layer_tensors(
    value_entries: dict[tuple[int, int], float],
    output_entries: dict[tuple[int, int], float],
) -> dict[str, tuple[str, list[int], list[float]]]:
    ones = [1.0] * HIDDEN
    return {
        "layers.0.input_layernorm.weight": ("F32", [HIDDEN], ones),
        "layers.0.post_attention_layernorm.weight": ("F32", [HIDDEN], ones),
        "layers.0.self_attn.q_norm.weight": ("F32", [HIDDEN], ones),
        "layers.0.self_attn.k_norm.weight": ("F32", [HIDDEN], ones),
        "layers.0.self_attn.q_proj.weight": (
            "F32",
            [HIDDEN, HIDDEN],
            matrix(HIDDEN, HIDDEN, {}),
        ),
        "layers.0.self_attn.k_proj.weight": (
            "F32",
            [HIDDEN, HIDDEN],
            matrix(HIDDEN, HIDDEN, {}),
        ),
        "layers.0.self_attn.v_proj.weight": (
            "F32",
            [HIDDEN, HIDDEN],
            matrix(HIDDEN, HIDDEN, value_entries),
        ),
        "layers.0.self_attn.o_proj.weight": (
            "F32",
            [HIDDEN, HIDDEN],
            matrix(HIDDEN, HIDDEN, output_entries),
        ),
        "layers.0.mlp.gate_proj.weight": (
            "F32",
            [INTERMEDIATE, HIDDEN],
            matrix(INTERMEDIATE, HIDDEN, {}),
        ),
        "layers.0.mlp.up_proj.weight": (
            "F32",
            [INTERMEDIATE, HIDDEN],
            matrix(INTERMEDIATE, HIDDEN, {}),
        ),
        "layers.0.mlp.down_proj.weight": (
            "F32",
            [HIDDEN, INTERMEDIATE],
            matrix(HIDDEN, INTERMEDIATE, {}),
        ),
        "norm.weight": ("F32", [HIDDEN], ones),
    }


def write_target(directory: Path) -> None:
    directory.mkdir()
    (directory / "config.json").write_text(json.dumps(layer_config()))
    embedding = {(token, token): 1.0 for token in range(VOCABULARY) if token != DECOY}
    embedding[(DECOY, DECOY_DIMENSION)] = 1.0
    # Row s reads every token whose successor is s; the poison row reads the
    # poison direction.
    output = {(successor, token): 1.0 for token, successor in SUCCESSORS.items()}
    output[(POISON, POISON_DIMENSION)] = 1.0
    write_safetensors(
        directory / "model.safetensors",
        {
            **layer_tensors(
                {(DECOY_DIMENSION, DECOY_DIMENSION): 1.0},
                {(POISON_DIMENSION, DECOY_DIMENSION): POISON_GAIN},
            ),
            "embed_tokens.weight": (
                "F32",
                [VOCABULARY, HIDDEN],
                matrix(VOCABULARY, HIDDEN, embedding),
            ),
            "output_projection.weight": (
                "F32",
                [VOCABULARY, HIDDEN],
                matrix(VOCABULARY, HIDDEN, output),
            ),
        },
    )


def write_draft(directory: Path, proposed_token: int) -> None:
    """A draft whose one-token vocabulary is proposed_token, at every position."""
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "transformer_layer_config": layer_config(),
                "mask_token_id": MASK,
                "block_size": BLOCK_SIZE,
                "draft_vocab_size": 1,
                "aux_hidden_state_layer_ids": [0],
                "sample_from_anchor": False,
                "sliding_window_non_causal": False,
            }
        )
    )
    write_safetensors(
        directory / "model.safetensors",
        {
            **layer_tensors({}, {}),
            "fc.weight": ("F32", [HIDDEN, HIDDEN], matrix(HIDDEN, HIDDEN, {})),
            "hidden_norm.weight": ("F32", [HIDDEN], [1.0] * HIDDEN),
            "lm_head.weight": ("F32", [1, HIDDEN], matrix(1, HIDDEN, {(0, MASK): 1.0})),
            "d2t": ("I64", [1], [float(proposed_token)]),
        },
    )


def probe(
    binary: Path, tmp_path: Path, proposed_token: int, requests: list[Request]
) -> list[ProbeRow]:
    target = tmp_path / "target"
    if not target.exists():
        write_target(target)
    else:
        pass
    draft = tmp_path / f"draft-{proposed_token}"
    write_draft(draft, proposed_token)
    requests_path = tmp_path / f"requests-{proposed_token}.json"
    requests_path.write_text(json.dumps(requests))
    completed = subprocess.run(
        [str(binary), str(target), str(draft), str(requests_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


@pytest.mark.parametrize(
    ("proposed_token", "prompt_ids", "guessed_ids"),
    [
        (DECOY, [10, 11, 12], []),
        (30, [27, 28, 29], []),
        (40, [38, 39], []),
        (DECOY, [10, 11, 12], [13, 14, DECOY]),
        (40, [38, 39], [40, DECOY, 40]),
    ],
    ids=["rejected", "alternating", "accepted", "guesses-then-decoy", "guess-rejected"],
)
def test_drafted_decoding_keeps_both_caches_exact(
    dflash_cache_probe_binary: Path,
    tmp_path: Path,
    proposed_token: int,
    prompt_ids: list[int],
    guessed_ids: list[int],
) -> None:
    requests: list[Request] = [
        {"prompt_ids": prompt_ids, "guessed_ids": guessed_ids, "budget": budget}
        for budget in BUDGETS
    ]
    rows = probe(dflash_cache_probe_binary, tmp_path, proposed_token, requests)
    for budget, row in zip(BUDGETS, rows):
        assert row["plain_ids"] == row["drafted_ids"], budget
        assert POISON not in row["plain_ids"]
        # Both caches cover the prompt and every committed token but the last,
        # which is cached too when the round ended on an accepted proposal.
        committed_count = len(row["drafted_ids"])
        assert row["target_cache_length"] == row["draft_cache_length"], budget
        assert (
            len(prompt_ids) + committed_count - 1
            <= row["target_cache_length"]
            <= len(prompt_ids) + committed_count
        ), budget
    long_request = rows[-1]
    assert long_request["verify_round_count"] > 0
    if proposed_token == DECOY:
        assert long_request["accepted_token_count"] == 0
    elif proposed_token == 30:
        assert (
            0
            < long_request["accepted_token_count"]
            < long_request["proposed_token_count"]
        )
    else:
        assert (
            long_request["accepted_token_count"] == long_request["proposed_token_count"]
        )
