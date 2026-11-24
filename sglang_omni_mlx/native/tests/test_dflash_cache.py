# SPDX-License-Identifier: Apache-2.0
"""DFlash decoding keeps the target's cache exact and the draft's aligned.

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
    # -1 for none.
    stop_id: int
    prefill_step_tokens: int


class ProbeRow(TypedDict):
    plain_ids: list[int]
    drafted_ids: list[int]
    plain_commit_count: int
    drafted_commit_count: int
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
    write_target(target)
    draft = tmp_path / "draft"
    write_draft(draft, proposed_token)
    requests_path = tmp_path / "requests.json"
    requests_path.write_text(json.dumps(requests))
    completed = subprocess.run(
        [str(binary), str(target), str(draft), str(requests_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout)


def check_caches(row: ProbeRow, prompt_length: int, is_last_token_cached: bool) -> None:
    """Both caches cover the prompt and the committed tokens, the last only
    when it was an accepted guess or proposal rather than the target's own
    prediction, which the next round verifies."""
    assert row["plain_ids"] == row["drafted_ids"]
    assert row["drafted_commit_count"] == row["plain_commit_count"]
    assert POISON not in row["drafted_ids"]
    assert row["target_cache_length"] == row["draft_cache_length"]
    assert row["target_cache_length"] == (
        prompt_length + row["drafted_commit_count"] - (0 if is_last_token_cached else 1)
    )


@pytest.mark.parametrize(
    (
        "proposed_token",
        "prompt_ids",
        "guessed_ids",
        "accepted_guess_count",
        "prefill_step_tokens",
    ),
    [
        (DECOY, [10, 11, 12], [], 0, 0),
        (DECOY, [10, 11, 12], [], 0, 1),
        (30, [27, 28, 29], [], 0, 0),
        (40, [38, 39], [], 0, 0),
        (DECOY, [10, 11, 12], [13, 14, DECOY], 2, 0),
        (DECOY, [10, 11, 12], [13, 14, 15], 3, 0),
        (40, [38, 39], [40, DECOY, 40], 1, 0),
    ],
    ids=[
        "rejected",
        "rejected-chunked-prefill",
        "alternating",
        "accepted",
        "guesses-then-decoy",
        "guesses-all-accepted",
        "guess-rejected",
    ],
)
def test_drafted_decoding_keeps_the_caches_on_the_committed_tokens(
    dflash_cache_probe_binary: Path,
    tmp_path: Path,
    proposed_token: int,
    prompt_ids: list[int],
    guessed_ids: list[int],
    accepted_guess_count: int,
    prefill_step_tokens: int,
) -> None:
    requests: list[Request] = [
        {
            "prompt_ids": prompt_ids,
            "guessed_ids": guessed_ids,
            "budget": budget,
            "stop_id": -1,
            "prefill_step_tokens": prefill_step_tokens,
        }
        for budget in BUDGETS
    ]
    rows = probe(dflash_cache_probe_binary, tmp_path, proposed_token, requests)
    for budget, row in zip(BUDGETS, rows):
        assert row["drafted_commit_count"] == budget
        # A budget ends on the target's own prediction: the block is cut so
        # the last proposal leaves room for it.
        check_caches(row, len(prompt_ids), budget <= accepted_guess_count)
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


@pytest.mark.parametrize(
    ("proposed_token", "prompt_ids", "stop_id", "is_stop_proposed"),
    [(40, [38], 40, True), (DECOY, [10, 11, 12], 15, False)],
    ids=["stop-accepted-as-proposal", "stop-as-prediction"],
)
def test_a_stop_token_ends_the_decode_once(
    dflash_cache_probe_binary: Path,
    tmp_path: Path,
    proposed_token: int,
    prompt_ids: list[int],
    stop_id: int,
    is_stop_proposed: bool,
) -> None:
    request: Request = {
        "prompt_ids": prompt_ids,
        "guessed_ids": [],
        "budget": 64,
        "stop_id": stop_id,
        "prefill_step_tokens": 0,
    }
    [row] = probe(dflash_cache_probe_binary, tmp_path, proposed_token, [request])
    # The stop token is the last commit, and nothing is committed after it.
    assert row["drafted_commit_count"] == len(row["drafted_ids"]) + 1
    assert row["accepted_token_count"] == (1 if is_stop_proposed else 0)
    check_caches(row, len(prompt_ids), is_stop_proposed)
