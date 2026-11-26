# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR's request budget, windowing and text joining on small checkpoints.

Both checkpoints have all-zero decoder layers, so every position keeps its
embedding and the next token is the argmax of RmsNorm(last embedding) against
the table. The loop checkpoint gives every row the same vector except a
strongly negative <|im_end|> row, so a decode never stops early and runs to
its token limit. The stop checkpoint's rows form the chain 259 -> 5 -> 7 ->
<|im_end|>, so every window decodes exactly "éx" and stops; the multi-window
requests then show how the budget is spent and how the window texts join.

The DFlash draft's one-token vocabulary maps to id 0, the loop checkpoint's
token, so its proposals are always accepted there and never on the stop
checkpoint; with either, decoding with the draft must equal plain decoding.
Attention adds nothing here, so this guards ARK-ASR's draft wiring (budget,
stop, windows, draft counts), not the shared draft's cache trimming and
context, which only the real-model DFlash golden checks.
"""

from __future__ import annotations

import json
import math
import random
import struct
import subprocess
from pathlib import Path

from native_server import EVENT_TIMEOUT_SECONDS, SupervisedServer

HIDDEN = 32
HEADS = 2
KV_HEADS = 2
MLP_HIDDEN = 64
MEL_BINS = 128
MERGE_FACTOR = 4
ADDED_TOKENS = [
    "<|user|>",
    "<|begin_of_audio|>",
    "<|end_of_audio|>",
    "<|assistant|>",
    "<|im_end|>",
    "<|audio|>",
]
VOCABULARY = 256 + len(ADDED_TOKENS) + 8
IM_END_ID = 256 + ADDED_TOKENS.index("<|im_end|>")
ASSISTANT_ID = 256 + ADDED_TOKENS.index("<|assistant|>")
AUDIO_TOKEN_ID = 256 + ADDED_TOKENS.index("<|audio|>")
ACCENTED_E_ID = 5
PLAIN_X_ID = 7
DRAFT_HEAD_DIM = 16
DRAFT_MASK_ID = VOCABULARY - 1


def write_safetensors(
    path: Path, tensors: dict[str, tuple[list[int], list[float]]]
) -> None:
    header = {}
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


def byte_level_vocabulary() -> dict[str, int]:
    printable = set(range(ord("!"), ord("~") + 1))
    printable |= set(range(0xA1, 0xAC + 1))
    printable |= set(range(0xAE, 0xFF + 1))
    vocabulary = {}
    next_code_point = 256
    for byte in range(256):
        if byte in printable:
            vocabulary[chr(byte)] = byte
        else:
            vocabulary[chr(next_code_point)] = byte
            next_code_point += 1
    return vocabulary


def write_checkpoint(
    directory: Path, embeddings: list[list[float]], extra_vocabulary: dict[str, int]
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "whisper_config": {
                    "encoder_layers": 1,
                    "encoder_attention_heads": HEADS,
                    "num_mel_bins": MEL_BINS,
                },
                "audio_token_id": AUDIO_TOKEN_ID,
                "merge_factor": MERGE_FACTOR,
                "use_rope": True,
                "adapter_type": "mlp",
                "mlp_adapter_act": "gelu",
                "tie_word_embeddings": True,
                "num_hidden_layers": 1,
                "num_attention_heads": HEADS,
                "num_key_value_heads": KV_HEADS,
                "hidden_size": HIDDEN,
                "vocab_size": VOCABULARY,
                "rms_norm_eps": 1e-6,
                "rope_theta": 10000.0,
            }
        )
    )
    vocabulary = byte_level_vocabulary() | extra_vocabulary
    (directory / "vocab.json").write_text(json.dumps(vocabulary))
    (directory / "merges.txt").write_text("#version: 0.2\n")
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    str(256 + i): {"content": content, "special": True}
                    for i, content in enumerate(ADDED_TOKENS)
                }
            }
        )
    )

    tensors: dict[str, tuple[list[int], list[float]]] = {}

    generator = random.Random(42)

    def normal(name: str, *shape: int) -> None:
        tensors[name] = (
            list(shape),
            [generator.gauss(0, 0.05) for _ in range(math.prod(shape))],
        )

    def constant(name: str, value: float, *shape: int) -> None:
        tensors[name] = (list(shape), [value] * math.prod(shape))

    whisper = "audio_encoder.whisper"
    normal(f"{whisper}.conv1.weight", HIDDEN, MEL_BINS, 3)
    constant(f"{whisper}.conv1.bias", 0, HIDDEN)
    normal(f"{whisper}.conv2.weight", HIDDEN, HIDDEN, 3)
    constant(f"{whisper}.conv2.bias", 0, HIDDEN)
    layer = f"{whisper}.layers.0"
    for norm in ("self_attn_layer_norm", "final_layer_norm"):
        constant(f"{layer}.{norm}.weight", 1, HIDDEN)
        constant(f"{layer}.{norm}.bias", 0, HIDDEN)
    for name in ("q_proj", "k_proj", "v_proj", "out_proj"):
        normal(f"{layer}.self_attn.{name}.weight", HIDDEN, HIDDEN)
        constant(f"{layer}.self_attn.{name}.bias", 0, HIDDEN)
    normal(f"{layer}.fc1.weight", MLP_HIDDEN, HIDDEN)
    constant(f"{layer}.fc1.bias", 0, MLP_HIDDEN)
    normal(f"{layer}.fc2.weight", HIDDEN, MLP_HIDDEN)
    constant(f"{layer}.fc2.bias", 0, HIDDEN)
    constant("audio_encoder.layer_norm.weight", 1, HIDDEN)
    constant("audio_encoder.layer_norm.bias", 0, HIDDEN)
    normal("audio_encoder.adapting.0.weight", MLP_HIDDEN, HIDDEN * MERGE_FACTOR)
    constant("audio_encoder.adapting.0.bias", 0, MLP_HIDDEN)
    normal("audio_encoder.adapting.2.weight", HIDDEN, MLP_HIDDEN)
    constant("audio_encoder.adapting.2.bias", 0, HIDDEN)

    tensors["model.embed_tokens.weight"] = (
        [VOCABULARY, HIDDEN],
        [value for embedding in embeddings for value in embedding],
    )
    decoder_layer = "model.layers.0"
    for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
        constant(f"{decoder_layer}.self_attn.{name}.weight", 0, HIDDEN, HIDDEN)
    for name in ("gate_proj", "up_proj"):
        constant(f"{decoder_layer}.mlp.{name}.weight", 0, MLP_HIDDEN, HIDDEN)
    constant(f"{decoder_layer}.mlp.down_proj.weight", 0, HIDDEN, MLP_HIDDEN)
    constant(f"{decoder_layer}.input_layernorm.weight", 1, HIDDEN)
    constant(f"{decoder_layer}.post_attention_layernorm.weight", 1, HIDDEN)
    constant("model.norm.weight", 1, HIDDEN)

    write_safetensors(directory / "model.safetensors", tensors)


def write_draft(directory: Path, hidden_size: int = HIDDEN) -> None:
    """A random one-layer DFlash draft that reads the target's only layer."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(
        json.dumps(
            {
                "aux_hidden_state_layer_ids": [0],
                "block_size": 8,
                "draft_vocab_size": 1,
                "mask_token_id": DRAFT_MASK_ID,
                "sample_from_anchor": False,
                "sliding_window_non_causal": False,
                "transformer_layer_config": {
                    "num_hidden_layers": 1,
                    "num_attention_heads": HEADS,
                    "num_key_value_heads": KV_HEADS,
                    "head_dim": DRAFT_HEAD_DIM,
                    "hidden_size": hidden_size,
                    "vocab_size": VOCABULARY,
                    "rms_norm_eps": 1e-6,
                    "rope_parameters": {"rope_theta": 1000000.0},
                },
            }
        )
    )
    generator = random.Random(7)
    tensors: dict[str, tuple[list[int], list[float]]] = {}

    def normal(name: str, *shape: int) -> None:
        tensors[name] = (
            list(shape),
            [generator.gauss(0, 0.05) for _ in range(math.prod(shape))],
        )

    def constant(name: str, value: float, *shape: int) -> None:
        tensors[name] = (list(shape), [value] * math.prod(shape))

    attention_width = HEADS * DRAFT_HEAD_DIM
    normal("fc.weight", hidden_size, hidden_size)
    constant("hidden_norm.weight", 1, hidden_size)
    layer = "layers.0"
    constant(f"{layer}.input_layernorm.weight", 1, hidden_size)
    constant(f"{layer}.post_attention_layernorm.weight", 1, hidden_size)
    normal(f"{layer}.self_attn.q_proj.weight", attention_width, hidden_size)
    normal(f"{layer}.self_attn.k_proj.weight", attention_width, hidden_size)
    normal(f"{layer}.self_attn.v_proj.weight", attention_width, hidden_size)
    normal(f"{layer}.self_attn.o_proj.weight", hidden_size, attention_width)
    constant(f"{layer}.self_attn.q_norm.weight", 1, DRAFT_HEAD_DIM)
    constant(f"{layer}.self_attn.k_norm.weight", 1, DRAFT_HEAD_DIM)
    normal(f"{layer}.mlp.gate_proj.weight", MLP_HIDDEN, hidden_size)
    normal(f"{layer}.mlp.up_proj.weight", MLP_HIDDEN, hidden_size)
    normal(f"{layer}.mlp.down_proj.weight", hidden_size, MLP_HIDDEN)
    constant("norm.weight", 1, hidden_size)
    normal("lm_head.weight", 1, hidden_size)
    # d2t holds each draft id's offset to its target id: draft id 0 is id 0.
    constant("d2t", 0, 1)
    write_safetensors(directory / "model.safetensors", tensors)


def loop_embeddings() -> list[list[float]]:
    # Every row the same vector, so the argmax is the lowest id; <|im_end|>'s
    # row is its strong negative, so a decode never stops early.
    row = [0.25] * HIDDEN
    embeddings = [row.copy() for _ in range(VOCABULARY)]
    embeddings[IM_END_ID] = [-4 * value for value in row]
    return embeddings


def stop_embeddings() -> list[list[float]]:
    # A chain: after the prompt's last token <|assistant|> (259) the argmax is
    # 5, after 5 it is 7, after 7 it is <|im_end|>. Each row's direction
    # differs from the previous, so RmsNorm(last) . table peaks at the next.
    embeddings = [[0.0] * HIDDEN for _ in range(VOCABULARY)]
    for token_id, row in (
        (ASSISTANT_ID, [1, 0, 0, 0]),
        (ACCENTED_E_ID, [2, 10, 0, 0]),
        (PLAIN_X_ID, [1, 22, 10, 0]),
        (IM_END_ID, [1, 20, 20, 0]),
    ):
        embeddings[token_id][: len(row)] = [float(value) for value in row]
    return embeddings


def test_arkasr_budget_windowing_and_join(
    tmp_path: Path, arkasr_probe_binary: Path
) -> None:
    loop_checkpoint = tmp_path / "arkasr-loop"
    write_checkpoint(loop_checkpoint, loop_embeddings(), {})
    stop_checkpoint = tmp_path / "arkasr-stop"
    write_checkpoint(
        stop_checkpoint,
        stop_embeddings(),
        {"Ã©": ACCENTED_E_ID, "x": PLAIN_X_ID},
    )
    draft_checkpoint = tmp_path / "arkasr-draft"
    write_draft(draft_checkpoint)
    completed = subprocess.run(
        [
            str(arkasr_probe_binary),
            str(loop_checkpoint),
            str(stop_checkpoint),
            str(draft_checkpoint),
        ],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    output = json.loads(completed.stdout)
    by_budget = {(row["checkpoint"], row["budget"]): row for row in output["plain"]}
    assert by_budget[("loop", 2)]["tokens"] == 2
    assert by_budget[("loop", 2)]["windows"] == 1
    assert by_budget[("loop", None)]["tokens"] == 3 * 256
    assert by_budget[("loop", None)]["windows"] == 1
    assert by_budget[("loop", 0)]["tokens"] == 3 * 256
    assert by_budget[("loop", 0)]["windows"] == 1
    assert by_budget[("stop", None)]["tokens"] == 3 * 2
    assert by_budget[("stop", None)]["windows"] == 3
    assert by_budget[("stop", None)]["text"] == "éx éx éx"
    assert by_budget[("stop", 5)]["tokens"] == 5
    assert by_budget[("stop", 5)]["windows"] == 3
    # Equality with plain decoding is checked in the probe; here, that the
    # draft really proposed: the loop accepts every proposal and the stop
    # none. Budgets 1 and 2 end before a proposal could commit.
    for row in output["drafted"]:
        if row["checkpoint"] == "stop":
            assert row["accepted"] == 0 < row["proposed"]
        elif row["budget"] not in (1, 2):
            assert row["accepted"] == row["proposed"] > 0


def test_arkasr_refuses_a_draft_for_another_target(
    tmp_path: Path, runtime_bin: Path
) -> None:
    checkpoint = tmp_path / "arkasr-stop"
    write_checkpoint(
        checkpoint, stop_embeddings(), {"Ã©": ACCENTED_E_ID, "x": PLAIN_X_ID}
    )
    draft = tmp_path / "wider-draft"
    write_draft(draft, hidden_size=2 * HIDDEN)
    server = SupervisedServer(
        runtime_bin / "arkasr_server",
        "arkasr",
        checkpoint,
        ("--draft-model-path", str(draft)),
    )
    assert server.first_event["event"] == "failed"
    assert (
        f"has hidden size {2 * HIDDEN}, the target's is {HIDDEN}"
        in server.first_event["reason"]
    )
    assert server.process.wait(timeout=EVENT_TIMEOUT_SECONDS) == 1
