# SPDX-License-Identifier: Apache-2.0
"""ARK-ASR's request budget, windowing and text joining on small checkpoints.

Both checkpoints have all-zero decoder layers, so every position keeps its
embedding and the next token is the argmax of RmsNorm(last embedding) against
the table. The loop checkpoint gives every row the same vector except a
strongly negative <|im_end|> row, so a decode never stops early and runs to
its token limit. The stop checkpoint's rows form the chain 259 -> 5 -> 7 ->
<|im_end|>, so every window decodes exactly "éx" and stops; the multi-window
requests then show how the budget is spent and how the window texts join.
"""

from __future__ import annotations

import json
import math
import random
import struct
import subprocess
from pathlib import Path

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
    completed = subprocess.run(
        [str(arkasr_probe_binary), str(loop_checkpoint), str(stop_checkpoint)],
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    rows = json.loads(completed.stdout)
    by_budget = {(row["checkpoint"], row["budget"]): row for row in rows}
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
