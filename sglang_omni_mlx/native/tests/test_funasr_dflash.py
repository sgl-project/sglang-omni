# SPDX-License-Identifier: Apache-2.0
"""DFlash decoding of small Fun-ASR-Nano checkpoints matches plain decoding.

The target's decoder layers are all zero, so each next token is the argmax of
RmsNorm(last embedding) against the tied table, and decoding follows the
prompt's last token, a newline, by its table rows alone: newline -> x -> x ...
in the loop checkpoint, newline -> x -> <|im_end|> in the stop checkpoint.
The draft's layers are zero too, so it proposes one token at every block
position: x, which the loop target accepts, or <|im_end|>, which it rejects
and the stop target accepts as its stop token. Attention adds nothing here,
so this guards Fun-ASR's draft wiring (budget, stop token, penalty dispatch,
draft counts), not the shared draft's cache trimming and context, which only
the real-model DFlash golden checks.
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

ENCODER_WIDTH = 8
HIDDEN = 32
HEADS = 2
HEAD_DIM = HIDDEN // HEADS
MLP_HIDDEN = 64
ADDED_TOKENS = ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "<|mask|>"]
VOCABULARY = 256 + len(ADDED_TOKENS)
NEWLINE_ID = 10
X_ID = ord("x")
IM_END_ID = 256 + ADDED_TOKENS.index("<|im_end|>")
MASK_ID = 256 + ADDED_TOKENS.index("<|mask|>")
BLOCK_SIZE = 8


class ProbeRow(TypedDict):
    budget: int
    text: str
    tokens: int
    finish_reason: str
    verify_rounds: int
    accepted_draft_tokens: int


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


def text_config(layer_count: int) -> dict[str, object]:
    return {
        "num_hidden_layers": layer_count,
        "num_attention_heads": HEADS,
        "num_key_value_heads": HEADS,
        "head_dim": HEAD_DIM,
        "hidden_size": HIDDEN,
        "intermediate_size": MLP_HIDDEN,
        "rms_norm_eps": 1e-6,
        "rope_parameters": {"rope_theta": 10000.0},
        "vocab_size": VOCABULARY,
    }


def zero_decoder_layers(
    tensors: dict[str, tuple[list[int], list[float]]], prefix: str, layer_count: int
) -> None:
    def constant(name: str, value: float, *shape: int) -> None:
        tensors[name] = (list(shape), [value] * math.prod(shape))

    for layer in range(layer_count):
        name = f"{prefix}layers.{layer}"
        for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
            constant(f"{name}.self_attn.{projection}.weight", 0, HIDDEN, HIDDEN)
        for norm in ("q_norm", "k_norm"):
            constant(f"{name}.self_attn.{norm}.weight", 1, HEAD_DIM)
        for projection in ("gate_proj", "up_proj"):
            constant(f"{name}.mlp.{projection}.weight", 0, MLP_HIDDEN, HIDDEN)
        constant(f"{name}.mlp.down_proj.weight", 0, HIDDEN, MLP_HIDDEN)
        for norm in ("input_layernorm", "post_attention_layernorm"):
            constant(f"{name}.{norm}.weight", 1, HIDDEN)
    constant(f"{prefix}norm.weight", 1, HIDDEN)


def row(*values: float) -> list[float]:
    return [float(value) for value in values] + [0.0] * (HIDDEN - len(values))


def write_target(directory: Path, im_end_row: list[float]) -> None:
    directory.mkdir()
    audio_layer_count = 3
    (directory / "config.json").write_text(
        json.dumps(
            {
                "model_type": "fun_asr_nano",
                "tie_word_embeddings": True,
                "audio_config": {
                    "num_mel_bins": 80,
                    "num_stacked_frames": 7,
                    "hidden_size": ENCODER_WIDTH,
                    "num_attention_heads": HEADS,
                    "num_hidden_layers": audio_layer_count,
                    "num_timestamp_prediction_layers": 1,
                    "fsmn_kernel_size": 3,
                    "hidden_act": "relu",
                    "layer_norm_eps": 1e-5,
                },
                "adaptor_config": {
                    "num_hidden_layers": 1,
                    "num_attention_heads": HEADS,
                    "layer_norm_eps": 1e-5,
                    "hidden_act": "relu",
                    "projector_hidden_act": "relu",
                },
                "text_config": text_config(2),
            }
        )
    )
    (directory / "processor_config.json").write_text(
        json.dumps(
            {
                "feature_extractor": {
                    "sampling_rate": 16000,
                    "feature_size": 80,
                    "frame_length": 25,
                    "frame_shift": 10,
                    "num_frames_lfr": 7,
                    "stride_lfr": 6,
                }
            }
        )
    )
    (directory / "vocab.json").write_text(json.dumps(byte_level_vocabulary()))
    (directory / "merges.txt").write_text("#version: 0.2\n")
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "added_tokens_decoder": {
                    str(256 + index): {"content": content, "special": True}
                    for index, content in enumerate(ADDED_TOKENS)
                }
            }
        )
    )

    generator = random.Random(42)
    tensors: dict[str, tuple[list[int], list[float]]] = {}

    def linear(name: str, out_width: int, in_width: int) -> None:
        tensors[f"{name}.weight"] = (
            [out_width, in_width],
            [generator.gauss(0, 0.05) for _ in range(out_width * in_width)],
        )
        tensors[f"{name}.bias"] = ([out_width], [0.0] * out_width)

    def layer_norm(name: str, width: int) -> None:
        tensors[f"{name}.weight"] = ([width], [1.0] * width)
        tensors[f"{name}.bias"] = ([width], [0.0] * width)

    for layer in range(audio_layer_count):
        name = f"model.audio_tower.layers.{layer}"
        in_width = 80 * 7 if layer == 0 else ENCODER_WIDTH
        layer_norm(f"{name}.input_layernorm", in_width)
        for projection in ("q_proj", "k_proj", "v_proj"):
            linear(f"{name}.self_attn.{projection}", ENCODER_WIDTH, in_width)
        linear(f"{name}.self_attn.o_proj", ENCODER_WIDTH, ENCODER_WIDTH)
        tensors[f"{name}.self_attn.fsmn.conv.weight"] = (
            [ENCODER_WIDTH, 1, 3],
            [generator.gauss(0, 0.05) for _ in range(ENCODER_WIDTH * 3)],
        )
        layer_norm(f"{name}.post_attention_layernorm", ENCODER_WIDTH)
        linear(f"{name}.mlp.fc1", 16, ENCODER_WIDTH)
        linear(f"{name}.mlp.fc2", ENCODER_WIDTH, 16)
    for layer in (audio_layer_count - 2, audio_layer_count - 1):
        layer_norm(f"model.audio_tower.layers.{layer}.final_layernorm", ENCODER_WIDTH)
    adaptor = "model.multi_modal_projector"
    linear(f"{adaptor}.linear_1", 16, ENCODER_WIDTH)
    linear(f"{adaptor}.linear_2", HIDDEN, 16)
    name = f"{adaptor}.layers.0"
    for norm in ("input_layernorm", "post_attention_layernorm"):
        layer_norm(f"{name}.{norm}", HIDDEN)
    for projection in ("q_proj", "k_proj", "v_proj", "o_proj"):
        linear(f"{name}.self_attn.{projection}", HIDDEN, HIDDEN)
    linear(f"{name}.mlp.fc1", 16, HIDDEN)
    linear(f"{name}.mlp.fc2", HIDDEN, 16)

    embeddings = [row() for _ in range(VOCABULARY)]
    embeddings[NEWLINE_ID] = row(1, 0)
    embeddings[X_ID] = row(2, 1)
    embeddings[IM_END_ID] = im_end_row
    embeddings[MASK_ID] = row(0, 0, 0, 1)
    tensors["model.language_model.embed_tokens.weight"] = (
        [VOCABULARY, HIDDEN],
        [value for embedding in embeddings for value in embedding],
    )
    zero_decoder_layers(tensors, "model.language_model.", 2)
    write_safetensors(directory / "model.safetensors", tensors)


def write_draft(directory: Path, proposed_id: int) -> None:
    directory.mkdir()
    (directory / "config.json").write_text(
        json.dumps(
            {
                "aux_hidden_state_layer_ids": [0, 1],
                "block_size": BLOCK_SIZE,
                "draft_vocab_size": VOCABULARY,
                "mask_token_id": MASK_ID,
                "sample_from_anchor": False,
                "sliding_window_non_causal": False,
                "transformer_layer_config": text_config(1),
            }
        )
    )
    tensors: dict[str, tuple[list[int], list[float]]] = {
        "fc.weight": ([HIDDEN, 2 * HIDDEN], [0.0] * (2 * HIDDEN * HIDDEN)),
        "hidden_norm.weight": ([HIDDEN], [1.0] * HIDDEN),
        "d2t": ([VOCABULARY], [0.0] * VOCABULARY),
    }
    # Every block position is the mask token's embedding, which only the
    # proposed token's row of the head scores above zero.
    head = [row() for _ in range(VOCABULARY)]
    head[proposed_id] = row(0, 0, 0, 1)
    tensors["lm_head.weight"] = (
        [VOCABULARY, HIDDEN],
        [value for head_row in head for value in head_row],
    )
    zero_decoder_layers(tensors, "", 1)
    write_safetensors(directory / "model.safetensors", tensors)


@pytest.fixture(scope="module")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("funasr-dflash")
    # After x, the loop target's <|im_end|> row scores far below x's own; the
    # stop target's scores above it, but below x after the newline.
    write_target(root / "loop", row(-4, -4))
    write_target(root / "stop", row(1.5, 4))
    write_draft(root / "draft-x", X_ID)
    write_draft(root / "draft-im-end", IM_END_ID)
    return root


def probe(binary: Path, target: Path, draft: Path) -> dict[int, ProbeRow]:
    completed = subprocess.run(
        [str(binary), str(target), str(draft)], capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    return {row["budget"]: row for row in json.loads(completed.stdout)}


def test_accepted_proposals_match_plain_decoding(
    funasr_dflash_probe_binary: Path, checkpoints: Path
) -> None:
    rows = probe(
        funasr_dflash_probe_binary, checkpoints / "loop", checkpoints / "draft-x"
    )
    for budget, result in rows.items():
        assert result["text"] == "x" * budget
        assert result["finish_reason"] == "length"
    # The first x comes from the prompt's pass; each round then commits the
    # accepted proposals and the target's own token, at most a block.
    assert rows[1]["verify_rounds"] == 0
    assert rows[9]["verify_rounds"] == 1
    assert rows[9]["accepted_draft_tokens"] == BLOCK_SIZE - 1
    assert rows[17]["verify_rounds"] == 2


def test_rejected_proposals_match_plain_decoding(
    funasr_dflash_probe_binary: Path, checkpoints: Path
) -> None:
    rows = probe(
        funasr_dflash_probe_binary, checkpoints / "loop", checkpoints / "draft-im-end"
    )
    assert rows[17]["text"] == "x" * 17
    assert rows[17]["verify_rounds"] == 16
    assert all(result["accepted_draft_tokens"] == 0 for result in rows.values())


@pytest.mark.parametrize(("draft", "accepted"), [("draft-x", 0), ("draft-im-end", 1)])
def test_the_stop_token_ends_a_drafted_request(
    funasr_dflash_probe_binary: Path, checkpoints: Path, draft: str, accepted: int
) -> None:
    rows = probe(funasr_dflash_probe_binary, checkpoints / "stop", checkpoints / draft)
    assert rows[1]["finish_reason"] == "length"
    for budget in (2, 3, 7, 8, 9, 17):
        assert rows[budget]["text"] == "x"
        assert rows[budget]["finish_reason"] == "stop"
        assert rows[budget]["verify_rounds"] == 1
        # A budget of 2 leaves the round no proposal to verify.
        assert rows[budget]["accepted_draft_tokens"] == (accepted if budget > 2 else 0)
