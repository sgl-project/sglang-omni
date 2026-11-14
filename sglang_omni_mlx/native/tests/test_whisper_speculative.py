# SPDX-License-Identifier: Apache-2.0
"""Greedy speculation must preserve output across rejection, EOS, budgets and a detected language."""

from __future__ import annotations

import json
import math
import random
import struct
import subprocess
from pathlib import Path
from typing import NamedTuple

import pytest
from native_server import silent_wav

WIDTH = 8
VOCABULARY_SIZE = 32
TEXT_CONTEXT_LENGTH = 32
MEL_BIN_COUNT = 80
# The tiny checkpoints' vocab.json spells token i as chr(FIRST_TOKEN_CODE + i).
FIRST_TOKEN_CODE = 65


class Tensor(NamedTuple):
    shape: tuple[int, ...]
    values: list[float]


def write_safetensors(path: Path, tensors: dict[str, Tensor]) -> None:
    """Float32 tensors in the safetensors layout: header size, header JSON, data."""
    header: dict[str, dict[str, str | list[int]]] = {}
    data = bytearray()
    for name, tensor in tensors.items():
        start = len(data)
        data += struct.pack(f"<{len(tensor.values)}f", *tensor.values)
        header[name] = {
            "dtype": "F32",
            "shape": list(tensor.shape),
            "data_offsets": [start, len(data)],
        }
    encoded_header = json.dumps(header).encode()
    path.write_bytes(struct.pack("<Q", len(encoded_header)) + encoded_header + data)


@pytest.fixture
def checkpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    """A one-layer target, a draft equal to it, and a decoder-only draft unlike it."""
    generator = random.Random(42)

    def normal(standard_deviation: float, shape: tuple[int, ...]) -> Tensor:
        return Tensor(
            shape,
            [generator.gauss(0.0, standard_deviation) for _ in range(math.prod(shape))],
        )

    def constant(value: float, shape: tuple[int, ...]) -> Tensor:
        return Tensor(shape, [value] * math.prod(shape))

    config = {
        "n_mels": MEL_BIN_COUNT,
        "n_audio_ctx": 1500,
        "n_audio_state": WIDTH,
        "n_audio_head": 2,
        "n_audio_layer": 1,
        "n_vocab": VOCABULARY_SIZE,
        "n_text_ctx": TEXT_CONTEXT_LENGTH,
        "n_text_head": 2,
        "n_text_layer": 1,
    }
    weights: dict[str, Tensor] = {}
    for side in ("encoder", "decoder"):
        for attention in ("attn",) if side == "encoder" else ("attn", "cross_attn"):
            for projection in ("query", "key", "value", "out"):
                prefix = f"{side}.blocks.0.{attention}.{projection}"
                weights[f"{prefix}.weight"] = normal(0.15, (WIDTH, WIDTH))
                if projection != "key":
                    weights[f"{prefix}.bias"] = constant(0.0, (WIDTH,))
                else:
                    pass
        norms = (
            ("attn_ln", "mlp_ln")
            if side == "encoder"
            else ("attn_ln", "mlp_ln", "cross_attn_ln")
        )
        for norm in norms:
            weights[f"{side}.blocks.0.{norm}.weight"] = constant(1.0, (WIDTH,))
            weights[f"{side}.blocks.0.{norm}.bias"] = constant(0.0, (WIDTH,))
        for projection, shape in (
            ("mlp1", (4 * WIDTH, WIDTH)),
            ("mlp2", (WIDTH, 4 * WIDTH)),
        ):
            weights[f"{side}.blocks.0.{projection}.weight"] = normal(0.15, shape)
            weights[f"{side}.blocks.0.{projection}.bias"] = constant(0.0, (shape[0],))
        final_norm = "encoder.ln_post" if side == "encoder" else "decoder.ln"
        weights[f"{final_norm}.weight"] = constant(1.0, (WIDTH,))
        weights[f"{final_norm}.bias"] = constant(0.0, (WIDTH,))
    for convolution, input_channels in (("conv1", MEL_BIN_COUNT), ("conv2", WIDTH)):
        weights[f"encoder.{convolution}.weight"] = normal(
            0.05, (WIDTH, 3, input_channels)
        )
        weights[f"encoder.{convolution}.bias"] = constant(0.0, (WIDTH,))
    weights["decoder.token_embedding.weight"] = normal(0.4, (VOCABULARY_SIZE, WIDTH))
    weights["decoder.positional_embedding"] = normal(0.5, (TEXT_CONTEXT_LENGTH, WIDTH))
    different_weights = {
        name: tensor for name, tensor in weights.items() if name.startswith("decoder.")
    }
    embedding = weights["decoder.token_embedding.weight"]
    different_weights["decoder.token_embedding.weight"] = Tensor(
        embedding.shape, [-value for value in embedding.values]
    )
    paths = [tmp_path / name for name in ("target", "identical", "different")]
    for path in paths:
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
        (path / "vocab.json").write_text(
            json.dumps({chr(FIRST_TOKEN_CODE + i): i for i in range(VOCABULARY_SIZE)})
        )
        (path / "merges.txt").write_text("#version: 0.2\n")
        (path / "tokenizer_config.json").write_text(
            json.dumps({"added_tokens_decoder": {}})
        )
        (path / "generation_config.json").write_text(
            json.dumps(
                {
                    "decoder_start_token_id": 29,
                    "eos_token_id": 30,
                    "no_timestamps_token_id": 31,
                    "suppress_tokens": [29, 31],
                    "begin_suppress_tokens": [30],
                }
            )
        )
        write_safetensors(
            path / "model.safetensors",
            different_weights if path.name == "different" else weights,
        )
    return paths[0], paths[1], paths[2]


@pytest.fixture
def audio(tmp_path: Path) -> Path:
    path = tmp_path / "silence.wav"
    path.write_bytes(silent_wav(1600))
    return path


@pytest.mark.parametrize("budget", [1, 2, 7, 23])
def test_speculative_matches_greedy(
    runtime_bin: Path,
    checkpoints: tuple[Path, Path, Path],
    audio: Path,
    budget: int,
) -> None:
    """Drafted decoding keeps plain greedy output whether proposals are accepted or rejected, up to the budget."""
    target, identical, different = checkpoints
    command = [
        str(runtime_bin / "whisper_transcribe"),
        "--model-path",
        str(target),
        "--max-new-tokens",
        str(budget),
    ]
    plain = json.loads(subprocess.check_output([*command, str(audio)], text=True))
    for draft in (identical, different):
        rows = subprocess.check_output(
            [
                *command,
                "--draft-model-path",
                str(draft),
                "--draft-steps",
                "3",
                str(audio),
                str(audio),
            ],
            text=True,
        )
        for line in rows.splitlines():
            actual = json.loads(line)
            for field in ("text", "generated_token_count", "finish_reason"):
                assert actual[field] == plain[field]
            assert (
                actual["accepted_draft_token_count"]
                <= actual["proposed_draft_token_count"]
            )
            if budget > 2 and draft == identical:
                assert actual["accepted_draft_token_count"] > 0
            elif budget > 2:
                assert (
                    actual["accepted_draft_token_count"]
                    < actual["proposed_draft_token_count"]
                )
            else:
                pass


def test_a_draft_with_fewer_text_positions_fails_at_load(
    runtime_bin: Path, checkpoints: tuple[Path, Path, Path]
) -> None:
    """A draft that cannot place every position the target decodes is refused at load."""
    target, _, draft = checkpoints
    config = json.loads((draft / "config.json").read_text())
    config["n_text_ctx"] = 16
    (draft / "config.json").write_text(json.dumps(config))
    result = subprocess.run(
        [
            str(runtime_bin / "whisper_transcribe"),
            "--model-path",
            str(target),
            "--draft-model-path",
            str(draft),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert f"fewer than the target's {TEXT_CONTEXT_LENGTH}" in result.stderr


def test_immediate_eos(
    runtime_bin: Path, checkpoints: tuple[Path, Path, Path], audio: Path
) -> None:
    """An end of text as the first token gives empty output, with or without the draft."""
    target, draft, _ = checkpoints
    command = [str(runtime_bin / "whisper_transcribe"), "--model-path", str(target)]
    first = json.loads(
        subprocess.check_output(
            [*command, "--max-new-tokens", "1", str(audio)], text=True
        )
    )
    assert len(first["text"]) == 1
    generation = json.loads((target / "generation_config.json").read_text())
    generation["eos_token_id"] = ord(first["text"]) - FIRST_TOKEN_CODE
    generation["begin_suppress_tokens"] = []
    (target / "generation_config.json").write_text(json.dumps(generation))
    for flags in ([], ["--draft-model-path", str(draft)]):
        row = json.loads(
            subprocess.check_output([*command, *flags, str(audio)], text=True)
        )
        assert row["text"] == ""
        assert row["generated_token_count"] == 0
        assert row["finish_reason"] == "stop"


def test_an_omitted_language_is_detected(
    runtime_bin: Path, checkpoints: tuple[Path, Path, Path], audio: Path
) -> None:
    """A missing or unknown language decodes as the detected one does when requested, with or without the draft."""
    target, draft, _ = checkpoints
    generation = json.loads((target / "generation_config.json").read_text())
    generation["lang_to_id"] = {"<|en|>": 26, "<|zh|>": 27}
    generation["task_to_id"] = {"transcribe": 28}
    generation["suppress_tokens"] = [26, 27, 28, 29, 31]
    (target / "generation_config.json").write_text(json.dumps(generation))
    command = [
        str(runtime_bin / "whisper_transcribe"),
        "--model-path",
        str(target),
        "--max-new-tokens",
        "7",
    ]

    def transcribe(flags: list[str]) -> dict[str, str | int]:
        row = json.loads(
            subprocess.check_output([*command, *flags, str(audio)], text=True)
        )
        return {
            field: row[field]
            for field in ("text", "language", "generated_token_count", "finish_reason")
        }

    detected = transcribe([])
    assert detected["language"] in ("en", "zh")
    other = "zh" if detected["language"] == "en" else "en"
    assert transcribe(["--language", other])["language"] == other
    for draft_flags in ([], ["--draft-model-path", str(draft)]):
        for language_flags in (
            [],
            ["--language", "unknown"],
            ["--language", str(detected["language"])],
        ):
            assert transcribe([*draft_flags, *language_flags]) == detected
