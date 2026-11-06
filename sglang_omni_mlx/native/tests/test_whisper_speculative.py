# SPDX-License-Identifier: Apache-2.0
"""Greedy speculation must preserve output across rejection, EOS and budgets."""

import json
import os
import subprocess
import wave
from pathlib import Path

import mlx.core as mx
import numpy as np
import pytest


@pytest.fixture
def checkpoints(tmp_path: Path) -> tuple[Path, Path, Path]:
    generator = np.random.default_rng(42)
    width, vocabulary = 8, 32
    config = dict(
        n_mels=80,
        n_audio_ctx=1500,
        n_audio_state=width,
        n_audio_head=2,
        n_audio_layer=1,
        n_vocab=vocabulary,
        n_text_ctx=32,
        n_text_head=2,
        n_text_layer=1,
    )
    weights: dict[str, mx.array] = {}
    for side in ("encoder", "decoder"):
        for attention in ("attn",) if side == "encoder" else ("attn", "cross_attn"):
            for projection in ("query", "key", "value", "out"):
                prefix = f"{side}.blocks.0.{attention}.{projection}"
                weights[f"{prefix}.weight"] = mx.array(
                    generator.normal(0, 0.15, (width, width)).astype(np.float32)
                )
                if projection != "key":
                    weights[f"{prefix}.bias"] = mx.zeros((width,))
                else:
                    pass
        norms = (
            ("attn_ln", "mlp_ln")
            if side == "encoder"
            else ("attn_ln", "mlp_ln", "cross_attn_ln")
        )
        for norm in norms:
            weights[f"{side}.blocks.0.{norm}.weight"] = mx.ones((width,))
            weights[f"{side}.blocks.0.{norm}.bias"] = mx.zeros((width,))
        for projection, shape in (
            ("mlp1", (4 * width, width)),
            ("mlp2", (width, 4 * width)),
        ):
            weights[f"{side}.blocks.0.{projection}.weight"] = mx.array(
                generator.normal(0, 0.15, shape).astype(np.float32)
            )
            weights[f"{side}.blocks.0.{projection}.bias"] = mx.zeros((shape[0],))
        final_norm = "encoder.ln_post" if side == "encoder" else "decoder.ln"
        weights[f"{final_norm}.weight"] = mx.ones((width,))
        weights[f"{final_norm}.bias"] = mx.zeros((width,))
    for convolution, inputs in (("conv1", 80), ("conv2", width)):
        weights[f"encoder.{convolution}.weight"] = mx.array(
            generator.normal(0, 0.05, (width, 3, inputs)).astype(np.float32)
        )
        weights[f"encoder.{convolution}.bias"] = mx.zeros((width,))
    weights["decoder.token_embedding.weight"] = mx.array(
        generator.normal(0, 0.4, (vocabulary, width)).astype(np.float32)
    )
    weights["decoder.positional_embedding"] = mx.array(
        generator.normal(0, 0.5, (32, width)).astype(np.float32)
    )
    paths = [tmp_path / name for name in ("target", "identical", "different")]
    for path in paths:
        path.mkdir()
        (path / "config.json").write_text(json.dumps(config))
        (path / "vocab.json").write_text(
            json.dumps({chr(65 + i): i for i in range(vocabulary)})
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
        if path.name == "different":
            weights["decoder.token_embedding.weight"] = -weights[
                "decoder.token_embedding.weight"
            ]
        else:
            pass
        mx.save_safetensors(str(path / "model.safetensors"), weights)
    return tuple(paths)


@pytest.mark.parametrize("budget", [1, 2, 7, 23])
def test_speculative_matches_greedy(
    checkpoints: tuple[Path, Path, Path], tmp_path: Path, budget: int
) -> None:
    target, identical, different = checkpoints
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(3200))
    command = [
        str(Path(os.environ["NATIVE_RUNTIME_BIN"]) / "whisper_transcribe"),
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


def test_incompatible_encoder_fails_before_decode(
    checkpoints: tuple[Path, Path, Path]
) -> None:
    target, _, draft = checkpoints
    weights = mx.load(str(draft / "model.safetensors"))
    weights["encoder.conv1.weight"] = weights["encoder.conv1.weight"] + 1
    mx.save_safetensors(str(draft / "model.safetensors"), weights)
    binary = Path(os.environ["NATIVE_RUNTIME_BIN"]) / "whisper_transcribe"
    result = subprocess.run(
        [str(binary), "--model-path", str(target), "--draft-model-path", str(draft)],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "draft encoder differs" in result.stderr


def test_immediate_eos(checkpoints: tuple[Path, Path, Path], tmp_path: Path) -> None:
    target, draft, _ = checkpoints
    audio = tmp_path / "silence.wav"
    with wave.open(str(audio), "wb") as output:
        output.setparams((1, 2, 16000, 0, "NONE", "not compressed"))
        output.writeframes(bytes(3200))
    binary = Path(os.environ["NATIVE_RUNTIME_BIN"]) / "whisper_transcribe"
    command = [str(binary), "--model-path", str(target)]
    first = json.loads(
        subprocess.check_output(
            [*command, "--max-new-tokens", "1", str(audio)], text=True
        )
    )
    assert len(first["text"]) == 1
    generation = json.loads((target / "generation_config.json").read_text())
    generation["eos_token_id"] = ord(first["text"]) - 65
    generation["begin_suppress_tokens"] = []
    (target / "generation_config.json").write_text(json.dumps(generation))
    for flags in ([], ["--draft-model-path", str(draft)]):
        row = json.loads(
            subprocess.check_output([*command, *flags, str(audio)], text=True)
        )
        assert row["text"] == ""
        assert row["generated_token_count"] == 0
        assert row["finish_reason"] == "stop"
