"""Numerical recurrence and tokenizer contracts used by Ming native inference."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest
from safetensors.numpy import load_file, save_file
from server_test_utils import DATA_ROOT, RUNTIME_BIN, Server, clip, sse_events
from tokenizers import Tokenizer

pytestmark = pytest.mark.skipif(
    not RUNTIME_BIN or not DATA_ROOT, reason="set NATIVE_RUNTIME_BIN and CI_DATA_ROOT"
)


@pytest.mark.parametrize(
    "eos_source",
    [
        "checkpoint_int",
        "checkpoint_list",
        "role_end",
        "tokenizer_string",
        "tokenizer_object",
    ],
)
@pytest.mark.parametrize("alternate_audio_tokens", [False, True])
def test_ming_checkpoint_eos_stops_before_emitting_token(
    tmp_path: Path, eos_source: str, alternate_audio_tokens: bool
) -> None:
    """Checkpoint and role-marker EOS must stop before emitting the predicted token."""
    model = Path(DATA_ROOT) / "models" / "mlx-community_Ming-3.0-asr-tiny-4bit"
    for checkpoint_file in model.iterdir():
        if checkpoint_file.is_file() and checkpoint_file.name not in (
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
        ):
            (tmp_path / checkpoint_file.name).symlink_to(checkpoint_file)
        else:
            pass
    tokenizer = Tokenizer.from_file(str(model / "tokenizer.json"))
    eos_token_id = tokenizer.encode("surely", add_special_tokens=False).ids[0]
    tokenizer_json = json.loads((model / "tokenizer.json").read_text())
    if alternate_audio_tokens:
        replacements = {
            "<audio>": "<|audio_start|>",
            "<audioPatch>": "<|audio_pad|>",
            "</audio>": "<|audio_end|>",
        }
        for added_token in tokenizer_json["added_tokens"]:
            added_token["content"] = replacements.get(
                added_token["content"], added_token["content"]
            )
    else:
        pass
    if eos_source == "role_end":
        for added_token in tokenizer_json["added_tokens"]:
            if added_token["content"] == "<|role_end|>":
                added_token["id"] = eos_token_id
            else:
                pass
    else:
        pass
    (tmp_path / "tokenizer.json").write_text(json.dumps(tokenizer_json))
    config = json.loads((model / "config.json").read_text())
    if eos_source == "checkpoint_int":
        config["text_config"]["eos_token_id"] = eos_token_id
    elif eos_source == "checkpoint_list":
        config["text_config"]["eos_token_id"] = [
            config["text_config"]["eos_token_id"],
            eos_token_id,
        ]
    else:
        pass
    (tmp_path / "config.json").write_text(json.dumps(config))
    tokenizer_config = json.loads((model / "tokenizer_config.json").read_text())
    if eos_source.startswith("tokenizer_"):
        eos_token = tokenizer.id_to_token(eos_token_id)
        tokenizer_config["eos_token"] = (
            eos_token if eos_source == "tokenizer_string" else {"content": eos_token}
        )
    else:
        pass
    (tmp_path / "tokenizer_config.json").write_text(json.dumps(tokenizer_config))
    running = Server("ming3_asr_server", "ming3_asr", tmp_path)
    try:
        assert running.ready["event"] == "ready", running.ready
        status, body = running.post_form(
            {"language": "en", "stream": "true", "include_generation_metadata": "true"},
            clip("0006_en_short"),
        )
        assert status == 200
        event = sse_events(body)[0]
        assert event["text"] == ""
        assert event["generation_metadata"]["generated_token_count"] == 0
        assert event["generation_metadata"]["finish_reason"] == "stop"
    finally:
        running.stop()


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("text_config", "eos_token_id", 4295124188),
        ("text_config", "eos_token_id", [156892, 4295124188]),
        ("text_config", "eos_token_id", -4294810404),
        ("text_config", "eos_token_id", 18446744073709551615),
        ("text_config", "eos_token_id", [156892, True]),
        ("text_config", "eos_token_id", []),
        ("text_config", "first_k_dense_replace", 1.5),
        ("text_config", "first_k_dense_replace", True),
        ("text_config", "first_k_dense_replace", 4294967297),
        ("text_config", "hidden_size", 0),
        ("", "audio_token_id", 4295124464),
        ("", "audio_token_id", 157168.5),
        ("text_config", "rms_norm_eps", True),
        ("text_config", "rms_norm_eps", 0),
        ("text_config", "rope_theta", "10000"),
        ("text_config", "rope_theta", 1e100),
        ("text_config", "routed_scaling_factor", -1),
        ("text_config", "kda_lower_bound", 1),
        ("quantization", "bits", 4294967300),
        ("quantization", "group_size", 64.5),
    ],
)
def test_ming_rejects_invalid_config_numbers(
    tmp_path: Path, section: str, field: str, value: int | float | str | list[int]
) -> None:
    """Invalid numeric configuration must not truncate or coerce into a serving model."""
    model = Path(DATA_ROOT) / "models" / "mlx-community_Ming-3.0-asr-tiny-4bit"
    for checkpoint_file in model.iterdir():
        if checkpoint_file.is_file() and checkpoint_file.name != "config.json":
            (tmp_path / checkpoint_file.name).symlink_to(checkpoint_file)
        else:
            pass
    config = json.loads((model / "config.json").read_text())
    (config[section] if section else config)[field] = value
    (tmp_path / "config.json").write_text(json.dumps(config))
    running = Server("ming3_asr_server", "ming3_asr", tmp_path)
    try:
        assert running.ready["event"] == "failed"
        assert field in running.ready["reason"]
        assert running.process.wait(timeout=30) != 0
    finally:
        running.stop()


def test_ming_delta_matches_recurrence_and_chunked_state(tmp_path: Path) -> None:
    generator = np.random.default_rng(42)
    shape = (2, 9, 2, 128)
    queries = generator.normal(size=shape).astype(np.float32)
    keys = generator.normal(size=shape).astype(np.float32)
    queries /= np.linalg.norm(queries, axis=-1, keepdims=True) * np.sqrt(128)
    keys /= np.linalg.norm(keys, axis=-1, keepdims=True)
    tensors = {
        "queries": queries,
        "keys": keys,
        "values": generator.normal(size=shape).astype(np.float32),
        "decay": generator.uniform(0.1, 0.95, size=shape).astype(np.float32),
        "beta": generator.uniform(size=shape[:-1]).astype(np.float32),
        "state": generator.normal(size=(2, 2, 128, 128)).astype(np.float32),
    }
    state = tensors["state"].copy()
    expected = []
    for token in range(shape[1]):
        state *= tensors["decay"][:, token, :, None, :]
        key = keys[:, token, :, None, :]
        prediction = (state * key).sum(-1)
        delta = (tensors["values"][:, token] - prediction) * tensors["beta"][
            :, token, :, None
        ]
        state += delta[..., None] * key
        expected.append((state * queries[:, token, :, None, :]).sum(-1))
    save_file(tensors, tmp_path / "input.safetensors")
    subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
            "delta",
            str(tmp_path / "input.safetensors"),
            str(tmp_path / "output.safetensors"),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    actual = load_file(tmp_path / "output.safetensors")
    np.testing.assert_allclose(
        actual["output"], np.stack(expected, axis=1), atol=2e-6, rtol=2e-5
    )
    np.testing.assert_allclose(actual["state"], state, atol=2e-6, rtol=2e-5)
    carried = tensors["state"]
    outputs = []
    for start, end in ((0, 4), (4, 9)):
        chunk = {
            name: np.ascontiguousarray(value[:, start:end])
            for name, value in tensors.items()
            if name != "state"
        }
        chunk["state"] = carried
        save_file(chunk, tmp_path / "chunk.safetensors")
        subprocess.run(
            [
                str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
                "delta",
                str(tmp_path / "chunk.safetensors"),
                str(tmp_path / "chunk-output.safetensors"),
            ],
            check=True,
            capture_output=True,
            timeout=30,
        )
        result = load_file(tmp_path / "chunk-output.safetensors")
        carried = result["state"]
        outputs.append(result["output"])
    np.testing.assert_array_equal(np.concatenate(outputs, axis=1), actual["output"])
    np.testing.assert_array_equal(carried, actual["state"])


def test_ming_delta_rejects_head_width_not_supported_by_kernel(
    tmp_path: Path,
) -> None:
    shape = (1, 1, 1, 31)
    tensors = {
        "queries": np.zeros(shape, dtype=np.float32),
        "keys": np.zeros(shape, dtype=np.float32),
        "values": np.zeros(shape, dtype=np.float32),
        "decay": np.ones(shape, dtype=np.float32),
        "beta": np.ones(shape[:-1], dtype=np.float32),
        "state": np.zeros((1, 1, 31, 31), dtype=np.float32),
    }
    input_path = tmp_path / "input.safetensors"
    save_file(tensors, input_path)

    completed = subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
            "delta",
            str(input_path),
            str(tmp_path / "output.safetensors"),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert completed.returncode != 0
    assert "head_width divisible by 32" in completed.stderr


@pytest.mark.parametrize("layout", ["tokenizer_json", "vocab_merges"])
def test_byte_level_tokenizer_matches_checkpoint(tmp_path: Path, layout: str) -> None:
    directory = Path(DATA_ROOT) / "models" / "mlx-community_Ming-3.0-asr-tiny-4bit"
    tokenizer = Tokenizer.from_file(str(directory / "tokenizer.json"))
    if layout == "vocab_merges":
        checkpoint = json.loads((directory / "tokenizer.json").read_text())
        (tmp_path / "vocab.json").write_text(json.dumps(checkpoint["model"]["vocab"]))
        (tmp_path / "merges.txt").write_text(
            "#version: 0.2\n"
            + "\n".join(" ".join(pair) for pair in checkpoint["model"]["merges"])
            + "\n"
        )
        (tmp_path / "tokenizer_config.json").write_bytes(
            (directory / "tokenizer_config.json").read_bytes()
        )
        directory = tmp_path
    else:
        pass
    texts = [
        "Please recognize the language of this speech and transcribe it, placing a `\t` between the identified language and the transcribe.<audio>",
        "English\t你好世界 1234567890",
        "I'm WE'RE they've isn't 42.5!\n\n\t  ",
        "Cafe\u0301 naïve 中文🙂\r\n!  \n",
        "<audio><audioPatch><audioPatch></audio><|role_end|>",
        "",
    ]
    (tmp_path / "texts.json").write_text(json.dumps(texts))
    subprocess.run(
        [
            str(Path(RUNTIME_BIN) / "ming3_asr_probe"),
            "tokenizer",
            str(tmp_path / "texts.json"),
            str(tmp_path / "tokens.json"),
            str(directory),
        ],
        check=True,
        capture_output=True,
        timeout=30,
    )
    actual = json.loads((tmp_path / "tokens.json").read_text())
    for text, result in zip(texts, actual, strict=True):
        expected = tokenizer.encode(text, add_special_tokens=False).ids
        assert result["ids"] == expected
        assert result["decoded"] == tokenizer.decode(
            expected, skip_special_tokens=False
        )
