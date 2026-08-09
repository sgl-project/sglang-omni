# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import base64
import json

from sglang_omni.models.kimi_audio.text_tokenizer import KimiTextTokenizer


def _write_tokenizer_fixture(tmp_path) -> None:
    ranks = [
        f"{base64.b64encode(bytes([value])).decode('ascii')} {value}"
        for value in range(256)
    ]
    (tmp_path / "tiktoken.model").write_text("\n".join(ranks))
    specials = ["[BOS]", "[EOS]", "[PAD]", "[UNK]", "<|custom|>"]
    (tmp_path / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "pat_str": r"\p{L}+|\p{N}+|[^\s\p{L}\p{N}]+|\s+",
                "bos_token": "[BOS]",
                "eos_token": "[EOS]",
                "pad_token": "[PAD]",
                "unk_token": "[UNK]",
                "added_tokens_decoder": {
                    str(256 + index): {"content": token}
                    for index, token in enumerate(specials)
                },
            }
        )
    )


def test_kimi_text_tokenizer_round_trips_and_uses_checkpoint_special_ids(
    tmp_path,
) -> None:
    _write_tokenizer_fixture(tmp_path)
    tokenizer = KimiTextTokenizer(str(tmp_path))

    encoded = tokenizer.encode("hello", bos=True, eos=True)

    assert encoded[0] == 256
    assert encoded[-1] == 257
    assert tokenizer.decode(encoded[1:-1]) == "hello"
    assert tokenizer.convert_tokens_to_ids("<|custom|>") == 260
    assert tokenizer.convert_tokens_to_ids("missing") == tokenizer.unk_token_id
    assert len(tokenizer) == 261
    assert tokenizer.base_eos_token_id == 257
    assert tokenizer.eos_token_id is None
    assert tokenizer.additional_stop_token_ids is None


def test_kimi_text_tokenizer_can_skip_special_tokens(tmp_path) -> None:
    _write_tokenizer_fixture(tmp_path)
    tokenizer = KimiTextTokenizer(str(tmp_path))
    token_ids = tokenizer.encode("hi") + [tokenizer.base_eos_token_id]

    assert tokenizer.decode(token_ids, skip_special_tokens=True) == "hi"
