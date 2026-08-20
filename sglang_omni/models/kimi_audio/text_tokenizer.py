# SPDX-License-Identifier: Apache-2.0
"""Minimal, version-independent Kimi text tokenizer."""

from __future__ import annotations

import json
from collections.abc import Collection, Iterator, Sequence
from collections.abc import Set as AbstractSet
from pathlib import Path
from typing import Literal

import tiktoken
from tiktoken.load import load_tiktoken_bpe

_MAX_ENCODE_CHARS = 400_000
_MAX_CONSECUTIVE_CHARS = 25_000


class KimiTextTokenizer:
    """Tiktoken wrapper exposing the subset used by Kimi serving."""

    def __init__(self, checkpoint_dir: str) -> None:
        root = Path(checkpoint_dir)
        with (root / "tokenizer_config.json").open(encoding="utf-8") as handle:
            config = json.load(handle)

        decoder = config.get("added_tokens_decoder")
        if not isinstance(decoder, dict) or not decoder:
            raise ValueError("Kimi-Audio tokenizer_config.json has no special tokens")
        self.special_tokens = {
            str(value["content"]): int(token_id) for token_id, value in decoder.items()
        }
        ranks = load_tiktoken_bpe(str(root / "tiktoken.model"))
        self.model = tiktoken.Encoding(
            name=f"kimi_audio_{root.name}",
            pat_str=str(config["pat_str"]),
            mergeable_ranks=ranks,
            special_tokens=self.special_tokens,
            explicit_n_vocab=len(ranks) + len(self.special_tokens),
        )
        self._special_token_ids = set(self.special_tokens.values())
        self.bos_token_id = self._required_special(str(config["bos_token"]))
        self.base_eos_token_id = self._required_special(str(config["eos_token"]))
        # Kimi generation uses <|im_kimia_text_eos|>, not the base tokenizer
        # EOS. Declaring the latter to SGLang would terminate on a normal Kimi
        # decode token before the model emits its stream-specific EOS.
        self.eos_token_id: None = None
        self.pad_token_id = self._required_special(str(config["pad_token"]))
        self.unk_token_id = self._required_special(str(config["unk_token"]))
        # SGLang probes this attribute while updating finish state. Kimi's
        # text EOS is supplied explicitly in each request's stop_token_ids.
        self.additional_stop_token_ids: set[int] | None = None

    @property
    def vocab_size(self) -> int:
        return self.model.n_vocab

    def __len__(self) -> int:
        return self.vocab_size

    def _required_special(self, token: str) -> int:
        try:
            return self.special_tokens[token]
        except KeyError as exc:
            raise ValueError(f"Kimi-Audio tokenizer is missing {token}") from exc

    def convert_tokens_to_ids(self, token: str) -> int:
        return self.special_tokens.get(token, self.unk_token_id)

    def encode(
        self,
        text: str,
        *,
        bos: bool = False,
        eos: bool = False,
        add_special_tokens: bool | None = None,
        allowed_special: Literal["all"] | AbstractSet[str] = frozenset(),
        disallowed_special: Literal["all"] | Collection[str] = (),
    ) -> list[int]:
        if not isinstance(text, str):
            raise TypeError("Kimi-Audio tokenizer input must be a string")
        if add_special_tokens:
            bos = True
            eos = True
        token_ids: list[int] = []
        for start in range(0, len(text), _MAX_ENCODE_CHARS):
            chunk = text[start : start + _MAX_ENCODE_CHARS]
            for substring in self._split_long_runs(chunk):
                token_ids.extend(
                    self.model.encode(
                        substring,
                        allowed_special=allowed_special,
                        disallowed_special=disallowed_special,
                    )
                )
        if bos:
            token_ids.insert(0, self.bos_token_id)
        if eos:
            token_ids.append(self.base_eos_token_id)
        return token_ids

    def decode(
        self,
        token_ids: Sequence[int],
        *,
        skip_special_tokens: bool = False,
        **_: object,
    ) -> str:
        ids = [int(token_id) for token_id in token_ids]
        if skip_special_tokens:
            ids = [
                token_id for token_id in ids if token_id not in self._special_token_ids
            ]
        return self.model.decode(ids)

    @staticmethod
    def _split_long_runs(text: str) -> Iterator[str]:
        if not text:
            yield text
            return
        run_length = 0
        run_is_space = text[0].isspace()
        slice_start = 0
        for index, character in enumerate(text):
            is_space = character.isspace()
            if run_is_space != is_space:
                run_length = 1
                run_is_space = is_space
                continue
            run_length += 1
            if run_length > _MAX_CONSECUTIVE_CHARS:
                yield text[slice_start:index]
                slice_start = index
                run_length = 1
        yield text[slice_start:]


__all__ = ["KimiTextTokenizer"]
