# SPDX-License-Identifier: Apache-2.0
"""Mandarin replies: a Mandarin system prompt and a Han opening word in place of a Latin one."""

import unicodedata

import torch
from tokenizers import Tokenizer

MANDARIN_PROMPT = "你是一位普通话语音助手。即使问题很短，也用有主语和谓语的完整中文句子，让听众单凭声音就能听懂。在回答中自然提到问题的主题，不只说孤立的一个字、字母或数字。说一次，答完就安静等待。"


def is_han(character: str) -> bool:
    return character == "〇" or unicodedata.name(character, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
    )


class MandarinOpening:
    def __init__(
        self,
        tokenizer: Tokenizer,
        excluded_ids: frozenset[int],
        response_id: int,
        device: torch.device,
    ) -> None:
        self.response_id = response_id
        self.neutral_ids: set[int] = set()
        self.latin_ids: set[int] = set()
        han_ids: list[int] = []
        size = tokenizer.get_vocab_size(with_added_tokens=True)
        for start in range(0, size, 1024):
            ids = range(start, min(start + 1024, size))
            pieces = tokenizer.decode_batch(
                [[i] for i in ids], skip_special_tokens=False
            )
            for token_id, piece in zip(ids, pieces, strict=True):
                if token_id in excluded_ids:
                    continue
                else:
                    letters = [character for character in piece if character.isalpha()]
                if not letters or "\ufffd" in piece:
                    # Note (cuzimi): A byte-level BPE fragment can precede the first complete word.
                    self.neutral_ids.add(token_id)
                elif all(unicodedata.name(c, "").startswith("LATIN ") for c in letters):
                    self.latin_ids.add(token_id)
                elif len(letters) >= 2 and all(is_han(c) for c in letters):
                    if all(is_han(c) for c in piece.lstrip()[:2]):
                        han_ids.append(token_id)
                    else:
                        pass
                else:
                    pass
        if not han_ids:
            raise ValueError(
                "D2 tokenizer has no lexical tokens starting with two Han characters"
            )
        else:
            self.han_ids = torch.tensor(han_ids, device=device, dtype=torch.long)

    def mask(self, logits: torch.Tensor, generated_text: list[int]) -> torch.Tensor:
        """Steer only the first Latin word after RESPONSE, using the native greedy decision."""
        for token_id in reversed(generated_text):
            if token_id == self.response_id:
                break
            elif token_id not in self.neutral_ids:
                return logits
            else:
                pass
        else:
            return logits
        if int(logits.argmax().item()) not in self.latin_ids:
            return logits
        else:
            han_scores = logits.index_select(-1, self.han_ids)
        if torch.isfinite(han_scores).any().item():
            return torch.full_like(logits, -torch.inf).index_copy_(
                -1, self.han_ids, han_scores
            )
        else:
            return logits
