# SPDX-License-Identifier: Apache-2.0
"""Choose a Mandarin opening word from the D2 Thinker's legal lexical tokens."""

import unicodedata

import torch
from tokenizers import Tokenizer


def is_han(character: str) -> bool:
    return character == "〇" or unicodedata.name(character, "").startswith(
        ("CJK UNIFIED IDEOGRAPH", "CJK COMPATIBILITY IDEOGRAPH")
    )


class MandarinOpening:
    def __init__(
        self,
        tokenizer: Tokenizer,
        excluded_ids: set[int],
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
