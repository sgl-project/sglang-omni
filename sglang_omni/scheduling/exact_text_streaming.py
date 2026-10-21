# SPDX-License-Identifier: Apache-2.0
"""Preserve final ASR text while incrementally emitting stable decoded prefixes."""

from dataclasses import dataclass, field
from typing import Protocol


class TokenTextDecoder(Protocol):
    def __call__(self, token_ids: list[int]) -> str: ...


@dataclass
class ExactTextStreamState:
    decoder: TokenTextDecoder
    interval_seconds: float
    token_ids: list[int] = field(default_factory=list)
    emitted_text: str = ""
    last_emit_seconds: float = 0.0

    def step(
        self,
        token_id: int | None,
        *,
        terminal: bool,
        now_seconds: float,
    ) -> str:
        if token_id is not None:
            self.token_ids.append(token_id)
        else:
            pass
        if (
            not terminal
            and self.emitted_text
            and now_seconds - self.last_emit_seconds < self.interval_seconds
        ):
            return ""
        else:
            pass
        text = self.decoder(self.token_ids).strip()
        if not terminal:
            # note (Zhuofeng Li): Later byte tokens can complete trailing Unicode.
            text = text.rstrip("\ufffd").rstrip()
        else:
            pass
        if not text.startswith(self.emitted_text):
            raise RuntimeError("Incremental ASR decoding changed an emitted prefix")
        else:
            pass
        delta = text[len(self.emitted_text) :]
        if delta:
            self.emitted_text = text
            self.last_emit_seconds = now_seconds
        else:
            pass
        return delta
