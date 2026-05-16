# SPDX-License-Identifier: Apache-2.0
"""Small FishAudio S2-Pro fakes for model-specific unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch

from sglang_omni.models.fishaudio_s2_pro.payload_types import S2ProState
from sglang_omni.proto import OmniRequest, StagePayload


def make_s2pro_state(**kwargs: Any) -> S2ProState:
    defaults: dict[str, Any] = {
        "input_ids": [10, 11, 12],
        "vq_mask_tokens": [False, True, False],
        "vq_parts": [torch.tensor([[1], [2]], dtype=torch.long)],
        "num_codebooks": 2,
        "codebook_size": 4096,
        "max_new_tokens": 4,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 20,
        "repetition_penalty": 1.05,
    }
    defaults.update(kwargs)
    return S2ProState(**defaults)


def make_s2pro_payload(
    state: S2ProState | None = None,
    *,
    request_id: str = "req-fish",
    inputs: Any = "hello",
    params: dict[str, Any] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs=inputs, params=params or {}),
        data=(state or make_s2pro_state()).to_dict(),
    )


class FakeFishTokenizer:
    vocab_size = 512

    def __init__(self) -> None:
        self.additional_stop_token_ids: list[int] = []
        self.encoded_texts: list[str] = []
        self._ids = {
            "<|im_end|>": 99,
            "<|semantic:0|>": 200,
            "<|semantic:4095|>": 295,
            "<|voice|>": 120,
            "<|im_start|>": 121,
        }

    def convert_tokens_to_ids(self, token):
        if isinstance(token, list):
            return [self.convert_tokens_to_ids(item) for item in token]
        if isinstance(token, str) and token.startswith("<|semantic:"):
            value = int(token.split(":", 1)[1].split("|>", 1)[0])
            return 200 + value
        return self._ids.get(token, 123)

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        self.encoded_texts.append(text)
        return (
            [self.convert_tokens_to_ids(text)]
            if text in self._ids
            else [1000 + ord(char) % 100 for char in text]
        )


class FakeFishCodec:
    sample_rate = 44100

    def __init__(self, *, frame_length: int | None = 4) -> None:
        if frame_length is not None:
            self.frame_length = frame_length
        self.calls: list[tuple[int, ...]] = []

    def from_indices(self, indices: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(indices.shape))
        batch_size = indices.shape[0]
        samples_per_token = int(getattr(self, "frame_length", 5) or 5)
        samples = int(indices.shape[-1]) * samples_per_token
        rows = [
            torch.full((1, samples), float(row_idx + 1), dtype=torch.float32)
            for row_idx in range(batch_size)
        ]
        return torch.stack(rows, dim=0)


class FakeFishReq:
    def __init__(
        self,
        *,
        rid: str = "req-fish",
        is_chunked: int = 0,
        extend_input_len: int = 0,
        prefix_indices: list[int] | None = None,
        decode_batch_idx: int = 0,
    ) -> None:
        self.rid = rid
        self.is_chunked = is_chunked
        self.extend_input_len = extend_input_len
        self.prefix_indices = prefix_indices or []
        self.decode_batch_idx = decode_batch_idx
        self.output_ids: list[int] = []

    def finished(self) -> bool:
        return False


class FakeFishModel:
    def __init__(self) -> None:
        self._semantic_begin_id = 200
        self._semantic_end_id = 295
        self._vq_mask = torch.zeros(4, dtype=torch.bool)
        self._vq_codes = torch.zeros((4, 2), dtype=torch.long)
        self._output_semantic_ids = torch.tensor([201, 202, 203, 204], dtype=torch.long)
        self._output_codes = torch.tensor(
            [[201, 1, 2], [202, 3, 4], [203, 5, 6], [204, 7, 8]],
            dtype=torch.long,
        )
        self._audio_decoder = SimpleNamespace(embed_text_dim=self._embed_text_dim)

    def get_embed_tokens(self):
        def _embed(input_ids: torch.Tensor) -> torch.Tensor:
            return input_ids.to(dtype=torch.float32).unsqueeze(-1).repeat(1, 2)

        return _embed

    def _embed_text_dim(
        self,
        req_embeds: torch.Tensor,
        vq_slice: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        del mask
        return vq_slice.to(dtype=req_embeds.dtype) + 1000.0
