# SPDX-License-Identifier: Apache-2.0
"""Small Qwen3-Omni fakes for model-specific unit tests."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch
from torch import nn

from sglang_omni_v1.models.qwen3_omni.payload_types import PipelineState
from sglang_omni_v1.proto import OmniRequest, StagePayload


class FakeQwenTokenizer:
    def __init__(
        self,
        *,
        eos_token_id: int = 99,
        pieces: dict[int, str] | None = None,
        vocab_size: int = 32000,
    ) -> None:
        self.eos_token_id = eos_token_id
        self.vocab_size = vocab_size
        self.pieces = pieces or {}

    def decode(self, token_ids: Any, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        ids = token_ids.tolist() if isinstance(token_ids, torch.Tensor) else token_ids
        return "".join(
            self.pieces.get(int(token_id), f"<{int(token_id)}>") for token_id in ids
        )


def make_qwen_state(**kwargs: Any) -> PipelineState:
    defaults: dict[str, Any] = {
        "raw_inputs": {"text": "hello"},
        "prompt": {
            "prompt_text": "hello",
            "input_ids": torch.tensor([11, 12, 13], dtype=torch.long),
            "attention_mask": torch.ones(3, dtype=torch.long),
        },
        "mm_inputs": {},
        "encoder_inputs": {},
        "stream_state": {},
    }
    defaults.update(kwargs)
    return PipelineState(**defaults)


def make_qwen_payload(
    state: PipelineState | None = None,
    *,
    request_id: str = "req-qwen",
    inputs: Any | None = None,
    params: dict[str, Any] | None = None,
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"text": "hello"} if inputs is None else inputs,
            params=params or {},
        ),
        data=(state or make_qwen_state()).to_dict(),
    )


class FakeCodecEmbedding(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.hidden_size = hidden_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        values = token_ids.to(dtype=torch.float32).unsqueeze(-1)
        return values.expand(token_ids.shape[0], self.hidden_size)


class FakeTalkerProjectionModel(nn.Module):
    def __init__(self, hidden_size: int = 2) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1, dtype=torch.float32))
        self.config = SimpleNamespace(codec_eos_token_id=2150)
        self._codec_embedding = FakeCodecEmbedding(hidden_size)

    def text_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 100.0

    def hidden_projection(self, tensor: torch.Tensor) -> torch.Tensor:
        return tensor + 200.0

    def get_input_embeddings(self) -> FakeCodecEmbedding:
        return self._codec_embedding


class FakeImageEncoderModel:
    spatial_merge_size = 1
    out_hidden_size = 2
    deepstack_layers = 1
    visual_dtype_bytes = 4

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        result: dict[str, Any] = {}
        if isinstance(kwargs.get("image_grid_thw"), torch.Tensor):
            grid = kwargs["image_grid_thw"].to(dtype=torch.long)
            counts = grid.prod(dim=-1).to(dtype=torch.long)
            total = int(counts.sum().item())
            embeds = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
            result.update(
                {
                    "image_grid_thw": grid,
                    "image_token_counts": counts,
                    "image_embeds": embeds,
                    "deepstack_visual_embeds_image": [embeds + 1000.0],
                }
            )
        if isinstance(kwargs.get("video_grid_thw"), torch.Tensor):
            grid = kwargs["video_grid_thw"].to(dtype=torch.long)
            counts = grid.prod(dim=-1).to(dtype=torch.long)
            total = int(counts.sum().item())
            embeds = (
                torch.arange(total * 2, dtype=torch.float32).reshape(total, 2) + 500.0
            )
            result.update(
                {
                    "video_grid_thw": grid,
                    "video_token_counts": counts,
                    "video_embeds": embeds,
                    "deepstack_visual_embeds_video": [embeds + 1000.0],
                }
            )
        return result


class FakeAudioEncoderModel:
    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def __call__(self, **kwargs: Any) -> dict[str, Any]:
        self.calls.append(kwargs)
        lengths = kwargs["audio_feature_lengths"].to(dtype=torch.long)
        output_lengths = lengths.clone()
        total = int(output_lengths.sum().item())
        embeds = torch.arange(total * 2, dtype=torch.float32).reshape(total, 2)
        return {
            "audio_embeds": embeds,
            "audio_feature_lengths": lengths,
            "audio_output_lengths": output_lengths,
        }


class FakeCode2WavModel:
    def __init__(self, *, total_upsample: int = 2) -> None:
        self.total_upsample = total_upsample
        self.calls: list[tuple[int, ...]] = []

    def __call__(self, codes: torch.Tensor) -> torch.Tensor:
        self.calls.append(tuple(codes.shape))
        samples = int(codes.shape[-1]) * self.total_upsample
        base = codes.to(dtype=torch.float32).sum().item()
        return torch.arange(samples, dtype=torch.float32).view(1, 1, samples) + base
