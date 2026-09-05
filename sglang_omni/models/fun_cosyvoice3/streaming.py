# SPDX-License-Identifier: Apache-2.0
"""CosyVoice3 causal Flow streaming hop math."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.proto import StagePayload

TOKEN_HOP_LEN = 25
PRE_LOOKAHEAD_LEN = 3
TOKEN_MEL_RATIO = 2
STREAM_SCALE_FACTOR = 2
TOKEN_MAX_HOP_LEN = TOKEN_HOP_LEN * 4
SAMPLE_RATE = 24000

# note (guozhihao-224): pad=0 first flush is hop+lookahead; producer adds
# prompt_token_pad so the first message matches the vocoder causal boundary.
AR_INITIAL_FLUSH_TOKENS = TOKEN_HOP_LEN + PRE_LOOKAHEAD_LEN
AR_FOLLOWUP_FLUSH_TOKENS = TOKEN_HOP_LEN


def prompt_token_len(prompt_token: Any) -> int:
    """Time-axis length of a Flow prompt-token tensor, or 0 when missing."""
    if prompt_token is None:
        return 0
    token = torch.as_tensor(prompt_token)
    if token.ndim == 0:
        return int(token.numel())
    return int(token.shape[-1])


def prompt_token_pad(prompt_token_len: int, *, hop_len: int = TOKEN_HOP_LEN) -> int:
    """Pad prompt length up to the next hop multiple."""
    if hop_len <= 0:
        raise ValueError(f"hop_len must be positive, got {hop_len}")
    length = max(int(prompt_token_len), 0)
    if length == 0:
        return 0
    return int((length + hop_len - 1) // hop_len * hop_len - length)


def stream_hop_len(
    token_offset: int,
    *,
    hop_len: int,
    prompt_pad: int,
) -> int:
    """Generated-token hop for the next causal Flow window."""
    if token_offset < 0:
        raise ValueError(f"token_offset must be >= 0, got {token_offset}")
    hop = int(hop_len)
    if hop <= 0:
        raise ValueError(f"hop_len must be positive, got {hop_len}")
    if int(token_offset) == 0:
        return hop + max(int(prompt_pad), 0)
    return hop


def next_stream_hop_len(
    hop_len: int,
    *,
    max_hop_len: int = TOKEN_MAX_HOP_LEN,
    scale: int = STREAM_SCALE_FACTOR,
) -> int:
    """Grow hop after a successful causal chunk, matching CosyVoice3Model."""
    hop = int(hop_len)
    if hop <= 0:
        raise ValueError(f"hop_len must be positive, got {hop_len}")
    if scale < 1:
        raise ValueError(f"scale must be >= 1, got {scale}")
    return min(int(max_hop_len), hop * int(scale))


def tokens_needed_for_causal_chunk(
    token_offset: int,
    *,
    hop_len: int,
    prompt_pad: int,
    lookahead: int = PRE_LOOKAHEAD_LEN,
) -> int:
    """Minimum generated-token count to run one non-final causal chunk."""
    hop = stream_hop_len(token_offset, hop_len=hop_len, prompt_pad=prompt_pad)
    extra = max(int(lookahead), 0)
    return int(token_offset) + hop + extra


def first_ar_flush_tokens(prompt_len: int) -> int:
    """Generated-token count for the first causal AR flush."""
    return tokens_needed_for_causal_chunk(
        0,
        hop_len=TOKEN_HOP_LEN,
        prompt_pad=prompt_token_pad(int(prompt_len)),
    )


def as_flow_prompt_token(value: Any | None) -> torch.Tensor:
    if value is None:
        return torch.zeros(1, 0, dtype=torch.int32)
    token = torch.as_tensor(value, dtype=torch.int32)
    if token.ndim == 1:
        token = token.unsqueeze(0)
    elif token.ndim != 2:
        raise ValueError(
            f"Fun-CosyVoice3 prompt speech token must be 1-D or 2-D, "
            f"got {tuple(token.shape)}"
        )
    return token


def as_flow_prompt_feat(value: Any | None) -> torch.Tensor:
    if value is None:
        return torch.zeros(1, 0, 80)
    feat = torch.as_tensor(value)
    if feat.ndim == 2:
        feat = feat.unsqueeze(0)
    elif feat.ndim != 3:
        raise ValueError(
            f"Fun-CosyVoice3 prompt speech feat must be 2-D or 3-D, "
            f"got {tuple(feat.shape)}"
        )
    return feat


def as_flow_embedding(value: Any | None) -> torch.Tensor:
    if value is None:
        return torch.zeros(1, 192)
    embedding = torch.as_tensor(value)
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    elif embedding.ndim != 2:
        raise ValueError(
            f"Fun-CosyVoice3 speaker embedding must be 1-D or 2-D, "
            f"got {tuple(embedding.shape)}"
        )
    return embedding


def build_cosyvoice3_stream_metadata(payload: StagePayload) -> dict[str, Any] | None:
    """Static per-chunk metadata, or None when the request is not streaming."""
    params = payload.request.params
    if not isinstance(params, dict):
        raise TypeError(
            f"Fun-CosyVoice3 request params must be a dict, got {type(params).__name__}"
        )
    if not bool(params.get("stream", False)):
        return None
    return {
        "modality": "audio_codes",
        "stream": True,
    }
