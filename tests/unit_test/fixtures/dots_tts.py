# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import torch

from sglang_omni.models.dots_tts.flow_head import DotsTTSFlowHead

LLM_HIDDEN = 48
FM_HIDDEN = 32
LATENT_DIM = 6
PATCH_SIZE = 2


def tiny_flow_head(
    tmp_path: Path, *, meanflow: bool, optimize: bool
) -> DotsTTSFlowHead:
    torch.save(
        {"mean": torch.zeros(LATENT_DIM), "var": torch.ones(LATENT_DIM)},
        tmp_path / "latent_stats.pt",
    )
    config = {
        "latent_dim": LATENT_DIM,
        "patch_size": PATCH_SIZE,
        "PatchEncoder": {
            "num_layers": 1,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "causal": True,
        },
        "DiT": {
            "num_layers": 2,
            "num_heads": 2,
            "hidden_size": FM_HIDDEN,
            "ffn_hidden_size": 64,
            "modulation": True,
            "qk_norm": True,
            "rotary_bias": True,
        },
        "vocoder": {"sample_rate": 48000},
        "meanflow": (
            {"enabled": True, "use_duration_embedding": True} if meanflow else None
        ),
    }
    flow = DotsTTSFlowHead(
        config,
        llm_hidden_size=LLM_HIDDEN,
        latent_stats_path=str(tmp_path / "latent_stats.pt"),
        optimize=optimize,
    )
    with torch.no_grad():
        for parameter in flow.parameters():
            parameter.normal_(0.0, 0.2)
    return flow.eval()


__all__ = ["FM_HIDDEN", "LATENT_DIM", "LLM_HIDDEN", "PATCH_SIZE", "tiny_flow_head"]
