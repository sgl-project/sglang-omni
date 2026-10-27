# SPDX-License-Identifier: Apache-2.0
"""Hopper depth-decoder regression: full generate_frame must not abort."""

from __future__ import annotations

import pytest
import torch

from sglang_omni.models.csm_tts.hf_config import (
    CsmTtsHfConfig,
    build_depth_decoder_config,
)
from sglang_omni.models.csm_tts.modeling import CsmDepthDecoder, CsmFrameEmbedding


@pytest.mark.accelerator
def test_generate_frame_survives_cudnn_enabled_hopper() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    device = "cuda"
    dtype = torch.bfloat16
    cfg = build_depth_decoder_config(CsmTtsHfConfig())
    frame_emb = CsmFrameEmbedding(
        num_codebooks=cfg.num_codebooks,
        codebook_vocab=cfg.vocab_size,
        hidden_size=cfg.backbone_hidden_size,
    ).to(device=device, dtype=dtype)
    decoder = CsmDepthDecoder(cfg, frame_emb, max_slots=2).to(
        device=device, dtype=dtype
    )
    decoder.eval()
    hidden = torch.randn(1, cfg.backbone_hidden_size, device=device, dtype=dtype)
    cb0 = torch.randint(0, 2048, (1,), device=device)
    temps = torch.full((1,), 0.9, device=device, dtype=torch.float32)
    top_ks = torch.full((1,), 50, device=device, dtype=torch.long)
    with torch.inference_mode():
        codes = decoder.generate_frame(hidden, cb0, temps, top_ks, bs=1)
    assert tuple(codes.shape) == (1, 32)
    assert codes.dtype == torch.int64
    assert int(codes.min()) >= 0
    assert int(codes.max()) < cfg.vocab_size
