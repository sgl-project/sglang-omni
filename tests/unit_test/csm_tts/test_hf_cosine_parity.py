# SPDX-License-Identifier: Apache-2.0
"""HF vs ported depth decoder: greedy code match + waveform cosine.

Mirrors MiniCPM-O / AUK golden cosine checks: same input, compare tensors
with ``F.cosine_similarity``. Uses the real ``eustlb/csm-1b`` checkpoint.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from sglang_omni.models.csm_tts.audio_codec import CsmMimiCodec
from sglang_omni.models.csm_tts.hf_config import (
    CsmTtsHfConfig,
    build_depth_decoder_config,
)
from sglang_omni.models.csm_tts.modeling import CsmDepthDecoder, CsmFrameEmbedding
from sglang_omni.models.csm_tts.weight_loader import CsmWeightMapper

CKPT = Path("/weights/eustlb-csm-1b")
# One-frame greedy decode of random-but-fixed hidden should stay near-identical
# after bf16 cast; waveform cosine is looser because Mimi is conv-sensitive.
# 31/32 greedy codes typically match; one argmax flip under bf16 is expected.
CODE_MATCH_MIN = 30 / 32
CODE_COS_MIN = 0.93
WAVE_COS_MIN = 0.95


def _checkpoint_available() -> bool:
    return CKPT.is_dir() and (CKPT / "config.json").is_file()


def _load_mapped_tensors() -> dict[str, torch.Tensor]:
    from safetensors import safe_open

    mapper = CsmWeightMapper()
    loaded: dict[str, torch.Tensor] = {}
    shards = sorted(CKPT.glob("*.safetensors"))
    if not shards:
        pytest.skip(f"no safetensors under {CKPT}")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                dest = mapper.map(name)
                if dest is None:
                    continue
                if dest.startswith("backbone."):
                    continue
                loaded[dest] = handle.get_tensor(name)
    return loaded


def _build_ported_depth(device: torch.device, dtype: torch.dtype):
    cfg = CsmTtsHfConfig.from_pretrained(str(CKPT), local_files_only=True)
    depth_cfg = build_depth_decoder_config(cfg)
    frame_emb = CsmFrameEmbedding(
        num_codebooks=depth_cfg.num_codebooks,
        codebook_vocab=depth_cfg.vocab_size,
        hidden_size=depth_cfg.backbone_hidden_size,
    )
    decoder = CsmDepthDecoder(depth_cfg, frame_emb, max_slots=2)
    codebook0 = torch.nn.Linear(
        depth_cfg.backbone_hidden_size, depth_cfg.vocab_size, bias=False
    )
    weights = _load_mapped_tensors()
    own = dict(decoder.named_parameters())
    own.update({f"frame_embedding.{k}": v for k, v in frame_emb.named_parameters()})
    own["codebook0_head.weight"] = codebook0.weight
    loaded = 0
    for dest, tensor in weights.items():
        key = (
            dest[len("depth_decoder.") :] if dest.startswith("depth_decoder.") else dest
        )
        param = own.get(key)
        if param is None:
            param = own.get(dest)
        if param is None:
            continue
        if param.shape != tensor.shape:
            raise AssertionError(
                f"{dest}: {tuple(param.shape)} vs {tuple(tensor.shape)}"
            )
        param.data.copy_(tensor.to(param.dtype))
        loaded += 1
    if loaded < 10:
        raise AssertionError(f"only loaded {loaded} depth/codec-adjacent tensors")
    frame_emb.to(device=device, dtype=dtype)
    decoder.to(device=device, dtype=dtype)
    codebook0.to(device=device, dtype=dtype)
    decoder.eval()
    codebook0.eval()
    return decoder, codebook0, depth_cfg


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    flat_a = a.reshape(-1).float()
    flat_b = b.reshape(-1).float()
    n = min(flat_a.numel(), flat_b.numel())
    return float(F.cosine_similarity(flat_a[:n], flat_b[:n], dim=0).item())


@pytest.mark.accelerator
def test_ported_depth_matches_hf_greedy_codes_and_waveform_cosine() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    if not _checkpoint_available():
        pytest.skip(f"checkpoint missing: {CKPT}")

    from transformers import CsmForConditionalGeneration

    device = torch.device("cuda")
    dtype = torch.bfloat16
    torch.manual_seed(0)
    torch.cuda.manual_seed_all(0)

    hf = CsmForConditionalGeneration.from_pretrained(
        str(CKPT),
        torch_dtype=dtype,
        local_files_only=True,
    ).to(device)
    hf.eval()

    decoder, codebook0, depth_cfg = _build_ported_depth(device, dtype)
    hidden = torch.randn(1, depth_cfg.backbone_hidden_size, device=device, dtype=dtype)
    with torch.inference_mode():
        logits0 = codebook0(hidden).float()
        cb0 = logits0.argmax(dim=-1)
        temps = torch.zeros(1, device=device, dtype=torch.float32)
        top_ks = torch.ones(1, device=device, dtype=torch.long)
        ported_codes = decoder.generate_frame(hidden, cb0, temps, top_ks, bs=1)

        # HF depth: same backbone hidden + greedy codebook 0.
        hf_codes = _hf_generate_frame_greedy(hf.depth_decoder, hidden, cb0)

    assert ported_codes.shape == hf_codes.shape == (1, 32)
    equal = ported_codes == hf_codes
    match = float(equal.float().mean().item())
    mismatch = (~equal[0]).nonzero(as_tuple=False).flatten().tolist()
    code_cos = _cosine(
        F.one_hot(ported_codes.clamp(min=0), num_classes=int(depth_cfg.vocab_size)),
        F.one_hot(hf_codes.clamp(min=0), num_classes=int(depth_cfg.vocab_size)),
    )
    assert match >= CODE_MATCH_MIN, (
        f"greedy code match {match:.4f} < {CODE_MATCH_MIN} " f"mismatch_cb={mismatch}"
    )
    assert code_cos >= CODE_COS_MIN, f"code cosine {code_cos:.6f} < {CODE_COS_MIN}"

    codec = CsmMimiCodec.from_pretrained(
        str(CKPT), device=str(device), dtype=torch.float32
    )
    with torch.inference_mode():
        wav_ported = codec.decode(ported_codes[0].cpu().unsqueeze(0))
        wav_hf = codec.decode(hf_codes[0].cpu().unsqueeze(0))
    if wav_ported.ndim > 1:
        wav_ported = wav_ported.reshape(-1)
    if wav_hf.ndim > 1:
        wav_hf = wav_hf.reshape(-1)
    wave_cos = _cosine(wav_ported, wav_hf)
    assert wave_cos >= WAVE_COS_MIN, f"waveform cosine {wave_cos:.6f} < {WAVE_COS_MIN}"


def _hf_generate_frame_greedy(depth, hidden_B2048: torch.Tensor, cb0_B: torch.Tensor):
    """Greedy 32-code frame from HF's depth decoder (same inputs as port)."""
    from torch.nn.attention import SDPBackend, sdpa_kernel

    first = cb0_B.view(-1, 1)
    depth_input = torch.nn.functional.pad(first, (1, 0), value=0)
    # Same Hopper guard as the port: HF's depth generate also hits cuDNN SDPA.
    with sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    ):
        out = depth.generate(
            input_ids=depth_input,
            backbone_last_hidden_state=hidden_B2048.clone(),
            do_sample=False,
            max_new_tokens=31,
            min_new_tokens=31,
        )
    codes = out if torch.is_tensor(out) else out.sequences
    codes = codes[:, 1:]  # drop the placeholder column
    if codes.ndim == 1:
        codes = codes.unsqueeze(0)
    return codes.to(torch.long)
