# SPDX-License-Identifier: Apache-2.0
"""Mimi codec wrapper for CSM.

Loaded from the checkpoint codec_model.* subtree. Defaults to fp32.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Callable

import torch
import torch.nn.functional as F
from safetensors import safe_open

logger = logging.getLogger(__name__)

# Local copies: utils.py imports this module, so do not import utils here.
_NUM_CODEBOOKS = 32
_MIMI_CODEBOOK_SIZE = 2048
_CODEC_IN_CKPT_PREFIX = "codec_model."
_SHARD_INDEX_CANDIDATES = (
    "transformers.safetensors.index.json",
    "model.safetensors.index.json",
)


def sanitize_for_mimi(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Clamp codes to [0, 2047] and return how many values were out of range."""
    if codes.numel() == 0:
        return codes, 0
    num_clamped = int((codes >= _MIMI_CODEBOOK_SIZE).sum().item())
    codes.clamp_(0, _MIMI_CODEBOOK_SIZE - 1)
    return codes, num_clamped


def trim_at_first_all_zero_frame(codes_F32: torch.Tensor) -> torch.Tensor:
    """Cut at the first all-zero frame. An EOS frame with cb31 != 0 is kept."""
    if codes_F32.numel() == 0:
        return codes_F32
    zero_frames = (codes_F32 == 0).all(dim=-1)
    hits = torch.nonzero(zero_frames)
    if hits.numel() == 0:
        return codes_F32
    return codes_F32[: int(hits[0].item())]


def load_codec_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load codec_model.* tensors with the Mimi prefix stripped."""
    shards: dict[str, list[str] | None] = {}
    for index_name in _SHARD_INDEX_CANDIDATES:
        index_path = os.path.join(checkpoint_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for full_name, shard in weight_map.items():
            if full_name.startswith(_CODEC_IN_CKPT_PREFIX):
                shards.setdefault(shard, []).append(full_name)  # type: ignore[union-attr]
        break
    else:
        for name in sorted(os.listdir(checkpoint_dir)):
            if name.endswith(".safetensors"):
                shards[name] = None

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        shard_path = os.path.join(checkpoint_dir, shard)
        with safe_open(shard_path, framework="pt") as f:
            keys = (
                names
                if names is not None
                else [k for k in f.keys() if k.startswith(_CODEC_IN_CKPT_PREFIX)]
            )
            for full_name in keys:
                state[full_name[len(_CODEC_IN_CKPT_PREFIX) :]] = f.get_tensor(full_name)
    return state


def to_codes_B32F(codes_F32: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[F, 32] to HF Mimi layout [1, 32, F]."""
    if codes_F32.ndim != 2 or codes_F32.shape[1] != _NUM_CODEBOOKS:
        raise ValueError(
            f"codes must be 2-D [F, {_NUM_CODEBOOKS}], got {tuple(codes_F32.shape)}"
        )
    return codes_F32.transpose(0, 1).unsqueeze(0).to(device=device, dtype=torch.long)


class CsmMimiCodec:
    """Wraps transformers.MimiModel for encode and decode."""

    SAMPLE_RATE = 24000
    SAMPLES_PER_FRAME = 1920  # asserted from config sampling_rate / frame_rate

    def __init__(self, model: Any, device: str, dtype: torch.dtype) -> None:
        """Hold the realized MimiModel and assert 1920 samples/frame."""
        cfg = model.config
        samples_per_frame = int(round(cfg.sampling_rate / cfg.frame_rate))
        if (
            int(cfg.sampling_rate) != self.SAMPLE_RATE
            or samples_per_frame != self.SAMPLES_PER_FRAME
        ):
            raise ValueError(
                f"Mimi config mismatch: sampling_rate={cfg.sampling_rate}, "
                f"frame_rate={cfg.frame_rate} → {samples_per_frame} samples/frame; "
                f"expected {self.SAMPLE_RATE} Hz / {self.SAMPLES_PER_FRAME}."
            )
        self.model = model
        self.device = torch.device(device)
        self.dtype = dtype
        self.clamp_count = 0

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "CsmMimiCodec":
        """Build a MimiModel from the checkpoint's codec_model.* tensors."""
        from transformers import MimiConfig, MimiModel

        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            csm_cfg = json.load(f)
        codec_cfg = csm_cfg.get("codec_config")
        if not isinstance(codec_cfg, dict):
            raise ValueError(
                f"{config_path} has no embedded codec_config dict; cannot "
                "build the bundled Mimi codec."
            )
        codec_cfg = dict(codec_cfg)
        for key in ("architectures", "torch_dtype", "transformers_version"):
            codec_cfg.pop(key, None)
        config = MimiConfig(**codec_cfg)
        model = MimiModel(config).eval()

        state = load_codec_state_dict(checkpoint_dir)
        if not state:
            raise FileNotFoundError(
                f"No codec weights found under {_CODEC_IN_CKPT_PREFIX!r} in "
                f"{checkpoint_dir}; this checkpoint doesn't bundle Mimi."
            )
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Mimi codec load got {len(unexpected)} unexpected tensors "
                f"(e.g. {unexpected[:3]}); checkpoint/transformers mismatch."
            )
        # Non-persistent buffers are regenerated at init.
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"Mimi codec load is too sparse: {len(missing)} missing / "
                f"{len(state)} loaded; embedded codec_config may be out of sync."
            )
        model = model.to(device=torch.device(device), dtype=dtype)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device, dtype=dtype)

    @torch.no_grad()
    def encode_reference(self, wav_11S: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode one 24 kHz mono waveform to [F, 32] Mimi codes."""
        if sample_rate != self.SAMPLE_RATE:
            raise ValueError(
                f"encode_reference expects {self.SAMPLE_RATE} Hz audio "
                f"(caller resamples), got {sample_rate}"
            )
        wav = wav_11S
        if wav.ndim == 1:
            wav = wav.view(1, 1, -1)
        elif wav.ndim == 2:
            wav = wav.unsqueeze(0)
        if wav.ndim != 3 or wav.shape[0] != 1 or wav.shape[1] != 1:
            raise ValueError(
                f"waveform must be mono [1, 1, S], got {tuple(wav_11S.shape)}"
            )
        if wav.shape[-1] < self.SAMPLES_PER_FRAME:
            wav = F.pad(wav, (0, self.SAMPLES_PER_FRAME - wav.shape[-1]))
        wav = wav.to(device=self.device, dtype=self.dtype)
        codes_B32F = self.model.encode(wav).audio_codes
        return codes_B32F.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    @torch.no_grad()
    def decode(self, codes_F32: torch.Tensor) -> torch.Tensor:
        """Decode [F, 32] codes to a waveform."""
        # Clone so the in-place clamp cannot mutate the caller.
        codes = codes_F32.detach().to(torch.long).clone()
        codes = self.sanitize_counted(codes, "decode")
        codes_B32F = to_codes_B32F(codes, self.device)
        wave = self.model.decode(codes_B32F).audio_values.squeeze(0)
        return self.exact_frames(wave, int(codes.shape[0])).to(torch.float32).cpu()

    @torch.no_grad()
    def decode_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch decode, grouped by frame count."""

        def _batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
            stacked = torch.stack([it.detach().to(torch.long) for it in batch_items])
            stacked = self.sanitize_counted(stacked, "decode_batch")
            codes_B32F = stacked.transpose(1, 2).to(
                device=self.device, dtype=torch.long
            )
            audio = self.model.decode(codes_B32F).audio_values
            num_frames = int(stacked.shape[1])
            return [
                self.exact_frames(audio[j], num_frames).to(torch.float32).cpu()
                for j in range(len(batch_items))
            ]

        return self.bucketed_batch(
            items,
            bucket_key_fn=lambda c: int(c.shape[0]),
            single_fn=self.decode,
            batch_fn=_batch_fn,
            error_label="CSM Mimi decode_batch",
        )

    def sanitize_counted(self, codes: torch.Tensor, where: str) -> torch.Tensor:
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            self.clamp_count += num_clamped
            logger.warning(
                "CSM Mimi %s clamped %d OOB codes (>= %d); total=%d",
                where,
                num_clamped,
                _MIMI_CODEBOOK_SIZE,
                self.clamp_count,
            )
        return codes

    def exact_frames(self, wave: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Trim or reject decode output that is not F * 1920 samples."""
        expected = num_frames * self.SAMPLES_PER_FRAME
        if wave.shape[-1] > expected:
            return wave[..., :expected]
        if wave.shape[-1] < expected:
            raise RuntimeError(
                f"Mimi decode returned {int(wave.shape[-1])} samples for "
                f"{num_frames} frames; expected {expected}."
            )
        return wave

    def bucketed_batch(
        self,
        items: list[torch.Tensor],
        *,
        bucket_key_fn: Callable[[torch.Tensor], int],
        single_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_fn: Callable[[list[torch.Tensor]], list[torch.Tensor]],
        error_label: str,
    ) -> list[torch.Tensor]:
        """Run single_fn on singleton buckets and batch_fn on multi-item buckets."""
        if not items:
            return []
        if len(items) == 1:
            return [single_fn(items[0])]

        buckets: dict[int, list[int]] = {}
        for i, item in enumerate(items):
            buckets.setdefault(bucket_key_fn(item), []).append(i)

        results: list[torch.Tensor | None] = [None] * len(items)
        for indices in buckets.values():
            if len(indices) == 1:
                results[indices[0]] = single_fn(items[indices[0]])
            else:
                batch_results = batch_fn([items[i] for i in indices])
                for idx, result in zip(indices, batch_results):
                    results[idx] = result

        out: list[torch.Tensor] = []
        for i, result in enumerate(results):
            if result is None:
                raise RuntimeError(f"{error_label} did not produce result for item {i}")
            out.append(result)
        return out


__all__ = ["CsmMimiCodec", "sanitize_for_mimi", "trim_at_first_all_zero_frame"]
