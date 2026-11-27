# SPDX-License-Identifier: Apache-2.0
"""Mimi codec wrapper for CSM.

Loaded from the checkpoint codec_model.* subtree. Defaults to fp32.
"""

from __future__ import annotations

import json
import logging
import os
from typing import Protocol

import torch
import torch.nn.functional as F
from safetensors import safe_open
from transformers import MimiConfig, MimiModel

from sglang_omni.models.csm_tts.utils import (
    MIMI_CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    SAMPLE_RATE,
    SAMPLES_PER_FRAME,
)

logger = logging.getLogger(__name__)


class BucketKey(Protocol):
    def __call__(self, item: torch.Tensor) -> int: ...


class SingleDecode(Protocol):
    def __call__(self, item: torch.Tensor) -> torch.Tensor: ...


class BatchDecode(Protocol):
    def __call__(self, items: list[torch.Tensor]) -> list[torch.Tensor]: ...


CODEC_CACHE: dict[tuple[str, str, str], CsmAudioCodec] = {}
CODEC_DTYPES: dict[str, torch.dtype] = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}
CODEC_IN_CKPT_PREFIX = "codec_model."
SHARD_INDEX_CANDIDATES = (
    "transformers.safetensors.index.json",
    "model.safetensors.index.json",
)


def sanitize_for_mimi(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Clamp codes to [0, 2047] and return how many values were out of range."""
    if codes.numel() == 0:
        return codes, 0
    else:
        pass
    num_clamped = int((codes >= MIMI_CODEBOOK_SIZE).sum().item())
    codes.clamp_(0, MIMI_CODEBOOK_SIZE - 1)
    return codes, num_clamped


def load_codec_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Load codec_model.* tensors with the Mimi prefix stripped."""
    shards: dict[str, list[str] | None] = {}
    for index_name in SHARD_INDEX_CANDIDATES:
        index_path = os.path.join(checkpoint_dir, index_name)
        if not os.path.isfile(index_path):
            continue
        else:
            pass
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        for full_name, shard in weight_map.items():
            if full_name.startswith(CODEC_IN_CKPT_PREFIX):
                shards.setdefault(shard, []).append(full_name)  # type: ignore[union-attr]
            else:
                pass
        break
    else:
        for name in sorted(os.listdir(checkpoint_dir)):
            if name.endswith(".safetensors"):
                shards[name] = None
            else:
                pass

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        shard_path = os.path.join(checkpoint_dir, shard)
        with safe_open(shard_path, framework="pt") as f:
            keys = (
                names
                if names is not None
                else [k for k in f.keys() if k.startswith(CODEC_IN_CKPT_PREFIX)]
            )
            for full_name in keys:
                state[full_name[len(CODEC_IN_CKPT_PREFIX) :]] = f.get_tensor(full_name)
    return state


def to_codes_B32F(codes_F32: torch.Tensor, device: torch.device) -> torch.Tensor:
    """[F, 32] to HF Mimi layout [1, 32, F]."""
    if codes_F32.ndim != 2 or codes_F32.shape[1] != NUM_CODEBOOKS:
        raise ValueError(
            f"codes must be 2-D [F, {NUM_CODEBOOKS}], got {tuple(codes_F32.shape)}"
        )
    else:
        pass
    return codes_F32.transpose(0, 1).unsqueeze(0).to(device=device, dtype=torch.long)


class CsmAudioCodec:
    """Wraps transformers.MimiModel for encode and decode."""

    def __init__(self, model: MimiModel, device: str, dtype: torch.dtype) -> None:
        mimi_config = model.config
        samples_per_frame = int(
            round(mimi_config.sampling_rate / mimi_config.frame_rate)
        )
        if (
            int(mimi_config.sampling_rate) != SAMPLE_RATE
            or samples_per_frame != SAMPLES_PER_FRAME
        ):
            raise ValueError(
                f"Mimi config mismatch: "
                f"sampling_rate={mimi_config.sampling_rate}, "
                f"frame_rate={mimi_config.frame_rate} -> {samples_per_frame} "
                f"samples/frame; expected {SAMPLE_RATE} Hz / {SAMPLES_PER_FRAME}."
            )
        else:
            pass
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
    ) -> CsmAudioCodec:
        """Build a MimiModel from the checkpoint's codec_model.* tensors."""
        config_path = os.path.join(checkpoint_dir, "config.json")
        with open(config_path) as f:
            checkpoint_config = json.load(f)
        codec_config = checkpoint_config.get("codec_config")
        if not isinstance(codec_config, dict):
            raise ValueError(
                f"{config_path} has no embedded codec_config dict; cannot "
                "build the bundled Mimi codec."
            )
        else:
            pass
        codec_config = dict(codec_config)
        for key in ("architectures", "torch_dtype", "transformers_version"):
            codec_config.pop(key, None)
        model = MimiModel(MimiConfig(**codec_config)).eval()

        state = load_codec_state_dict(checkpoint_dir)
        if not state:
            raise FileNotFoundError(
                f"No codec weights found under {CODEC_IN_CKPT_PREFIX!r} in "
                f"{checkpoint_dir}; this checkpoint doesn't bundle Mimi."
            )
        else:
            pass
        missing, unexpected = model.load_state_dict(state, strict=False)
        if unexpected:
            raise RuntimeError(
                f"Mimi codec load got {len(unexpected)} unexpected tensors "
                f"(e.g. {unexpected[:3]}); checkpoint/transformers mismatch."
            )
        else:
            pass
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"Mimi codec load is too sparse: {len(missing)} missing / "
                f"{len(state)} loaded; embedded codec_config may be out of sync."
            )
        else:
            pass
        model = model.to(device=torch.device(device), dtype=dtype)
        for parameter in model.parameters():
            parameter.requires_grad_(False)
        return cls(model, device=device, dtype=dtype)

    @torch.no_grad()
    def encode_reference(self, wav_11S: torch.Tensor, sample_rate: int) -> torch.Tensor:
        """Encode one 24 kHz mono waveform to [F, 32] Mimi codes."""
        if sample_rate != SAMPLE_RATE:
            raise ValueError(
                f"encode_reference expects {SAMPLE_RATE} Hz audio "
                f"(caller resamples), got {sample_rate}"
            )
        else:
            pass
        waveform = wav_11S
        if waveform.ndim == 1:
            waveform = waveform.view(1, 1, -1)
        elif waveform.ndim == 2:
            waveform = waveform.unsqueeze(0)
        else:
            pass
        if waveform.ndim != 3 or waveform.shape[0] != 1 or waveform.shape[1] != 1:
            raise ValueError(
                f"waveform must be mono [1, 1, S], got {tuple(wav_11S.shape)}"
            )
        else:
            pass
        if waveform.shape[-1] < SAMPLES_PER_FRAME:
            waveform = F.pad(waveform, (0, SAMPLES_PER_FRAME - waveform.shape[-1]))
        else:
            pass
        waveform = waveform.to(device=self.device, dtype=self.dtype)
        codes_B32F = self.model.encode(waveform).audio_codes
        return codes_B32F.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    @torch.no_grad()
    def decode(self, codes_F32: torch.Tensor) -> torch.Tensor:
        # clone so the in-place clamp cannot mutate the caller.
        codes = codes_F32.detach().to(torch.long).clone()
        codes = self.sanitize_counted(codes, "decode")
        codes_B32F = to_codes_B32F(codes, self.device)
        wave = self.model.decode(codes_B32F).audio_values.squeeze(0)
        return self.exact_frames(wave, int(codes.shape[0])).to(torch.float32).cpu()

    @torch.no_grad()
    def decode_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch decode, grouped by frame count."""

        def batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
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
            batch_fn=batch_fn,
            error_label="CSM Mimi decode_batch",
        )

    def sanitize_counted(self, codes: torch.Tensor, where: str) -> torch.Tensor:
        codes, num_clamped = sanitize_for_mimi(codes)
        if num_clamped:
            self.clamp_count += num_clamped
            logger.warning(
                f"CSM Mimi {where} clamped {num_clamped} OOB codes "
                f"(>= {MIMI_CODEBOOK_SIZE}); total={self.clamp_count}"
            )
        else:
            pass
        return codes

    def exact_frames(self, wave: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Trim or reject decode output that is not F * 1920 samples."""
        expected = num_frames * SAMPLES_PER_FRAME
        if wave.shape[-1] > expected:
            return wave[..., :expected]
        else:
            pass
        if wave.shape[-1] < expected:
            raise RuntimeError(
                f"Mimi decode returned {int(wave.shape[-1])} samples for "
                f"{num_frames} frames; expected {expected}."
            )
        else:
            pass
        return wave

    def bucketed_batch(
        self,
        items: list[torch.Tensor],
        *,
        bucket_key_fn: BucketKey,
        single_fn: SingleDecode,
        batch_fn: BatchDecode,
        error_label: str,
    ) -> list[torch.Tensor]:
        """Run single_fn on singleton buckets and batch_fn on multi-item buckets."""
        if not items:
            return []
        else:
            pass
        if len(items) == 1:
            return [single_fn(items[0])]
        else:
            pass

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
            else:
                pass
            out.append(result)
        return out


def load_cached_codec(checkpoint_dir: str, device: str, dtype: str) -> CsmAudioCodec:
    """Process-wide cached CsmAudioCodec per (path, device, dtype)."""
    cache_key = (str(checkpoint_dir), str(device), str(dtype))
    cached = CODEC_CACHE.get(cache_key)
    if cached is not None:
        return cached
    else:
        pass
    torch_dtype = CODEC_DTYPES.get(dtype)
    if torch_dtype is None:
        raise ValueError(
            f"CSM codec dtype must be one of {sorted(CODEC_DTYPES)}, got {dtype!r}"
        )
    else:
        pass
    codec = CsmAudioCodec.from_pretrained(
        checkpoint_dir, device=device, dtype=torch_dtype
    )
    CODEC_CACHE[cache_key] = codec
    return codec


__all__ = ["CsmAudioCodec", "load_cached_codec", "sanitize_for_mimi"]
