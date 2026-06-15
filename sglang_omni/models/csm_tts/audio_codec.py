# SPDX-License-Identifier: Apache-2.0
# Wraps the Mimi neural audio codec via `transformers.MimiModel` (Apache-2.0; Kyutai Moshi/Mimi).
"""Mimi codec wrapper for CSM — loaded from the SAME checkpoint's
``codec_model.*`` subtree (one-artifact pattern; the engine's weight loader
skips that subtree).

fp32 BY DEFAULT (conv-transpose decode stability; bf16 opt-in). Mimi runs at
24 kHz / 12.5 Hz ⇒ one frame = 80 ms = 1920 samples; 32 quantizers with
2048-row codebooks — codes 2048/2049/2050 are OUT OF BOUNDS and must be
clamped before any Mimi call (hard SIGSEGV-class hazard on some kernels,
garbage on others).


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

# Local copies of the CSM constants — utils.py imports THIS module, so this
# module must not import utils (cycle).
_NUM_CODEBOOKS = 32
_MIMI_CODEBOOK_SIZE = 2048  # valid Mimi codes are [0, 2047]

# Codec weights live under this prefix inside the CSM checkpoint shards.
_CODEC_IN_CKPT_PREFIX = "codec_model."
# CSM ships a NONSTANDARD index name (`transformers.safetensors.index.json`,
# shards `transformers-0000x-of-00002.safetensors`).
_SHARD_INDEX_CANDIDATES = (
    "transformers.safetensors.index.json",
    "model.safetensors.index.json",
)


def sanitize_for_mimi(codes: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Fence #2 of 3: count then ``codes.clamp_(0, 2047)``.

    The clamp count is exported as a counter: under greedy/temp=0 runs
    ``count == 0`` is a hard parity-gate assertion; under
    sampling it is telemetry ONLY — the 2051-way heads can legally emit
    2048-2050 with nonzero probability.

    Args:
        codes: int64 ``[F, 32]`` (any device).

    Returns:
        ``(clamped codes [F, 32], num_clamped int)``.
    """
    if codes.numel() == 0:
        return codes, 0
    num_clamped = int((codes >= _MIMI_CODEBOOK_SIZE).sum().item())
    codes.clamp_(0, _MIMI_CODEBOOK_SIZE - 1)
    return codes, num_clamped


def trim_at_first_all_zero_frame(codes_F32: torch.Tensor) -> torch.Tensor:
    """Fence #3 of 3: cut at the first frame whose ALL 32 codebooks
    are 0 (HF cutoff semantics, generation_csm.py:471-477 — mirrors HF's
    stop(31)/trim(32) inconsistency exactly: an EOS frame with cb31 != 0 IS
    kept and decoded).

    Args:
        codes_F32: int ``[F, 32]``.

    Returns:
        ``codes_F32[:cutoff]`` (possibly 0 frames).
    """
    if codes_F32.numel() == 0:
        return codes_F32
    zero_frames = (codes_F32 == 0).all(dim=-1)
    hits = torch.nonzero(zero_frames)
    if hits.numel() == 0:
        return codes_F32
    return codes_F32[: int(hits[0].item())]


def _load_codec_state_dict(checkpoint_dir: str) -> dict[str, torch.Tensor]:
    """Pull all ``codec_model.*`` tensors out of the CSM checkpoint shards
    into a state dict keyed by Mimi's own parameter names (prefix stripped)."""
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
        # No index file: scan every shard in the dir for the prefix.
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


def _to_codes_B32F(codes_F32: torch.Tensor, device: torch.device) -> torch.Tensor:
    """Validated ``[F, 32]`` → HF Mimi layout ``[1, 32, F]`` long on device."""
    if codes_F32.ndim != 2 or codes_F32.shape[1] != _NUM_CODEBOOKS:
        raise ValueError(
            f"codes must be 2-D [F, {_NUM_CODEBOOKS}], got {tuple(codes_F32.shape)}"
        )
    return codes_F32.transpose(0, 1).unsqueeze(0).to(device=device, dtype=torch.long)


class CsmMimiCodec:
    """Wraps ``transformers.MimiModel`` for encode (audio_encoder stage) and
    decode (vocoder stage). One process-wide instance per (path, device,
    dtype) via :func:`sglang_omni.models.csm_tts.utils.get_or_load_codec`."""

    SAMPLE_RATE = 24000
    SAMPLES_PER_FRAME = 1920  # asserted from config sampling_rate / frame_rate

    def __init__(self, model: Any, device: str, dtype: torch.dtype) -> None:
        """Hold the realized ``MimiModel`` + device/dtype; assert
        ``sampling_rate / frame_rate == 1920``."""
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
        # Process-wide sanitize telemetry (fence #2); the vocoder additionally
        # tracks per-request counts. Gate-able only at temp=0.
        self.clamp_count = 0

    @classmethod
    def from_pretrained(
        cls,
        checkpoint_dir: str,
        device: str = "cuda",
        dtype: torch.dtype = torch.float32,
    ) -> "CsmMimiCodec":
        """Build a ``MimiModel`` from the CSM checkpoint's ``codec_model.*``
        tensors (NOT a separate kyutai repo download).

        Returns:
            :class:`CsmMimiCodec` on ``device`` in ``dtype`` (fp32 default).
        """
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

        state = _load_codec_state_dict(checkpoint_dir)
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
        # Non-persistent buffers (rope inv_freq etc.) are regenerated at init;
        # anything beyond that means a real architecture mismatch.
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
        """Encode one context waveform to Mimi codes.

        Pads to >= 1 frame; transposes HF's ``[1, 32, F]`` output.

        Args:
            wav_11S: float32 ``[1, 1, S]`` mono.
            sample_rate: must be 24000 (caller resamples).

        Returns:
            int64 ``[F, 32]``, values in ``[0, 2047]``.
        """
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
        """Decode a code matrix to a waveform (stateless path).

        Transposes to ``[1, 32, F]``, applies the :func:`sanitize_for_mimi`
        clamp guard, decodes.

        Args:
            codes_F32: int64 ``[F, 32]``.

        Returns:
            float32 ``[1, S]`` with ``S == F * 1920``.
        """
        # Clone so the caller's tensor is never mutated by the in-place clamp.
        codes = codes_F32.detach().to(torch.long).clone()
        codes = self._sanitize_counted(codes, "decode")
        codes_B32F = _to_codes_B32F(codes, self.device)
        wave = self.model.decode(codes_B32F).audio_values.squeeze(0)
        return self._exact_frames(wave, int(codes.shape[0])).to(torch.float32).cpu()

    @torch.no_grad()
    def decode_batch(self, items: list[torch.Tensor]) -> list[torch.Tensor]:
        """Bucketed batch decode (exact-frame-count buckets, higgs
        ``_bucketed_batch`` copy) for the non-streaming path.

        Args:
            items: list of int64 ``[F_i, 32]``.

        Returns:
            list of float32 ``[1, S_i]`` aligned with the input order.
        """

        def _batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
            stacked = torch.stack(
                [it.detach().to(torch.long) for it in batch_items]
            )  # [B, F, 32]; stack copies, so the in-place clamp is safe
            stacked = self._sanitize_counted(stacked, "decode_batch")
            codes_B32F = stacked.transpose(1, 2).to(
                device=self.device, dtype=torch.long
            )
            audio = self.model.decode(codes_B32F).audio_values  # [B, 1, S]
            num_frames = int(stacked.shape[1])
            return [
                self._exact_frames(audio[j], num_frames).to(torch.float32).cpu()
                for j in range(len(batch_items))
            ]

        return self._bucketed_batch(
            items,
            bucket_key_fn=lambda c: int(c.shape[0]),
            single_fn=self.decode,
            batch_fn=_batch_fn,
            error_label="CSM Mimi decode_batch",
        )

    @torch.no_grad()
    def streaming_decode(
        self, codes_F32: torch.Tensor, past_key_values: Any | None
    ) -> tuple[torch.Tensor, Any]:
        """Stateful incremental decode wrapping
        ``MimiModel.decode(..., decoder_past_key_values=...)``; the
        per-request state object is OWNED by the vocoder scheduler and dropped
        in ``clear_stream_state``. This stateful path is optional and not the
        shipped default; the default vocoder uses stateless overlap-trim decode.

        Args:
            codes_F32: int64 ``[F_new, 32]`` delta frames.
            past_key_values: Mimi decoder KV state or ``None`` on first call.

        Returns:
            ``(wave float32 [1, F_new * 1920], new past_key_values)``.
        """
        if past_key_values is None:
            # Force the cache path (the bundled codec_config ships
            # use_cache=False); only the decoder TRANSFORMER KV persists — the
            # CNN/upsample ladder has no conv-state cache (HF docstring,
            # modeling_mimi.py:1641-1643), hence the trailing-trim below.
            from transformers.cache_utils import DynamicCache

            past_key_values = DynamicCache()
        codes = codes_F32.detach().to(torch.long).clone()
        codes = self._sanitize_counted(codes, "streaming_decode")
        codes_B32F = _to_codes_B32F(codes, self.device)
        out = self.model.decode(codes_B32F, decoder_past_key_values=past_key_values)
        wave = out.audio_values.squeeze(0)
        num_frames = int(codes.shape[0])
        expected = num_frames * self.SAMPLES_PER_FRAME
        if wave.shape[-1] > expected:
            wave = wave[..., :expected]  # CNN edge overrun; extra steps trimmed
        return wave.to(torch.float32).cpu(), out.decoder_past_key_values

    # --- internals ------------------------------------------------------------

    def _sanitize_counted(self, codes: torch.Tensor, where: str) -> torch.Tensor:
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

    def _exact_frames(self, wave: torch.Tensor, num_frames: int) -> torch.Tensor:
        """Enforce the ``S == F * 1920`` contract the vocoder framing math
        relies on (Mimi full decode is exact; spec)."""
        expected = num_frames * self.SAMPLES_PER_FRAME
        if wave.shape[-1] > expected:
            return wave[..., :expected]
        if wave.shape[-1] < expected:
            raise RuntimeError(
                f"Mimi decode returned {int(wave.shape[-1])} samples for "
                f"{num_frames} frames; expected {expected}."
            )
        return wave

    def _bucketed_batch(
        self,
        items: list[torch.Tensor],
        *,
        bucket_key_fn: Callable[[torch.Tensor], int],
        single_fn: Callable[[torch.Tensor], torch.Tensor],
        batch_fn: Callable[[list[torch.Tensor]], list[torch.Tensor]],
        error_label: str,
    ) -> list[torch.Tensor]:
        """Run single_fn on singleton buckets and batch_fn on multi-item
        buckets (higgs audio_codec._bucketed_batch copy)."""
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
