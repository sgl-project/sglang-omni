"""Thin facade over Transformers' Higgs Audio V2 tokenizer.

The codec weights are bundled inside the Higgs TTS checkpoint under the
prefix ``tied.embedding.modality_embeddings.0.model.*`` (acoustic encoder,
acoustic decoder, quantizer, semantic model, etc.).
:meth:`HiggsAudioCodec.from_pretrained` extracts those keys directly from
the TTS ``model.safetensors`` so a single checkpoint serves both the AR
engine and the codec.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchaudio
from huggingface_hub import snapshot_download
from safetensors import safe_open
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

WaveformInput = torch.Tensor | np.ndarray

# Codec weights live under this prefix inside the TTS checkpoint.
_CODEC_IN_TTS_CKPT_PREFIX = "tied.embedding.modality_embeddings.0.model."

# Bundled codec config (taken byte-for-byte from
# https://huggingface.co/bosonai/higgs-audio-v2-tokenizer/blob/main/config.json).
_BUNDLED_CODEC_CONFIG_PATH = os.path.join(
    os.path.dirname(__file__),
    "configs",
    "higgs_audio_v2_tokenizer.json",
)


def _to_mono_3d(waveform: WaveformInput) -> torch.Tensor:
    """Normalise (Tensor | ndarray) waveform to mono ``[1, 1, L]``."""
    if isinstance(waveform, np.ndarray):
        wav = torch.from_numpy(np.ascontiguousarray(waveform))
    elif isinstance(waveform, torch.Tensor):
        wav = waveform
    else:
        raise TypeError(
            f"waveform must be Tensor or ndarray, got {type(waveform).__name__}"
        )

    if wav.ndim == 1:
        return wav.view(1, 1, -1)
    if wav.ndim == 2:
        return wav[:1].unsqueeze(0)
    if wav.ndim == 3:
        if wav.shape[1] != 1:
            raise ValueError(f"audio must be mono, got shape {tuple(wav.shape)}")
        return wav
    raise ValueError(f"waveform must be 1-, 2- or 3-D, got {wav.ndim}-D")


def _resolve_ckpt_dir(model_path: str | Path) -> str:
    """Local dir or HF repo id → local snapshot dir."""
    p = str(model_path)
    if os.path.isdir(p):
        return p
    return snapshot_download(p)


def _load_codec_state_dict(tts_ckpt_dir: str) -> dict[str, torch.Tensor]:
    """Pull all ``tied.embedding.modality_embeddings.0.model.*`` tensors out of
    the TTS checkpoint's shards into a state dict keyed by the codec's own
    parameter names (prefix stripped)."""
    index_path = os.path.join(tts_ckpt_dir, "model.safetensors.index.json")
    if os.path.isfile(index_path):
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
        shards: dict[str, list[str]] = {}
        for full_name, shard in weight_map.items():
            if full_name.startswith(_CODEC_IN_TTS_CKPT_PREFIX):
                shards.setdefault(shard, []).append(full_name)
    else:
        shards = {"model.safetensors": None}  # single-shard layout

    state: dict[str, torch.Tensor] = {}
    for shard, names in shards.items():
        shard_path = os.path.join(tts_ckpt_dir, shard)
        with safe_open(shard_path, framework="pt") as f:
            keys = (
                names
                if names is not None
                else [k for k in f.keys() if k.startswith(_CODEC_IN_TTS_CKPT_PREFIX)]
            )
            for full_name in keys:
                state[full_name[len(_CODEC_IN_TTS_CKPT_PREFIX) :]] = f.get_tensor(
                    full_name
                )
    return state


class HiggsAudioCodec:
    """Frozen encode/decode wrapper around :class:`HiggsAudioV2TokenizerModel`."""

    SAMPLE_RATE: int = 24_000

    def __init__(
        self, model: HiggsAudioV2TokenizerModel, *, device: torch.device
    ) -> None:
        self.model = model
        self.device = device
        self._dtype = next(model.parameters()).dtype

    @classmethod
    def from_pretrained(
        cls,
        model_path: str | Path,
        *,
        device: str | torch.device = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> "HiggsAudioCodec":
        """Load codec from a Higgs TTS checkpoint (local dir or HF repo id).

        Codec weights are bundled inside the TTS ``model.safetensors`` under
        the ``tied.embedding.modality_embeddings.0.model.*`` prefix; we use
        the bundled codec config for the codec architecture and load the codec
        tensors out of the TTS shards directly.

        ``dtype`` defaults to fp32; decode ConvTranspose is unstable in bf16
        — opt in only if you've validated quality at your sample rate.
        """
        device = torch.device(device)
        ckpt_dir = _resolve_ckpt_dir(model_path)

        config = HiggsAudioV2TokenizerConfig.from_json_file(_BUNDLED_CODEC_CONFIG_PATH)
        model = HiggsAudioV2TokenizerModel(config).to(dtype=dtype).eval()

        state = _load_codec_state_dict(ckpt_dir)
        if not state:
            raise FileNotFoundError(
                f"No codec weights found under {_CODEC_IN_TTS_CKPT_PREFIX!r} in "
                f"{ckpt_dir}; this checkpoint doesn't bundle the audio codec."
            )

        missing, unexpected = model.load_state_dict(state, strict=False)
        # Some upstream init keys (e.g. `weight_g`/`weight_v` weight-norm parts)
        # are regenerated by post_init; the substantive codec tensors all load.
        if len(missing) > len(state) // 2:
            raise RuntimeError(
                f"Codec weight load is too sparse: {len(missing)} missing / "
                f"{len(state)} loaded; bundled codec config may be incompatible "
                "with the installed Transformers version."
            )
        model = model.to(device=device)
        for p in model.parameters():
            p.requires_grad_(False)
        return cls(model, device=device)

    @torch.no_grad()
    def encode_reference(
        self,
        waveform: WaveformInput,
        *,
        sample_rate: int | None = None,
    ) -> torch.Tensor:
        """Reference clip → ``[T, num_codebooks]`` int64 codes (CPU).

        Accepts 1-D ``[L]``, 2-D ``[C, L]``, or 3-D ``[1, 1, L]`` float32 in
        ``[-1, 1]``. Resamples to 24 kHz; clips < 1 s are zero-padded since
        the encoder errors otherwise.
        """
        wav = _to_mono_3d(waveform).to(torch.float32)
        sr = sample_rate or self.SAMPLE_RATE
        if sr != self.SAMPLE_RATE:
            wav = torchaudio.functional.resample(wav, sr, self.SAMPLE_RATE)
        if wav.shape[-1] < self.SAMPLE_RATE:
            wav = F.pad(wav, (0, self.SAMPLE_RATE - wav.shape[-1]))

        wav = wav.to(device=self.device, dtype=self._dtype)
        codes_BNT = self.model.encode(wav).audio_codes
        return codes_BNT.squeeze(0).transpose(0, 1).to(torch.long).cpu()

    def _bucketed_batch(
        self,
        items: list[torch.Tensor],
        *,
        bucket_key_fn,
        single_fn,
        batch_fn,
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

    @torch.no_grad()
    def encode_batch(self, waveforms: list[torch.Tensor]) -> list[torch.Tensor]:
        # Offline/bulk encoding utility.
        padded: list[torch.Tensor] = []
        for w in waveforms:
            wav = _to_mono_3d(w).to(torch.float32)
            if wav.shape[-1] < self.SAMPLE_RATE:
                wav = F.pad(wav, (0, self.SAMPLE_RATE - wav.shape[-1]))
            padded.append(wav)

        def _batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
            batch = torch.cat(batch_items).to(device=self.device, dtype=self._dtype)
            codes_BNT = self.model.encode(batch).audio_codes.to(torch.long).cpu()
            return [codes_BNT[j].transpose(0, 1) for j in range(len(batch_items))]

        return self._bucketed_batch(
            padded,
            bucket_key_fn=lambda w: w.shape[-1],
            single_fn=self.encode_reference,
            batch_fn=_batch_fn,
            error_label="encode_batch",
        )

    @torch.no_grad()
    def decode(self, codes_TN: torch.Tensor) -> torch.Tensor:
        """``[T, num_codebooks]`` → mono waveform ``[L]``."""
        if codes_TN.ndim != 2:
            raise ValueError(
                f"codes must be 2-D [T, num_codebooks], got {tuple(codes_TN.shape)}"
            )
        codes_BNT = (
            codes_TN.transpose(0, 1)
            .unsqueeze(0)
            .to(device=self.device, dtype=torch.long)
        )
        return self.model.decode(codes_BNT).audio_values.squeeze(0).squeeze(0).cpu()

    @torch.no_grad()
    def decode_batch(self, codes_list: list[torch.Tensor]) -> list[torch.Tensor]:
        """Batch-decode variable-length ``[T_i, N]`` tensors into ``[L_i]`` waveforms."""

        def _batch_fn(batch_items: list[torch.Tensor]) -> list[torch.Tensor]:
            stacked = torch.stack(batch_items)
            codes_BNT = stacked.transpose(1, 2).to(device=self.device, dtype=torch.long)
            audio = self.model.decode(codes_BNT).audio_values.cpu()
            return [audio[j, 0] for j in range(len(batch_items))]

        return self._bucketed_batch(
            codes_list,
            bucket_key_fn=lambda c: c.shape[0],
            single_fn=self.decode,
            batch_fn=_batch_fn,
            error_label="decode_batch",
        )


__all__ = ["HiggsAudioCodec"]
