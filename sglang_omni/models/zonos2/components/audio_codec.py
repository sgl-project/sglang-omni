# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 DAC vocoder: de-shear delayed 9-codebook frames and DAC-decode to 44.1 kHz PCM.

Non-streaming path; mirrors the reference zonos2/tokenizer/vocoder.py decode_all.
"""

from __future__ import annotations

import torch

from ..payload_types import N_CODEBOOKS, ZONOS2_SAMPLE_RATE

DAC_HOP_LENGTH = 512
# eoa_id=1024, audio_pad_id=1025 are sentinels, not real codes; clamp >=1024 before decode.
_MAX_VALID_CODE = 1023
_AUDIO_PAD_ID = 1025

_dac_cache: tuple[str, torch.nn.Module] | None = None


def _get_dac(device: str):
    """Lazily load and cache the DAC 44 kHz model on ``device``."""
    global _dac_cache
    if _dac_cache is None or _dac_cache[0] != device:
        import dac as dac_module

        dac_model = (
            dac_module.DAC.load(dac_module.utils.download(model_type="44khz"))
            .eval()
            .to(device)
        )
        # note (Yue Yin): the DAC convs use legacy weight_norm (78 modules) that
        # recomputes g*v/||v|| every forward. For inference the reparam is fixed,
        # so fold it into the conv weight once at load -> ~2e-4 fp-reassoc on the
        # output (WER-neutral, verified), fewer ops per (streaming) vocoder call.
        for _m in dac_model.modules():
            try:
                torch.nn.utils.remove_weight_norm(_m)
            except (ValueError, RuntimeError):
                # Modules without weight normalization raise here.
                pass
        _dac_cache = (device, dac_model)
    return _dac_cache[1]


def shear_up(codes: torch.Tensor, pad_id: int = _AUDIO_PAD_ID) -> torch.Tensor:
    """Remove the AR delay pattern: column ``j`` is shifted up by ``j`` rows.

    Inverse of ``tts/prompt.py::shear``. The trailing ``W - 1`` rows become a
    flush region filled with ``pad_id`` and are dropped before decoding.
    """
    H, W = codes.shape[-2:]
    out = codes.new_full(codes.shape, pad_id)
    for j in range(W):
        if H > j:
            out[..., : H - j, j] = codes[..., j:, j]
    return out


class Zonos2DACVocoder:
    """Decode delayed 9-codebook ZONOS2 frames into mono float32 PCM @ 44.1 kHz."""

    def __init__(self, device: str = "cuda"):
        self.device = device
        self.n_codebooks = N_CODEBOOKS
        self.sample_rate = ZONOS2_SAMPLE_RATE
        self.hop_length = DAC_HOP_LENGTH
        self.audio_pad_id = _AUDIO_PAD_ID
        self._dac = _get_dac(device)

    @torch.inference_mode()
    def decode(
        self,
        audio_codes: torch.Tensor,
        eos_frame: int | None = None,
    ) -> torch.Tensor:
        """Decode delayed per-frame codes ``[T, 9]`` (or ``[B, T, 9]``) to a 1-D
        float32 PCM tensor at 44.1 kHz on CPU. ``eos_frame`` caps the number of
        aligned frames kept before EOS.
        """
        codes = torch.as_tensor(audio_codes, dtype=torch.long, device=self.device)
        if codes.ndim == 2:
            codes = codes.unsqueeze(0)
        elif codes.ndim != 3:
            raise ValueError(
                f"audio_codes must be [T, 9] or [B, T, 9], got shape {tuple(codes.shape)}"
            )

        codes = shear_up(codes, self.audio_pad_id)

        # Trailing (n_codebooks - 1) rows are the flush region.
        valid = codes.shape[1] - (self.n_codebooks - 1)
        if eos_frame is not None:
            valid = min(valid, max(0, int(eos_frame)))

        if valid <= 0:
            return torch.zeros(0, dtype=torch.float32)
        codes = codes[:, :valid, :]

        codes = torch.clamp(codes, max=_MAX_VALID_CODE)

        # DAC expects (batch, codebooks, seq).
        codes = codes.permute(0, 2, 1).contiguous()

        # float32: bf16 ConvTranspose is numerically unstable.
        z = self._dac.quantizer.from_codes(codes)[0]
        audio = self._dac.decode(z).float().squeeze(1).cpu()

        return audio[0].contiguous()

    @torch.inference_mode()
    def decode_batch(
        self,
        audio_codes_list: list[torch.Tensor],
        eos_frames: list[int | None],
    ) -> list[torch.Tensor]:
        """Decode many delayed ``[T_i, 9]`` code tensors in ONE DAC forward.

        Returns one 1-D float32 CPU PCM tensor per input, trimmed to its own
        valid frame count (empty when the item has no aligned frames). Each item
        is sheared and trimmed individually (the delay pattern is per-item),
        then right-padded to the longest valid length before the shared decode.
        """
        if len(audio_codes_list) != len(eos_frames):
            raise ValueError("audio_codes_list and eos_frames length mismatch")
        results: list[torch.Tensor] = [
            torch.zeros(0, dtype=torch.float32) for _ in audio_codes_list
        ]
        sheared: list[torch.Tensor] = []
        valids: list[int] = []
        keep: list[int] = []
        for i, (item, eos_frame) in enumerate(zip(audio_codes_list, eos_frames)):
            codes = torch.as_tensor(item, dtype=torch.long, device=self.device)
            if codes.ndim != 2:
                raise ValueError(
                    f"each audio_codes entry must be [T, 9], got shape "
                    f"{tuple(codes.shape)}"
                )
            codes = shear_up(codes, self.audio_pad_id)
            valid = codes.shape[0] - (self.n_codebooks - 1)
            if eos_frame is not None:
                valid = min(valid, max(0, int(eos_frame)))
            if valid <= 0:
                continue
            sheared.append(codes[:valid, :])
            valids.append(valid)
            keep.append(i)

        if not keep:
            return results

        t_max = max(valids)
        batch = torch.full(
            (len(keep), t_max, self.n_codebooks),
            self.audio_pad_id,
            dtype=torch.long,
            device=self.device,
        )
        for j, codes in enumerate(sheared):
            batch[j, : codes.shape[0], :] = codes
        batch = torch.clamp(batch, max=_MAX_VALID_CODE)

        # DAC expects (batch, codebooks, seq); float32 for stable ConvTranspose.
        batch = batch.permute(0, 2, 1).contiguous()
        z = self._dac.quantizer.from_codes(batch)[0]
        audio = self._dac.decode(z).float().squeeze(1).cpu()

        for j, i in enumerate(keep):
            results[i] = audio[j, : valids[j] * self.hop_length].contiguous()
        return results
