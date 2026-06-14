# SPDX-License-Identifier: Apache-2.0
"""ZONOS2 audio codec utilities.

Provides DAC (Descript Audio Codec) encoding/decoding at 44.1kHz
for reference audio processing and output audio generation.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)

# DAC model cache
_dac_model = None
_dac_device: str = "cpu"


def load_dac_model(device: str = "cuda:0") -> None:
    """Load and cache the DAC 44kHz model."""
    global _dac_model, _dac_device
    if _dac_model is not None and _dac_device == device:
        return
    try:
        import dac as dac_module

        _dac_model = (
            dac_module.DAC.load(dac_module.utils.download(model_type="44khz"))
            .eval()
            .to(device)
        )
        _dac_device = device
        logger.info("DAC 44kHz model loaded on %s", device)
    except ImportError:
        raise RuntimeError(
            "ZONOS2 TTS requires the 'dac' package. "
            "Install with: pip install descript-audio-codec==1.0.0"
        )


def get_dac_model():
    """Get the cached DAC model."""
    if _dac_model is None:
        load_dac_model()
    return _dac_model


def encode_audio(
    waveform: torch.Tensor,
    sample_rate: int = 44100,
    n_codebooks: int = 9,
) -> torch.Tensor:
    """Encode audio waveform to DAC codes.

    Args:
        waveform: Audio tensor of shape (samples,) or (1, samples)
        sample_rate: Input sample rate (will be resampled to 44.1kHz if needed)
        n_codebooks: Number of codebooks to use

    Returns:
        codes: Tensor of shape (seq_len, n_codebooks)
    """
    import torchaudio

    dac = get_dac_model()
    device = next(dac.parameters()).device

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.dim() == 2:
        waveform = waveform.unsqueeze(0)  # Add batch dim

    # Resample to 44.1kHz if needed
    if sample_rate != 44100:
        resampler = torchaudio.transforms.Resample(sample_rate, 44100).to(device)
        waveform = resampler(waveform.to(device))
    else:
        waveform = waveform.to(device)

    with torch.no_grad(), torch.inference_mode():
        z, codes, _, _, _ = dac.encode(waveform)
        # codes: (batch, n_codebooks, seq_len)
        codes = codes[0, :n_codebooks, :].T  # (seq_len, n_codebooks)

    return codes.cpu()


def decode_audio(
    codes: torch.Tensor,
    n_codebooks: int = 9,
    codebook_size: int = 1024,
) -> torch.Tensor:
    """Decode DAC codes to audio waveform.

    Args:
        codes: Tensor of shape (seq_len, n_codebooks) or (batch, seq_len, n_codebooks)

    Returns:
        waveform: Audio tensor of shape (samples,) at 44.1kHz
    """
    dac = get_dac_model()
    device = next(dac.parameters()).device

    if codes.dim() == 2:
        codes = codes.unsqueeze(0)  # Add batch dim

    # Clamp to valid range
    codes = torch.clamp(codes, max=codebook_size - 1)

    # DAC expects (batch, codebooks, seq_len)
    codes = codes.permute(0, 2, 1).contiguous().to(device=device, dtype=torch.long)

    with torch.no_grad(), torch.inference_mode():
        z = dac.quantizer.from_codes(codes)[0]
        audio = dac.decode(z).float().squeeze(0).squeeze(0).cpu()

    return audio
