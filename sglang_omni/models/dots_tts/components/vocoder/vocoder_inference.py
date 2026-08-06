from __future__ import annotations

import torch
import torch.nn as nn

from sglang_omni.models.dots_tts.components.vocoder.bigvgan import AudioVAE


class VocoderInference:
    """AudioVAE encode/decode entry points used by the dots.tts stages."""

    def __init__(self, vocoder: nn.Module):
        if not isinstance(vocoder, AudioVAE):
            raise TypeError(
                f"dots.tts vocoder must be an AudioVAE, got {type(vocoder)!r}"
            )
        self.vocoder = vocoder

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def extract_latents(self, x: torch.Tensor) -> torch.Tensor:
        """Posterior ``[batch, 2 * latent_dim, frames]`` for a waveform batch."""
        x = self.vocoder.audio_encoder(x.float())
        x = self.vocoder.enc_mi_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.vocoder.pre_proj(x)

    @torch.no_grad()
    @torch.autocast(enabled=False, device_type="cuda")
    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        """Waveform for denormalized latents shaped ``[batch, frames, latent_dim]``."""
        x = latents.transpose(1, 2).float()
        latent_dim = int(self.vocoder.h.latent_dim)
        if x.size(1) != latent_dim:
            raise ValueError(
                f"AudioVAE latent channel count must be {latent_dim}, got {x.size(1)}"
            )
        x = self.vocoder.post_proj(x)
        x = self.vocoder.dec_mi_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        return self.vocoder.decoder(x)


__all__ = ["VocoderInference"]
