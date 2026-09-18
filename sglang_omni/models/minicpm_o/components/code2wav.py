# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with a cached speaker reference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

if TYPE_CHECKING:
    from sglang_omni.models.minicpm_o.components.token2wav.vocoder import SpeakerPrompt

OUTPUT_SAMPLE_RATE = 24000


class MiniCPMOCode2Wav(nn.Module):
    """Convert codec tokens into a float32 waveform with Token2wav."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        float16: bool = False,
        n_timesteps: int = 10,
        prompt_wav: str | None = None,
    ) -> None:
        super().__init__()
        from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"Token2wav requires a CUDA device, got {device}")
        self.device_context = torch.cuda.device(dev.index or 0)

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        with self.device_context:
            self.token2wav = Token2Wav(
                Path(asset_dir), device=dev, float16=float16, n_timesteps=n_timesteps
            )

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        self.default_prompt_wav = prompt_wav
        self.prompt_cache_key: str | None = None

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        prompt_wav: str | bytes | None = None,
        **_: object,
    ) -> dict[str, object]:
        """Vocode EOS-stripped codec tokens using the supplied or default reference."""
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            return {
                "waveform": np.zeros(0, dtype=np.float32),
                "sample_rate": OUTPUT_SAMPLE_RATE,
            }
        with self.device_context:
            reference = self.default_prompt_wav if prompt_wav is None else prompt_wav
            waveform = self.vocode(tokens, reference)
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def speaker_prompt(self, prompt_wav: str | bytes | None) -> SpeakerPrompt:
        if prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        prompt_key = (
            f"bytes:{hash_bytes(prompt_wav)}"
            if isinstance(prompt_wav, bytes)
            else reference_path_cache_key(prompt_wav)
        )
        t2w = self.token2wav
        if (
            t2w.cache is None
            or prompt_key is None
            or prompt_key != self.prompt_cache_key
        ):
            if isinstance(prompt_wav, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav") as reference:
                    reference.write(prompt_wav)
                    reference.flush()
                    prompt = t2w.prepare_prompt(reference.name)
            else:
                prompt = t2w.prepare_prompt(prompt_wav)
            t2w.cache = prompt
            self.prompt_cache_key = prompt_key
        return t2w.cache

    def vocode(self, tokens: list[int], prompt_wav: str | bytes | None) -> np.ndarray:
        """Return the waveform directly, avoiding the vocoder's file encoder."""
        t2w = self.token2wav
        (
            prompt_speech_tokens,
            prompt_speech_tokens_lens,
            spk_emb,
            prompt_mels,
        ) = self.speaker_prompt(prompt_wav)

        speech_tokens = torch.tensor([tokens], dtype=torch.int32, device=t2w.device)
        speech_tokens_lens = torch.tensor(
            [speech_tokens.shape[1]], dtype=torch.int32, device=t2w.device
        )
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=t2w.float16):
            mel = t2w.flow.inference(
                speech_tokens,
                speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                spk_emb,
                t2w.n_timesteps,
            )
        # note (MayDomine): HiFT stays FP32 when the flow runs in half precision.
        wav, _ = t2w.hift(speech_feat=mel.float())
        return wav.reshape(-1).float().cpu().numpy()
