# SPDX-License-Identifier: Apache-2.0
"""Code2Wav component for MiniCPM-o.

Wraps stepaudio2's ``Token2wav`` (the remote code's ``init_tts`` dependency)
to turn s3tokenizer codec tokens into a 24 kHz waveform. The vocoder assets
live in the checkpoint directory under ``assets/token2wav``; the default
prompt (speaker reference) wav is ``assets/HT_ref_audio.wav`` when present,
matching the remote demo's default voice.
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch
import torch.nn as nn

from sglang_omni.models.weight_loader import resolve_model_path

logger = logging.getLogger(__name__)

OUTPUT_SAMPLE_RATE = 24000


class MiniCPMOCode2Wav(nn.Module):
    """stepaudio2 Token2wav wrapper: codec tokens → float32 waveform."""

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
        del device  # Token2wav manages its own device placement (cuda).
        try:
            from stepaudio2 import Token2wav
        except ImportError as exc:
            raise ImportError(
                "MiniCPM-o audio output requires stepaudio2; install via "
                "pip install minicpmo-utils[all]"
            ) from exc

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        self.token2wav = Token2wav(asset_dir, float16=float16, n_timesteps=n_timesteps)

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        self._prompt_wav = prompt_wav

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        prompt_wav: str | None = None,
        **_: object,
    ) -> dict[str, object]:
        """Vocode one utterance.

        Args:
            codec_tokens: ``(N,)`` s3tokenizer codes (EOS already stripped).
            prompt_wav: optional path to a 16 kHz speaker-reference wav;
                falls back to the component default.

        Returns:
            ``waveform``: ``(samples,)`` float32 at 24 kHz; ``sample_rate``.
        """
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            return {
                "waveform": np.zeros(0, dtype=np.float32),
                "sample_rate": OUTPUT_SAMPLE_RATE,
            }
        waveform = self._vocode(tokens, prompt_wav or self._prompt_wav)
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def _vocode(self, tokens: list[int], prompt_wav: str | None) -> np.ndarray:
        """``Token2wav.__call__`` minus its final ``torchaudio.save`` — newer
        torchaudio (torchcodec backend) cannot encode into ``BytesIO``, and we
        want the raw waveform anyway."""
        t2w = self.token2wav
        if t2w.cache is None:
            t2w.cache = t2w._prepare_prompt(prompt_wav)
        (
            prompt_speech_tokens,
            prompt_speech_tokens_lens,
            spk_emb,
            prompt_mels,
            prompt_mels_lens,
        ) = t2w.cache

        speech_tokens = torch.tensor([tokens], dtype=torch.int32, device="cuda")
        speech_tokens_lens = torch.tensor(
            [speech_tokens.shape[1]], dtype=torch.int32, device="cuda"
        )
        with torch.amp.autocast(
            "cuda", dtype=torch.float16 if t2w.float16 else torch.float32
        ):
            mel = t2w.flow.inference(
                speech_tokens,
                speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                prompt_mels_lens,
                spk_emb,
                t2w.n_timesteps,
            )
        wav, _ = t2w.hift(speech_feat=mel)
        return wav.reshape(-1).float().cpu().numpy()
