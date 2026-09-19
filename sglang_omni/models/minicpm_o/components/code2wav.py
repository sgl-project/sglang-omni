# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with a cached speaker reference."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import numpy as np
import torch
import torch.nn as nn

from sglang_omni.models.weight_loader import resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

if TYPE_CHECKING:
    from sglang_omni.models.minicpm_o.components.token2wav.vocoder import SpeakerPrompt

OUTPUT_SAMPLE_RATE = 24000
CODEC_TOKEN_RATE = 25


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
        self.sample_rate = OUTPUT_SAMPLE_RATE

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
            reference = self.resolve_prompt_wav(prompt_wav)
            waveform = self.vocode(tokens, reference)
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def resolve_prompt_wav(self, prompt_wav: str | bytes | None) -> str | bytes:
        if prompt_wav is not None:
            return prompt_wav
        if self.default_prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        return self.default_prompt_wav

    def speaker_prompt(self, prompt_wav: str | bytes | None) -> SpeakerPrompt:
        t2w = self.token2wav
        prompt_wav = self.resolve_prompt_wav(prompt_wav)
        prompt_key = (
            f"bytes:{hash_bytes(prompt_wav)}"
            if isinstance(prompt_wav, bytes)
            else reference_path_cache_key(prompt_wav)
        )
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
        return self.vocode_many([tokens], prompt_wav)[0]

    def vocode_many(
        self, token_batches: Sequence[list[int]], prompt_wav: str | bytes | None
    ) -> list[np.ndarray]:
        """Vocode a prompt-homogeneous batch of codec-token sequences."""
        if not token_batches:
            return []
        if any(not tokens for tokens in token_batches):
            return [
                (
                    np.zeros(0, dtype=np.float32)
                    if not tokens
                    else self.vocode(tokens, prompt_wav)
                )
                for tokens in token_batches
            ]

        t2w = self.token2wav
        (
            prompt_speech_tokens,
            prompt_speech_tokens_lens,
            spk_emb,
            prompt_mels,
        ) = self.speaker_prompt(prompt_wav)

        batch_size = len(token_batches)
        token_lens = [len(tokens) for tokens in token_batches]
        max_token_len = max(token_lens)
        speech_tokens = torch.zeros(
            (batch_size, max_token_len), dtype=torch.int32, device=t2w.device
        )
        for i, tokens in enumerate(token_batches):
            speech_tokens[i, : len(tokens)] = torch.tensor(
                tokens, dtype=torch.int32, device=t2w.device
            )
        speech_tokens_lens = torch.tensor(
            token_lens, dtype=torch.int32, device=t2w.device
        )
        prompt_speech_tokens = prompt_speech_tokens.expand(batch_size, -1).contiguous()
        prompt_speech_tokens_lens = prompt_speech_tokens_lens.expand(
            batch_size
        ).contiguous()
        spk_emb = spk_emb.expand(batch_size, -1).contiguous()
        prompt_mels = prompt_mels.expand(batch_size, -1, -1).contiguous()
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
        samples_per_token = OUTPUT_SAMPLE_RATE // CODEC_TOKEN_RATE
        mel_lens = [length * t2w.flow.up_rate for length in token_lens]
        outputs: list[np.ndarray | None] = [None] * batch_size
        for mel_len in sorted(set(mel_lens)):
            indices = [idx for idx, length in enumerate(mel_lens) if length == mel_len]
            speech_feat = torch.stack(
                [mel[idx, :, :mel_len] for idx in indices],
                dim=0,
            ).float()
            # note (MayDomine): HiFT stays FP32 when the flow runs in half precision.
            wav, _ = t2w.hift(speech_feat=speech_feat)
            wav = wav.float().cpu()
            for local_idx, batch_idx in enumerate(indices):
                outputs[batch_idx] = (
                    wav[local_idx]
                    .reshape(-1)[: token_lens[batch_idx] * samples_per_token]
                    .numpy()
                )
        return [output for output in outputs if output is not None]
