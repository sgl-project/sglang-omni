# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with per-request speaker conditioning."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.vocoder import (
    Token2Wav,
    resolve_token2wav_assets,
)
from sglang_omni.models.minicpm_o.payload_types import SpeakerPromptInputs
from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.scheduling.vocoder_base import group_by_padding_waste

FLOW_DTYPES = (torch.float32, torch.float16, torch.bfloat16)

OUTPUT_SAMPLE_RATE = 24000
CODEC_TOKEN_RATE = 25
SAMPLES_PER_CODEC_TOKEN = OUTPUT_SAMPLE_RATE // CODEC_TOKEN_RATE


class MiniCPMOCode2Wav(nn.Module):
    """Convert codec tokens into a float32 waveform with Token2wav."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str = "cuda",
        dtype: str | torch.dtype | None = None,
        n_timesteps: int = 10,
        hift_max_padding_waste: float,
    ) -> None:
        super().__init__()
        if hift_max_padding_waste < 1.0:
            raise ValueError("hift_max_padding_waste must be at least 1.0")
        else:
            pass
        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"Token2wav requires a CUDA device, got {device}")
        else:
            pass
        self.device_context = torch.cuda.device(dev.index or 0)

        asset_dir, _ = resolve_token2wav_assets(model_path)
        if dtype is None:
            torch_dtype = torch.float32
        elif isinstance(dtype, torch.dtype):
            torch_dtype = dtype
        else:
            torch_dtype = resolve_dtype(dtype)
        if torch_dtype not in FLOW_DTYPES:
            raise ValueError(
                f"Code2Wav dtype must be float32, float16, or bfloat16, got {dtype}"
            )
        else:
            pass
        with self.device_context:
            self.token2wav = Token2Wav(
                asset_dir, device=dev, dtype=torch_dtype, n_timesteps=n_timesteps
            )
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.hift_max_padding_waste = hift_max_padding_waste
        self.eval()

    @torch.inference_mode()
    def forward(
        self,
        *,
        codec_tokens: torch.Tensor,
        speaker_prompt: SpeakerPromptInputs,
    ) -> dict[str, object]:
        """Vocode EOS-stripped codec tokens with one request's speaker conditioning."""
        tokens = codec_tokens.reshape(-1).tolist()
        if not tokens:
            waveform = np.zeros(0, dtype=np.float32)
        else:
            with self.device_context:
                waveform = self.vocode([tokens], [speaker_prompt])[0]
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def vocode(
        self,
        token_sequences: Sequence[Sequence[int]],
        speaker_prompts: Sequence[SpeakerPromptInputs],
    ) -> list[np.ndarray]:
        """Batch flow across references and preserve each HiFT sequence boundary."""
        if not token_sequences:
            return []
        else:
            pass
        if any(len(tokens) == 0 for tokens in token_sequences):
            raise ValueError("codec token sequences must be non-empty")
        else:
            pass
        if len(speaker_prompts) != len(token_sequences):
            raise ValueError(
                f"speaker prompt count {len(speaker_prompts)} does not match "
                f"token sequence count {len(token_sequences)}"
            )
        else:
            pass

        batch_size = len(token_sequences)
        token_lens = [len(tokens) for tokens in token_sequences]
        device = self.token2wav.device
        up_rate = self.token2wav.flow.up_rate
        speech_tokens = pad_sequence(
            [
                torch.tensor(tokens, dtype=torch.int32, device=device)
                for tokens in token_sequences
            ],
            batch_first=True,
        )
        speech_tokens_lens = torch.tensor(token_lens, dtype=torch.int32, device=device)
        # Rows pad to a common prompt width; the flow re-derives each row's real
        # width from prompt_speech_tokens_lens.
        prompt_speech_tokens = pad_sequence(
            [prompt["speech_tokens"].reshape(-1) for prompt in speaker_prompts],
            batch_first=True,
        ).to(device)
        prompt_speech_tokens_lens = torch.cat(
            [prompt["speech_token_len"].reshape(1) for prompt in speaker_prompts]
        ).to(device)
        speaker_embedding = torch.cat(
            [prompt["speaker_embedding"] for prompt in speaker_prompts]
        ).to(device)
        # note (liuqihao): prompt mel repeats its last frame to cover every prompt token.
        prompt_mels = pad_sequence(
            [
                torch.nn.functional.pad(
                    prompt["prompt_mel"],
                    (
                        0,
                        0,
                        0,
                        prompt["speech_tokens"].numel() * up_rate
                        - prompt["prompt_mel"].shape[1],
                    ),
                    mode="replicate",
                )[0]
                for prompt in speaker_prompts
            ],
            batch_first=True,
        ).to(device)

        with torch.amp.autocast(
            "cuda",
            dtype=self.token2wav.dtype,
            enabled=self.token2wav.dtype != torch.float32,
        ):
            mel = self.token2wav.flow.inference(
                speech_tokens,
                speech_tokens_lens,
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                prompt_mels,
                speaker_embedding,
                self.token2wav.n_timesteps,
            )

        mel_lens = [token_len * up_rate for token_len in token_lens]
        waveform_rows: dict[int, torch.Tensor] = {}
        for indices in group_by_padding_waste(mel_lens, self.hift_max_padding_waste):
            group_mel_lens = [mel_lens[idx] for idx in indices]
            speech_feat = mel[indices, :, : max(group_mel_lens)].float().contiguous()
            wav, _ = self.token2wav.hift(
                speech_feat=speech_feat, mel_lengths=group_mel_lens
            )
            for row, idx in enumerate(indices):
                waveform_rows[idx] = wav[row].reshape(-1)[
                    : token_lens[idx] * SAMPLES_PER_CODEC_TOKEN
                ]
        wav = (
            pad_sequence(
                [waveform_rows[idx] for idx in range(batch_size)], batch_first=True
            )
            .float()
            .cpu()
        )
        return [
            wav[idx, : token_len * SAMPLES_PER_CODEC_TOKEN].numpy()
            for idx, token_len in enumerate(token_lens)
        ]
