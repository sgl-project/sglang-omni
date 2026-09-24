# SPDX-License-Identifier: Apache-2.0
"""Vocode MiniCPM-o codec tokens with a cached speaker reference."""

from __future__ import annotations

import os
import tempfile
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from sglang_omni.models.minicpm_o.components.token2wav.flow_cuda_graph import (
    FlowCudaGraphRunner,
)
from sglang_omni.models.weight_loader import resolve_dtype, resolve_model_path
from sglang_omni.preprocessing.cache_key import hash_bytes, reference_path_cache_key

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
        prompt_wav: str | None = None,
        flow_cuda_graph_capture_shapes: tuple[tuple[int, int], ...] = (),
        flow_cuda_graph_frame_bucket: int | None = None,
    ) -> None:
        super().__init__()
        if flow_cuda_graph_capture_shapes and flow_cuda_graph_frame_bucket is None:
            raise ValueError(
                "flow_cuda_graph_frame_bucket is required with capture shapes"
            )
        else:
            pass
        from sglang_omni.models.minicpm_o.components.token2wav.vocoder import Token2Wav

        dev = torch.device(device)
        if dev.type != "cuda":
            raise ValueError(f"Token2wav requires a CUDA device, got {device}")
        else:
            pass
        self.device_context = torch.cuda.device(dev.index or 0)

        model_dir = str(resolve_model_path(model_path))
        asset_dir = os.path.join(model_dir, "assets", "token2wav")
        if not os.path.isdir(asset_dir):
            raise FileNotFoundError(
                f"token2wav assets not found at {asset_dir}; copy the "
                "checkpoint's assets/token2wav directory next to the weights"
            )
        else:
            pass
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
                Path(asset_dir), device=dev, dtype=torch_dtype, n_timesteps=n_timesteps
            )
            if flow_cuda_graph_capture_shapes:
                assert flow_cuda_graph_frame_bucket is not None
                runner = FlowCudaGraphRunner(
                    self.token2wav.flow.decoder,
                    capture_shapes=flow_cuda_graph_capture_shapes,
                    frame_bucket=flow_cuda_graph_frame_bucket,
                    n_timesteps=n_timesteps,
                    # note (Codex): Encoder LayerNorm keeps conditioning in FP32 under autocast.
                    conditioning_dtype=torch.float32,
                )
                runner.capture_all()
                self.token2wav.flow.decoder.cuda_graph_runner = runner
            else:
                pass

        if prompt_wav is None:
            default_wav = os.path.join(model_dir, "assets", "HT_ref_audio.wav")
            prompt_wav = default_wav if os.path.isfile(default_wav) else None
        else:
            pass
        self.default_prompt_wav = prompt_wav
        self.prompt_cache_key: str | None = None
        self.sample_rate = OUTPUT_SAMPLE_RATE
        self.eval()

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
            waveform = np.zeros(0, dtype=np.float32)
        else:
            with self.device_context:
                reference = self.resolve_prompt_wav(prompt_wav)
                waveform = self.vocode([tokens], reference)[0]
        return {"waveform": waveform, "sample_rate": OUTPUT_SAMPLE_RATE}

    def resolve_prompt_wav(self, prompt_wav: str | bytes | None) -> str | bytes:
        if prompt_wav is not None:
            resolved = prompt_wav
        elif self.default_prompt_wav is None:
            raise ValueError("No speaker-reference audio supplied or default available")
        else:
            resolved = self.default_prompt_wav
        return resolved

    def speaker_prompt(
        self, prompt_wav: str | bytes | None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        prompt_wav = self.resolve_prompt_wav(prompt_wav)
        prompt_key = (
            f"bytes:{hash_bytes(prompt_wav)}"
            if isinstance(prompt_wav, bytes)
            else reference_path_cache_key(prompt_wav)
        )
        if (
            self.token2wav.cache is None
            or prompt_key is None
            or prompt_key != self.prompt_cache_key
        ):
            if isinstance(prompt_wav, bytes):
                with tempfile.NamedTemporaryFile(suffix=".wav") as reference:
                    reference.write(prompt_wav)
                    reference.flush()
                    prompt = self.token2wav.prepare_prompt(reference.name)
            else:
                prompt = self.token2wav.prepare_prompt(prompt_wav)
            self.token2wav.cache = prompt
            self.prompt_cache_key = prompt_key
        else:
            pass
        return self.token2wav.cache

    def vocode(
        self,
        token_sequences: Sequence[Sequence[int]],
        prompt_wav: str | bytes | None,
    ) -> list[np.ndarray]:
        """Vocode a prompt-homogeneous batch of codec-token sequences."""
        if not token_sequences:
            waveforms: list[np.ndarray] = []
        elif any(len(tokens) == 0 for tokens in token_sequences):
            raise ValueError("codec token sequences must be non-empty")
        else:
            (
                prompt_speech_tokens,
                prompt_speech_tokens_lens,
                speaker_embedding,
                prompt_mels,
            ) = self.speaker_prompt(prompt_wav)
            batch_size = len(token_sequences)
            token_lens = [len(tokens) for tokens in token_sequences]
            speech_tokens = pad_sequence(
                [
                    torch.tensor(
                        tokens, dtype=torch.int32, device=self.token2wav.device
                    )
                    for tokens in token_sequences
                ],
                batch_first=True,
            )
            speech_tokens_lens = torch.tensor(
                token_lens, dtype=torch.int32, device=self.token2wav.device
            )
            prompt_speech_tokens = prompt_speech_tokens.expand(
                batch_size, -1
            ).contiguous()
            prompt_speech_tokens_lens = prompt_speech_tokens_lens.expand(
                batch_size
            ).contiguous()
            speaker_embedding = speaker_embedding.expand(batch_size, -1).contiguous()
            prompt_mels = prompt_mels.expand(batch_size, -1, -1).contiguous()
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
            length_groups: dict[int, list[int]] = defaultdict(list)
            for idx, token_len in enumerate(token_lens):
                length_groups[token_len * self.token2wav.flow.up_rate].append(idx)
            waveforms_by_index: dict[int, np.ndarray] = {}
            for mel_len, indices in length_groups.items():
                speech_feat = torch.stack(
                    [mel[idx, :, :mel_len] for idx in indices],
                    dim=0,
                ).float()
                # note (MayDomine): HiFT stays FP32 when the flow runs in half precision.
                wav, _ = self.token2wav.hift(speech_feat=speech_feat)
                wav = wav.float().cpu()
                for local_idx, batch_idx in enumerate(indices):
                    n_samples = token_lens[batch_idx] * SAMPLES_PER_CODEC_TOKEN
                    waveforms_by_index[batch_idx] = (
                        wav[local_idx].reshape(-1)[:n_samples].numpy()
                    )
            waveforms = [waveforms_by_index[idx] for idx in range(batch_size)]
        return waveforms
