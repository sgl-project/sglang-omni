# SPDX-License-Identifier: Apache-2.0
"""Fixed-address decode inputs and outputs for the EasyMagpie talker.

A captured decode CUDA graph replays against the tensors it saw at capture.
SGLang refreshes its own batch fields before each replay, but EasyMagpie's
conditioning and per-row sampling controls are model inputs SGLang does not
know about, so they live here, at addresses that never change. The graph also
writes its predictions here, where the runner reads them after the step.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from sglang_omni.models.easymagpie_tts.hf_config import EasyMagpieTTSConfig


@dataclass
class EasyMagpieDecodeBuffers:
    conditioning: torch.Tensor
    audio_valid: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor
    codes: torch.Tensor
    phonemes: torch.Tensor
    eos: torch.Tensor
    # Top-k width of the sampling kernels; rows may ask for any k up to it.
    max_top_k: int

    @classmethod
    def allocate(
        cls,
        config: EasyMagpieTTSConfig,
        *,
        max_batch: int,
        max_top_k: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> EasyMagpieDecodeBuffers:
        if max_batch < 1 or max_top_k < 1:
            raise ValueError("decode buffers need a positive batch and top-k width")
        else:
            pass

        def zeros(*shape: int, dtype: torch.dtype = torch.long) -> torch.Tensor:
            return torch.zeros(shape, device=device, dtype=dtype)

        return cls(
            conditioning=zeros(max_batch, config.embedding_dim, dtype=dtype),
            audio_valid=zeros(max_batch, dtype=torch.bool),
            temperatures=torch.ones(max_batch, device=device, dtype=torch.float32),
            top_ks=torch.ones(max_batch, device=device, dtype=torch.long),
            seeds=zeros(max_batch),
            positions=zeros(max_batch),
            codes=zeros(max_batch, config.num_stacked_codebooks),
            phonemes=zeros(max_batch, config.phoneme_stacking_factor),
            eos=zeros(max_batch, dtype=torch.bool),
            max_top_k=min(int(max_top_k), config.codebook_vocab_size),
        )

    @property
    def capacity(self) -> int:
        return int(self.conditioning.shape[0])

    def stage(
        self,
        *,
        conditioning: torch.Tensor,
        audio_valid: torch.Tensor,
        temperatures: torch.Tensor,
        top_ks: torch.Tensor,
        seeds: torch.Tensor,
        positions: torch.Tensor,
    ) -> None:
        """Copy one step's rows in; rows past the batch become inert padding."""
        batch = int(conditioning.shape[0])
        if batch > self.capacity:
            raise ValueError(
                f"decode batch {batch} exceeds the buffer capacity {self.capacity}"
            )
        else:
            pass
        # Padding rows stay valid sampling inputs (temperature 1, k 1) and can
        # never stop a request (audio_valid False).
        for buffer, values, pad in (
            (self.conditioning, conditioning, 0),
            (self.audio_valid, audio_valid, False),
            (self.temperatures, temperatures, 1.0),
            (self.top_ks, top_ks, 1),
            (self.seeds, seeds, 0),
            (self.positions, positions, 0),
        ):
            buffer[:batch].copy_(values, non_blocking=True)
            buffer[batch:].fill_(pad)


__all__ = ["EasyMagpieDecodeBuffers"]
