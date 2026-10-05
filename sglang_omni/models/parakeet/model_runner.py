# SPDX-License-Identifier: Apache-2.0
"""Batched greedy Parakeet inference on the Hugging Face implementation.

Parakeet has no language-model decoder, so it runs outside the SGLang engine:
one FastConformer encoder pass per batch, then the checkpoint's own greedy
CTC, RNN-T, or TDT decode. The serving stage runs it on Apple MPS only.
"""

from __future__ import annotations

import logging
import math
from collections.abc import Sequence

import numpy as np
import torch

from sglang_omni.models.weight_loader import resolve_dtype

logger = logging.getLogger(__name__)

PARAKEET_ARCHITECTURES = ("ParakeetForCTC", "ParakeetForRNNT", "ParakeetForTDT")
# One second of silence: long enough for the subsampling stack and for the MPS
# kernels the real requests hit to be built before the first request arrives.
WARMUP_SECONDS = 1.0


def pad_to_min_samples(waveform: np.ndarray, min_samples: int) -> np.ndarray:
    """Zero-pad clips shorter than one STFT window; silence decodes to nothing."""
    if waveform.shape[0] >= min_samples:
        return waveform
    else:
        return np.pad(waveform, (0, min_samples - waveform.shape[0]))


def encoder_output_lengths(
    feature_lengths: torch.Tensor, encoder_config
) -> torch.Tensor:
    """Frame count after the strided subsampling convolutions."""
    kernel = encoder_config.subsampling_conv_kernel_size
    padding = (kernel - 1) // 2
    lengths = feature_lengths.to(torch.long)
    for _ in range(int(math.log2(encoder_config.subsampling_factor))):
        lengths = (
            lengths + 2 * padding - kernel
        ) // encoder_config.subsampling_conv_stride + 1
    return lengths


def drop_tokens_past_valid_frames(
    sequences: torch.Tensor,
    durations: torch.Tensor,
    valid_frames: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Blank out transducer steps that start on a padding frame.

    Batched Transformers RNN-T/TDT ``generate`` keeps decoding a shorter
    utterance until the longest one finishes, reading its padding frames; the
    per-step durations locate every step, so those emissions can be removed.
    """
    starts = durations.cumsum(dim=-1) - durations
    return sequences.masked_fill(starts >= valid_frames[:, None], pad_token_id)


def resolve_parakeet_architecture(architectures: Sequence[str] | None) -> str:
    for architecture in architectures or ():
        if architecture in PARAKEET_ARCHITECTURES:
            return architecture
        else:
            pass
    raise ValueError(
        f"Parakeet ASR supports {list(PARAKEET_ARCHITECTURES)} checkpoints in "
        f"Hugging Face format, got architectures={list(architectures or ())}"
    )


class ParakeetModelRunner:
    """Load one Parakeet checkpoint and transcribe padded batches greedily."""

    def __init__(
        self,
        model_path: str,
        *,
        device: str,
        dtype: str = "float32",
        warmup: bool = True,
    ) -> None:
        import transformers
        from transformers import AutoConfig, AutoProcessor

        config = AutoConfig.from_pretrained(model_path)
        self.architecture = resolve_parakeet_architecture(config.architectures)
        self.device = torch.device(device)
        self.dtype = resolve_dtype(dtype)
        self.processor = AutoProcessor.from_pretrained(model_path)
        feature_extractor = self.processor.feature_extractor
        self.sample_rate = int(feature_extractor.sampling_rate)
        # The STFT needs one full window of samples to produce a frame.
        self.min_samples = int(feature_extractor.n_fft)
        model_cls = getattr(transformers, self.architecture)
        self.model = (
            model_cls.from_pretrained(model_path, dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(
            "Loaded Parakeet %s from %s on %s (%s)",
            self.architecture,
            model_path,
            self.device,
            self.dtype,
        )
        if warmup:
            self.transcribe(
                [np.zeros(int(WARMUP_SECONDS * self.sample_rate), dtype=np.float32)]
            )
        else:
            pass

    def pad_short_waveform(self, waveform: np.ndarray) -> np.ndarray:
        return pad_to_min_samples(waveform, self.min_samples)

    @torch.inference_mode()
    def transcribe(self, waveforms: Sequence[np.ndarray]) -> list[str]:
        """Transcribe mono waveforms at ``sample_rate``; one string per input."""
        if not waveforms:
            return []
        else:
            pass
        features = self.processor(
            [self.pad_short_waveform(waveform) for waveform in waveforms],
            sampling_rate=self.sample_rate,
            return_tensors="pt",
        )
        output = self.model.generate(
            input_features=features.input_features.to(
                device=self.device, dtype=self.dtype
            ),
            attention_mask=features.attention_mask.to(device=self.device),
        )
        # CTC returns token ids directly; the transducers return a ModelOutput.
        if isinstance(output, torch.Tensor):
            sequences = output
        else:
            sequences = drop_tokens_past_valid_frames(
                output.sequences,
                output.durations,
                encoder_output_lengths(
                    features.attention_mask.sum(-1), self.model.config.encoder_config
                ).to(output.sequences.device),
                self.model.config.pad_token_id,
            )
        texts = self.processor.batch_decode(sequences.cpu(), skip_special_tokens=True)
        return [text.strip() for text in texts]


__all__ = [
    "PARAKEET_ARCHITECTURES",
    "WARMUP_SECONDS",
    "ParakeetModelRunner",
    "drop_tokens_past_valid_frames",
    "encoder_output_lengths",
    "pad_to_min_samples",
    "resolve_parakeet_architecture",
]
