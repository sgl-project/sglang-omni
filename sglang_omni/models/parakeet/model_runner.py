# SPDX-License-Identifier: Apache-2.0
"""Batched greedy Parakeet inference on the Hugging Face implementation.

Parakeet has no language-model decoder, so it runs outside the SGLang engine:
one FastConformer encoder pass per batch, then the checkpoint's own greedy
CTC, RNN-T, or TDT decode. The same code serves CUDA, Apple MPS, and CPU.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence

import numpy as np
import torch

from sglang_omni.models.weight_loader import resolve_dtype

logger = logging.getLogger(__name__)

PARAKEET_ARCHITECTURES = ("ParakeetForCTC", "ParakeetForRNNT", "ParakeetForTDT")
# One second of silence: long enough for the subsampling stack and for the MPS
# kernels the real requests hit to be built before the first request arrives.
WARMUP_SECONDS = 1.0


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
        """Zero-pad clips shorter than one STFT window; silence decodes to nothing."""
        if waveform.shape[0] >= self.min_samples:
            return waveform
        else:
            return np.pad(waveform, (0, self.min_samples - waveform.shape[0]))

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
        sequences = output if isinstance(output, torch.Tensor) else output.sequences
        texts = self.processor.batch_decode(sequences.cpu(), skip_special_tokens=True)
        return [text.strip() for text in texts]


__all__ = [
    "PARAKEET_ARCHITECTURES",
    "ParakeetModelRunner",
    "resolve_parakeet_architecture",
]
