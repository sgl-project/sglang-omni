# SPDX-License-Identifier: Apache-2.0
"""Bound offline mel extraction without changing checkpoint normalization."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from transformers import BatchFeature, WhisperFeatureExtractor


class BoundedAudioFeatureExtractor:
    """Retain the checkpoint extractor and omit its unused thirty-second tail."""

    def __init__(self, extractor: WhisperFeatureExtractor) -> None:
        self.extractor = extractor
        self.hop_length = extractor.hop_length

    def __call__(
        self,
        audios: npt.NDArray[np.float32] | list[npt.NDArray[np.float32]],
        *,
        sampling_rate: int,
        return_attention_mask: bool,
        padding: str,
        return_tensors: str,
    ) -> BatchFeature:
        sequences = [audios] if isinstance(audios, np.ndarray) else audios
        if sequences and padding == "max_length" and self.extractor.dither == 0:
            # note (koppx): a full FFT of zeros preserves the original tail boundary.
            maximum_samples = min(
                self.extractor.n_samples,
                max(len(sequence) for sequence in sequences) + self.extractor.n_fft,
            )
        else:
            maximum_samples = self.extractor.n_samples
        return self.extractor(
            audios,
            sampling_rate=sampling_rate,
            return_attention_mask=return_attention_mask,
            padding=padding,
            return_tensors=return_tensors,
            max_length=maximum_samples,
        )
