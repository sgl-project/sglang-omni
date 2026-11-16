# SPDX-License-Identifier: Apache-2.0
"""Trim zero-only STFT frames, with a runtime-checked Whisper reference fallback."""

from __future__ import annotations

import logging

import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from transformers.feature_extraction_utils import FeatureExtractionMixin

logger = logging.getLogger(__name__)

# Frames per STFT bucket; smaller spans reach small-matrix kernels whose
# rounding differs from the full-window computation.
FRAME_BUCKET = 64
LOG_FLOOR = -10.0
DYNAMIC_RANGE = 8.0


class TrimmedWhisperFbank:
    """Exact fast path for single-window Whisper log-mel extraction."""

    def __init__(self, feature_extractor: WhisperFeatureExtractor) -> None:
        self.feature_extractor = feature_extractor
        self.n_fft = int(feature_extractor.n_fft)
        self.hop_length = int(feature_extractor.hop_length)
        self.n_samples = int(feature_extractor.n_samples)
        self.n_frames = self.n_samples // self.hop_length
        self.mel_filters = torch.from_numpy(feature_extractor.mel_filters).to(
            torch.float32
        )
        self.window = torch.hann_window(self.n_fft)
        self.enabled = False

    @classmethod
    def verified(
        cls, feature_extractor: FeatureExtractionMixin
    ) -> "TrimmedWhisperFbank | None":
        """A self-checked instance for a Whisper extractor, else None."""
        if not isinstance(feature_extractor, WhisperFeatureExtractor):
            return None
        else:
            pass
        fbank = cls(feature_extractor)
        return fbank if fbank.self_check() else None

    def supports(self, num_samples: int) -> bool:
        return self.enabled and 0 < num_samples <= self.n_samples

    def reference(self, audio: np.ndarray) -> np.ndarray:
        """The feature extractor's output for one zero-padded window."""
        chunk = np.pad(audio, (0, self.n_samples - audio.shape[0]))
        return self.feature_extractor(
            [chunk],
            sampling_rate=int(self.feature_extractor.sampling_rate),
            padding="max_length",
            return_tensors="np",
        )["input_features"][0]

    def compute(self, audio: np.ndarray) -> np.ndarray:
        """Log-mel features [num_mel_bins, n_frames] for audio of at most one window."""
        half_window = self.n_fft // 2
        last_real_frame = (audio.shape[0] - 1 + half_window) // self.hop_length + 1
        needed_frames = min(self.n_frames, last_real_frame + 1)
        frames = min(self.n_frames, -(-needed_frames // FRAME_BUCKET) * FRAME_BUCKET)
        span_samples = min(
            self.n_samples, (frames - 1) * self.hop_length + half_window + 1
        )
        waveform = np.zeros(max(span_samples, self.n_fft), dtype=np.float32)
        waveform[: audio.shape[0]] = audio
        stft = torch.stft(
            torch.from_numpy(waveform),
            self.n_fft,
            self.hop_length,
            window=self.window,
            return_complex=True,
        )[..., :frames]
        log_spec = torch.clamp(
            self.mel_filters.T @ (stft.abs() ** 2), min=1e-10
        ).log10()
        features = torch.full(
            (self.mel_filters.shape[1], self.n_frames), LOG_FLOOR, dtype=torch.float32
        )
        features[:, :frames] = log_spec
        features = torch.maximum(features, features.max() - DYNAMIC_RANGE)
        return ((features + 4.0) / 4.0).numpy()

    def self_check(self) -> bool:
        """Enable the fast path only if it matches the extractor bit for bit."""
        extractor = self.feature_extractor
        if (
            float(extractor.dither) != 0.0
            or self.n_samples % self.hop_length != 0
            or self.mel_filters.shape[1] != int(extractor.feature_size)
        ):
            logger.info("MOSS-TD fast log-mel disabled: unsupported extractor config")
            return False
        else:
            pass
        rng = np.random.default_rng(0)
        lengths = [1, self.hop_length, self.n_fft, self.n_samples]
        lengths += [
            max(1, frames * self.hop_length - self.n_fft // 2 - offset)
            for frames in range(FRAME_BUCKET, self.n_frames + 1, FRAME_BUCKET)
            for offset in (0, self.hop_length // 2)
        ]
        for num_samples in lengths:
            audio = (rng.standard_normal(num_samples) * 0.3).astype(np.float32)
            if not np.array_equal(self.compute(audio), self.reference(audio)):
                logger.warning(
                    f"MOSS-TD fast log-mel disabled: mismatch at {num_samples} samples"
                )
                return False
            else:
                pass
        self.enabled = True
        logger.info(f"MOSS-TD fast log-mel enabled ({len(lengths)} lengths verified)")
        return True
