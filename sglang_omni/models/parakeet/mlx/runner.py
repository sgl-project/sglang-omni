# SPDX-License-Identifier: Apache-2.0
"""Batched greedy Parakeet inference on native MLX.

Reads the same Transformers-format checkpoint as the Torch MPS path. Log-mel
features still come from the checkpoint's Transformers feature extractor, so
both backends see identical model inputs.
"""

from __future__ import annotations

import glob
import json
import logging
import os
from collections.abc import Sequence

import mlx.core as mx
import numpy as np

from sglang_omni.models.parakeet.mlx.model import (
    ParakeetMlxConfig,
    ParakeetModel,
    torch_layout_to_mlx,
)
from sglang_omni.models.parakeet.model_runner import WARMUP_SECONDS, pad_to_min_samples

logger = logging.getLogger(__name__)

MLX_DTYPES = {
    "float32": mx.float32,
    "fp32": mx.float32,
    "bfloat16": mx.bfloat16,
    "bf16": mx.bfloat16,
    "float16": mx.float16,
    "fp16": mx.float16,
}
# Everything a Transformers-format Parakeet repository needs; skips the .nemo
# and GGUF copies that some repositories also ship.
CHECKPOINT_PATTERNS = ["*.json", "*.safetensors"]


def resolve_checkpoint_dir(model_path: str) -> str:
    if os.path.isdir(model_path):
        return model_path
    else:
        from huggingface_hub import snapshot_download

        return snapshot_download(model_path, allow_patterns=CHECKPOINT_PATTERNS)


def resolve_mlx_dtype(dtype: str) -> mx.Dtype:
    try:
        return MLX_DTYPES[dtype.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype string: {dtype}") from exc


def load_parakeet_mlx_model(checkpoint_dir: str, dtype: mx.Dtype) -> ParakeetModel:
    with open(os.path.join(checkpoint_dir, "config.json")) as reader:
        config = ParakeetMlxConfig.from_dict(json.load(reader))
    model = ParakeetModel(config)
    weights: dict[str, mx.array] = {}
    for weight_file in sorted(glob.glob(os.path.join(checkpoint_dir, "*.safetensors"))):
        weights.update(mx.load(weight_file))
    model.load_weights(list(torch_layout_to_mlx(weights).items()), strict=True)
    model.set_dtype(dtype)
    model.eval()
    mx.eval(model.parameters())
    return model


class ParakeetMlxModelRunner:
    """MLX counterpart of ``ParakeetModelRunner`` with the same interface."""

    def __init__(
        self,
        model_path: str,
        *,
        dtype: str = "float32",
        warmup: bool = True,
    ) -> None:
        from transformers import AutoProcessor

        checkpoint_dir = resolve_checkpoint_dir(model_path)
        self.dtype = resolve_mlx_dtype(dtype)
        self.model = load_parakeet_mlx_model(checkpoint_dir, self.dtype)
        self.architecture = self.model.config.architecture
        self.processor = AutoProcessor.from_pretrained(checkpoint_dir)
        feature_extractor = self.processor.feature_extractor
        self.sample_rate = int(feature_extractor.sampling_rate)
        self.min_samples = int(feature_extractor.n_fft)
        logger.info(
            "Loaded Parakeet %s from %s on MLX (%s)",
            self.architecture,
            model_path,
            self.dtype,
        )
        if warmup:
            self.transcribe(
                [np.zeros(int(WARMUP_SECONDS * self.sample_rate), dtype=np.float32)]
            )
        else:
            pass

    def transcribe(self, waveforms: Sequence[np.ndarray]) -> list[str]:
        """Transcribe mono waveforms at ``sample_rate``; one string per input."""
        if not waveforms:
            return []
        else:
            pass
        features = self.processor(
            [pad_to_min_samples(waveform, self.min_samples) for waveform in waveforms],
            sampling_rate=self.sample_rate,
            return_tensors="np",
        )
        input_features = mx.array(features["input_features"]).astype(self.dtype)
        # A lone utterance has no padding, so it skips the masks entirely.
        lengths = (
            mx.array(features["attention_mask"].sum(-1).astype(np.int32))
            if len(waveforms) > 1
            else None
        )
        token_ids = self.model.greedy_decode(input_features, lengths)
        texts = self.processor.batch_decode(token_ids, skip_special_tokens=True)
        return [text.strip() for text in texts]


__all__ = ["ParakeetMlxModelRunner", "load_parakeet_mlx_model"]
