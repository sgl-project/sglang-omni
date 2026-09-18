# SPDX-License-Identifier: Apache-2.0
"""MLX backend for the shared model-owned Nemotron transcription stage."""

from __future__ import annotations

import threading
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.models.weight_loader import resolve_dtype
from sglang_omni.utils.checkpoint import resolve_checkpoint

from ..hf_compat import Nemotron3_5AsrConfig, Nemotron3_5AsrProcessor
from ..model_runner import Nemotron3_5ASRModelRunner
from .model import Model, sanitize_weights


class Nemotron3_5ASRMLXRunner(Nemotron3_5ASRModelRunner):
    """Reuse audio preparation, locale prompts, output metadata and lifecycle.

    The CPU processor computes log-mel features. All neural network operations,
    including the predictor LSTM and joint network, execute in MLX on Metal.
    """

    def __init__(
        self, model_path, *, device="mps", dtype="float32", num_lookahead_tokens=3
    ):
        if resolve_dtype(dtype) != torch.float32:
            raise ValueError("Nemotron MLX currently requires dtype=float32")
        if not str(device).startswith("mps") or not mx.metal.is_available():
            raise RuntimeError("Nemotron MLX requires Apple Metal")
        checkpoint = Path(resolve_checkpoint(model_path)).resolve()
        self.processor = Nemotron3_5AsrProcessor.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.processor.set_num_lookahead_tokens(num_lookahead_tokens)
        config = Nemotron3_5AsrConfig.from_pretrained(checkpoint, local_files_only=True)
        self.model = Model(config)
        files = sorted(checkpoint.glob("*.safetensors"))
        if not files:
            raise ValueError(
                "Nemotron MLX requires official-format safetensors; convert NeMo weights first"
            )
        weights = {}
        for file in files:
            weights.update(mx.load(str(file)))
        self.model.load_weights(list(sanitize_weights(weights).items()), strict=True)
        self.model.eval()
        mx.eval(self.model.parameters())
        self.model_lock = threading.Lock()

    def _generate_sequences(self, processor_inputs, *, max_new_tokens):
        def array(name):
            return mx.array(np.asarray(processor_inputs[name]))

        features = array("input_features")
        lengths = array("attention_mask").sum(axis=-1)
        prompt_ids = array("prompt_ids")
        encoded, lengths = self.model.encode(
            features, lengths, prompt_ids, int(processor_inputs["num_lookahead_tokens"])
        )
        return [
            self.model.decode(encoded[i], int(lengths[i].item()), max_new_tokens)
            for i in range(encoded.shape[0])
        ]

    def close(self):
        with self.model_lock:
            super().close()
            mx.clear_cache()
