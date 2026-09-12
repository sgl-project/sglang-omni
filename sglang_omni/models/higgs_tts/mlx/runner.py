# SPDX-License-Identifier: Apache-2.0
"""Public MLX worker factory and Higgs model initialization."""
from __future__ import annotations

from pathlib import Path

import torch

from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
from sglang_omni.models.higgs_tts.mlx.model import load_mlx_language_model


def make_higgs_mlx_runner_class():
    """Factory for Omni's public MLX worker registry."""
    return HiggsMlxWorkerModel


class HiggsMlxWorkerModel:
    """Native model owner; the Higgs scheduler runner owns generation hooks."""

    def __init__(
        self,
        *,
        model_path,
        pool_size=4096,
        dtype="bfloat16",
        quantization=None,
        enable_sampling=False,
        revision=None,
        disable_radix_cache=True,
        **kwargs,
    ):
        if dtype not in ("bfloat16", "bf16", torch.bfloat16):
            raise ValueError("Higgs MLX requires dtype='bfloat16'")

        from safetensors import safe_open

        from sglang_omni.models.higgs_tts.model import HiggsTTSModel
        from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper

        if not disable_radix_cache:
            raise ValueError("Higgs MLX requires disabled radix cache")
        if quantization is not None or enable_sampling:
            raise ValueError(
                "Higgs MLX requires unquantized weights and its own sampler"
            )
        if not Path(model_path).is_dir():
            from huggingface_hub import snapshot_download

            model_path = snapshot_download(model_path, revision=revision)
        self.request_caches = {}
        self.pool_size = pool_size
        config = HiggsMultimodalQwen3Config.from_pretrained(model_path)
        text_config = config.get_text_config()
        with torch.device("mps"):
            backbone = torch.nn.Module()
            backbone.config = text_config
            backbone.model = torch.nn.Module()
            backbone.model.embed_tokens = torch.nn.Embedding(
                text_config.vocab_size, text_config.hidden_size, dtype=torch.bfloat16
            )
            model = HiggsTTSModel(config, backbone=backbone)
        mapper = DiscreteWeightMapper(
            text_prefix_map={
                "tied.embedding.text_embedding.": "backbone.model.embed_tokens."
            },
            tie_modality=model._tie_modality,
        )
        state = {}
        expected = set(dict(model.named_parameters(remove_duplicate=False)))
        for path in sorted(Path(model_path).glob("*.safetensors")):
            with safe_open(path, framework="pt", device="cpu") as weights:
                for key in weights.keys():
                    name = mapper.map(key)
                    if name in expected:
                        if name in state:
                            raise ValueError(f"Duplicate Higgs weight: {name}")
                        state[name] = weights.get_tensor(key)
        if model._tie_modality:
            state["modality_head.weight"] = state[
                "multimodal_embedding.modality_embedding_0.weight"
            ]
        # Persistent sampler buffers are initialized, never loaded from checkpoint.
        missing = expected - state.keys()
        if missing:
            raise ValueError(
                f"Missing Higgs audio/embedding weights: {sorted(missing)}"
            )
        with torch.no_grad():
            for name, param in model.named_parameters(remove_duplicate=False):
                if param.shape != state[name].shape:
                    raise ValueError(f"Higgs weight shape mismatch: {name}")
                param.copy_(state[name])
        model.mlx_language_model = load_mlx_language_model(model_path)
        self.scheduler_model = model.eval()

    def has_request(self, request_id):
        return request_id in self.request_caches

    def remove_request(self, request_id):
        self.request_caches.pop(request_id, None)
        self.scheduler_model.reset_request(request_id)

    def store_auxiliary_state_for_request(self, request_id):
        # Prefix/radix reuse is disabled; nothing may survive request completion.
        pass
