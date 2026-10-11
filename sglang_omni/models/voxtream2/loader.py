# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Nikita Torgashov
# Construction and graph-safe cache operations adapted from herimor/voxtream
# commit 8ec2d62159dae4716ae7058827244a962d40603c. See LICENSE-MIT.voxtream.
"""Explicit checkpoint/device loading for the author VoXtream2 implementation.

No hub-download replacement, global dtype override, or global KVCache method
replacement is used. Optional dependencies are imported only while loading.
"""

from __future__ import annotations

import json
import logging
from dataclasses import fields
from pathlib import Path
from types import MethodType

import torch


def bind_graph_safe_caches(model):
    """Adapt only this model's KV caches; other torchtune users stay untouched."""
    from torchtune.modules.kv_cache import KVCache

    for cache in model.modules():
        if not isinstance(cache, KVCache) or getattr(cache, "omni_graph_safe", False):
            continue
        else:
            pass
        original_update, original_reset = cache.update, cache.reset

        def update(self, keys, values, _original=original_update):
            if not self.k_cache.is_cuda or not torch.cuda.is_current_stream_capturing():
                return _original(keys, values)
            else:
                pass
            batch, _, length, _ = keys.shape
            if batch > self.k_cache.shape[0]:
                raise ValueError("KV cache batch capacity exceeded")
            else:
                pass
            self.k_cache[:, :, self.cache_pos[:length]] = keys
            self.v_cache[:, :, self.cache_pos[:length]] = values
            self.cache_pos += length
            return self.k_cache, self.v_cache

        def reset(self, _original=original_reset):
            if not self.k_cache.is_cuda or not torch.cuda.is_current_stream_capturing():
                return _original()
            else:
                pass
            self.k_cache.zero_()
            self.v_cache.zero_()
            self.cache_pos.copy_(self.cache_pos - self.cache_pos[0])

        cache.update = MethodType(update, cache)
        cache.reset = MethodType(reset, cache)
        cache.omni_graph_safe = True


def load_generator(
    *,
    model_path: str,
    mimi_checkpoint: str,
    speaker_checkpoint: str,
    generator_config_path: str,
    speaking_rate_config_path: str,
    device: str,
    dtype: str = "float16",
    use_cuda_graph: bool = True,
):
    """Load local weights using the author's model and generation loop.

    Generator/rate JSON files are explicit operator configuration. The author
    source dependency must be installed separately; importing this module is
    safe when that optional dependency is unavailable.
    """
    if dtype not in {"float16", "bfloat16", "float32"}:
        raise ValueError("dtype must be float16, bfloat16 or float32")
    else:
        pass
    target = torch.device(device)
    if target.type not in {"cpu", "cuda"}:
        raise ValueError("VoXtream2 supports CPU or CUDA devices")
    else:
        pass
    if target.type == "cpu" and (dtype != "float32" or use_cuda_graph):
        raise ValueError("CPU loading requires float32 and use_cuda_graph=False")
    else:
        pass
    if target.type == "cuda":
        torch.cuda.set_device(target)
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16; use float16")
        else:
            pass
    else:
        pass

    config_data = json.loads(Path(generator_config_path).read_text())
    rate_config = json.loads(Path(speaking_rate_config_path).read_text())
    directory = Path(model_path)
    paths = {
        "model config": directory / config_data["model_config_name"],
        "phoneme dictionary": directory / config_data["phoneme_dict_name"],
        "model weights": directory / config_data["model_name"],
        "Mimi weights": Path(mimi_checkpoint),
        "speaker weights": Path(speaker_checkpoint),
    }
    for name, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"Missing {name}: {path}")
        else:
            pass

    from moshi.models import loaders
    from moshi.utils.compile import CUDAGraphed
    from redimnet.model import ReDimNetWrap
    from safetensors.torch import load_file
    from voxtream.config import SpeechGeneratorConfig
    from voxtream.generator import SpeechGenerator
    from voxtream.model import Model, ModelConfig
    from voxtream.utils.generator.context import GenerationContext
    from voxtream.utils.generator.helpers import autocast_ctx, configure_cpu_threads
    from voxtream.utils.text.phonemizer import ESpeak

    config = SpeechGeneratorConfig(**config_data)
    # This first deployment has an operator-owned reference. Disable optional
    # enhancement/VAD and pickle-based prompt caches, as in the tested baseline.
    config.enhance_prompt = config.apply_vad = config.cache_prompt = False
    tensor_dtype = getattr(torch, dtype)
    batch_size = 1 if config.cfg_gamma is None else 2
    values = json.loads(paths["model config"].read_text())
    valid = {field.name for field in fields(ModelConfig)}
    model = Model(
        ModelConfig(**{key: value for key, value in values.items() if key in valid})
    )
    model.load_state_dict(load_file(str(paths["model weights"])), strict=True)
    model = model.eval().to(target, dtype=tensor_dtype)
    model.setup_caches(max_batch_size=batch_size, dtype=tensor_dtype)

    def mimi():
        return (
            loaders.get_mimi(
                filename=str(paths["Mimi weights"]),
                device=target,
                num_codebooks=config.num_codebooks,
            )
            .eval()
            .to(dtype=tensor_dtype)
        )

    speaker_data = torch.load(
        paths["speaker weights"], map_location="cpu", weights_only=True
    )
    speaker = ReDimNetWrap(**speaker_data["model_config"])
    speaker.load_state_dict(speaker_data["state_dict"], strict=True)
    speaker = speaker.to(target, dtype=tensor_dtype).eval()
    speaker.spec.float()
    speaker.bn.float()

    # The upstream constructor has no component injection or dtype parameter.
    # Populate its version-specific generation context without hub loaders.
    generator = SpeechGenerator.__new__(SpeechGenerator)
    generator.config = config
    generator.logger = logging.getLogger("voxtream")
    generator.spk_rate_config = rate_config
    generator.model = model
    generator.mimi = mimi()
    generator.ctx = GenerationContext(
        config=config,
        logger=generator.logger,
        device=target.type,
        dtype=tensor_dtype,
        batch_size=batch_size,
        extract_phone_embeddings=model.extract_phoneme_embeddings,
        phonemizer=ESpeak(),
        phone_to_token=json.loads(paths["phoneme dictionary"].read_text()),
        phoneme_index_map=config.phoneme_index_map,
        mimi_prompt=mimi(),
        spk_enc=speaker,
        sidon_se=None,
        vad=None,
    )
    generator._autocast_ctx = autocast_ctx(
        device=target.type, dtype=tensor_dtype
    )  # noqa: leading-underscore - pinned author model contract
    if use_cuda_graph:
        bind_graph_safe_caches(model)
        model._temp_former = CUDAGraphed(
            model._temp_former
        )  # noqa: leading-underscore - pinned author model contract
        model._dep_former_init = CUDAGraphed(
            model._dep_former
        )  # noqa: leading-underscore - pinned author model contract
        model._dep_former = CUDAGraphed(
            model._dep_former
        )  # noqa: leading-underscore - pinned author model contract
    elif target.type == "cpu":
        configure_cpu_threads()
    else:
        pass
    generator._mimi_stream_ctx = (
        None  # noqa: leading-underscore - pinned author model contract
    )
    generator._mimi_streaming_started = (
        False  # noqa: leading-underscore - pinned author model contract
    )
    return generator
