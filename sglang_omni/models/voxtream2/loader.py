# SPDX-License-Identifier: MIT
# Copyright (c) 2025 Nikita Torgashov
# Construction and graph-safe cache operations adapted from herimor/voxtream
# commit 8ec2d62159dae4716ae7058827244a962d40603c. See LICENSE-MIT.voxtream.
"""Model construction; import only in the configured model process."""

from __future__ import annotations

import json
import logging
from dataclasses import fields

import torch
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

from sglang_omni.models.voxtream2.assets import resolve_assets
from sglang_omni.models.voxtream2.cache import bind_graph_safe_caches


def load_generator(
    *,
    model_path: str,
    mimi_checkpoint: str,
    speaker_checkpoint: str,
    generator_config_path: str,
    speaking_rate_config_path: str,
    device: str,
    dtype: str,
    use_cuda_graph: bool,
) -> SpeechGenerator:
    """Construct a generator inside the configured model process."""
    assets = resolve_assets(
        model_path=model_path,
        mimi_checkpoint=mimi_checkpoint,
        speaker_checkpoint=speaker_checkpoint,
        generator_config_path=generator_config_path,
        speaking_rate_config_path=speaking_rate_config_path,
        device=device,
        dtype=dtype,
        use_cuda_graph=use_cuda_graph,
    )
    target = assets.device
    paths = assets.paths
    config_data = assets.generator_config
    rate_config = assets.speaking_rate_config
    if target.type == "cuda":
        torch.cuda.set_device(target)
        if dtype == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("This CUDA device does not support bfloat16; use float16")
        else:
            pass
    else:
        pass

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

    def mimi() -> torch.nn.Module:
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
