# SPDX-License-Identifier: MIT
# Copyright (c) 2024 Prince Canuma and contributors.
# Derived from Blaizzy/mlx-audio CosyVoice3 PR #861 (commit 5272f213f8cc).
# Based on FunAudioLLM/CosyVoice (Apache-2.0, Copyright 2024-2025 Alibaba Inc).
# Modified for the non-streaming sglang-omni vocoder contract.

"""Loader and stable stage-facing API for the native MLX vocoder."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from .config import VocoderConfig
from .flow import CausalMaskedDiffWithDiT
from .hift import CausalHiFTGenerator

_MLX_DTYPES = {
    "bfloat16": mx.bfloat16,
    "float16": mx.float16,
    "float32": mx.float32,
}


def _normalize_dtype_name(value: Any) -> str:
    name = str(value).lower().removeprefix("torch.").removeprefix("mlx.core.")
    if name not in _MLX_DTYPES:
        raise ValueError(
            f"unsupported Fun-CosyVoice3 MLX artifact dtype {value!r}; "
            f"expected one of {sorted(_MLX_DTYPES)}"
        )
    return name


def _resolve_model_directory(model_path: str, revision: str | None) -> Path:
    local_path = Path(model_path).expanduser()
    if local_path.is_dir():
        return local_path.resolve()
    from sglang.srt.hardware_backend.mlx.remote_code_gate import resolve_model_directory

    return Path(resolve_model_directory(model_path, revision=revision))


def _map_flow_weight(name: str) -> str:
    """Map the public mlx-community artifact onto the PR #861 module tree."""
    name = name.removeprefix("flow.")
    replacements = (
        (".time_embed.time_mlp_0.", ".time_embed.time_mlp.0."),
        (".time_embed.time_mlp_2.", ".time_embed.time_mlp.1."),
        (".attn.to_out_0.", ".attn.to_out.0."),
        (".ff.ff_0_0.", ".ff.ff.0."),
        (".ff.ff_2.", ".ff.ff.1."),
    )
    for old, new in replacements:
        name = name.replace(old, new)
    return name


def _map_hift_weight(name: str) -> str:
    """Map wrapped-convolution names onto the PR #861 HiFT module tree."""
    name = name.removeprefix("hifigan.")
    name = name.removeprefix("hift.")
    for source_index, target_index in zip((0, 2, 4, 6, 8), range(5), strict=True):
        name = name.replace(
            f"f0_predictor.condnet_{source_index}.conv.",
            f"f0_predictor.condnet.{target_index}.",
        )
    return name.replace(".conv.weight", ".weight").replace(".conv.bias", ".bias")


def _as_batch(
    value: Any,
    *,
    name: str,
    dtype: mx.Dtype,
    feature_size: int | None = None,
) -> mx.array:
    array = value if isinstance(value, mx.array) else mx.array(value)
    if feature_size is None:
        if array.ndim == 1:
            array = array[None, :]
        if array.ndim != 2 or array.shape[0] != 1:
            raise ValueError(f"{name} must have shape [T] or [1, T], got {array.shape}")
    else:
        expected_rank = 2 if name == "embedding" else 3
        if name == "embedding" and array.ndim == 1:
            array = array[None, :]
        elif name != "embedding" and array.ndim == 2:
            array = array[None, :, :]
        if array.ndim != expected_rank or array.shape[0] != 1:
            raise ValueError(f"{name} must have batch size 1, got {array.shape}")
        if array.shape[-1] != feature_size:
            raise ValueError(
                f"{name} must have feature size {feature_size}, got {array.shape}"
            )
    return array.astype(dtype)


class FunCosyVoice3MlxVocoder:
    """Non-streaming batch-one Flow + HiFT implementation for Omni stages."""

    def __init__(
        self,
        *,
        flow: CausalMaskedDiffWithDiT,
        hift: CausalHiFTGenerator,
        config: VocoderConfig,
        dtype: mx.Dtype,
    ) -> None:
        self.flow = flow
        self.hift = hift
        self.config = config
        self.dtype = dtype

    @property
    def sample_rate(self) -> int:
        return self.config.hift.sampling_rate

    @property
    def token_mel_ratio(self) -> int:
        return self.config.flow.token_mel_ratio

    @classmethod
    def from_pretrained(
        cls,
        model_path: str,
        revision: str | None = None,
        *,
        expected_dtype: str | None = None,
    ) -> "FunCosyVoice3MlxVocoder":
        """Load sanitized Flow/HiFT weights from a converted MLX artifact.

        The official Fun-CosyVoice3 bundle contains PyTorch ``flow.pt`` and
        ``hift.pt`` and is intentionally not converted here. It should remain
        Omni's main checkpoint for ONNX preprocessing assets; this loader gets
        the separately converted MLX artifact.
        """
        model_dir = _resolve_model_directory(model_path, revision)
        config_path = model_dir / "config.json"
        weights_path = model_dir / "model.safetensors"
        if not config_path.is_file() or not weights_path.is_file():
            raise FileNotFoundError(
                "Fun-CosyVoice3 native MLX vocoder requires a converted artifact "
                f"with config.json and model.safetensors: {model_dir}"
            )

        raw_config = json.loads(config_path.read_text(encoding="utf-8"))
        dtype_name = _normalize_dtype_name(raw_config.get("dtype", "float16"))
        if expected_dtype is not None:
            requested_dtype = _normalize_dtype_name(expected_dtype)
            if requested_dtype != dtype_name:
                raise ValueError(
                    "Fun-CosyVoice3 native MLX vocoder dtype is owned by the "
                    f"converted artifact ({dtype_name}), but factory dtype "
                    f"requested {requested_dtype}"
                )
        dtype = _MLX_DTYPES[dtype_name]
        config = VocoderConfig.from_dict(raw_config)
        all_weights = mx.load(str(weights_path))

        flow_weights = {
            _map_flow_weight(name): value
            for name, value in all_weights.items()
            if name.startswith("flow.") and "rotary_embed.inv_freq" not in name
        }
        hift_prefix = (
            "hifigan."
            if any(name.startswith("hifigan.") for name in all_weights)
            else "hift."
        )
        raw_hift_names = [name for name in all_weights if name.startswith(hift_prefix)]
        if any(
            ".parametrizations.weight.original" in name
            or name.endswith(".weight_g")
            or name.endswith(".weight_v")
            for name in raw_hift_names
        ):
            raise ValueError(
                "Fun-CosyVoice3 MLX vocoder does not accept raw unsanitized "
                "HiFT weights; convert/fold weight normalization first"
            )
        hift_weights = {
            _map_hift_weight(name): value
            for name, value in all_weights.items()
            if name.startswith(hift_prefix)
        }
        if not flow_weights or not hift_weights:
            raise ValueError(
                "Fun-CosyVoice3 MLX artifact must contain flow.* and sanitized "
                "hifigan.* or hift.* weights"
            )

        flow = CausalMaskedDiffWithDiT(config.flow)
        hift = CausalHiFTGenerator(config.hift)
        flow.load_weights(list(flow_weights.items()), strict=True)
        hift.load_weights(list(hift_weights.items()), strict=True)
        del all_weights, flow_weights, hift_weights

        # These runtime buffers use underscore names and are therefore not in
        # ``parameters()``. Materialize them on the construction thread so a
        # scheduler thread does not inherit a lazy graph tied to stream 0.
        mx.eval(
            flow.parameters(),
            hift.parameters(),
            flow.decoder._rand_noise,
            hift._stft_window,
            hift.m_source.l_sin_gen._rand_ini,
        )
        return cls(flow=flow, hift=hift, config=config, dtype=dtype)

    def decode_mx(
        self,
        *,
        token: Any,
        prompt_token: Any,
        prompt_feat: Any,
        embedding: Any,
    ) -> mx.array:
        """Decode one request to a rank-one MLX waveform."""
        token = _as_batch(token, name="token", dtype=mx.int32)
        prompt_token = _as_batch(
            prompt_token,
            name="prompt_token",
            dtype=mx.int32,
        )
        prompt_feat = _as_batch(
            prompt_feat,
            name="prompt_feat",
            dtype=self.dtype,
            feature_size=self.config.flow.output_size,
        )
        embedding = _as_batch(
            embedding,
            name="embedding",
            dtype=self.dtype,
            feature_size=self.config.flow.spk_embed_dim,
        )
        if token.shape[1] == 0:
            raise ValueError("token must contain at least one generated speech token")
        expected_prompt_frames = prompt_token.shape[1] * self.token_mel_ratio
        if prompt_feat.shape[1] != expected_prompt_frames:
            raise ValueError(
                "prompt_feat must contain token_mel_ratio frames per prompt token "
                f"({prompt_feat.shape[1]} != {expected_prompt_frames})"
            )

        token_len = mx.array([token.shape[1]], dtype=mx.int32)
        prompt_token_len = mx.array([prompt_token.shape[1]], dtype=mx.int32)
        prompt_feat_len = mx.array([prompt_feat.shape[1]], dtype=mx.int32)
        mel = self.flow.inference(
            token=token,
            token_len=token_len,
            prompt_token=prompt_token,
            prompt_token_len=prompt_token_len,
            prompt_feat=prompt_feat,
            prompt_feat_len=prompt_feat_len,
            embedding=embedding,
            n_timesteps=self.config.flow.n_timesteps,
        )
        waveform, _ = self.hift.inference(mel)
        waveform = waveform.reshape(-1).astype(mx.float32)
        mx.eval(waveform)
        return waveform

    def decode(
        self,
        *,
        token: Any,
        prompt_token: Any,
        prompt_feat: Any,
        embedding: Any,
    ) -> np.ndarray:
        """Decode one request and return a contiguous float32 NumPy waveform."""
        waveform = np.asarray(
            self.decode_mx(
                token=token,
                prompt_token=prompt_token,
                prompt_feat=prompt_feat,
                embedding=embedding,
            ),
            dtype=np.float32,
        )
        return np.ascontiguousarray(waveform)
