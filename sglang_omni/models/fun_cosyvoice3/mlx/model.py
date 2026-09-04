# SPDX-License-Identifier: Apache-2.0
"""Small native-MLX speech-token model used by Fun-CosyVoice3.

The Flow/HiFT stages remain the existing Torch implementation.  This module
owns only the Qwen2 speech-token LLM, which keeps the framework boundary at an
integer token list and avoids importing an external ``mlx-audio`` package.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.qwen2 import ModelArgs, Qwen2Model

logger = logging.getLogger(__name__)

SPEECH_TOKEN_SIZE = 6561
EXTENDED_VOCAB_SIZE = 200
TOTAL_VOCAB_SIZE = SPEECH_TOKEN_SIZE + EXTENDED_VOCAB_SIZE
_MLX_QUANTIZATION_PRESETS: dict[str, tuple[int, int]] = {
    "mlx_q4": (4, 64),
    "mlx_q8": (8, 64),
}


def _qwen2_args(config: dict[str, Any]) -> ModelArgs:
    """Build the fixed 0.5B Qwen2 shape used by Fun-CosyVoice3."""
    # The converted MLX checkpoint carries CosyVoice's flow/HiFT config, not
    # the nested ``CosyVoice-BlankEN`` Qwen2 config.  Keep the architecture
    # explicit and validate any supplied values so a mismatched checkpoint
    # fails at load time instead of producing corrupted speech tokens.
    expected = {
        "hidden_size": 896,
        "intermediate_size": 4864,
        "num_hidden_layers": 24,
        "num_attention_heads": 14,
        "num_key_value_heads": 2,
        "vocab_size": 151936,
    }
    llm_config = config.get("llm", {})
    for name, value in expected.items():
        supplied = llm_config.get(name)
        if supplied is not None and int(supplied) != value:
            raise ValueError(
                f"Fun-CosyVoice3 MLX checkpoint has unsupported {name}={supplied}; "
                f"expected {value} for the 0.5B model"
            )
    return ModelArgs(
        model_type="qwen2",
        hidden_size=expected["hidden_size"],
        intermediate_size=expected["intermediate_size"],
        num_hidden_layers=expected["num_hidden_layers"],
        num_attention_heads=expected["num_attention_heads"],
        num_key_value_heads=expected["num_key_value_heads"],
        rms_norm_eps=1e-6,
        vocab_size=expected["vocab_size"],
        max_position_embeddings=32768,
        rope_theta=1_000_000.0,
        tie_word_embeddings=True,
    )


def _strip_qwen2_prefix(name: str) -> str:
    name = name.removeprefix("qwen2.")
    # Converted checkpoints have either ``qwen2.model.layers.*`` (the
    # mlx-lm wrapper layout) or ``qwen2.layers.*`` (the inner model layout).
    return name.removeprefix("model.")


class CosyVoice3MlxModel(nn.Module):
    """Qwen2 trunk with CosyVoice3's speech embedding and output head."""

    def __init__(
        self,
        backbone: Qwen2Model,
        speech_embedding: mx.array,
        llm_decoder: mx.array,
    ) -> None:
        super().__init__()
        self.model = backbone
        self.speech_embedding = nn.Embedding(
            TOTAL_VOCAB_SIZE, backbone.args.hidden_size
        )
        self.speech_embedding.weight = speech_embedding
        self.llm_decoder = nn.Linear(
            backbone.args.hidden_size, TOTAL_VOCAB_SIZE, bias=False
        )
        self.llm_decoder.weight = llm_decoder

    @property
    def speech_token_size(self) -> int:
        return SPEECH_TOKEN_SIZE

    @property
    def lm_head(self) -> nn.Linear:
        """Alias expected by SGLang's generic MLX runner introspection."""
        return self.llm_decoder

    def build_prompt_embeddings(
        self,
        text_token_ids: list[int],
        prompt_speech_token_ids: list[int],
    ) -> mx.array:
        """Construct ``[SOS, text, TASK, prompt speech]`` embeddings."""
        text_ids = mx.array([text_token_ids], dtype=mx.int32)
        pieces = [
            self.speech_embedding.weight[SPEECH_TOKEN_SIZE + 0][None, None, :],
            self.model.embed_tokens(text_ids),
            self.speech_embedding.weight[SPEECH_TOKEN_SIZE + 2][None, None, :],
        ]
        if prompt_speech_token_ids:
            speech_ids = mx.array([prompt_speech_token_ids], dtype=mx.int32)
            pieces.append(self.speech_embedding(speech_ids))
        return mx.concatenate(pieces, axis=1)

    def forward_embeddings(self, embeddings: mx.array, cache=None) -> mx.array:
        hidden = self.model(
            inputs=None,
            input_embeddings=embeddings,
            cache=cache,
        )
        # Prefill only needs the distribution after the final prompt token.
        # Avoid projecting every text/reference position through the 6,761-way
        # speech head.
        return self.llm_decoder(hidden[:, -1:, :])

    def __call__(self, input_ids: mx.array, cache=None) -> mx.array:
        # Decode ids are speech-code ids (including the extended stop range),
        # not Qwen text-token ids.
        embeddings = self.speech_embedding(input_ids)
        return self.forward_embeddings(embeddings, cache=cache)


def _find_weight(weights: dict[str, mx.array], *names: str) -> mx.array:
    for name in names:
        if name in weights:
            return weights[name]
    raise ValueError(f"Fun-CosyVoice3 MLX checkpoint is missing {names[0]!r}")


def _load_converted_backbone(
    args: ModelArgs,
    config: dict[str, Any],
    weights: dict[str, mx.array],
) -> Qwen2Model:
    backbone = Qwen2Model(args)
    quantization = config.get("quantization")
    if isinstance(quantization, dict):
        bits = int(quantization.get("bits", 4))
        group_size = int(quantization.get("group_size", 64))
        if bits not in (2, 3, 4, 6, 8):
            raise ValueError(f"unsupported MLX quantization bits: {bits}")

        def quantize_layers(path: str, module: Any) -> bool:
            return (
                isinstance(module, nn.Linear)
                and "layers" in path
                and module.weight.shape[-1] % group_size == 0
            )

        nn.quantize(
            backbone,
            bits=bits,
            group_size=group_size,
            class_predicate=quantize_layers,
        )
    backbone_weights = {
        _strip_qwen2_prefix(name): value
        for name, value in weights.items()
        if name.startswith("qwen2.") and not name.endswith("lm_head.weight")
    }
    if not backbone_weights:
        raise ValueError("Fun-CosyVoice3 MLX checkpoint has no qwen2 weights")
    backbone.load_weights(list(backbone_weights.items()))
    return backbone


def _quantize_loaded_backbone(
    backbone: Qwen2Model,
    quantization: str | None,
) -> None:
    if quantization is None:
        return
    try:
        bits, group_size = _MLX_QUANTIZATION_PRESETS[quantization]
    except KeyError as exc:
        raise ValueError(
            "Fun-CosyVoice3 MLX quantization must be one of "
            f"{sorted(_MLX_QUANTIZATION_PRESETS)}, got {quantization!r}"
        ) from exc

    def quantize_layers(path: str, module: Any) -> bool:
        return (
            isinstance(module, nn.Linear)
            and "layers" in path
            and module.weight.shape[-1] % group_size == 0
        )

    nn.quantize(
        backbone,
        bits=bits,
        group_size=group_size,
        class_predicate=quantize_layers,
    )


def _to_mlx_float(tensor: Any, dtype: mx.Dtype) -> mx.array:
    import numpy as np

    # ``bfloat16`` tensors cannot be converted to NumPy directly.
    import torch

    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().to(device="cpu", dtype=torch.float32).numpy()
    return mx.array(np.asarray(tensor, dtype=np.float32)).astype(dtype)


def _load_raw_backbone(
    checkpoint_root: Path,
    *,
    dtype: mx.Dtype,
) -> tuple[ModelArgs, Qwen2Model, mx.array, mx.array]:
    """Convert the fine-tuned Qwen2 trunk and custom heads from ``llm.pt``."""
    import torch

    nested_dir = checkpoint_root / "CosyVoice-BlankEN"
    if not nested_dir.is_dir():
        raise FileNotFoundError(f"Fun-CosyVoice3 checkpoint is missing {nested_dir}")
    llm_path = checkpoint_root / "llm.pt"
    if not llm_path.is_file():
        raise FileNotFoundError(f"Fun-CosyVoice3 checkpoint is missing {llm_path}")

    nested_config = json.loads((nested_dir / "config.json").read_text(encoding="utf-8"))
    args = _qwen2_args({"llm": nested_config})
    state = torch.load(llm_path, map_location="cpu", weights_only=True)
    backbone = Qwen2Model(args)
    backbone_weights = {
        name.removeprefix("llm.model.model."): _to_mlx_float(value, dtype)
        for name, value in state.items()
        if name.startswith("llm.model.model.")
    }
    if not backbone_weights:
        raise ValueError("Fun-CosyVoice3 llm.pt has no fine-tuned Qwen2 weights")
    backbone.load_weights(list(backbone_weights.items()))
    try:
        speech_embedding = _to_mlx_float(state["speech_embedding.weight"], dtype)
        llm_decoder = _to_mlx_float(state["llm_decoder.weight"], dtype)
    except KeyError as exc:
        raise ValueError(
            "Fun-CosyVoice3 llm.pt is missing speech_embedding/llm_decoder weights"
        ) from exc
    del state, backbone_weights
    return args, backbone, speech_embedding, llm_decoder


def load_cosyvoice3_mlx_model(
    model_path: str | Path,
    *,
    dtype: mx.Dtype = mx.float16,
    quantization: str | None = None,
) -> CosyVoice3MlxModel:
    """Load either a converted MLX artifact or an official CosyVoice bundle."""
    model_dir = Path(model_path).expanduser().resolve()
    config_path = model_dir / "config.json"
    weights_path = model_dir / "model.safetensors"
    if weights_path.is_file() and config_path.is_file():
        config = json.loads(config_path.read_text(encoding="utf-8"))
        all_weights = mx.load(str(weights_path))
        args = _qwen2_args(config)
        backbone = _load_converted_backbone(args, config, all_weights)
        speech_embedding = _find_weight(
            all_weights,
            "llm.speech_embedding.weight",
            "speech_embedding.weight",
        )
        llm_decoder = _find_weight(
            all_weights,
            "llm.llm_decoder.weight",
            "llm_decoder.weight",
        )
        # Converted artifacts can include Flow/HiFT/CAMPPlus tensors that this
        # LLM process never uses. Drop those references before an optional
        # on-load quantization creates its packed Qwen weights.
        del all_weights
        if quantization is not None and not isinstance(
            config.get("quantization"), dict
        ):
            _quantize_loaded_backbone(backbone, quantization)
        elif quantization is not None:
            logger.info(
                "Fun-CosyVoice3 MLX artifact is already quantized; ignoring %s",
                quantization,
            )
    else:
        args, backbone, speech_embedding, llm_decoder = _load_raw_backbone(
            model_dir,
            dtype=dtype,
        )
        _quantize_loaded_backbone(backbone, quantization)
    if tuple(speech_embedding.shape) != (TOTAL_VOCAB_SIZE, args.hidden_size):
        raise ValueError(
            "unexpected Fun-CosyVoice3 speech embedding shape: "
            f"{tuple(speech_embedding.shape)}"
        )
    if tuple(llm_decoder.shape) != (TOTAL_VOCAB_SIZE, args.hidden_size):
        raise ValueError(
            "unexpected Fun-CosyVoice3 decoder shape: " f"{tuple(llm_decoder.shape)}"
        )

    speech_embedding = speech_embedding.astype(dtype)
    llm_decoder = llm_decoder.astype(dtype)
    model = CosyVoice3MlxModel(backbone, speech_embedding, llm_decoder)
    mx.eval(model.parameters())
    logger.info("Loaded native MLX Fun-CosyVoice3 0.5B model from %s", model_dir)
    return model


__all__ = [
    "CosyVoice3MlxModel",
    "EXTENDED_VOCAB_SIZE",
    "SPEECH_TOKEN_SIZE",
    "TOTAL_VOCAB_SIZE",
    "load_cosyvoice3_mlx_model",
]
