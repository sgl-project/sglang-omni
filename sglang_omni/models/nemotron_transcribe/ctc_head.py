# SPDX-License-Identifier: Apache-2.0
"""Bundled CTC timestamp head.

The consolidated checkpoint stores this head under ``ctc_timestamp.`` and
records its config in the safetensors metadata. Those tensors are not part of
the language model. A head is usable only when ``load_state_dict`` matches
strictly. Anything else leaves word timestamps unavailable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch
from torch import nn

from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerBlock,
)

logger = logging.getLogger(__name__)

HEAD_FORMAT = "nemo_ctc_timestamp_artifact_v1"
DECODER_PREFIX = "ctc_timestamp.decoder."
PROTO_KEY = "ctc_timestamp.tokenizer.model_proto"


class CtcTimestampHead(nn.Module):
    """Transformer bridge plus the 1x1 conv that emits blank-augmented logits."""

    def __init__(
        self,
        *,
        feat_in: int,
        class_count: int,
        vocabulary: list[str],
        n_layers: int,
        n_heads: int,
        ff_expansion: float,
        use_transformer: bool,
        qkv_bias: bool,
        qk_norm: bool,
        with_final_norm: bool,
        with_out_proj: bool,
        with_embed_norm: bool,
        with_residual_scale: bool,
        apply_residual: bool,
        tokenizer_proto: bytes,
    ) -> None:
        super().__init__()
        self.vocabulary = list(vocabulary)
        self.blank_id = len(self.vocabulary)
        self.use_transformer = use_transformer
        self.apply_residual = apply_residual
        self.tokenizer_proto = bytes(tokenizer_proto)
        self.usable = False
        self.decoder_layers = nn.Sequential(
            nn.Conv1d(feat_in, class_count, kernel_size=1, bias=True)
        )
        if with_residual_scale:
            self.residual_scale = nn.Parameter(torch.zeros(()))
        else:
            self.register_parameter("residual_scale", None)
        if use_transformer:
            self.transformer = CtcBridge(
                feat_in=feat_in,
                n_layers=n_layers,
                n_heads=n_heads,
                ff_expansion=ff_expansion,
                qkv_bias=qkv_bias,
                qk_norm=qk_norm,
                with_final_norm=with_final_norm,
                with_out_proj=with_out_proj,
                with_embed_norm=with_embed_norm,
            )
        else:
            self.transformer = None

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        """``encoded`` is ``(batch, width, time)``. Returns log-probs ``(batch, time, classes)``."""
        if not self.usable:
            raise RuntimeError("CTC timestamp head is not usable")
        else:
            pass
        hidden = encoded
        if self.transformer is not None:
            bridged = self.transformer(encoded.transpose(1, 2))
            if self.apply_residual and self.residual_scale is not None:
                hidden = encoded + self.residual_scale.to(dtype=bridged.dtype) * bridged
            else:
                hidden = bridged
        else:
            pass
        logits = self.decoder_layers(
            hidden.to(dtype=self.decoder_layers[0].weight.dtype)
        )
        return torch.log_softmax(logits, dim=1).transpose(1, 2)


class CtcBridge(nn.Module):
    """Pre-norm stack matching the checkpoint names under ``transformer.``."""

    def __init__(
        self,
        *,
        feat_in: int,
        n_layers: int,
        n_heads: int,
        ff_expansion: float,
        qkv_bias: bool,
        qk_norm: bool,
        with_final_norm: bool,
        with_out_proj: bool,
        with_embed_norm: bool,
    ) -> None:
        super().__init__()
        self.embed_norm = nn.LayerNorm(feat_in) if with_embed_norm else nn.Identity()
        self.layers = nn.ModuleList(
            [
                SpeechTransformerBlock(
                    feat_in,
                    n_heads,
                    ff_expansion,
                    drop_rate=0.0,
                    qkv_bias=qkv_bias,
                    qk_norm=qk_norm,
                    rope=None,
                )
                for _ in range(n_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(feat_in) if with_final_norm else nn.Identity()
        self.out_proj = nn.Linear(feat_in, feat_in) if with_out_proj else None

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` is ``(batch, time, width)``. Returns ``(batch, width, time)``."""
        hidden = self.embed_norm(hidden)
        for layer in self.layers:
            hidden = layer(hidden, None)
        hidden = self.final_norm(hidden)
        if self.out_proj is not None:
            hidden = self.out_proj(hidden)
        else:
            pass
        return hidden.transpose(1, 2)


def load_ctc_head(checkpoint_dir: str | Path) -> CtcTimestampHead | None:
    """Read the bundled head. ``None`` means the checkpoint has no head metadata."""
    directory = Path(checkpoint_dir)
    files = sorted(directory.glob("*.safetensors"))
    if not files:
        return None
    else:
        pass
    try:
        from safetensors import safe_open
    except ImportError:
        logger.warning("safetensors is not installed; word timestamps are unavailable")
        return None
    else:
        pass

    metadata: dict[str, str] | None = None
    weights: dict[str, torch.Tensor] = {}
    for path in files:
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            file_metadata = handle.metadata() or {}
            if file_metadata.get("ctc_timestamp_format") == HEAD_FORMAT:
                metadata = file_metadata
            else:
                pass
            for name in handle.keys():
                if name.startswith("ctc_timestamp."):
                    weights[name] = handle.get_tensor(name)
                else:
                    pass
    if metadata is None:
        return None
    else:
        pass
    try:
        config = json.loads(metadata["ctc_timestamp_decoder_config"])
    except (KeyError, json.JSONDecodeError, TypeError) as exc:
        logger.warning("CTC timestamp decoder config is unreadable: %s", exc)
        return None
    else:
        pass
    proto = weights.get(PROTO_KEY)
    decoder_state = {
        name[len(DECODER_PREFIX) :]: tensor
        for name, tensor in weights.items()
        if name.startswith(DECODER_PREFIX)
    }
    if not isinstance(config, dict) or proto is None or not decoder_state:
        logger.warning("CTC timestamp head metadata is incomplete")
        return None
    else:
        pass
    try:
        head = build_ctc_head(config, decoder_state, proto)
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("CTC timestamp head could not be built: %s", exc)
        return None
    else:
        pass
    missing, unexpected = head.load_state_dict(decoder_state, strict=False)
    projection = decoder_state.get("decoder_layers.0.weight")
    class_count = int(projection.shape[0]) if projection is not None else -1
    vocabulary = config.get("vocabulary")
    vocabulary_ok = isinstance(vocabulary, list) and class_count == len(vocabulary) + 1
    residual_ok = not bool(config.get("residual")) or "residual_scale" in decoder_state
    head.usable = not missing and not unexpected and vocabulary_ok and residual_ok
    if not head.usable:
        logger.warning(
            "CTC timestamp head is not usable "
            "(missing=%s, unexpected=%s, vocabulary_ok=%s)",
            list(missing)[:8],
            list(unexpected)[:8],
            vocabulary_ok and residual_ok,
        )
    else:
        logger.info(
            "CTC timestamp head loaded (%d classes, blank=%d)",
            class_count,
            head.blank_id,
        )
    return head


def build_ctc_head(
    config: dict[str, Any],
    state: dict[str, torch.Tensor],
    proto: torch.Tensor,
) -> CtcTimestampHead:
    vocabulary = config["vocabulary"]
    if not isinstance(vocabulary, list) or not vocabulary:
        raise ValueError("CTC vocabulary must be a non-empty list")
    else:
        pass
    feat_in = int(config["feat_in"])
    num_classes = int(config["num_classes"])
    if num_classes != len(vocabulary):
        raise ValueError("CTC num_classes must equal the vocabulary length")
    else:
        pass
    names = set(state)
    use_transformer = bool(config.get("use_transformer", False))
    n_heads = int(config.get("n_heads") or 1)
    if use_transformer and feat_in % n_heads != 0:
        raise ValueError("CTC feat_in must be divisible by n_heads")
    else:
        pass
    return CtcTimestampHead(
        feat_in=feat_in,
        class_count=num_classes + 1,
        vocabulary=[str(piece) for piece in vocabulary],
        n_layers=int(config.get("n_layers") or 0),
        n_heads=n_heads,
        ff_expansion=float(config.get("ff_expansion") or 4.0),
        use_transformer=use_transformer,
        qkv_bias=any(name.endswith("attn.w_qkv.bias") for name in names),
        qk_norm=any(".attn.q_norm." in name for name in names),
        with_final_norm="transformer.final_norm.weight" in names,
        with_out_proj="transformer.out_proj.weight" in names,
        with_embed_norm="transformer.embed_norm.weight" in names,
        with_residual_scale="residual_scale" in names,
        apply_residual=bool(config.get("residual")) and "residual_scale" in names,
        tokenizer_proto=bytes(proto.detach().cpu().contiguous().numpy().tobytes()),
    )
