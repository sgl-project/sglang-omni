# SPDX-License-Identifier: Apache-2.0
"""ByT5 text encoder with mapper for ZImage diffusion.

Loads the ByT5 base encoder from the Ming model's byt5/ directory,
applies the T5EncoderBlock mapper to project from d_model to cap_feat_dim (2560),
and provides a unified encode() interface for the ZImage pipeline.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoTokenizer, T5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import (
    T5LayerFF,
    T5LayerNorm,
    T5LayerSelfAttention,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Mapper layers (reimplemented from Ming's T5EncoderBlockByT5Mapper)
# ---------------------------------------------------------------------------


class _T5EncoderBlock(nn.Module):
    """Single T5 encoder block: self-attention + feed-forward."""

    def __init__(self, config, has_relative_attention_bias: bool = False):
        super().__init__()
        self.layer = nn.ModuleList(
            [
                T5LayerSelfAttention(
                    config, has_relative_attention_bias=has_relative_attention_bias
                ),
                T5LayerFF(config),
            ]
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_bias: torch.Tensor | None = None,
        **kwargs,
    ) -> tuple[torch.Tensor, ...]:
        seq_len = hidden_states.shape[1]
        cache_position = torch.arange(seq_len, device=hidden_states.device)
        sa_out = self.layer[0](
            hidden_states,
            attention_mask=attention_mask,
            position_bias=position_bias,
            cache_position=cache_position,
        )
        hidden_states = sa_out[0]
        position_bias = sa_out[1] if len(sa_out) > 1 else None
        hidden_states = self.layer[1](hidden_states)
        return (hidden_states, position_bias)


class ByT5Mapper(nn.Module):
    """Projects ByT5 encoder outputs to cap_feat_dim via additional T5 blocks + linear.

    Architecture (from Ming's T5EncoderBlockByT5Mapper):
        ByT5 hidden (d_model) → [optional T5 encoder blocks] → LayerNorm
        → Linear(d_model, sdxl_channels) → LayerNorm → output (sdxl_channels)
    """

    def __init__(self, byt5_config, num_layers: int, sdxl_channels: int | None = None):
        super().__init__()
        if num_layers > 0:
            self.blocks = nn.ModuleList(
                [
                    _T5EncoderBlock(byt5_config, has_relative_attention_bias=(i == 0))
                    for i in range(num_layers)
                ]
            )
        else:
            self.blocks = None

        self.layer_norm = T5LayerNorm(
            byt5_config.d_model, eps=byt5_config.layer_norm_epsilon
        )
        if sdxl_channels is not None:
            self.channel_mapper = nn.Linear(byt5_config.d_model, sdxl_channels)
            self.final_layer_norm = T5LayerNorm(
                sdxl_channels, eps=byt5_config.layer_norm_epsilon
            )
        else:
            self.channel_mapper = None
            self.final_layer_norm = None

    def forward(
        self, inputs_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        # Expand attention mask: [B, S] → [B, 1, 1, S]
        if attention_mask.dim() == 2:
            ext_mask = attention_mask[:, None, None, :]
        else:
            ext_mask = attention_mask[:, None, :, :]
        ext_mask = ext_mask.to(dtype=inputs_embeds.dtype)
        ext_mask = (1.0 - ext_mask) * torch.finfo(inputs_embeds.dtype).min

        hidden_states = inputs_embeds
        position_bias = None
        if self.blocks is not None:
            for block in self.blocks:
                hidden_states, position_bias = block(
                    hidden_states,
                    attention_mask=ext_mask,
                    position_bias=position_bias,
                )

        hidden_states = self.layer_norm(hidden_states)
        if self.channel_mapper is not None:
            hidden_states = self.channel_mapper(hidden_states)
            hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


# ---------------------------------------------------------------------------
# ByT5TextEncoder: composite ByT5 + Mapper
# ---------------------------------------------------------------------------


class ByT5TextEncoder(nn.Module):
    """Composite ByT5 base encoder + mapper for ZImage text conditioning.

    Encodes text to cap_feat_dim (2560) embeddings ready for the ZImage
    transformer's cap_embedder.

    Text format: ByT5 expects render text wrapped as ``Text "...". `` — this
    matches Ming's production code (``processing_bailingmm2.py:get_text_from_prompt``).
    The ``encode()`` method applies this formatting automatically.
    """

    def __init__(self, byt5_encoder: nn.Module, mapper: ByT5Mapper):
        super().__init__()
        self.byt5_encoder = byt5_encoder
        self.mapper = mapper

    @staticmethod
    def format_render_text(text: str) -> str:
        """Wrap render text in the format expected by the fine-tuned ByT5 model.

        Ming's production code (``processing_bailingmm2.py:get_text_from_prompt``)
        extracts quoted text from user prompts and formats it as::

            Text "百灵 Ming Omni".

        Without this wrapper, the byte-level tokenization produces incorrect
        embeddings and garbled text rendering.
        """
        if not text:
            return ""
        return f'Text "{text}". '

    @torch.no_grad()
    def encode(
        self,
        text: str | list[str],
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int = 256,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        """Encode render text into padded embedding lists.

        The text is automatically formatted as ``Text "...". `` before encoding.

        Returns:
            (prompt_embeds, negative_embeds): Each is a list of tensors,
            one per batch item, with shape ``[max_length, cap_feat_dim]``.
            Padding positions are zeroed out.  Negative embeds are all zeros
            (matching production: ``byt5_prompt_embeds * 0.0``).
        """
        if isinstance(text, str):
            text = [text]

        # Format text for ByT5 (production: Text "...". )
        formatted = [self.format_render_text(t) for t in text]

        # Encode positive prompt
        prompt_embeds = self._encode_batch(formatted, tokenizer, device, max_length)

        # Negative = zeros with same shape (production: byt5_prompt_embeds * 0.0)
        neg_embeds = [torch.zeros_like(e) for e in prompt_embeds]

        return prompt_embeds, neg_embeds

    def _encode_batch(
        self,
        texts: list[str],
        tokenizer: AutoTokenizer,
        device: torch.device,
        max_length: int,
    ) -> list[torch.Tensor]:
        inputs = tokenizer(
            texts,
            padding="max_length",
            max_length=max_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        input_ids = inputs.input_ids.to(device)
        attn_mask = inputs.attention_mask.to(device)

        # ByT5 encoder forward
        byt5_out = self.byt5_encoder(input_ids=input_ids, attention_mask=attn_mask)
        base_hidden = byt5_out.last_hidden_state

        # Mapper: d_model → cap_feat_dim
        mapped = self.mapper(base_hidden, attn_mask)

        # Zero out padding positions (production: byt5_prompt_embeds *= attn_mask)
        # Keep full padded tensor shape [max_length, cap_feat_dim] per item.
        mapped = mapped * attn_mask.unsqueeze(-1)
        return [mapped[b] for b in range(mapped.shape[0])]


# ---------------------------------------------------------------------------
# Loading utility
# ---------------------------------------------------------------------------


def load_byt5_text_encoder(
    model_path: str,
    device: torch.device,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[ByT5TextEncoder, AutoTokenizer]:
    """Load ByT5 text encoder + mapper + tokenizer from a Ming model directory.

    Expects the model directory to contain:
        byt5/
            byt5.json                    - config with byt5_config and byt5_mapper_config
            google__byt5-smal/           - base ByT5 model + tokenizer
            byt5_model/byt5_model.pt     - fine-tuned encoder weights
            byt5_mapper/byt5_mapper.pt   - mapper weights
            font_uni_10-lang_idx.json    - font special tokens
            color_idx.json               - color special tokens

    Returns:
        (text_encoder, tokenizer)
    """
    model_root = Path(model_path)
    byt5_dir = model_root / "byt5"
    byt5_json = json.loads((byt5_dir / "byt5.json").read_text())
    byt5_cfg = byt5_json["byt5_config"]
    mapper_cfg = byt5_json["byt5_mapper_config"]

    # 1. Load base ByT5 model + tokenizer
    byt5_ckpt_path = str(byt5_dir / byt5_cfg["byt5_ckpt_path"])
    logger.info("[ByT5] Loading base model from %s", byt5_ckpt_path)
    tokenizer = AutoTokenizer.from_pretrained(byt5_ckpt_path)
    byt5_model = T5ForConditionalGeneration.from_pretrained(byt5_ckpt_path)
    byt5_encoder = byt5_model.get_encoder()
    logger.info("[ByT5] Base encoder loaded (d_model=%d)", byt5_encoder.config.d_model)

    # 2. Add special tokens (font / color)
    _add_special_tokens(tokenizer, byt5_encoder, byt5_cfg, byt5_dir)

    # 3. Load fine-tuned ByT5 weights
    finetuned_path = byt5_dir / "byt5_model" / "byt5_model.pt"
    if finetuned_path.exists():
        logger.info("[ByT5] Loading fine-tuned weights from %s", finetuned_path)
        state = torch.load(str(finetuned_path), map_location="cpu", weights_only=True)
        missing, unexpected = byt5_encoder.load_state_dict(state, strict=False)
        if missing:
            logger.warning(
                "[ByT5] Missing keys (%d): %s ...", len(missing), missing[:3]
            )
        if unexpected:
            logger.warning(
                "[ByT5] Unexpected keys (%d): %s ...", len(unexpected), unexpected[:3]
            )

    # 4. Load mapper
    logger.info(
        "[ByT5] Loading mapper (layers=%d, channels=%s)",
        mapper_cfg["num_layers"],
        mapper_cfg.get("sdxl_channels"),
    )
    mapper = ByT5Mapper(
        byt5_encoder.config,
        num_layers=mapper_cfg["num_layers"],
        sdxl_channels=mapper_cfg.get("sdxl_channels"),
    )
    mapper_path = byt5_dir / "byt5_mapper" / "byt5_mapper.pt"
    mapper_state = torch.load(str(mapper_path), map_location="cpu", weights_only=True)
    missing, unexpected = mapper.load_state_dict(mapper_state, strict=False)
    if missing:
        logger.warning("[ByT5] Mapper missing keys: %s", missing)
    if unexpected:
        logger.warning("[ByT5] Mapper unexpected keys: %s", unexpected)
    logger.info("[ByT5] Mapper loaded")

    # 5. Compose and move to device
    text_encoder = ByT5TextEncoder(byt5_encoder, mapper)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    return text_encoder, tokenizer


def _add_special_tokens(
    tokenizer: AutoTokenizer,
    encoder: nn.Module,
    byt5_cfg: dict,
    byt5_dir: Path,
) -> None:
    """Add font/color special tokens to the ByT5 tokenizer and resize embeddings."""
    if not byt5_cfg.get("special_token"):
        return

    font_ann_path = str(
        byt5_dir / byt5_cfg.get("font_ann_path", "font_uni_10-lang_idx.json")
    )
    color_ann_path = str(byt5_dir / byt5_cfg.get("color_ann_path", "color_idx.json"))

    additional: list[str] = []

    if byt5_cfg.get("color_special_token"):
        with open(color_ann_path) as f:
            idx_color = json.load(f)
        additional += [f"<color-{i}>" for i in range(len(idx_color))]

    if byt5_cfg.get("font_special_token"):
        with open(font_ann_path) as f:
            idx_font = json.load(f)
        if byt5_cfg.get("multilingual"):
            for code in idx_font:
                prefix = code[:3]
                if prefix in ("cn-", "en-", "jp-", "kr-"):
                    additional.append(f"<{prefix}font-{idx_font[code]}>")
                else:
                    additional.append(f"<font-{idx_font[code]}>")
        else:
            additional += [f"<font-{i}>" for i in range(len(idx_font))]

    if additional:
        tokenizer.add_tokens(additional, special_tokens=True)
        encoder.resize_token_embeddings(len(tokenizer))
        logger.info("[ByT5] Added %d special tokens", len(additional))
