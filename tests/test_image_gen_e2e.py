# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for ZImage backend with Ming-flash-omni-2.0.

Tests that the full pipeline (ByT5 text encoder + ZImage transformer + VAE)
can generate a valid image from a text prompt.

Run on remote with:
    cd /sgl-workspace/sglang-omni-dev
    source .venv/bin/activate
    python tests/test_image_gen_e2e.py [--test manual|full] [--device cuda:1]
"""

from __future__ import annotations

import json
import logging
import os
import time
from pathlib import Path

import pytest
import torch
import torch.nn as nn

logging.basicConfig(
    level="INFO",
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# Resolve model path: support both HF cache and direct path
_DEFAULT_MODEL_PATH = "inclusionAI/Ming-flash-omni-2.0"
MODEL_PATH = os.environ.get("MING_MODEL_PATH", _DEFAULT_MODEL_PATH)
DEVICE = os.environ.get("CUDA_DEVICE", "cuda:1")
OUTPUT_DIR = "/tmp"


def _resolve_model_root(model_path: str) -> Path:
    """Resolve model_path to a local directory, downloading from HF if needed."""
    p = Path(model_path)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model_path))


# ---------------------------------------------------------------------------
# Composite ByT5 + Mapper text encoder
# ---------------------------------------------------------------------------


class _HiddenStatesOutput:
    """Mimics HuggingFace model output with .hidden_states attribute."""

    def __init__(self, hidden_states: list[torch.Tensor]):
        self.hidden_states = hidden_states


class ByT5TextEncoder(nn.Module):
    """Composite text encoder: ByT5 base encoder + T5EncoderBlockByT5Mapper.

    This wraps the ByT5 base model and mapper to produce 2560-dim text features
    compatible with ZImagePipeline._encode_prompt().

    The forward() signature matches what ZImagePipeline expects:
        text_encoder(input_ids, attention_mask, output_hidden_states=True)
        → output with .hidden_states[-2] of shape [batch, seq_len, 2560]
    """

    def __init__(self, byt5_encoder: nn.Module, mapper: nn.Module):
        super().__init__()
        self.byt5_encoder = byt5_encoder
        self.mapper = mapper

    @property
    def dtype(self) -> torch.dtype:
        """Return dtype of the first parameter (needed by diffusers pipeline)."""
        p = next(self.parameters(), None)
        return p.dtype if p is not None else torch.float32

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        output_hidden_states: bool = True,
        **kwargs,
    ) -> _HiddenStatesOutput:
        # 1. Run ByT5 base encoder
        byt5_out = self.byt5_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        # byt5_out.last_hidden_state: [batch, seq_len, d_model]
        base_hidden = byt5_out.last_hidden_state

        # 2. Run mapper to project d_model → 2560
        mapped_hidden = self.mapper(base_hidden, attention_mask)
        # mapped_hidden: [batch, seq_len, 2560]

        # 3. Return in HuggingFace-compatible format
        #    _encode_prompt() accesses .hidden_states[-2]
        return _HiddenStatesOutput(
            hidden_states=[base_hidden, mapped_hidden, mapped_hidden]
        )


def load_byt5_mapper(
    model_root: Path, byt5_config, device: torch.device, dtype: torch.dtype
):
    """Load the T5EncoderBlockByT5Mapper from Ming model weights."""

    byt5_dir = model_root / "byt5"
    byt5_json = json.loads((byt5_dir / "byt5.json").read_text())
    mapper_config = byt5_json["byt5_mapper_config"]

    logger.info(
        "Mapper config: num_layers=%d, sdxl_channels=%s",
        mapper_config["num_layers"],
        mapper_config.get("sdxl_channels"),
    )

    # Build mapper using the same architecture as Ming
    # T5EncoderBlockByT5Mapper(byt5_config, num_layers=4, sdxl_channels=2560)
    num_layers = mapper_config["num_layers"]
    sdxl_channels = mapper_config.get("sdxl_channels")

    # Import T5 layer components from transformers
    from transformers.models.t5.modeling_t5 import (
        T5LayerFF,
        T5LayerNorm,
        T5LayerSelfAttention,
    )

    class T5EncoderBlock(nn.Module):
        def __init__(self, config, has_relative_attention_bias=False):
            super().__init__()
            self.layer = nn.ModuleList()
            self.layer.append(
                T5LayerSelfAttention(
                    config, has_relative_attention_bias=has_relative_attention_bias
                )
            )
            self.layer.append(T5LayerFF(config))

        def forward(
            self, hidden_states, attention_mask=None, position_bias=None, **kwargs
        ):
            seq_len = hidden_states.shape[1]
            cache_position = torch.arange(seq_len, device=hidden_states.device)
            self_attention_outputs = self.layer[0](
                hidden_states,
                attention_mask=attention_mask,
                position_bias=position_bias,
                cache_position=cache_position,
            )
            hidden_states = self_attention_outputs[0]
            attention_outputs = self_attention_outputs[1:]
            hidden_states = self.layer[-1](hidden_states)
            outputs = (hidden_states,) + attention_outputs
            return outputs

    class SimpleByT5Mapper(nn.Module):
        def __init__(self, config, num_layers, sdxl_channels=None):
            super().__init__()
            if num_layers > 0:
                self.blocks = nn.ModuleList(
                    [
                        T5EncoderBlock(config, has_relative_attention_bias=bool(i == 0))
                        for i in range(num_layers)
                    ]
                )
            else:
                self.blocks = None
            self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
            if sdxl_channels is not None:
                self.channel_mapper = nn.Linear(config.d_model, sdxl_channels)
                self.final_layer_norm = T5LayerNorm(
                    sdxl_channels, eps=config.layer_norm_epsilon
                )
            else:
                self.channel_mapper = None
                self.final_layer_norm = None

        def forward(self, inputs_embeds, attention_mask):
            # Expand attention mask
            if attention_mask.dim() == 2:
                extended_attention_mask = attention_mask[:, None, None, :]
            else:
                extended_attention_mask = attention_mask[:, None, :, :]
            extended_attention_mask = extended_attention_mask.to(
                dtype=inputs_embeds.dtype
            )
            extended_attention_mask = (1.0 - extended_attention_mask) * torch.finfo(
                inputs_embeds.dtype
            ).min

            hidden_states = inputs_embeds
            position_bias = None

            if self.blocks is not None:
                for layer_module in self.blocks:
                    layer_outputs = layer_module(
                        hidden_states,
                        attention_mask=extended_attention_mask,
                        position_bias=position_bias,
                    )
                    hidden_states = layer_outputs[0]
                    position_bias = layer_outputs[1] if len(layer_outputs) > 1 else None
            hidden_states = self.layer_norm(hidden_states)
            if self.channel_mapper is not None:
                hidden_states = self.channel_mapper(hidden_states)
                hidden_states = self.final_layer_norm(hidden_states)
            return hidden_states

    mapper = SimpleByT5Mapper(byt5_config, num_layers, sdxl_channels)

    # Load pretrained mapper weights
    mapper_path = byt5_dir / "byt5_mapper" / "byt5_mapper.pt"
    mapper_state = torch.load(str(mapper_path), map_location="cpu", weights_only=True)
    missing, unexpected = mapper.load_state_dict(mapper_state, strict=False)
    if missing:
        logger.warning("Mapper missing keys: %s", missing)
    if unexpected:
        logger.warning("Mapper unexpected keys: %s", unexpected)
    logger.info("Mapper loaded successfully")

    mapper = mapper.to(device=device, dtype=dtype)
    return mapper


def load_full_text_encoder(model_root: Path, device: torch.device, dtype: torch.dtype):
    """Load the complete ByT5 + Mapper text encoder from Ming model."""
    byt5_dir = model_root / "byt5"
    byt5_json = json.loads((byt5_dir / "byt5.json").read_text())
    byt5_config_dict = byt5_json["byt5_config"]

    # 1. Load ByT5 base model
    byt5_path = str(byt5_dir / byt5_config_dict["byt5_ckpt_path"])
    logger.info("Loading ByT5 base from %s", byt5_path)

    from transformers import AutoTokenizer, T5ForConditionalGeneration

    t0 = time.time()
    byt5_tokenizer = AutoTokenizer.from_pretrained(byt5_path)
    byt5_model = T5ForConditionalGeneration.from_pretrained(byt5_path)
    byt5_encoder = byt5_model.get_encoder()
    logger.info(
        "ByT5 base loaded in %.1fs (d_model=%d)",
        time.time() - t0,
        byt5_encoder.config.d_model,
    )

    # Add special tokens
    if byt5_config_dict.get("special_token"):
        font_ann_path = str(
            byt5_dir
            / byt5_config_dict.get("font_ann_path", "font_uni_10-lang_idx.json")
        )
        color_ann_path = str(
            byt5_dir / byt5_config_dict.get("color_ann_path", "color_idx.json")
        )
        with open(font_ann_path) as f:
            idx_font_dict = json.load(f)
        with open(color_ann_path) as f:
            idx_color_dict = json.load(f)

        additional_tokens = []
        if byt5_config_dict.get("color_special_token"):
            additional_tokens += [f"<color-{i}>" for i in range(len(idx_color_dict))]
        if byt5_config_dict.get("font_special_token"):
            if byt5_config_dict.get("multilingual"):
                for font_code in idx_font_dict:
                    prefix = font_code[:3]
                    if prefix in ("cn-", "en-", "jp-", "kr-"):
                        additional_tokens.append(
                            f"<{prefix}font-{idx_font_dict[font_code]}>"
                        )
                    else:
                        additional_tokens.append(f"<font-{idx_font_dict[font_code]}>")
            else:
                additional_tokens += [f"<font-{i}>" for i in range(len(idx_font_dict))]

        if additional_tokens:
            byt5_tokenizer.add_tokens(additional_tokens, special_tokens=True)
            byt5_encoder.resize_token_embeddings(len(byt5_tokenizer))
            logger.info("Added %d special tokens to ByT5", len(additional_tokens))

    # Load fine-tuned ByT5 weights
    byt5_model_path = byt5_dir / "byt5_model" / "byt5_model.pt"
    if byt5_model_path.exists():
        logger.info("Loading fine-tuned ByT5 weights from %s", byt5_model_path)
        byt5_state = torch.load(
            str(byt5_model_path), map_location="cpu", weights_only=True
        )
        missing, unexpected = byt5_encoder.load_state_dict(byt5_state, strict=False)
        if missing:
            logger.warning(
                "ByT5 encoder missing keys (%d): %s...", len(missing), missing[:3]
            )
        if unexpected:
            logger.warning(
                "ByT5 encoder unexpected keys (%d): %s...",
                len(unexpected),
                unexpected[:3],
            )

    byt5_encoder = byt5_encoder.to(device=device, dtype=dtype)

    # 2. Load mapper
    mapper = load_byt5_mapper(model_root, byt5_encoder.config, device, dtype)

    # 3. Compose
    text_encoder = ByT5TextEncoder(byt5_encoder, mapper)
    text_encoder = text_encoder.to(device=device, dtype=dtype)
    text_encoder.eval()

    return text_encoder, byt5_tokenizer


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_manual_pipeline():
    """Test pipeline with random prompt_embeds (proves pipeline mechanics work)."""
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    logger.info("=== Manual pipeline test (random embeddings) ===")
    model_root = _resolve_model_root(MODEL_PATH)

    t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )
    scheduler.config["use_dynamic_shifting"] = True
    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    logger.info("Components loaded in %.1fs", time.time() - t0)

    pipe = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
    )
    pipe = pipe.to(DEVICE)
    logger.info("Pipeline on %s", DEVICE)

    cap_feat_dim = transformer.config.cap_feat_dim
    seq_len = 77
    prompt_embeds = [
        torch.randn(seq_len, cap_feat_dim, device=DEVICE, dtype=torch.bfloat16)
    ]
    neg_embeds = [
        torch.zeros(seq_len, cap_feat_dim, device=DEVICE, dtype=torch.bfloat16)
    ]

    logger.info("Generating 512x512, 10 steps...")
    t0 = time.time()
    result = pipe(
        prompt_embeds=prompt_embeds,
        negative_prompt_embeds=neg_embeds,
        height=512,
        width=512,
        num_inference_steps=10,
        guidance_scale=7.0,
        generator=torch.Generator(device=DEVICE).manual_seed(42),
        max_sequence_length=512,
    )
    image = result.images[0]
    logger.info(
        "Generated in %.1fs: %dx%d", time.time() - t0, image.width, image.height
    )

    out = os.path.join(OUTPUT_DIR, "test_zimage_manual.png")
    image.save(out)
    logger.info("Saved to %s", out)
    assert image.width == 512 and image.height == 512
    logger.info("=== Manual pipeline test PASSED ===")

    del pipe
    torch.cuda.empty_cache()


@pytest.mark.skip(reason="Requires large model download; run manually with --run-slow")
def test_full_pipeline():
    """Full test: ByT5 text encoder + ZImage pipeline → generate from text prompt."""
    from diffusers import (
        AutoencoderKL,
        FlowMatchEulerDiscreteScheduler,
        ZImagePipeline,
        ZImageTransformer2DModel,
    )

    logger.info("=== Full pipeline test (with ByT5 text encoder) ===")
    model_root = _resolve_model_root(MODEL_PATH)

    # Load diffusion components
    t0 = time.time()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        MODEL_PATH, subfolder="scheduler"
    )
    scheduler.config["use_dynamic_shifting"] = True
    vae = AutoencoderKL.from_pretrained(
        MODEL_PATH, subfolder="vae", torch_dtype=torch.bfloat16
    )
    transformer = ZImageTransformer2DModel.from_pretrained(
        MODEL_PATH, subfolder="transformer", torch_dtype=torch.bfloat16
    )
    logger.info("Diffusion components loaded in %.1fs", time.time() - t0)

    # Load text encoder
    t0 = time.time()
    text_encoder, tokenizer = load_full_text_encoder(
        model_root, torch.device(DEVICE), torch.bfloat16
    )
    logger.info("Text encoder loaded in %.1fs", time.time() - t0)

    # Build pipeline WITHOUT text_encoder (we'll encode manually and pass prompt_embeds)
    pipe = ZImagePipeline(
        scheduler=scheduler,
        vae=vae,
        transformer=transformer,
        text_encoder=None,
        tokenizer=None,
    )
    pipe = pipe.to(DEVICE)
    logger.info("Full pipeline assembled on %s", DEVICE)

    # Test with real prompts
    prompts = [
        "A cat sitting on a windowsill watching the sunset, digital art",
        "一幅水墨画，画中有竹子和远山",
    ]

    for i, prompt in enumerate(prompts):
        logger.info("Generating image %d: %r", i, prompt[:80])

        # Manually encode text with ByT5 + mapper
        t0 = time.time()
        with torch.no_grad():
            text_inputs = tokenizer(
                [prompt],
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            input_ids = text_inputs.input_ids.to(DEVICE)
            attn_mask = text_inputs.attention_mask.to(DEVICE)

            enc_out = text_encoder(
                input_ids=input_ids,
                attention_mask=attn_mask,
                output_hidden_states=True,
            )
            prompt_embeds_tensor = enc_out.hidden_states[-2]  # [1, seq_len, 2560]
            # Apply attention mask to get variable-length embeddings
            mask_bool = attn_mask.bool()
            prompt_embeds_list = [
                prompt_embeds_tensor[b][mask_bool[b]]
                for b in range(prompt_embeds_tensor.shape[0])
            ]

            # Encode negative prompt (empty string)
            neg_inputs = tokenizer(
                [""],
                padding="max_length",
                max_length=256,
                truncation=True,
                return_tensors="pt",
            )
            neg_ids = neg_inputs.input_ids.to(DEVICE)
            neg_mask = neg_inputs.attention_mask.to(DEVICE)

            neg_out = text_encoder(
                input_ids=neg_ids,
                attention_mask=neg_mask,
                output_hidden_states=True,
            )
            neg_embeds_tensor = neg_out.hidden_states[-2]
            neg_mask_bool = neg_mask.bool()
            neg_embeds_list = [
                neg_embeds_tensor[b][neg_mask_bool[b]]
                for b in range(neg_embeds_tensor.shape[0])
            ]

        logger.info("Text encoded in %.1fs", time.time() - t0)

        t0 = time.time()
        result = pipe(
            prompt_embeds=prompt_embeds_list,
            negative_prompt_embeds=neg_embeds_list,
            height=1024,
            width=1024,
            num_inference_steps=28,
            guidance_scale=7.0,
            generator=torch.Generator(device=DEVICE).manual_seed(42 + i),
            max_sequence_length=256,
        )
        image = result.images[0]
        elapsed = time.time() - t0
        logger.info("Generated in %.1fs: %dx%d", elapsed, image.width, image.height)

        out = os.path.join(OUTPUT_DIR, f"test_zimage_full_{i}.png")
        image.save(out)
        logger.info("Saved to %s", out)
        assert image.width == 1024 and image.height == 1024

    logger.info("=== Full pipeline test PASSED ===")

    del pipe
    torch.cuda.empty_cache()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["manual", "full", "all"], default="full")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    args = parser.parse_args()

    if args.model_path:
        MODEL_PATH = args.model_path
    if args.device:
        DEVICE = args.device

    if args.test in ("manual", "all"):
        test_manual_pipeline()
    if args.test in ("full", "all"):
        test_full_pipeline()
