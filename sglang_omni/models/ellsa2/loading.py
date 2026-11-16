# SPDX-License-Identifier: Apache-2.0
"""Load the ELLSA 2 generator, its motion tokenizers and the style clip, from the Hub or local copies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from safetensors.torch import load_file
from transformers import AutoTokenizer, Qwen2Config

from sglang_omni.models.ellsa2.audio_encoder import encoder_config
from sglang_omni.models.ellsa2.fbank import LogMelFilterbank
from sglang_omni.models.ellsa2.generator import (
    AUDIO_END,
    AUDIO_START,
    PROMPT_HEAD,
    Ellsa2Generator,
    PromptTokens,
)
from sglang_omni.models.ellsa2.motion import (
    MotionDecoder,
    Normalization,
    Tokenizers,
    style_codes,
)
from sglang_omni.models.ellsa2.rvq import RVQVAE, FlameTokenizer

GENERATOR_DIR = "standalone_generator"
TOKENIZER_DIR = "tokenizers"
# Feature widths of the tokenizers' normalized 6D rotations: 13 upper-body joints, 2 × 15 hand joints.
UPPER_FEATURES = 78
HAND_FEATURES = 180
BACKBONE_PREFIX = "base_llm.model."
# Tied to the backbone's token embeddings; the generator reads hidden states, never text logits.
TEXT_HEAD = "base_llm.lm_head.weight"
BACKBONE_ATTENTION = "sdpa"


@dataclass(kw_only=True)
class MotionModels:
    """What every Session of the stage shares: the generator, the tokenizers, the style's codes and the filterbank."""

    generator: Ellsa2Generator
    prompt: PromptTokens
    tokenizers: Tokenizers
    normalization: Normalization
    style: torch.Tensor
    filterbank: LogMelFilterbank
    device: torch.device

    def new_decoder(self) -> MotionDecoder:
        """The decoding state of one Session."""
        return MotionDecoder(self.tokenizers, self.normalization)


def fetch(location: str, patterns: list[str], repo_type: str | None = None) -> Path:
    """A local directory as is, else the files of the Hub repository that match patterns."""
    path = Path(location)
    if path.is_dir():
        return path
    else:
        return Path(
            snapshot_download(location, repo_type=repo_type, allow_patterns=patterns)
        )


def load_generator(
    directory: Path,
    device: torch.device,
    encoder_attn_implementation: str,
    cudnn_attention: bool = False,
) -> tuple[Ellsa2Generator, Qwen2Config]:
    # Note (Junnan Li): The checkpoint's config asks for flash-attention, but the reference's transformers builds the Qwen2 backbone from it with SDPA; the checkpoint's codes come from SDPA there and flash-attention in the audio encoder.
    config = Qwen2Config.from_pretrained(
        directory, attn_implementation=BACKBONE_ATTENTION
    )
    generator = Ellsa2Generator(
        config, encoder_config(encoder_attn_implementation), cudnn_attention
    )
    weights = {
        (
            "backbone." + name.removeprefix(BACKBONE_PREFIX)
            if name.startswith(BACKBONE_PREFIX)
            else name
        ): tensor
        for name, tensor in load_file(directory / "model.safetensors").items()
        if name != TEXT_HEAD
    }
    generator.load_state_dict(weights, strict=True)
    generator.to(
        device=device,
        dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
    )
    generator.restore_rope_precision()
    return generator.eval(), config


def load_rvq(
    path: Path, features: int, context_frames: int, device: torch.device
) -> RVQVAE:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = RVQVAE(features, checkpoint["num_quantizers"], context_frames)
    model.load_state_dict(checkpoint["net"], strict=True)
    return model.to(device).eval()


def load_flame(path: Path, context_frames: int, device: torch.device) -> FlameTokenizer:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    model = FlameTokenizer(
        checkpoint["net"]["mouth_basis"],
        checkpoint["net"]["mouth_vertex_matrix"],
        checkpoint["nonmouth_num_quantizers"],
        checkpoint["mouth_num_quantizers"],
        context_frames,
    )
    model.load_state_dict(checkpoint["net"], strict=True)
    return model.to(device).eval()


def load_motion_models(
    *,
    model_path: str,
    data_path: str,
    style: str,
    encoder_attn_implementation: str,
    device: torch.device,
    cudnn_attention: bool = False,
) -> MotionModels:
    root = fetch(model_path, [f"{GENERATOR_DIR}/*", f"{TOKENIZER_DIR}/*"])
    generator, config = load_generator(
        root / GENERATOR_DIR, device, encoder_attn_implementation, cudnn_attention
    )
    tokenizer = AutoTokenizer.from_pretrained(root / GENERATOR_DIR)
    prompt = PromptTokens(
        head=torch.tensor(
            tokenizer(PROMPT_HEAD, add_special_tokens=False)["input_ids"],
            device=device,
        ),
        audio_start=tokenizer.convert_tokens_to_ids(AUDIO_START),
        audio_end=tokenizer.convert_tokens_to_ids(AUDIO_END),
    )
    tokenizers = Tokenizers(
        upper=load_rvq(
            root / TOKENIZER_DIR / "upper.pth",
            UPPER_FEATURES,
            config.rvq_gesture_context_frames,
            device,
        ),
        hands=load_rvq(
            root / TOKENIZER_DIR / "hands.pth",
            HAND_FEATURES,
            config.rvq_gesture_context_frames,
            device,
        ),
        flame=load_flame(
            root / TOKENIZER_DIR / "flame.pth",
            config.rvq_expression_context_frames,
            device,
        ),
    )
    style_pose = f"testset/style/{style}.npz"
    style_flame = f"testset/style_flame/{style}_flame.npz"
    data = fetch(data_path, [style_pose, style_flame], repo_type="dataset")
    normalization = Normalization.release()
    return MotionModels(
        generator=generator,
        prompt=prompt,
        tokenizers=tokenizers,
        normalization=normalization,
        style=style_codes(
            data / style_pose, data / style_flame, tokenizers, normalization
        ),
        filterbank=LogMelFilterbank(),
        device=device,
    )


__all__ = ["MotionModels", "load_motion_models"]
