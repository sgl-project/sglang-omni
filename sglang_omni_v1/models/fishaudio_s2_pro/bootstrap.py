# SPDX-License-Identifier: Apache-2.0
"""Bootstrap helpers for Fish Audio S2-Pro SGLang execution."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

logger = logging.getLogger(__name__)


def patch_fish_config_for_sglang() -> None:
    """Patch Fish config classes with the aliases SGLang expects."""
    from sglang_omni_v1.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3Config,
        FishQwen3OmniConfig,
    )

    if hasattr(FishQwen3Config, "_sglang_patched"):
        return

    original_text_init = FishQwen3Config.__init__

    def _patched_text_init(self, *args, **kwargs):
        original_text_init(self, *args, **kwargs)
        self.num_attention_heads = self.n_head
        self.hidden_size = self.dim
        self.num_hidden_layers = self.n_layer
        self.num_key_value_heads = self.n_local_heads
        self.torch_dtype = torch.bfloat16
        if self.architectures is None:
            self.architectures = ["S2ProSGLangTextModel"]

    FishQwen3Config.__init__ = _patched_text_init
    FishQwen3Config._sglang_patched = True

    original_omni_init = FishQwen3OmniConfig.__init__

    def _patched_omni_init(self, *args, **kwargs):
        original_omni_init(self, *args, **kwargs)
        if self.architectures is None:
            self.architectures = ["S2ProSGLangTextModel"]

    FishQwen3OmniConfig.__init__ = _patched_omni_init


def truncate_rope_to_bf16(model: torch.nn.Module) -> None:
    """Match the old Fish runtime's rope-cache precision behavior."""
    for module in model.modules():
        if hasattr(module, "cos_sin_cache"):
            module.cos_sin_cache.data = module.cos_sin_cache.data.to(torch.bfloat16).to(
                torch.float32
            )


def load_audio_decoder(
    checkpoint_dir: str,
    *,
    device: str,
) -> tuple[torch.nn.Module, int, int, Any]:
    """Load the Fish audio decoder and return it with metadata + tokenizer."""
    from transformers import PreTrainedTokenizerFast

    from sglang_omni_v1.models.fishaudio_s2_pro.fish_speech.models.text2semantic.configuration import (
        FishQwen3OmniConfig,
    )
    from sglang_omni_v1.models.fishaudio_s2_pro.fish_speech.models.text2semantic.modeling import (
        FishQwen3OmniForCausalLM,
    )

    logger.info("Loading Fish audio decoder from %s", checkpoint_dir)
    start = time.perf_counter()

    config = FishQwen3OmniConfig.from_pretrained(checkpoint_dir)
    model = FishQwen3OmniForCausalLM.from_pretrained(checkpoint_dir, config=config)
    model = model.to(dtype=torch.bfloat16).eval()

    audio_decoder = model.audio_decoder
    if audio_decoder is None:
        raise RuntimeError("Fish checkpoint did not contain an audio decoder")
    audio_decoder = audio_decoder.to(device=device)

    tokenizer = PreTrainedTokenizerFast.from_pretrained(checkpoint_dir)
    num_codebooks = int(config.audio_decoder_config.num_codebooks)
    codebook_size = int(config.audio_decoder_config.vocab_size)

    del model
    if str(device).startswith("cuda"):
        torch.cuda.empty_cache()

    logger.info(
        "Fish audio decoder loaded in %.2fs (num_codebooks=%d, codebook_size=%d)",
        time.perf_counter() - start,
        num_codebooks,
        codebook_size,
    )
    return audio_decoder, num_codebooks, codebook_size, tokenizer


def bootstrap_text_model_for_decode(
    *,
    text_model: Any,
    audio_decoder: torch.nn.Module,
    semantic_begin_id: int,
    semantic_end_id: int,
    im_end_token_id: int,
    max_batch_size: int,
    num_codebooks: int,
    codebook_size: int,
) -> None:
    """Attach the fast codebook head and allocate persistent decode buffers."""
    audio_decoder.setup_caches(max_batch_size=max_batch_size, dtype=torch.bfloat16)
    text_model.setup_vq_decode(
        audio_decoder,
        num_codebooks=num_codebooks,
        codebook_size=codebook_size,
        semantic_begin_id=semantic_begin_id,
        semantic_end_id=semantic_end_id,
        im_end_token_id=im_end_token_id,
        max_batch_size=max_batch_size,
    )
