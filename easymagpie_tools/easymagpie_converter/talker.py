# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Talker conversion: ``config.json``, weights, and tokenizers.

Adapted from NeMo ``tools/easymagpie_vllm_omni/scripts/convert_to_vllm.py``
(see ``upstream/SOURCE.txt``). The character-aware subword encoder is baked
into one ``text_embedding`` lookup table, and the backbone token-embedding
table is replaced by a two-row placeholder because the backbone always runs
from ``inputs_embeds``.
"""

from __future__ import annotations

import json
import logging
from math import prod
from pathlib import Path
from typing import Any

import torch
from omegaconf import OmegaConf
from safetensors.torch import save_file

LOG = logging.getLogger(__name__)

WEIGHTS_FILE = "model.safetensors"
PHONEME_TEXT_TOKENIZER_FILE = "phoneme_text_tokenizer/tokenizer.json"
# Serving reads vocab_size 2 as "no backbone token table".
BACKBONE_VOCAB_SIZE = 2

TTS_PREFIXES = (
    "audio_embeddings.",
    "audio_in_projection.",
    "local_transformer.",
    "local_transformer_in_projection.",
    "local_transformer_audio_out_projection.",
    "local_transformer_out_projections.",
    "phoneme_embeddings.",
    "phoneme_final_proj.",
    "task_embedding.",
)

NEMOTRON_CONFIG_FIELDS = (
    "hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "attention_dropout",
    "attention_bias",
    "max_position_embeddings",
    "mamba_num_heads",
    "mamba_head_dim",
    "ssm_state_size",
    "conv_kernel",
    "n_groups",
    "chunk_size",
    "mamba_hidden_act",
    "use_conv_bias",
    "use_bias",
    "intermediate_size",
    "mlp_hidden_act",
    "mlp_bias",
    "n_routed_experts",
    "num_experts_per_tok",
    "moe_intermediate_size",
    "moe_shared_expert_intermediate_size",
    "n_group",
    "topk_group",
    "routed_scaling_factor",
    "norm_topk_prob",
    "hybrid_override_pattern",
    "layer_norm_epsilon",
    "residual_in_fp32",
)

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def load_model(
    nemo_checkpoint: Path,
    codec_checkpoint: Path,
    phoneme_tokenizer: Path | None,
    *,
    disable_cas_for_context_text: bool,
    device: str,
) -> Any:
    from nemo.collections.tts.modules.magpietts_inference.utils import (
        ModelLoadConfig,
        load_easy_magpie_model,
    )

    model, name = load_easy_magpie_model(
        ModelLoadConfig(
            nemo_file=str(nemo_checkpoint),
            codecmodel_path=str(codec_checkpoint),
            phoneme_tokenizer_path=None if phoneme_tokenizer is None else str(phoneme_tokenizer),
            disable_cas_for_context_text=disable_cas_for_context_text,
        ),
        device=device,
    )
    model.eval()
    LOG.info("Loaded EasyMagpie checkpoint %s", name)
    return model


def validate_model(model: Any) -> None:
    """Reject checkpoint features the SGLang-Omni EasyMagpie model does not implement."""
    cfg = model.cfg
    decoder_type = str(cfg.get("decoder_type", "huggingface"))
    if decoder_type != "nemotron_h":
        raise ValueError(f"only decoder_type='nemotron_h' is supported, got {decoder_type!r}")
    embedding_dim = int(cfg.embedding_dim)
    if int(cfg.hidden_dim) != embedding_dim:
        raise ValueError(
            f"hidden_dim ({cfg.hidden_dim}) must equal embedding_dim ({embedding_dim})"
        )
    backbone_hidden = int(cfg.get("nemotron_h_config", {}).get("hidden_size", embedding_dim))
    if backbone_hidden != embedding_dim:
        raise ValueError(
            f"nemotron_h_config.hidden_size ({backbone_hidden}) must equal embedding_dim ({embedding_dim})"
        )
    local_transformer_type = str(cfg.get("local_transformer_type", "none"))
    if local_transformer_type not in {"ar", "autoregressive"}:
        raise ValueError(
            f"only an autoregressive local transformer is supported, got {local_transformer_type!r}"
        )
    mode = model.mode_name_to_mode.get(model.default_inference_mode)
    if mode is None:
        raise ValueError(f"default inference mode {model.default_inference_mode!r} is not defined")
    if mode.text_input_mode != "streaming":
        raise ValueError(
            f"only streaming text input is supported, got {mode.text_input_mode!r}"
        )
    if getattr(model, "phoneme_tokenizer", None) is None:
        raise ValueError("the checkpoint has no phoneme tokenizer; serving requires the phoneme stream")


@torch.no_grad()
def text_embedding_table(model: Any, batch_size: int) -> torch.Tensor:
    """Run the text embedding path once per text id: ``[text_vocab_size, embedding_dim]``."""
    device = next(model.parameters()).device
    if getattr(model, "text_vocab_size", None) is not None:
        vocab_size = int(model.text_vocab_size)
    elif getattr(model, "text_embedding", None) is not None:
        vocab_size = int(model.text_embedding.num_embeddings)
    else:
        last_special = max(
            int(model.cfg_unk_token_id),
            int(getattr(model, "interruption_token_id", model.cfg_unk_token_id)),
        )
        vocab_size = last_special + 1
    table = torch.zeros((vocab_size, int(model.cfg.embedding_dim)), dtype=torch.float32)
    LOG.info("Baking text embeddings for %d ids", vocab_size)
    for start in range(0, vocab_size, batch_size):
        end = min(start + batch_size, vocab_size)
        ids = torch.arange(start, end, dtype=torch.long, device=device).unsqueeze(0)
        lens = torch.tensor([end - start], dtype=torch.long, device=device)
        embeds = model.embed_text_tokens(ids, text_lens=lens, disable_cas_embedding=False)
        table[start:end] = embeds.squeeze(0).float().cpu()
    return table


def codec_layout(model: Any) -> tuple[int, list[int], int]:
    """Return ``(num_codebooks, num_levels_per_group, frame_stacking_factor)``."""
    converter = getattr(model, "_codec_converter", None)
    if converter is not None:
        quantizer = getattr(converter, "vector_quantizer_new", None)
    else:
        quantizer = getattr(getattr(model, "_codec_model", None), "vector_quantizer", None)
    fsqs = getattr(quantizer, "fsqs", None)
    if not fsqs:
        raise ValueError("only GroupFiniteScalarQuantizer codec layouts are supported")
    groups = [[int(level) for level in fsq.num_levels.reshape(-1).tolist()] for fsq in fsqs]
    levels = groups[0]
    if any(group != levels for group in groups[1:]):
        raise ValueError("every FSQ group must share one num_levels_per_group layout")
    num_codebooks = len(fsqs)
    if num_codebooks != int(model.num_audio_codebooks):
        raise ValueError(
            f"codec has {num_codebooks} codebooks, model expects {model.num_audio_codebooks}"
        )
    if prod(levels) != int(model.codebook_size):
        raise ValueError(
            f"codec levels imply codebook_size {prod(levels)}, model expects {model.codebook_size}"
        )
    return num_codebooks, levels, int(model.frame_stacking_factor)


def build_config(model: Any, text_vocab_size: int, dtype: str) -> dict[str, Any]:
    """Flat HF-style ``config.json``: Nemotron-H backbone fields plus EasyMagpie scalars."""
    from nemo.collections.tts.modules.nemotron_h_decoder import NemotronHConfig

    cfg = model.cfg
    embedding_dim = int(cfg.embedding_dim)
    backbone = dict(OmegaConf.to_container(cfg.nemotron_h_config, resolve=True))
    backbone.setdefault("hidden_size", embedding_dim)
    backbone_cfg = NemotronHConfig(**backbone)

    config: dict[str, Any] = {
        "architectures": ["EasyMagpieTTSForConditionalGeneration"],
        "model_type": "nemotron_h",
    }
    for field in NEMOTRON_CONFIG_FIELDS:
        if hasattr(backbone_cfg, field):
            config[field] = getattr(backbone_cfg, field)
    mode = model.mode_name_to_mode[model.default_inference_mode]
    tokenizer = model.phoneme_tokenizer
    config.update(
        tie_word_embeddings=False,
        torch_dtype=dtype,
        vocab_size=BACKBONE_VOCAB_SIZE,
        text_vocab_size=text_vocab_size,
        text_eos_id=int(model.eos_id),
        embedding_dim=embedding_dim,
        audio_embedding_dim=int(cfg.get("audio_embedding_dim", embedding_dim)),
        num_audio_codebooks=int(model.num_audio_codebooks),
        codebook_size=int(model.codebook_size),
        frame_stacking_factor=int(model.frame_stacking_factor),
        phoneme_stacking_factor=int(model.phoneme_stacking_factor),
        phoneme_vocab_size=int(model.phoneme_vocab_size),
        phoneme_bos_id=int(tokenizer.bos_token_id),
        phoneme_eos_id=int(tokenizer.eos_token_id),
        phoneme_unk_id=int(tokenizer.unk_token_id),
        phoneme_confidence_unk_threshold=float(
            getattr(model, "phoneme_confidence_unk_threshold", 0.0)
        ),
        streaming_phonemes_delay=int(mode.streaming_phonemes_delay),
        streaming_speech_delay=int(mode.streaming_speech_delay),
        num_task_embeddings=len(model.training_modes) if model.task_embedding is not None else 0,
        local_transformer_n_layers=int(cfg.get("local_transformer_n_layers", 2)),
        local_transformer_n_heads=int(cfg.get("local_transformer_n_heads", 1)),
        local_transformer_hidden_dim=int(cfg.get("local_transformer_hidden_dim", embedding_dim)),
        forced_audio_bos_id=int(model.audio_bos_id),
        forced_audio_eos_id=int(model.audio_eos_id),
        forced_mask_token_id=int(model.mask_token_id),
        use_multiturn_dataset=bool(cfg.get("use_multiturn_dataset", False)),
        condition_on_user_speech=bool(cfg.get("condition_on_user_speech", False)),
        use_user_speaking_token=bool(cfg.get("use_user_speaking_token", False)),
        use_user_speaking_end_token=bool(cfg.get("use_user_speaking_end_token", False)),
        forced_audio_user_speaking_id=int(
            getattr(model, "audio_user_speaking_id", model.codebook_size + 5)
        ),
        forced_audio_user_speaking_end_id=int(
            getattr(model, "audio_user_speaking_end_id", model.codebook_size + 6)
        ),
        codec_encoder_bundled=False,
        add_language_to_context_text=bool(cfg.get("add_language_to_context_text", False)),
        ignore_phoneme_languages=[
            str(language) for language in cfg.get("ignore_phoneme_languages", None) or []
        ],
    )
    if hasattr(model, "interruption_token_id"):
        config["text_interruption_id"] = int(model.interruption_token_id)

    config["enable_phoneme_text_input"] = bool(getattr(model, "enable_phoneme_text_input", False))
    if config["enable_phoneme_text_input"]:
        offset = int(model.text_phoneme_token_offset)
        size = int(model.text_phoneme_vocab_size)
        if offset + size != text_vocab_size:
            raise ValueError(
                "pronunciation-control ids must be the trailing text range: "
                f"offset={offset}, size={size}, text_vocab_size={text_vocab_size}"
            )
        config.update(
            text_phoneme_token_offset=offset,
            text_phoneme_vocab_size=size,
            phoneme_text_bop_marker=str(model.phoneme_text_bop_marker),
            phoneme_text_eop_marker=str(model.phoneme_text_eop_marker),
            text_phoneme_tokenizer_file=PHONEME_TEXT_TOKENIZER_FILE,
        )
    return config


def select_weights(
    state_dict: dict[str, torch.Tensor], embedding_dim: int, dtype: torch.dtype
) -> dict[str, torch.Tensor]:
    """Keep the backbone and TTS heads under the reference EasyMagpie key layout."""

    def cast(value: torch.Tensor) -> torch.Tensor:
        return value.to(dtype) if value.is_floating_point() else value

    weights = {}
    for key, value in state_dict.items():
        if key.endswith(".causal_mask") or key == "decoder.embeddings.weight":
            continue
        if key.startswith("decoder.") or key.startswith(TTS_PREFIXES):
            weights[key] = cast(value).contiguous()
    weights["decoder.embeddings.weight"] = torch.zeros(BACKBONE_VOCAB_SIZE, embedding_dim, dtype=dtype)
    return weights


def save_weights(weights: dict[str, torch.Tensor], output: Path) -> None:
    save_file(weights, output / WEIGHTS_FILE, metadata={"format": "pt"})
    index = {
        "metadata": {"total_size": sum(w.numel() * w.element_size() for w in weights.values())},
        "weight_map": {name: WEIGHTS_FILE for name in weights},
    }
    write_json(output / f"{WEIGHTS_FILE}.index.json", index)
    LOG.info("Saved %d talker tensors", len(weights))


def save_text_tokenizer(model: Any, output: Path, override: str | None) -> None:
    """Export the text-conditioning tokenizer used for target and context text."""
    from transformers import AutoTokenizer

    source = override
    if source is None:
        tokenizer_cfg = model.cfg.text_tokenizers[model.text_conditioning_tokenizer_name]
        if tokenizer_cfg.get("_target_") != "AutoTokenizer" or not tokenizer_cfg.get("pretrained_model"):
            raise ValueError("cannot infer the text tokenizer from the checkpoint; pass --text-tokenizer")
        source = str(tokenizer_cfg.pretrained_model)
    LOG.info("Saving text tokenizer %s", source)
    AutoTokenizer.from_pretrained(source).save_pretrained(output)


def save_phoneme_text_tokenizer(model: Any, output: Path) -> None:
    """Export the IPA BPE used by ``<bop>...<eop>`` spans in target text."""
    if not bool(getattr(model, "enable_phoneme_text_input", False)):
        return
    raw = getattr(model.phoneme_tokenizer, "_tokenizer", None)
    if raw is None:
        raise ValueError("pronunciation control needs an IPABPETokenizer with a raw tokenizer")
    if len(raw.get_vocab()) + 3 != int(model.text_phoneme_vocab_size):
        raise ValueError(
            f"IPA tokenizer has {len(raw.get_vocab())} entries; "
            f"text range expects {int(model.text_phoneme_vocab_size) - 3}"
        )
    path = output / PHONEME_TEXT_TOKENIZER_FILE
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(path))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def convert_talker(
    model: Any,
    output: Path,
    *,
    dtype: str,
    precompute_batch_size: int,
    text_tokenizer: str | None,
) -> dict[str, Any]:
    validate_model(model)
    table = text_embedding_table(model, precompute_batch_size)
    config = build_config(model, int(table.shape[0]), dtype)
    weights = select_weights(model.state_dict(), int(model.cfg.embedding_dim), DTYPES[dtype])
    weights["text_embedding.weight"] = table.to(DTYPES[dtype])
    save_weights(weights, output)
    del weights
    write_json(output / "config.json", config)
    save_text_tokenizer(model, output, text_tokenizer)
    save_phoneme_text_tokenizer(model, output)
    return config


__all__ = ["DTYPES", "codec_layout", "convert_talker", "load_model", "write_json"]
