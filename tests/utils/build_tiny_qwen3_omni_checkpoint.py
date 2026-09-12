# SPDX-License-Identifier: Apache-2.0
"""Build a deterministic, test-sized, structurally complete Qwen3-Omni checkpoint."""

from __future__ import annotations

import argparse
import contextlib
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
import torch.utils.deterministic
from safetensors import safe_open
from transformers import Qwen3OmniMoeConfig, Qwen3OmniMoeForConditionalGeneration

__all__ = [
    "OFFICIAL_CONTEXT_LENGTHS",
    "OFFICIAL_SPECIAL_TOKEN_IDS",
    "OFFICIAL_VOCAB_SIZES",
    "NATIVE_MLX_COMPONENTS",
    "TINY_DIMS",
    "build_tiny_config",
    "build_tiny_qwen3_omni_checkpoint",
]

SEED = 0

# Canonical ownership order for the deterministic MLX export derived from this
# dense fixture. Each component is written to ``<component>/model.safetensors``.
NATIVE_MLX_COMPONENTS = ("vision", "audio", "thinker", "talker", "code2wav")

TINY_DIMS: dict[str, int] = {
    "thinker_hidden_size": 32,
    # Keep layer 24 because the production speech path captures layers 0 and 24.
    "thinker_layers": 25,
    "thinker_heads": 4,
    "thinker_kv_heads": 2,
    "thinker_experts": 4,
    "thinker_top_k": 2,
    "talker_hidden_size": 32,
    "talker_layers": 2,
    "talker_experts": 4,
    "talker_top_k": 2,
    "predictor_layers": 2,
    "code_groups": 16,
    "code2wav_hidden_size": 32,
    "code2wav_layers": 2,
    "head_dim": 8,
    "intermediate_size": 64,
    "vision_depth": 2,
    "vision_heads": 4,
    "audio_layers": 2,
    "audio_heads": 4,
    "code2wav_heads": 4,
    "code2wav_decoder_dim": 64,
}

# The official component-specific autoregressive context lengths. These are
# preserved even in the tiny fixture because they are part of the production
OFFICIAL_CONTEXT_LENGTHS: dict[str, int] = {
    "thinker_text": 65536,
    "talker_text": 65536,
    "code_predictor": 32768,
}

OFFICIAL_CODE2WAV_MAX_POSITION_EMBEDDINGS = 8000

OFFICIAL_VOCAB_SIZES: dict[str, int] = {
    "thinker_text": 152064,
    "talker_text": 3072,
    "code_predictor": 2048,
    "codec_codebook": 2048,
}

# Dotted paths into the serialized ``config.json``.
OFFICIAL_SPECIAL_TOKEN_IDS: dict[str, int] = {
    "im_start_token_id": 151644,
    "im_end_token_id": 151645,
    "tts_pad_token_id": 151671,
    "tts_bos_token_id": 151672,
    "tts_eos_token_id": 151673,
    "system_token_id": 8948,
    "user_token_id": 872,
    "assistant_token_id": 77091,
    "thinker_config.audio_token_id": 151675,
    "thinker_config.image_token_id": 151655,
    "thinker_config.video_token_id": 151656,
    "thinker_config.audio_start_token_id": 151669,
    "thinker_config.audio_end_token_id": 151670,
    "thinker_config.vision_start_token_id": 151652,
    "thinker_config.vision_end_token_id": 151653,
    "thinker_config.user_token_id": 872,
    "talker_config.audio_token_id": 151675,
    "talker_config.image_token_id": 151655,
    "talker_config.video_token_id": 151656,
    "talker_config.vision_start_token_id": 151652,
    "talker_config.audio_start_token_id": 151669,
    "talker_config.audio_end_token_id": 151670,
    "talker_config.codec_pad_id": 2148,
    "talker_config.codec_bos_id": 2149,
    "talker_config.codec_eos_token_id": 2150,
    "talker_config.codec_nothink_id": 2155,
    "talker_config.codec_think_bos_id": 2156,
    "talker_config.codec_think_eos_id": 2157,
}

# Special IDs indexed against the retained text vocabulary vs the retained
# talker codec vocabulary.
_TEXT_SCOPED_ID_PREFIXES = (
    "im_",
    "tts_",
    "system_",
    "user_",
    "assistant_",
    "thinker_config.",
)
_CODEC_SCOPED_IDS = (
    "talker_config.codec_pad_id",
    "talker_config.codec_bos_id",
    "talker_config.codec_eos_token_id",
    "talker_config.codec_nothink_id",
    "talker_config.codec_think_bos_id",
    "talker_config.codec_think_eos_id",
)

OFFICIAL_SPEAKER_IDS: dict[str, int] = {"chelsie": 2301, "ethan": 2302, "aiden": 2303}

_REQUIRED_WEIGHT_PREFIXES = (
    "thinker.model.",
    "thinker.lm_head.",
    "thinker.visual.",
    "thinker.audio_tower.",
    "talker.model.",
    "talker.codec_head.",
    "talker.text_projection.",
    "talker.hidden_projection.",
    "talker.code_predictor.",
    "code2wav.pre_transformer.",
    "code2wav.code_embedding.",
    "code2wav.upsample.",
    "code2wav.decoder.",
)

# Matches the official checkpoint layout; deliberately excludes weight shards.
_PROCESSOR_ASSET_PATTERNS = (
    "added_tokens.json",
    "chat_template.jinja",
    "chat_template.json",
    "merges.txt",
    "preprocessor_config.json",
    "processor_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "video_preprocessor_config.json",
    "vocab.json",
)

# The official checkpoint ships interleaved MRoPE; the section widths must sum
# to head_dim // 2 so the reduced head dimension stays rotationally valid.
_MROPE_SECTION = [2, 1, 1]


def _thinker_rope_parameters() -> dict[str, Any]:
    return {
        "rope_type": "default",
        "rope_theta": 1000000.0,
        "mrope_section": list(_MROPE_SECTION),
        "interleaved": True,
        "mrope_interleaved": True,
    }


def _thinker_text_config() -> dict[str, Any]:
    return {
        "vocab_size": OFFICIAL_VOCAB_SIZES["thinker_text"],
        "hidden_size": TINY_DIMS["thinker_hidden_size"],
        "intermediate_size": TINY_DIMS["intermediate_size"],
        "moe_intermediate_size": TINY_DIMS["intermediate_size"],
        "shared_expert_intermediate_size": 0,
        "num_hidden_layers": TINY_DIMS["thinker_layers"],
        "num_attention_heads": TINY_DIMS["thinker_heads"],
        "num_key_value_heads": TINY_DIMS["thinker_kv_heads"],
        "head_dim": TINY_DIMS["head_dim"],
        "num_experts": TINY_DIMS["thinker_experts"],
        "num_experts_per_tok": TINY_DIMS["thinker_top_k"],
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "max_position_embeddings": OFFICIAL_CONTEXT_LENGTHS["thinker_text"],
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "use_cache": True,
        "rope_parameters": _thinker_rope_parameters(),
    }


def _vision_config() -> dict[str, Any]:
    return {
        "depth": TINY_DIMS["vision_depth"],
        "hidden_size": TINY_DIMS["thinker_hidden_size"],
        "out_hidden_size": TINY_DIMS["thinker_hidden_size"],
        "intermediate_size": TINY_DIMS["intermediate_size"],
        "num_heads": TINY_DIMS["vision_heads"],
        "in_channels": 3,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2,
        "num_position_embeddings": 2304,
        "deepstack_visual_indexes": list(range(TINY_DIMS["vision_depth"])),
        "tokens_per_second": 2,
    }


def _audio_config() -> dict[str, Any]:
    return {
        "d_model": TINY_DIMS["thinker_hidden_size"],
        "output_dim": TINY_DIMS["thinker_hidden_size"],
        "downsample_hidden_size": TINY_DIMS["thinker_hidden_size"],
        "encoder_layers": TINY_DIMS["audio_layers"],
        "encoder_attention_heads": TINY_DIMS["audio_heads"],
        "encoder_ffn_dim": TINY_DIMS["intermediate_size"],
        "num_mel_bins": 128,
        "max_source_positions": 1500,
        "n_window": 50,
        "n_window_infer": 800,
        "conv_chunksize": 500,
        "scale_embedding": False,
    }


def _thinker_config() -> dict[str, Any]:
    return {
        "audio_config": _audio_config(),
        "vision_config": _vision_config(),
        "text_config": _thinker_text_config(),
        "audio_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.audio_token_id"],
        "image_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.image_token_id"],
        "video_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.video_token_id"],
        "audio_start_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "thinker_config.audio_start_token_id"
        ],
        "audio_end_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "thinker_config.audio_end_token_id"
        ],
        "vision_start_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "thinker_config.vision_start_token_id"
        ],
        "vision_end_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "thinker_config.vision_end_token_id"
        ],
        "user_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["thinker_config.user_token_id"],
        "position_id_per_seconds": 13,
        "seconds_per_chunk": 2,
    }


def _talker_text_config() -> dict[str, Any]:
    return {
        "vocab_size": OFFICIAL_VOCAB_SIZES["talker_text"],
        "hidden_size": TINY_DIMS["talker_hidden_size"],
        "intermediate_size": TINY_DIMS["intermediate_size"],
        "moe_intermediate_size": TINY_DIMS["intermediate_size"],
        # Required by Qwen3OmniMoeTalkerTextSparseMoeBlock's shared expert.
        "shared_expert_intermediate_size": TINY_DIMS["intermediate_size"],
        "num_hidden_layers": TINY_DIMS["talker_layers"],
        "num_attention_heads": TINY_DIMS["thinker_heads"],
        "num_key_value_heads": TINY_DIMS["thinker_kv_heads"],
        "head_dim": TINY_DIMS["head_dim"],
        "num_experts": TINY_DIMS["talker_experts"],
        "num_experts_per_tok": TINY_DIMS["talker_top_k"],
        "norm_topk_prob": True,
        "decoder_sparse_step": 1,
        "mlp_only_layers": [],
        "max_position_embeddings": OFFICIAL_CONTEXT_LENGTHS["talker_text"],
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "use_cache": True,
        "rope_parameters": _thinker_rope_parameters(),
    }


def _code_predictor_config() -> dict[str, Any]:
    return {
        "vocab_size": OFFICIAL_VOCAB_SIZES["code_predictor"],
        "hidden_size": TINY_DIMS["talker_hidden_size"],
        "intermediate_size": TINY_DIMS["intermediate_size"],
        "num_hidden_layers": TINY_DIMS["predictor_layers"],
        "num_attention_heads": TINY_DIMS["thinker_heads"],
        "num_key_value_heads": TINY_DIMS["thinker_kv_heads"],
        "head_dim": TINY_DIMS["head_dim"],
        "num_code_groups": TINY_DIMS["code_groups"],
        "max_position_embeddings": OFFICIAL_CONTEXT_LENGTHS["code_predictor"],
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": False,
        "use_cache": True,
        "rope_parameters": {"rope_type": "default", "rope_theta": 1000000.0},
    }


def _talker_config() -> dict[str, Any]:
    return {
        "text_config": _talker_text_config(),
        "code_predictor_config": _code_predictor_config(),
        "num_code_groups": TINY_DIMS["code_groups"],
        "thinker_hidden_size": TINY_DIMS["thinker_hidden_size"],
        # The production speech path captures thinker hidden layers [0, 24].
        "accept_hidden_layer": 24,
        # Not a declared field on Qwen3OmniMoeTalkerConfig, but read by
        # Qwen3OmniMoeTalkerForConditionalGeneration.__init__.
        "spatial_merge_size": _vision_config()["spatial_merge_size"],
        "speaker_id": dict(OFFICIAL_SPEAKER_IDS),
        "position_id_per_seconds": 13,
        "seconds_per_chunk": 2,
        "output_router_logits": False,
        "audio_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.audio_token_id"],
        "image_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.image_token_id"],
        "video_token_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.video_token_id"],
        "vision_start_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.vision_start_token_id"
        ],
        "audio_start_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.audio_start_token_id"
        ],
        "audio_end_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.audio_end_token_id"
        ],
        "codec_pad_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.codec_pad_id"],
        "codec_bos_id": OFFICIAL_SPECIAL_TOKEN_IDS["talker_config.codec_bos_id"],
        "codec_eos_token_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_eos_token_id"
        ],
        "codec_nothink_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_nothink_id"
        ],
        "codec_think_bos_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_think_bos_id"
        ],
        "codec_think_eos_id": OFFICIAL_SPECIAL_TOKEN_IDS[
            "talker_config.codec_think_eos_id"
        ],
    }


def _code2wav_config() -> dict[str, Any]:
    return {
        "codebook_size": OFFICIAL_VOCAB_SIZES["codec_codebook"],
        "num_quantizers": TINY_DIMS["code_groups"],
        "hidden_size": TINY_DIMS["code2wav_hidden_size"],
        "intermediate_size": TINY_DIMS["intermediate_size"],
        "num_hidden_layers": TINY_DIMS["code2wav_layers"],
        "num_attention_heads": TINY_DIMS["code2wav_heads"],
        "num_key_value_heads": TINY_DIMS["code2wav_heads"],
        "decoder_dim": TINY_DIMS["code2wav_decoder_dim"],
        "upsample_rates": [8, 5, 4, 3],
        "upsampling_ratios": [2, 2],
        "max_position_embeddings": OFFICIAL_CODE2WAV_MAX_POSITION_EMBEDDINGS,
        "sliding_window": 72,
        "rms_norm_eps": 1e-5,
        "layer_scale_initial_scale": 0.01,
        "rope_parameters": {"rope_type": "default", "rope_theta": 10000.0},
    }


def build_tiny_config() -> Qwen3OmniMoeConfig:
    """Return the reduced but internally valid full Qwen3-Omni config."""

    config = Qwen3OmniMoeConfig(
        thinker_config=_thinker_config(),
        talker_config=_talker_config(),
        code2wav_config=_code2wav_config(),
        enable_audio_output=True,
        im_start_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["im_start_token_id"],
        im_end_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["im_end_token_id"],
        tts_pad_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["tts_pad_token_id"],
        tts_bos_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["tts_bos_token_id"],
        tts_eos_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["tts_eos_token_id"],
        system_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["system_token_id"],
        user_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["user_token_id"],
        assistant_token_id=OFFICIAL_SPECIAL_TOKEN_IDS["assistant_token_id"],
    )
    _assert_config_is_consistent(config)
    return config


def _assert_config_is_consistent(config: Qwen3OmniMoeConfig) -> None:
    thinker_text = config.thinker_config.text_config
    talker_text = config.talker_config.text_config
    predictor = config.talker_config.code_predictor_config
    hidden = thinker_text.hidden_size

    if config.thinker_config.vision_config.out_hidden_size != hidden:
        raise ValueError("vision out_hidden_size must match thinker hidden size")
    if config.thinker_config.audio_config.output_dim != hidden:
        raise ValueError("audio output_dim must match thinker hidden size")
    if config.talker_config.thinker_hidden_size != hidden:
        raise ValueError("talker thinker_hidden_size must match thinker hidden size")
    if talker_text.hidden_size != predictor.hidden_size:
        raise ValueError("code predictor hidden size must match talker hidden size")
    if (
        config.talker_config.spatial_merge_size
        != config.thinker_config.vision_config.spatial_merge_size
    ):
        raise ValueError("talker spatial_merge_size must match the vision encoder")
    if config.talker_config.num_code_groups != predictor.num_code_groups:
        raise ValueError("talker and code predictor num_code_groups must match")
    if config.code2wav_config.num_quantizers != config.talker_config.num_code_groups:
        raise ValueError("code2wav num_quantizers must match talker num_code_groups")
    if talker_text.shared_expert_intermediate_size <= 0:
        raise ValueError("talker text shared_expert_intermediate_size must be positive")

    accept_hidden_layer = config.talker_config.accept_hidden_layer
    if thinker_text.num_hidden_layers <= accept_hidden_layer:
        raise ValueError(
            "thinker must retain more layers than the talker accept_hidden_layer "
            f"({accept_hidden_layer})"
        )

    if thinker_text.max_position_embeddings != OFFICIAL_CONTEXT_LENGTHS["thinker_text"]:
        raise ValueError(
            "thinker text max_position_embeddings must stay at the official 65536"
        )
    if talker_text.max_position_embeddings != OFFICIAL_CONTEXT_LENGTHS["talker_text"]:
        raise ValueError(
            "talker text max_position_embeddings must stay at the official 65536"
        )
    if predictor.max_position_embeddings != OFFICIAL_CONTEXT_LENGTHS["code_predictor"]:
        raise ValueError(
            "code predictor max_position_embeddings must stay at the official 32768"
        )
    if (
        config.get_text_config().max_position_embeddings
        != thinker_text.max_position_embeddings
    ):
        raise ValueError("root-derived context length disagrees with thinker text")

    if sum(_MROPE_SECTION) * 2 != thinker_text.head_dim:
        raise ValueError("mrope_section must sum to head_dim // 2")

    for dotted, expected in OFFICIAL_SPECIAL_TOKEN_IDS.items():
        actual = _resolve_config_attr(config, dotted)
        if actual != expected:
            raise ValueError(f"{dotted} must stay at the official id {expected}")
        limit = (
            talker_text.vocab_size
            if dotted in _CODEC_SCOPED_IDS
            else thinker_text.vocab_size
        )
        if not 0 <= expected < limit:
            raise ValueError(f"{dotted}={expected} exceeds retained vocabulary {limit}")

    for name, speaker_id in OFFICIAL_SPEAKER_IDS.items():
        if not 0 <= speaker_id < talker_text.vocab_size:
            raise ValueError(f"speaker {name}={speaker_id} exceeds talker vocabulary")


def _resolve_config_attr(config: Qwen3OmniMoeConfig, dotted: str) -> Any:
    node: Any = config
    for part in dotted.split("."):
        node = getattr(node, part)
    return node


@contextlib.contextmanager
def _nan_filled_uninitialized_memory() -> Iterator[None]:
    """Poison uninitialized allocations with NaN for the duration of the block."""

    previous_fill = torch.utils.deterministic.fill_uninitialized_memory
    previous_mode = torch.are_deterministic_algorithms_enabled()
    previous_warn_only = torch.is_deterministic_algorithms_warn_only_enabled()
    torch.utils.deterministic.fill_uninitialized_memory = True
    torch.use_deterministic_algorithms(True)
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(previous_mode, warn_only=previous_warn_only)
        torch.utils.deterministic.fill_uninitialized_memory = previous_fill


@torch.no_grad()
def _initialize_weights_upstream_missed(
    model: Qwen3OmniMoeForConditionalGeneration,
    config: Qwen3OmniMoeConfig,
) -> None:
    """Seed every float tensor that upstream ``_init_weights`` left untouched."""

    generator = torch.Generator(device="cpu").manual_seed(SEED)
    std = config.talker_config.text_config.initializer_range
    for parameter in model.parameters():
        if not parameter.is_floating_point() or torch.isfinite(parameter).all():
            continue
        parameter.normal_(mean=0.0, std=std, generator=generator)


# The vocoder is a deep upsampling conv stack: with every weight drawn at
# ``initializer_range`` (0.02) the per-layer gain compounds to roughly 1e-7 peak
_CODE2WAV_INIT_GAIN = 0.75

# Deterministic probe used by the build-time audibility guard.
_CODE2WAV_PROBE_FRAMES = 24
_CODE2WAV_PROBE_SEED = 0
_PCM_FULL_SCALE = 32767


@torch.no_grad()
def _rescale_code2wav_to_audible_gain(
    model: Qwen3OmniMoeForConditionalGeneration,
) -> None:
    """Give the randomly initialised vocoder a non-vanishing transfer gain."""

    from torch.nn.init import _calculate_fan_in_and_fan_out

    for name, parameter in model.code2wav.named_parameters():
        if parameter.dim() < 2 or not parameter.is_floating_point():
            continue
        current_std = float(parameter.std())
        if current_std == 0.0:
            raise RuntimeError(f"code2wav parameter {name} is constant; cannot rescale")
        fan_in, _ = _calculate_fan_in_and_fan_out(parameter)
        parameter.mul_(_CODE2WAV_INIT_GAIN / (fan_in**0.5) / current_std)


@torch.no_grad()
def _assert_code2wav_emits_audible_audio(
    model: Qwen3OmniMoeForConditionalGeneration,
    config: Qwen3OmniMoeConfig,
) -> None:
    """Refuse to save a checkpoint whose vocoder decodes to silence."""

    num_quantizers = int(config.code2wav_config.num_quantizers)
    generator = torch.Generator(device="cpu").manual_seed(_CODE2WAV_PROBE_SEED)
    codes = torch.randint(
        0,
        int(config.code2wav_config.codebook_size),
        (1, num_quantizers, _CODE2WAV_PROBE_FRAMES),
        generator=generator,
        dtype=torch.long,
    )
    waveform = model.code2wav(codes).float()
    if not torch.isfinite(waveform).all():
        raise RuntimeError("code2wav produced non-finite audio for the probe codes")
    pcm = (waveform.clamp(-1.0, 1.0) * _PCM_FULL_SCALE).round().to(torch.int16)
    nonzero = int((pcm != 0).sum())
    if nonzero <= pcm.numel() // 2:
        raise RuntimeError(
            "code2wav decodes to 16-bit silence: only "
            f"{nonzero}/{pcm.numel()} samples survive PCM quantisation"
        )
    peak = float(waveform.abs().max())
    if peak >= 1.0:
        raise RuntimeError(f"code2wav probe audio clips the 16-bit rail (peak {peak})")


def _assert_weights_are_initialized(
    model: Qwen3OmniMoeForConditionalGeneration,
) -> None:
    uninitialized = [
        name
        for name, tensor in (*model.named_parameters(), *model.named_buffers())
        if tensor.is_floating_point() and not torch.isfinite(tensor).all()
    ]
    if uninitialized:
        raise RuntimeError(
            "refusing to save a checkpoint with uninitialized weights: "
            f"{', '.join(uninitialized)}"
        )


def _assert_model_is_complete(model: Qwen3OmniMoeForConditionalGeneration) -> None:
    if not model.has_talker:
        raise RuntimeError("talker was not enabled; refusing to save a partial model")
    if getattr(model, "talker", None) is None:
        raise RuntimeError("talker module is missing; refusing to save a partial model")
    if getattr(model, "code2wav", None) is None:
        raise RuntimeError("code2wav module is missing; refusing to save partial model")
    if getattr(model.talker, "code_predictor", None) is None:
        raise RuntimeError("talker code predictor is missing from the model")


def _assert_saved_checkpoint_is_complete(output_dir: Path) -> None:
    keys: set[str] = set()
    shards = sorted(output_dir.rglob("*.safetensors"))
    if not shards:
        raise RuntimeError(f"no safetensors shard was written to {output_dir}")
    for shard in shards:
        with safe_open(str(shard), framework="pt", device="cpu") as handle:
            keys.update(handle.keys())

    missing = [
        prefix
        for prefix in _REQUIRED_WEIGHT_PREFIXES
        if not any(key.startswith(prefix) for key in keys)
    ]
    if missing:
        raise RuntimeError(
            f"saved checkpoint {output_dir} is missing weights for: "
            f"{', '.join(missing)}"
        )


def _clear_stale_shards(output_dir: Path) -> None:
    for stale in output_dir.glob("*.safetensors"):
        stale.unlink()
    for stale in output_dir.glob("*.safetensors.index.json"):
        stale.unlink()


def _copy_processor_assets(output_dir: Path, processor_source: str) -> None:
    source = Path(processor_source)
    if not source.is_dir():
        from huggingface_hub import snapshot_download

        source = Path(
            snapshot_download(
                processor_source,
                allow_patterns=list(_PROCESSOR_ASSET_PATTERNS),
            )
        )

    copied = 0
    for pattern in _PROCESSOR_ASSET_PATTERNS:
        asset = source / pattern
        if asset.is_file():
            shutil.copy2(asset, output_dir / pattern)
            copied += 1
    if copied == 0:
        raise RuntimeError(
            f"no tokenizer or processor assets were found in {processor_source!r}"
        )


def build_tiny_qwen3_omni_checkpoint(
    output_dir: Path,
    *,
    processor_source: str | None = None,
) -> Path:
    """Materialise the tiny checkpoint under ``output_dir`` and return the path."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _clear_stale_shards(output_dir)

    config = build_tiny_config()
    with _nan_filled_uninitialized_memory():
        torch.manual_seed(SEED)
        model = Qwen3OmniMoeForConditionalGeneration(config)
        _initialize_weights_upstream_missed(model, config)
    _rescale_code2wav_to_audible_gain(model)
    _assert_weights_are_initialized(model)
    _assert_model_is_complete(model)
    _assert_code2wav_emits_audible_audio(model, config)

    model.save_pretrained(output_dir, safe_serialization=True)
    _assert_saved_checkpoint_is_complete(output_dir)

    if processor_source is not None:
        _copy_processor_assets(output_dir, processor_source)

    return output_dir


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build a deterministic tiny Qwen3-Omni checkpoint.",
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Directory that receives the generated checkpoint.",
    )
    parser.add_argument(
        "--processor-source",
        default=None,
        help=(
            "Hugging Face repo id or local directory to copy tokenizer and "
            "processor assets from."
        ),
    )
    args = parser.parse_args(argv)

    path = build_tiny_qwen3_omni_checkpoint(
        args.output.expanduser(),
        processor_source=args.processor_source,
    )
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
