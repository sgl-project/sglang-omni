# SPDX-License-Identifier: Apache-2.0
"""Load Irodori model, tokenizer, and audio codec for inference."""

import json
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file

from sglang_omni.models.irodori_tts.codec import DACVAECodec
from sglang_omni.models.irodori_tts.dit import TextToLatentRFDiT
from sglang_omni.models.irodori_tts.model_config import (
    ModelConfig,
    merge_dataclass_overrides,
)
from sglang_omni.models.irodori_tts.tokenizer import PretrainedTextTokenizer
from sglang_omni.utils.checkpoint import resolve_checkpoint

MODEL_CONFIG_METADATA_KEY = "config_json"
TEXT_CONFIG_METADATA_KEY = "text_encoder_config_json"


def read_checkpoint_metadata(checkpoint: str | Path) -> dict[str, str]:
    """Read configuration metadata embedded in the Irodori safetensors header."""
    with safe_open(str(checkpoint), framework="pt", device="cpu") as handle:
        return dict(handle.metadata() or {})


def checkpoint_file(model_path: str) -> Path:
    requested_path = Path(model_path).expanduser()
    if requested_path.is_file():
        resolved_path = requested_path
    elif requested_path.is_dir():
        resolved_path = requested_path / "model.safetensors"
    else:
        resolved_path = Path(resolve_checkpoint(model_path))
        if resolved_path.is_dir():
            resolved_path = resolved_path / "model.safetensors"
        else:
            pass
    if not resolved_path.is_file():
        raise FileNotFoundError(f"Irodori model.safetensors not found: {resolved_path}")
    else:
        pass
    return resolved_path.resolve()


def checkpoint_configuration(
    path: Path,
) -> tuple[dict[str, object], dict[str, object] | None]:
    metadata = read_checkpoint_metadata(path)
    raw_model_configuration = metadata.get(MODEL_CONFIG_METADATA_KEY)
    if raw_model_configuration is None:
        raise ValueError(f"Irodori checkpoint is missing config_json metadata: {path}")
    else:
        pass
    model_configuration = json.loads(raw_model_configuration)
    if not isinstance(model_configuration, dict):
        raise ValueError(f"Irodori config_json metadata must be an object: {path}")
    else:
        pass
    raw_text_configuration = metadata.get(TEXT_CONFIG_METADATA_KEY)
    if raw_text_configuration is None:
        text_configuration = None
    else:
        text_configuration = json.loads(raw_text_configuration)
        if not isinstance(text_configuration, dict):
            raise ValueError(
                f"Irodori text_encoder_config_json metadata must be an object: {path}"
            )
        else:
            pass
    return model_configuration, text_configuration


def precision_dtype(precision: str, device: torch.device) -> torch.dtype:
    normalized_precision = precision.strip().lower()
    if normalized_precision == "fp32":
        return torch.float32
    elif normalized_precision == "bf16":
        if device.type not in ("cuda", "xpu"):
            raise ValueError("Irodori bf16 precision requires a CUDA or XPU device")
        else:
            pass
        return torch.bfloat16
    else:
        raise ValueError(f"Unsupported Irodori precision: {precision}")


def tokenizer_source(checkpoint_path: Path, fallback_repo: str) -> tuple[str, bool]:
    tokenizer_path = checkpoint_path.parent / "tokenizer"
    if (tokenizer_path / "tokenizer_config.json").is_file():
        return str(tokenizer_path), True
    else:
        return fallback_repo, False


def load_model(
    checkpoint_path: Path,
    device: torch.device,
    model_precision: str,
    codec_repo: str,
    codec_precision: str,
) -> tuple[
    ModelConfig,
    TextToLatentRFDiT,
    PretrainedTextTokenizer,
    PretrainedTextTokenizer | None,
    DACVAECodec,
    int,
    int,
    float,
]:
    raw_configuration, text_encoder_configuration = checkpoint_configuration(
        checkpoint_path
    )
    inference_configuration: dict[str, object] = {}
    for key in ("max_text_len", "max_caption_len", "ref_max_seconds"):
        value = raw_configuration.pop(key, None)
        if value is not None:
            inference_configuration[key] = value
        else:
            pass
    model_configuration = merge_dataclass_overrides(
        ModelConfig(), raw_configuration, section="Irodori checkpoint model_config"
    )
    model = TextToLatentRFDiT(
        model_configuration,
        pretrained_backbone_config=text_encoder_configuration,
        load_pretrained_backbone_weights=(
            not model_configuration.use_pretrained_text_encoder
        ),
    )
    model_state = load_file(str(checkpoint_path), device="cpu")
    model.load_state_dict(
        model_state,
        assign=model_configuration.use_pretrained_text_encoder,
    )
    model_dtype = precision_dtype(model_precision, device)
    model = model.to(device=device, dtype=model_dtype)
    model.eval()

    text_tokenizer_path, text_tokenizer_is_local = tokenizer_source(
        checkpoint_path, model_configuration.text_tokenizer_repo
    )
    tokenizer = PretrainedTextTokenizer.from_pretrained(
        text_tokenizer_path,
        add_bos=bool(model_configuration.text_add_bos),
        local_files_only=text_tokenizer_is_local,
        revision=(
            None
            if text_tokenizer_is_local
            else model_configuration.text_encoder_revision
        ),
    )
    if (
        not model_configuration.use_pretrained_text_encoder
        and tokenizer.vocab_size != model_configuration.text_vocab_size
    ):
        raise ValueError(
            "Irodori text tokenizer vocabulary size does not match the checkpoint"
        )
    else:
        pass

    caption_tokenizer = None
    if model_configuration.use_caption_condition:
        caption_tokenizer_path, caption_tokenizer_is_local = tokenizer_source(
            checkpoint_path, model_configuration.caption_tokenizer_repo_resolved
        )
        caption_tokenizer = PretrainedTextTokenizer.from_pretrained(
            caption_tokenizer_path,
            add_bos=model_configuration.caption_add_bos_resolved,
            local_files_only=caption_tokenizer_is_local,
            revision=(
                None
                if caption_tokenizer_is_local
                else model_configuration.text_encoder_revision
            ),
        )
    else:
        pass

    codec_device = str(device)
    codec_dtype = precision_dtype(codec_precision, device)
    codec = DACVAECodec.load(
        repo_id=codec_repo,
        device=codec_device,
        dtype=codec_dtype,
        deterministic_encode=True,
        deterministic_decode=True,
    )
    if model_configuration.latent_dim != codec.latent_dim:
        raise ValueError(
            "Irodori checkpoint latent dimension does not match "
            "the selected DACVAE codec"
        )
    else:
        pass
    default_text_max_len = int(inference_configuration.get("max_text_len", 256))
    default_caption_max_len = int(
        inference_configuration.get("max_caption_len", default_text_max_len)
    )
    default_max_ref_seconds = float(
        inference_configuration.get("ref_max_seconds", 30.0)
    )
    return (
        model_configuration,
        model,
        tokenizer,
        caption_tokenizer,
        codec,
        default_text_max_len,
        default_caption_max_len,
        default_max_ref_seconds,
    )
