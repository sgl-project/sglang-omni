# SPDX-License-Identifier: Apache-2.0
"""Hugging Face config adapter for Zyphra ZONOS2 checkpoints."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any

from transformers import AutoConfig, PretrainedConfig

ARCHITECTURE = "Zonos2SGLangModel"


def _round_ffn(dim: int, multiplier: float, multiple_of: int) -> int:
    hidden = int(dim * multiplier)
    return multiple_of * ((hidden + multiple_of - 1) // multiple_of)


class Zonos2Config(PretrainedConfig):
    """HF-style config for ZONOS2's native ``params.json``."""

    model_type = "zonos2"

    def __init__(
        self,
        *,
        n_layers: int = 28,
        dim: int = 2048,
        head_dim: int = 128,
        n_heads: int | None = None,
        n_kv_heads: int = 4,
        ffn_dim_multiplier: float = 1.5,
        multiple_of: int = 256,
        norm_eps: float = 1e-5,
        rope_theta: float = 10000.0,
        max_seqlen: int = 6144,
        dtype: str = "bfloat16",
        n_codebooks: int = 9,
        codebook_size: int = 1024,
        eoa_id: int = 1024,
        audio_pad_id: int = 1025,
        text_vocab: int = 519,
        loss_softcap: float = 15.0,
        speaker_enabled: bool = True,
        speaker_embedding_dim: int = 2048,
        speaker_lda_dim: int = 1024,
        speaker_background_token_enabled: bool = True,
        accurate_mode_token_enabled: bool = True,
        speaking_rate_num_buckets: int = 8,
        speaking_rate_buckets: list[str] | None = None,
        quality_num_buckets: int = 60,
        quality_features: list[str] | None = None,
        quality_buckets: dict[str, list[str]] | None = None,
        moe_impl: str = "sonic",
        moe_n_experts: int = 16,
        moe_router_topk: int = 1,
        special_topk_layers: dict[str, int] | None = None,
        moe_router_dim: int = 128,
        moe_start_from_layer: int = 3,
        moe_end_from_layer: int = 1,
        moe_balancing_strategy: str | None = None,
        **kwargs: Any,
    ) -> None:
        hidden_size = kwargs.pop("hidden_size", None)
        if hidden_size is not None and dim == 2048:
            dim = int(hidden_size)
        num_hidden_layers = kwargs.pop("num_hidden_layers", None)
        if num_hidden_layers is not None and n_layers == 28:
            n_layers = int(num_hidden_layers)
        num_attention_heads = kwargs.pop("num_attention_heads", None)
        if num_attention_heads is not None and n_heads is None:
            n_heads = int(num_attention_heads)
        num_key_value_heads = kwargs.pop("num_key_value_heads", None)
        if num_key_value_heads is not None and n_kv_heads == 4:
            n_kv_heads = int(num_key_value_heads)
        max_position_embeddings = kwargs.pop("max_position_embeddings", None)
        if max_position_embeddings is not None and max_seqlen == 6144:
            max_seqlen = int(max_position_embeddings)
        rms_norm_eps = kwargs.pop("rms_norm_eps", None)
        if rms_norm_eps is not None and norm_eps == 1e-5:
            norm_eps = float(rms_norm_eps)

        intermediate_size = kwargs.pop("intermediate_size", None)
        self.n_layers = n_layers
        self.dim = dim
        self.head_dim = head_dim
        self.n_heads = n_heads if n_heads else dim // head_dim
        self.n_kv_heads = n_kv_heads
        self.ffn_dim_multiplier = ffn_dim_multiplier
        self.multiple_of = multiple_of
        self.norm_eps = norm_eps
        self.rope_theta = rope_theta
        self.max_seqlen = max_seqlen
        self.zonos_dtype = dtype

        self.n_codebooks = n_codebooks
        self.codebook_size = codebook_size
        self.eoa_id = eoa_id
        self.audio_pad_id = audio_pad_id
        self.audio_vocab = codebook_size + 2
        self.text_vocab = text_vocab
        self.loss_softcap = loss_softcap

        self.speaker_enabled = speaker_enabled
        self.speaker_embedding_dim = speaker_embedding_dim
        self.speaker_lda_dim = speaker_lda_dim
        self.speaker_background_token_enabled = speaker_background_token_enabled
        self.accurate_mode_token_enabled = accurate_mode_token_enabled

        self.speaking_rate_num_buckets = speaking_rate_num_buckets
        self.speaking_rate_buckets = speaking_rate_buckets or []
        self.quality_num_buckets = quality_num_buckets
        self.quality_features = quality_features or []
        self.quality_buckets = quality_buckets or {}

        self.moe_impl = moe_impl
        self.moe_n_experts = moe_n_experts
        self.moe_router_topk = moe_router_topk
        self.special_topk_layers = {
            int(k): int(v) for k, v in (special_topk_layers or {}).items()
        }
        self.moe_router_dim = moe_router_dim
        self.moe_start_from_layer = moe_start_from_layer
        self.moe_end_from_layer = moe_end_from_layer
        self.moe_balancing_strategy = moe_balancing_strategy

        self.intermediate_size = (
            int(intermediate_size)
            if intermediate_size is not None
            else _round_ffn(dim, ffn_dim_multiplier, multiple_of)
        )

        self.hidden_size = dim
        self.num_hidden_layers = n_layers
        self.num_attention_heads = self.n_heads
        self.num_key_value_heads = n_kv_heads
        self.max_position_embeddings = max_seqlen
        self.rms_norm_eps = norm_eps
        self.vocab_size = self.audio_vocab

        kwargs.pop("vocab_size", None)
        kwargs.setdefault("architectures", [ARCHITECTURE])
        super().__init__(**kwargs)


def _normalize_special_topk_layers(value: Any) -> dict[int, int] | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("special_topk_layers must be a mapping")
    normalized: dict[int, int] = {}
    for layer_idx, topk in value.items():
        topk_i = int(topk)
        if topk_i < 1:
            raise ValueError(
                f"special_topk_layers[{layer_idx!r}] must be >= 1, got {topk_i}"
            )
        normalized[int(layer_idx)] = topk_i
    return normalized


def _normalize_moe_balancing_strategy(strategy: Any) -> str:
    normalized = str(strategy or "legacy").strip().lower().replace("-", "_")
    aliases = {
        "current": "quantile",
        "quantile": "quantile",
        "qbalancing": "quantile",
        "old": "legacy",
        "legacy": "legacy",
        "aux": "legacy",
        "aux_loss": "legacy",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(
            f"Unsupported ZONOS2 moe_balancing_strategy={strategy!r}"
        ) from exc


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, dict):
        return cfg.get(key, default)
    return getattr(cfg, key, default)


def _load_yaml_config(path: Path) -> Any:
    try:
        from omegaconf import OmegaConf

        return OmegaConf.load(path)
    except ImportError:
        import yaml

        with path.open(encoding="utf-8") as f:
            return yaml.safe_load(f)


def _conditioned_text_vocab_size(
    speaking_rate_num_buckets: int,
    quality_num_buckets: int,
    speaker_background_num_buckets: int,
    accurate_mode_num_buckets: int,
) -> int:
    return (
        448
        + int(speaking_rate_num_buckets)
        + int(quality_num_buckets)
        + int(speaker_background_num_buckets)
        + int(accurate_mode_num_buckets)
    )


def _apply_data_sidecar(model_params: dict[str, Any], data_cfg: Any) -> None:
    if data_cfg is None:
        return

    rate_buckets = _cfg_get(data_cfg, "speaking_rate_buckets", None) or []
    rate_buckets = [str(item) for item in rate_buckets]
    if rate_buckets:
        model_params["speaking_rate_buckets"] = rate_buckets

    rate_count = int(model_params.get("speaking_rate_num_buckets") or 0)
    if bool(_cfg_get(data_cfg, "speaking_rate_enabled", False)):
        sidecar_rate_count = len(rate_buckets) or int(
            _cfg_get(data_cfg, "speaking_rate_num_buckets", 0) or 0
        )
        if sidecar_rate_count > 0:
            rate_count = sidecar_rate_count
            if not int(model_params.get("speaking_rate_num_buckets") or 0):
                model_params["speaking_rate_num_buckets"] = rate_count

    if bool(_cfg_get(data_cfg, "quality_enabled", False)):
        raw_features = _cfg_get(data_cfg, "quality_features", None)
        if hasattr(raw_features, "items"):
            quality_features = [
                str(feature)
                for feature, enabled in raw_features.items()
                if bool(enabled)
            ]
        else:
            quality_features = [str(item) for item in (raw_features or ())]
        raw_buckets = _cfg_get(data_cfg, "quality_buckets", None) or {}
        quality_buckets = {
            str(feature): [
                str(item) for item in (raw_buckets.get(feature, None) or ())
            ]
            for feature in (quality_features or raw_buckets.keys())
        }
        if quality_buckets and "quality_buckets" not in model_params:
            model_params["quality_buckets"] = quality_buckets
            model_params["quality_features"] = quality_features or list(
                quality_buckets.keys()
            )
        raw_dropout = _cfg_get(data_cfg, "quality_dropout", None)
        if raw_dropout is not None and "quality_dropout" not in model_params:
            if hasattr(raw_dropout, "items"):
                model_params["quality_dropout"] = {
                    str(feature): float(dropout)
                    for feature, dropout in raw_dropout.items()
                }

    background_enabled = _cfg_get(
        data_cfg, "speaker_embedding_origin_token_enabled", None
    )
    if background_enabled is not None:
        model_params.setdefault(
            "speaker_background_token_enabled", bool(background_enabled)
        )
    accurate_enabled = _cfg_get(
        data_cfg, "speaker_embedding_cartesia_clone_source_token_enabled", None
    )
    if accurate_enabled is not None:
        model_params.setdefault("accurate_mode_token_enabled", bool(accurate_enabled))

    if rate_count > 0 and model_params.get("text_vocab") is None:
        quality_count = sum(
            len(buckets)
            for buckets in (model_params.get("quality_buckets") or {}).values()
        )
        background_count = (
            2 if model_params.get("speaker_background_token_enabled") else 0
        )
        accurate_count = (
            1
            if model_params.get("accurate_mode_token_enabled") and background_count
            else 0
        )
        model_params["text_vocab"] = _conditioned_text_vocab_size(
            rate_count, quality_count, background_count, accurate_count
        )


def _normalize_zonos2_params(params: dict[str, Any]) -> dict[str, Any]:
    model_type = params.get("model_type")
    if model_type is not None and str(model_type) != "zonos2":
        raise ValueError(f"Unsupported ZONOS2 model_type={model_type!r}")
    params = dict(params)
    if "special_topk_layers" in params:
        params["special_topk_layers"] = _normalize_special_topk_layers(
            params["special_topk_layers"]
        )
    params["moe_balancing_strategy"] = _normalize_moe_balancing_strategy(
        params.get("moe_balancing_strategy", "legacy")
    )
    return params


def register_zonos2_hf_config() -> None:
    """Register the local ZONOS2 config class with Transformers."""

    try:
        AutoConfig.register(Zonos2Config.model_type, Zonos2Config)
    except ValueError:
        pass


def load_zonos2_params(checkpoint_dir: str | os.PathLike[str]) -> dict[str, Any]:
    checkpoint = Path(checkpoint_dir)
    params_path = checkpoint / "params.json"
    if not params_path.is_file():
        raise FileNotFoundError(f"ZONOS2 params.json not found under {checkpoint_dir}")
    with params_path.open(encoding="utf-8") as f:
        raw_params = json.load(f)
    if not isinstance(raw_params, dict):
        raise ValueError(f"ZONOS2 params.json must contain an object: {params_path}")
    params = (
        raw_params.get("model")
        if isinstance(raw_params.get("model"), dict)
        else raw_params
    )
    params = dict(params)

    for parent in (checkpoint, checkpoint.parent, checkpoint.parent.parent):
        config_yaml = parent / "config.yaml"
        if not config_yaml.is_file():
            continue
        cfg = _load_yaml_config(config_yaml)
        _apply_data_sidecar(params, _cfg_get(cfg, "data", None))
        break

    return _normalize_zonos2_params(params)


def build_zonos2_hf_config_dict(params: dict[str, Any]) -> dict[str, Any]:
    cfg = Zonos2Config(**params)
    data = cfg.to_dict()
    data["model_type"] = Zonos2Config.model_type
    data["architectures"] = [ARCHITECTURE]
    data["enable_decode_state_pool"] = True
    return data


def ensure_zonos2_hf_layout(checkpoint_dir: str | os.PathLike[str]) -> str:
    """Return a temp directory with HF-style config and weight entrypoints.

    Zyphra publishes ``params.json`` + ``model.pth``. SGLang expects a
    Transformers-readable ``config.json`` and a standard weight filename, so we
    create a deterministic temp view with symlinks back to the original files.
    """

    checkpoint = Path(checkpoint_dir).resolve()
    params = load_zonos2_params(checkpoint)
    digest = hashlib.blake2b(
        str(checkpoint).encode("utf-8"), digest_size=12
    ).hexdigest()
    layout_dir = Path(tempfile.gettempdir()) / "sglang_omni_zonos2" / digest
    layout_dir.mkdir(parents=True, exist_ok=True)

    config_path = layout_dir / "config.json"
    config_data = build_zonos2_hf_config_dict(params)
    current = None
    if config_path.is_file():
        try:
            current = json.loads(config_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            current = None
    if current != config_data:
        config_path.write_text(
            json.dumps(config_data, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    params_link = layout_dir / "params.json"
    _ensure_link(params_link, checkpoint / "params.json")

    weight_path = _single_file_checkpoint(checkpoint)
    if weight_path is None:
        raise FileNotFoundError(
            f"ZONOS2 weights not found under {checkpoint}; expected model.pth, "
            "model.pt, or consolidated/consolidated.pth"
        )
    _ensure_link(layout_dir / "pytorch_model.bin", weight_path)
    return str(layout_dir)


def _single_file_checkpoint(checkpoint: Path) -> Path | None:
    for name in ("model.pth", "model.pt", "consolidated/consolidated.pth"):
        candidate = checkpoint / name
        if candidate.is_file():
            return candidate
    return None


def _ensure_link(link: Path, target: Path) -> None:
    if link.exists() or link.is_symlink():
        try:
            if link.resolve() == target.resolve():
                return
        except FileNotFoundError:
            pass
        link.unlink()
    try:
        link.symlink_to(target)
    except OSError:
        os.link(target, link)
