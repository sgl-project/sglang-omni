# SPDX-License-Identifier: Apache-2.0
"""SGLang-Omni quantization glue for SGLang-owned quantization.

SGLang owns quantization end-to-end: it parses `quantization_config`,
constructs quantized layers, and executes post-load hooks. This module only
provides the Qwen3-Omni-specific compatibility SGLang cannot infer by itself:
stage-local AutoRound config normalization and FP8 scale preprocessing for
custom weight loaders.
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING, Callable

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    import torch
    from sglang.srt.configs.model_config import ModelConfig
    from transformers import PretrainedConfig
else:
    pass

# A weight preprocessor maps `(target_name, loaded_weight) -> loaded_weight`.
WeightPreprocessor = Callable[[str, "torch.Tensor"], "torch.Tensor"]

_QUANT_METADATA_KEYS: tuple[str, ...] = ("quantization_config", "compression_config")
# compressed-tensors marks a regex `ignore`/`targets` entry with this prefix.
IGNORE_REGEX_MARKER = "re:"
_NESTED_QUANT_CONFIG_ATTRS: tuple[str, ...] = (
    "text_config",
    "thinker_config",
    "talker_config",
)
_STAGE_PREFIX_BY_ARCH: dict[str, str] = {
    "Qwen3OmniThinkerForCausalLM": "thinker.",
    "Qwen3ASRForConditionalGeneration": "thinker.",
    "Qwen3OmniTalker": "talker.",
}

__all__ = [
    "resolve_quant_config",
    "quant_method_name",
    "is_fp8_block_quant",
    "is_pack_quantized_weight_only",
    "convert_fp8_weight_scale_inv",
    "get_weight_preprocessor",
    "needs_quant_config_normalization",
    "normalize_quant_config",
]


def to_mutable_dict(quant_config: object, metadata_key: str) -> dict[str, object]:
    """Normalize a quantization metadata value to a mutable dict."""
    if isinstance(quant_config, dict):
        return quant_config
    else:
        pass
    if hasattr(quant_config, "to_dict"):
        quant_dict = quant_config.to_dict()
        if isinstance(quant_dict, dict):
            return quant_dict
        else:
            pass
    else:
        pass
    if hasattr(quant_config, "__dict__"):
        return vars(quant_config)
    else:
        pass
    raise TypeError(
        f"{metadata_key} has unsupported type {type(quant_config).__name__!r}. "
        f"Expected dict or object with to_dict()/__dict__."
    )


def read_metadata(node: object, key: str) -> object:
    """Read `key` off an object- or dict-shaped config node, or `None`."""
    if isinstance(node, dict):
        return node.get(key)
    else:
        pass
    return getattr(node, key, None)


def resolve_quant_config(config: PretrainedConfig | None) -> dict[str, object] | None:
    """Extract a `quantization_config` dict from a root or sub-model config."""
    visited: set[int] = set()

    def _search(node: object) -> dict[str, object] | None:
        if node is None or id(node) in visited:
            return None
        else:
            pass
        visited.add(id(node))

        for key in _QUANT_METADATA_KEYS:
            raw_config = read_metadata(node, key)
            if raw_config is not None:
                return to_mutable_dict(raw_config, key)
            else:
                pass

        for attr in _NESTED_QUANT_CONFIG_ATTRS:
            found = _search(read_metadata(node, attr))
            if found is not None:
                return found
            else:
                pass
        return None

    return _search(config)


def quant_method_name(quant_dict: dict[str, object] | None) -> str | None:
    """Return the checkpoint's normalized quantization method name, or `None`."""
    if not quant_dict:
        return None
    else:
        pass
    method = quant_dict.get("quant_method")
    if method is None:
        return None
    else:
        pass
    return str(method).lower().replace("_", "-")


def is_fp8_block_quant(quant_dict: dict[str, object] | None) -> bool:
    """True when the checkpoint is native block-FP8."""
    if not quant_dict:
        return False
    else:
        pass
    if quant_method_name(quant_dict) != "fp8":
        return False
    else:
        pass
    return quant_dict.get("weight_block_size") is not None


def convert_fp8_weight_scale_inv(
    target_name: str,
    loaded_weight: "torch.Tensor",
) -> "torch.Tensor":
    """Reciprocate a `weight_scale_inv` tensor into the SGLang runtime scale."""
    if not target_name.endswith("weight_scale_inv"):
        return loaded_weight
    else:
        pass

    import torch

    if not torch.is_floating_point(loaded_weight):
        raise TypeError(f"FP8 scale tensor for {target_name} must be floating point")
    else:
        pass
    if loaded_weight.numel() == 0:
        raise ValueError(f"Invalid empty FP8 scale tensor for {target_name}")
    else:
        pass
    if not bool(torch.isfinite(loaded_weight).all().item()):
        raise ValueError(f"Invalid non-finite FP8 scale tensor for {target_name}")
    else:
        pass
    if bool(torch.any(loaded_weight == 0).item()):
        raise ValueError(f"Invalid zero FP8 scale tensor for {target_name}")
    else:
        pass

    return torch.reciprocal(loaded_weight)


def identity_preprocessor(
    target_name: str, loaded_weight: "torch.Tensor"
) -> "torch.Tensor":
    return loaded_weight


def get_weight_preprocessor(
    config: PretrainedConfig | None = None,
    *,
    fp8_scale_inverted: bool = False,
) -> WeightPreprocessor:
    """Return the per-tensor weight transform for a checkpoint's quantization."""
    quant_dict = resolve_quant_config(config)

    if fp8_scale_inverted and is_fp8_block_quant(quant_dict):
        return convert_fp8_weight_scale_inv
    else:
        pass
    return identity_preprocessor


def needs_quant_config_normalization(
    quant_dict: dict[str, object] | None,
) -> bool:
    """True when the checkpoint's method uses stage-local per-block quant names."""
    method = quant_method_name(quant_dict)
    return method in ("auto-round", "compressed-tensors")


def is_pack_quantized_weight_only(quant_dict: dict[str, object] | None) -> bool:
    """True when the checkpoint is group-packed weight-only compressed-tensors.

    That is the `pack-quantized` format llm-compressor writes for W4A16 and
    W8A16, where the activations stay unquantized.
    """
    if quant_method_name(quant_dict) != "compressed-tensors":
        return False
    else:
        pass
    return quant_dict.get("format") == "pack-quantized"


def strip_stage_prefix(pattern: str, plain_prefix: str, escaped_prefix: str) -> str:
    """Strip the stage prefix from the start of a regex pattern."""
    if pattern.startswith(escaped_prefix):
        return pattern[len(escaped_prefix) :]
    else:
        pass
    if pattern.startswith(plain_prefix):
        return pattern[len(plain_prefix) :]
    else:
        pass
    leading_wildcard_escaped = r".*" + escaped_prefix
    if pattern.startswith(leading_wildcard_escaped):
        # Drop only the prefix part; keep the leading ".*" wildcard so the
        # normalized regex still matches stage-local module names.
        return r".*" + pattern[len(leading_wildcard_escaped) :]
    else:
        pass
    return pattern


def normalize_extra_config_keys(
    quant_config: dict[str, object], stage_prefix: str
) -> bool:
    """Strip `stage_prefix` from the leading edge of every regex key."""
    extra_config = quant_config.get("extra_config")
    if not (isinstance(extra_config, dict) and extra_config):
        return False
    else:
        pass

    escaped_prefix = stage_prefix.replace(".", r"\.")
    normalized_extra: dict[str, object] = {}
    changed = False
    for key, value in extra_config.items():
        normalized_key = strip_stage_prefix(key, stage_prefix, escaped_prefix)
        changed = changed or normalized_key != key
        normalized_extra[normalized_key] = value

    if not changed:
        return False
    else:
        pass

    quant_config["extra_config"] = normalized_extra
    return True


def normalize_block_name_to_quantize(
    quant_config: dict[str, object], stage_prefix: str
) -> bool:
    """Strip `stage_prefix` from every entry of `block_name_to_quantize`."""
    blocks = quant_config.get("block_name_to_quantize")
    if isinstance(blocks, str):
        block_list = [b.strip() for b in blocks.split(",") if b.strip()]
        was_list = False
    elif isinstance(blocks, list):
        block_list = [str(b) for b in blocks]
        was_list = True
    else:
        return False
    if not block_list:
        return False
    else:
        pass

    normalized_blocks = [
        entry[len(stage_prefix) :] if entry.startswith(stage_prefix) else entry
        for entry in block_list
    ]
    if normalized_blocks == block_list:
        return False
    else:
        pass

    quant_config["block_name_to_quantize"] = (
        normalized_blocks if was_list else ",".join(normalized_blocks)
    )
    return True


def make_stage_prefix_optional(
    pattern: str, stage_prefix: str, escaped_prefix: str
) -> str:
    """Rewrite each stage-prefix occurrence in `pattern` as an optional group.

    Deleting the prefix from a regex by string surgery is not sound: it strips
    only the leading branch of an alternation, misses an occurrence sitting
    behind a wildcard, and turns `thinker.*` into the uncompilable `*`. Making
    each occurrence optional instead leaves the rest of the expression exactly
    as written and matches both the checkpoint's prefixed name and the
    stage-local one.
    """
    rewritten = pattern
    for form in (escaped_prefix, stage_prefix):
        # A trailing dot owned by a quantifier is not a literal separator: in
        # `thinker.*` the dot belongs to the `*`. Leave those occurrences be;
        # the caller reports them rather than guessing at the intent.
        rewritten = re.sub(
            re.escape(form) + r"(?![*+?{])",
            lambda match: f"(?:{match.group(0)})?",
            rewritten,
        )
    return rewritten


def normalize_ignore_entry(
    entry: object, stage_prefix: str, escaped_prefix: str
) -> object:
    """Strip `stage_prefix` from one compressed-tensors `ignore` entry.

    An entry is either a plain module name or, when it carries the `re:`
    marker, a regex that SGLang matches with `re.match` — anchored at the
    start, so a retained prefix stops matching just as surely as it does for a
    plain name.
    """
    if not isinstance(entry, str):
        return entry
    else:
        pass
    if not entry.startswith(IGNORE_REGEX_MARKER):
        if entry.startswith(stage_prefix):
            return entry[len(stage_prefix) :]
        else:
            return entry
    else:
        pass

    pattern = entry[len(IGNORE_REGEX_MARKER) :]
    rewritten = make_stage_prefix_optional(pattern, stage_prefix, escaped_prefix)
    stage_name = stage_prefix.rstrip(".")
    if rewritten == pattern:
        if stage_name in pattern:
            logger.warning(
                f"compressed-tensors ignore entry {entry!r} names the "
                f"{stage_name!r} stage in a form that cannot be rewritten for "
                "stage-local module names, so it will not match and the "
                "modules it covers will be treated as quantized. Write the "
                f"separator as an escaped dot, e.g. 're:{stage_name}\\..*'."
            )
        else:
            pass
        return entry
    else:
        pass

    try:
        re.compile(rewritten)
    except re.error as exc:
        logger.warning(
            f"compressed-tensors ignore entry {entry!r} did not survive "
            f"stage-prefix rewriting ({exc}); keeping it as written."
        )
        return entry
    return IGNORE_REGEX_MARKER + rewritten


def normalize_ignore_list(quant_config: dict[str, object], stage_prefix: str) -> bool:
    """Strip `stage_prefix` from every entry of a compressed-tensors `ignore`.

    `ignore` is what keeps an unquantized submodule (the Qwen3-Omni audio
    tower, for instance) off the quantized scheme. A stage that loads a
    sub-model sees those module names without the checkpoint's stage prefix, so
    an unstripped entry silently stops matching and the submodule is handed a
    scheme its weights cannot satisfy.
    """
    ignored = quant_config.get("ignore")
    if not isinstance(ignored, list) or not ignored:
        return False
    else:
        pass

    escaped_prefix = stage_prefix.replace(".", r"\.")
    normalized = [
        normalize_ignore_entry(entry, stage_prefix, escaped_prefix) for entry in ignored
    ]
    if normalized == ignored:
        return False
    else:
        pass

    quant_config["ignore"] = normalized
    return True


def load_writable_quant_config(
    hf_config: PretrainedConfig,
) -> tuple[object, str, dict[str, object], bool] | None:
    """Return `(owner, metadata_key, quant_config, needs_writeback)` for the
    quant metadata discovered on `hf_config` or a nested stage sub-config,
    or `None` if none is found."""
    visited: set[int] = set()

    def _search(node: object) -> tuple[object, str, dict[str, object], bool] | None:
        if node is None or id(node) in visited:
            return None
        else:
            pass
        visited.add(id(node))

        for metadata_key in _QUANT_METADATA_KEYS:
            quant_config_raw = read_metadata(node, metadata_key)
            if quant_config_raw is None:
                continue
            else:
                pass

            quant_config = to_mutable_dict(quant_config_raw, metadata_key)
            # If we created a new dict from a non-dict object, we must write it
            # back after mutation so downstream consumers see the normalized names.
            needs_writeback = quant_config is not quant_config_raw
            return node, metadata_key, quant_config, needs_writeback

        for attr in _NESTED_QUANT_CONFIG_ATTRS:
            found = _search(read_metadata(node, attr))
            if found is not None:
                return found
            else:
                pass
        return None

    return _search(hf_config)


def resolve_stage_prefix(hf_config: PretrainedConfig) -> str | None:
    """Return the checkpoint prefix for the active stage architecture."""
    architectures = getattr(hf_config, "architectures", None) or []
    if not architectures:
        return None
    else:
        pass
    return _STAGE_PREFIX_BY_ARCH.get(architectures[0])


def normalize_quant_config(model_config: ModelConfig) -> None:
    """Strip the active stage's checkpoint prefix from the quant config"""
    hf_config = getattr(model_config, "hf_config", None)
    if hf_config is None:
        return
    else:
        pass

    loaded = load_writable_quant_config(hf_config)
    if loaded is None:
        return
    else:
        pass
    owner, metadata_key, quant_config, needs_writeback = loaded

    stage_prefix = resolve_stage_prefix(hf_config)
    if not stage_prefix:
        return
    else:
        pass

    blocks_changed = normalize_block_name_to_quantize(quant_config, stage_prefix)
    extra_changed = normalize_extra_config_keys(quant_config, stage_prefix)
    ignore_changed = normalize_ignore_list(quant_config, stage_prefix)
    if not (blocks_changed or extra_changed or ignore_changed):
        return
    else:
        pass

    if needs_writeback:
        setattr(owner, metadata_key, quant_config)
    else:
        pass
