"""Strict, component-owned native loading with checkpoint quantization intact."""

import json
import re
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeConfig,
)

from sglang_omni.models.qwen3_omni.mlx.audio import (
    Qwen3OmniMlxAudioEncoder,
    sanitize_audio_weights,
)
from sglang_omni.models.qwen3_omni.mlx.code2wav import (
    Qwen3OmniMlxCode2Wav,
    sanitize_code2wav_weights,
)
from sglang_omni.models.qwen3_omni.mlx.language import LanguageModel
from sglang_omni.models.qwen3_omni.mlx.talker import Talker
from sglang_omni.models.qwen3_omni.mlx.vision import (
    Qwen3OmniMlxVisionEncoder,
    sanitize_vision_weights,
)
from sglang_omni.models.weight_loader import resolve_model_path

Component = Literal["language", "vision", "audio", "talker", "code2wav", "embedding"]
PREFIXES = {
    "language": ("thinker.language_model.", "thinker.model.", "thinker.lm_head."),
    "vision": ("thinker.vision_tower.", "thinker.visual."),
    "audio": ("thinker.audio_tower.",),
    "talker": ("talker.",),
    "code2wav": ("code2wav.",),
    "embedding": (
        "thinker.language_model.model.embed_tokens.",
        "thinker.model.embed_tokens.",
    ),
}


def component_name(name: str, component: Component) -> str:
    if component == "language":
        return name.removeprefix("thinker.language_model.").removeprefix("thinker.")
    elif component == "embedding":
        for prefix in PREFIXES[component]:
            if name.startswith(prefix):
                return name[len(prefix) :]
            else:
                pass
    else:
        for prefix in PREFIXES[component]:
            if name.startswith(prefix):
                name = name[len(prefix) :]
                break
            else:
                pass
    if component == "vision":
        name = (
            name.replace("merger_list.", "deepstack_merger_list.", 1)
            if name.startswith("merger_list.")
            else name
        )
        name = name.replace(".ln_q.", ".norm.")
        name = name.replace(".mlp.0.", ".linear_fc1.").replace(
            ".mlp.2.", ".linear_fc2."
        )
    else:
        pass
    return name


def normalize_individual_experts(
    weights: dict[str, mx.array],
    overrides: dict,
    shapes: dict[str, tuple[int, ...]],
    quantization: dict | None,
) -> tuple[dict[str, mx.array], dict]:
    """Stack complete, compatible expert groups without discarding source tensors."""
    groups = {}
    for name, value in weights.items():
        match = re.fullmatch(
            r"(.+)\.experts\.(\d+)\.(gate_proj|up_proj|down_proj)\.(weight|scales|biases)",
            name,
        )
        if match is not None:
            prefix, index, projection, field = match.groups()
            target = f"{prefix}.switch_mlp.{projection}"
            groups.setdefault(target, {}).setdefault(int(index), {})[field] = (
                name,
                value,
            )
        else:
            pass
    updates = {}
    metadata = {}
    removed_weights = set()
    removed_overrides = set()
    for target, experts in groups.items():
        if target in overrides or any(
            f"{target}.{field}" in weights for field in ("weight", "scales", "biases")
        ):
            raise ValueError(f"Duplicate expert target: {target}")
        else:
            pass
        shape = shapes.get(f"{target}.weight")
        if shape is None or set(experts) != set(range(shape[0])):
            raise ValueError(
                f"Incomplete or unsupported individual expert group: {target}"
            )
        else:
            pass
        fields = set(experts[0])
        if fields not in ({"weight"}, {"weight", "scales", "biases"}) or any(
            set(expert) != fields for expert in experts.values()
        ):
            raise ValueError(f"Inconsistent individual expert tensors: {target}")
        else:
            pass
        packed = "scales" in fields
        if packed and quantization is None:
            raise ValueError("Packed weights require affine quantization metadata")
        else:
            pass
        settings = []
        for index in range(shape[0]):
            source = experts[index]["weight"][0].removesuffix(".weight")
            default = (
                {
                    "bits": quantization["bits"],
                    "group_size": quantization["group_size"],
                    "mode": quantization.get("mode", "affine"),
                }
                if packed
                else False
            )
            setting = overrides.get(source, default)
            if isinstance(setting, dict):
                setting = {**(default or {}), **setting}
            elif setting is True:
                setting = default
            else:
                pass
            if packed != isinstance(setting, dict):
                raise ValueError(
                    f"Individual expert quantization disagrees with tensors: {source}"
                )
            elif packed and setting.get("mode", "affine") != "affine":
                raise ValueError("Individual experts require affine quantization")
            else:
                pass
            settings.append(setting)
            if source in overrides:
                removed_overrides.add(source)
            else:
                pass
        if any(setting != settings[0] for setting in settings[1:]):
            raise ValueError(f"Incompatible individual expert quantization: {target}")
        else:
            pass
        metadata[target] = settings[0]
        for field in fields:
            tensors = [experts[index][field][1] for index in range(shape[0])]
            if any(
                tensor.shape != tensors[0].shape or tensor.dtype != tensors[0].dtype
                for tensor in tensors[1:]
            ):
                raise ValueError(
                    f"Incompatible individual expert shapes or dtypes: {target}.{field}"
                )
            else:
                pass
            updates[f"{target}.{field}"] = mx.stack(tensors)
            removed_weights.update(expert[field][0] for expert in experts.values())
    result = {
        name: value for name, value in weights.items() if name not in removed_weights
    }
    result.update(updates)
    normalized = {
        name: value
        for name, value in overrides.items()
        if name not in removed_overrides
    }
    normalized.update(metadata)
    return result, normalized


def load_component(model_path: str, component: Component) -> nn.Module:
    directory = resolve_model_path(model_path)
    raw = json.loads((directory / "config.json").read_text())
    if raw.get("model_type") != "qwen3_omni_moe" or raw.get("model_file"):
        raise ValueError(
            "Native Omni requires a full, built-in qwen3_omni_moe checkpoint"
        )
    else:
        pass
    config = Qwen3OmniMoeConfig.from_dict(raw)
    with mx.stream(mx.new_thread_local_stream(mx.default_device())):
        if component == "language":
            model = LanguageModel(
                config.thinker_config.text_config, config.thinker_config
            )
        elif component == "embedding":
            text = config.thinker_config.text_config
            model = nn.Embedding(text.vocab_size, text.hidden_size)
        elif component == "vision":
            model = Qwen3OmniMlxVisionEncoder(config.thinker_config.vision_config)
        elif component == "audio":
            model = Qwen3OmniMlxAudioEncoder(config.thinker_config.audio_config)
        elif component in ("talker", "code2wav"):
            if not config.enable_audio_output:
                raise ValueError("The checkpoint has no audio output components")
            elif component == "talker":
                model = Talker(config.talker_config)
            else:
                model = Qwen3OmniMlxCode2Wav(config.code2wav_config)
        else:
            raise ValueError(f"Unknown Omni component: {component}")

        index_path = directory / "model.safetensors.index.json"
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
            shards = {
                directory / shard
                for name, shard in weight_map.items()
                if name.startswith(PREFIXES[component])
            }
        else:
            shards = set(directory.glob("*.safetensors"))
        weights: dict[str, mx.array] = {}
        for shard in sorted(shards):
            for name, value in mx.load(str(shard)).items():
                if not name.startswith(PREFIXES[component]):
                    continue
                else:
                    pass
                name = component_name(name, component)
                if name in weights:
                    raise ValueError(f"Duplicate {component} weight: {name}")
                else:
                    weights[name] = value
        if not weights:
            raise ValueError(f"No {component} weights in {directory}")
        else:
            pass
        if (
            component == "language"
            and model.args.tie_word_embeddings
            and "lm_head.weight" in weights
        ):
            head = weights["lm_head.weight"]
            embedding = weights.get("model.embed_tokens.weight")
            if (
                embedding is None
                or head.shape != embedding.shape
                or head.dtype != embedding.dtype
                or not mx.array_equal(head, embedding).item()
            ):
                raise ValueError("Conflicting tied embedding alias: lm_head.weight")
            else:
                del weights["lm_head.weight"]
        else:
            pass
        shapes = {
            name: tuple(value.shape) for name, value in tree_flatten(model.parameters())
        }
        if component == "vision":
            weights = sanitize_vision_weights(weights, expected_shapes=shapes)
        elif component == "audio":
            weights = sanitize_audio_weights(weights, expected_shapes=shapes)
        elif component == "code2wav":
            weights = sanitize_code2wav_weights(weights, expected_shapes=shapes)
        elif component in ("language", "talker"):
            for name in list(weights):
                if ".experts.gate_up_proj" in name:
                    value = weights.pop(name)
                    suffix = name.split(".experts.gate_up_proj", 1)[1] or ".weight"
                    prefix = name.split(".experts.gate_up_proj", 1)[0]
                    for projection, tensor in zip(
                        ("gate_proj", "up_proj"),
                        mx.split(value, 2, axis=-2),
                        strict=True,
                    ):
                        target = f"{prefix}.switch_mlp.{projection}{suffix}"
                        if target in weights:
                            raise ValueError(f"Duplicate expert weight: {target}")
                        else:
                            weights[target] = tensor
                elif ".experts.down_proj" in name:
                    target = name.replace(".experts.down_proj", ".switch_mlp.down_proj")
                    target = (
                        f"{target}.weight" if target.endswith(".down_proj") else target
                    )
                    if target in weights:
                        raise ValueError(f"Duplicate expert weight: {target}")
                    else:
                        weights[target] = weights.pop(name)
                else:
                    pass
        else:
            pass

        quantization = raw.get("quantization", raw.get("quantization_config"))
        if quantization is not None:
            if quantization.get("mode", "affine") != "affine":
                raise ValueError(
                    "Native Omni supports affine checkpoint quantization only"
                )
            else:
                pass
            overrides = {
                component_name(f"{name}.weight", component)
                .removesuffix("weight")
                .rstrip("."): value
                for name, value in quantization.items()
                if isinstance(value, (dict, bool))
                and f"{name}.".startswith(PREFIXES[component])
            }
            for name in list(overrides):
                if ".experts.gate_up_proj" in name:
                    value = overrides.pop(name)
                    for projection in ("gate_proj", "up_proj"):
                        overrides[
                            name.replace(
                                ".experts.gate_up_proj", f".switch_mlp.{projection}"
                            )
                        ] = value
                elif ".experts.down_proj" in name:
                    overrides[
                        name.replace(".experts.down_proj", ".switch_mlp.down_proj")
                    ] = overrides.pop(name)
                else:
                    pass
        else:
            overrides = {}
        if component in ("language", "talker"):
            weights, overrides = normalize_individual_experts(
                weights, overrides, shapes, quantization
            )
        else:
            pass
        namespace = nn.Module()
        namespace.component = model

        def quantized_module(
            path: str, module: nn.Module
        ) -> bool | dict[str, int | str]:
            name = path.removeprefix("component").removeprefix(".")
            if name in overrides:
                return overrides[name]
            else:
                return (
                    hasattr(module, "to_quantized")
                    and (f"{name}.scales" if name else "scales") in weights
                )

        if quantization is not None:
            nn.quantize(
                namespace,
                bits=quantization["bits"],
                group_size=quantization["group_size"],
                mode="affine",
                class_predicate=quantized_module,
            )
        elif any(name.endswith("scales") for name in weights):
            raise ValueError("Packed weights require affine quantization metadata")
        else:
            pass
        namespace.load_weights(
            [(f"component.{name}", value) for name, value in weights.items()],
            strict=True,
        )
        namespace.eval()
        mx.eval(namespace.parameters())
        return namespace.component
