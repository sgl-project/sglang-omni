"""Component-owned MLX-VLM checkpoint loading."""

import json
from typing import Literal

import mlx.core as mx
import mlx.nn as nn
from mlx_vlm.models.qwen3_omni_moe import ModelConfig
from mlx_vlm.models.qwen3_omni_moe.code2wav import Code2WavModel
from mlx_vlm.models.qwen3_omni_moe.language import LanguageModel
from mlx_vlm.models.qwen3_omni_moe.talker import Talker
from mlx_vlm.models.qwen3_omni_moe.thinker import AudioModel, Thinker, VisionModel
from mlx_vlm.utils import load_model_weights

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


def load_component(model_path: str, component: Component) -> nn.Module:
    directory = resolve_model_path(model_path)
    raw = json.loads((directory / "config.json").read_text())
    if raw.get("model_type") != "qwen3_omni_moe":
        raise ValueError("MLX Omni requires a full qwen3_omni_moe checkpoint")
    else:
        pass
    config = ModelConfig.from_dict(raw)
    if component in ("talker", "code2wav") and not config.enable_audio_output:
        raise ValueError("The checkpoint has no audio output components")
    else:
        pass
    namespace = nn.Module()
    if component in ("language", "vision", "audio", "embedding"):
        namespace.thinker = nn.Module()
        if component == "language":
            model = LanguageModel(
                config.thinker_config.text_config, config.thinker_config
            )
            namespace.thinker.language_model = model
        elif component == "vision":
            model = VisionModel(config.thinker_config.vision_config)
            namespace.thinker.vision_tower = model
        elif component == "audio":
            model = AudioModel(config.thinker_config.audio_config)
            namespace.thinker.audio_tower = model
        else:
            text = config.thinker_config.text_config
            model = nn.Embedding(text.vocab_size, text.hidden_size)
            namespace.thinker.language_model = nn.Module()
            namespace.thinker.language_model.model = nn.Module()
            namespace.thinker.language_model.model.embed_tokens = model
    elif component == "talker":
        model = Talker(config.talker_config)
        namespace.talker = model
    else:
        model = Code2WavModel(config.code2wav_config)
        namespace.code2wav = model

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
        weights.update(
            (name, value)
            for name, value in mx.load(str(shard)).items()
            if name.startswith(PREFIXES[component])
        )
    if not weights:
        raise ValueError(f"No {component} weights in {directory}")
    else:
        pass
    # note (Copilot): Sanitizers and quantization need the full checkpoint namespace.
    if component in ("language", "vision", "audio", "embedding"):
        weights = Thinker.sanitize(namespace, weights)
    else:
        pass
    if component != "embedding":
        weights = model.sanitize(weights)
    else:
        pass
    load_model_weights(namespace, raw, weights, directory, strict=True)
    if component == "talker":
        # note (Copilot): This library rotary object is not an nn.Module; materialize
        # its lazy buffer before the model moves from loader to scheduler thread.
        mx.eval(model.code_predictor.model.rotary_emb.inv_freq)
    else:
        pass
    if component == "embedding":
        return namespace.thinker.language_model.model.embed_tokens
    else:
        return model
