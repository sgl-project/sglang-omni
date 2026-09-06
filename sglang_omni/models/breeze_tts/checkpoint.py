# SPDX-License-Identifier: Apache-2.0
"""Load only the requested components; never instantiate the legacy Mimi codec."""

import json
from pathlib import Path

from safetensors import safe_open


def read_config(checkpoint: str) -> dict:
    config = json.loads((Path(checkpoint) / "config.json").read_text())
    if (
        config.get("model_type") != "breeze"
        or config.get("backbone_model_type") != "qwen3"
        or config.get("text_encoder_proj_type") != "linear"
        or not config.get("tie_codebooks_embeddings")
    ):
        raise ValueError(
            "Only the Breeze-TTS-2 Qwen3/linear-projection checkpoint is supported"
        )
    return config


def backbone_weights(weights, num_layers: int):
    """Map the native Qwen3 weights, rejecting incomplete backbone checkpoints."""
    required = {"backbone_model.norm.weight", "lm_head.weight"}
    for layer in range(num_layers):
        prefix = f"backbone_model.layers.{layer}."
        required.update(
            prefix + name + ".weight"
            for name in (
                "input_layernorm",
                "post_attention_layernorm",
                "self_attn.q_proj",
                "self_attn.k_proj",
                "self_attn.v_proj",
                "self_attn.o_proj",
                "self_attn.q_norm",
                "self_attn.k_norm",
                "mlp.gate_proj",
                "mlp.up_proj",
                "mlp.down_proj",
            )
        )
    seen = set()
    for name, weight in weights:
        if name.startswith(("backbone_model.layers.", "backbone_model.norm.")):
            seen.add(name)
            yield "model." + name[len("backbone_model.") :], weight
        elif name == "lm_head.weight":
            seen.add(name)
            yield name, weight
    missing = required - seen
    if missing:
        raise ValueError(
            f"Breeze checkpoint is missing backbone weights: {sorted(missing)}"
        )


def load_component(module, checkpoint: str, prefix: str) -> None:
    """Strictly load a component from the checkpoint's safetensors shards."""
    root = Path(checkpoint)
    index = root / "model.safetensors.index.json"
    if index.exists():
        weight_map = json.loads(index.read_text())["weight_map"]
        files = sorted(
            {name for key, name in weight_map.items() if key.startswith(prefix)}
        )
    else:
        files = ["model.safetensors"]
    state = {}
    for filename in files:
        with safe_open(root / filename, framework="pt", device="cpu") as shard:
            for name in shard.keys():
                if name.startswith(prefix):
                    state[name[len(prefix) :]] = shard.get_tensor(name)
    # assign=True also permits a meta-device construction without a second
    # random-initialized copy of the text encoder or depth model.
    module.load_state_dict(state, strict=True, assign=True)
