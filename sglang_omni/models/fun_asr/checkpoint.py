# SPDX-License-Identifier: Apache-2.0
"""Translate split-layout checkpoints into the canonical HF flat layout."""

from __future__ import annotations

import re


def checkpoint_layout(audio_config: dict) -> str:
    if "num_timestamp_prediction_layers" in audio_config:
        if "num_timestamp_prediction_blocks" in audio_config:
            raise ValueError("Fun-ASR config mixes split and flat layer counts")
        return "flat"
    return "split"


def canonical_weight_name(
    name: str, *, layout: str, num_blocks: int, tp_blocks: int
) -> str:
    """Map a single key without buffering checkpoint tensors or guessing from order."""
    if layout not in {"split", "flat"}:
        raise ValueError(f"Unknown Fun-ASR checkpoint layout: {layout}")
    if not name.startswith(
        ("model.audio_tower.", "model.audio_adaptor.", "model.multi_modal_projector.")
    ):
        return name
    old = any(
        part in name
        for part in (
            ".stem.",
            ".timestamp_prediction_layers.",
            ".timestamp_prediction_layer_norm.",
            ".self_attn_layer_norm.",
            ".final_layer_norm.",
            ".blocks.",
            ".audio_adaptor.",
            ".audio_tower.layer_norm.",
        )
    )
    new = any(
        part in name
        for part in (
            ".input_layernorm.",
            ".post_attention_layernorm.",
            ".final_layernorm.",
            ".mlp.",
        )
    )
    if (layout == "flat" and old) or (layout == "split" and new):
        raise ValueError(
            f"Fun-ASR {layout} config disagrees with checkpoint weight {name}"
        )
    if layout == "split":
        name = name.replace("model.audio_adaptor.", "model.multi_modal_projector.")
    prefix = "model.audio_tower."
    if layout == "split" and name.startswith(prefix):
        suffix = name[len(prefix) :]
        if suffix.startswith("stem."):
            suffix = "layers.0." + suffix[len("stem.") :]
        elif suffix.startswith("timestamp_prediction_layer_norm."):
            suffix = (
                f"layers.{num_blocks + tp_blocks - 1}.final_layernorm."
                + suffix.split(".", 1)[1]
            )
        elif suffix.startswith("layer_norm."):
            suffix = (
                f"layers.{num_blocks - 1}.final_layernorm." + suffix.split(".", 1)[1]
            )
        else:
            match = re.match(
                r"(layers|timestamp_prediction_layers)\.(\d+)\.(.+)", suffix
            )
            if match:
                group, index, tail = match.groups()
                index = int(index)
                limit = num_blocks - 1 if group == "layers" else tp_blocks
                if not 0 <= index < limit:
                    raise ValueError(f"Fun-ASR checkpoint layer out of range: {name}")
                index += 1 if group == "layers" else num_blocks
                suffix = f"layers.{index}.{tail}"
        name = prefix + suffix
    elif layout == "split":
        name = name.replace(".blocks.", ".layers.")
    # Keep the local projection name stable; current HF calls it `o_proj`.
    name = name.replace(".self_attn.o_proj.", ".self_attn.out_proj.")
    if layout == "split":
        for old_part, new_part in (
            (".self_attn_layer_norm.", ".input_layernorm."),
            (".final_layer_norm.", ".post_attention_layernorm."),
            (".feedforward_sequential_memory.", ".self_attn.fsmn."),
            (".fsmn.", ".self_attn.fsmn."),
            (".fc1.", ".mlp.fc1."),
            (".fc2.", ".mlp.fc2."),
        ):
            # FSMN was a sibling of attention in both split export variants.
            if old_part == ".fsmn." and ".self_attn.fsmn." in name:
                continue
            name = name.replace(old_part, new_part)
    return name
