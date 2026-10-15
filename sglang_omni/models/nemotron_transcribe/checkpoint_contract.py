# SPDX-License-Identifier: Apache-2.0
"""Classify consolidated perception and language-model checkpoint tensors."""

from __future__ import annotations

from typing import Literal

CheckpointTensorKind = Literal["skip", "perception", "rename", "split_down", "split_up"]


def perception_checkpoint_keys(
    asr_layers: int,
    diar_layers: int,
    *,
    asr_qk_norm: bool,
    high_resolution: bool,
    learnable_silence: bool,
) -> set[str]:
    """Return every perception tensor the GA checkpoint is expected to contain."""
    names: set[str] = set()

    for frontend in (
        "perception.preprocessor.featurizer",
        "perception.encoder.diarization_model.preprocessor.featurizer",
    ):
        names.add(f"{frontend}.fb")
        names.add(f"{frontend}.window")
    names.add("perception.proj.weight")
    names.add("perception.proj.bias")
    names.add("perception.encoder.asr_norm.weight")
    names.add("perception.encoder.asr_norm.bias")
    names.add("perception.encoder.diar_norm.weight")
    names.add("perception.encoder.diar_norm.bias")
    add_transformer_keys(
        names,
        "perception.encoder.asr_encoder",
        asr_layers,
        qk_norm=asr_qk_norm,
    )
    add_transformer_keys(
        names,
        "perception.encoder.diarization_model.encoder",
        diar_layers,
        qk_norm=False,
    )
    modules = "perception.encoder.diarization_model.sortformer_modules"
    for stem in (
        "encoder_proj",
        "first_hidden_to_hidden",
        "hidden_to_spks",
        "single_hidden_to_spks",
    ):
        names.add(f"{modules}.{stem}.weight")
        names.add(f"{modules}.{stem}.bias")
    if learnable_silence:
        names.add(f"{modules}.learnable_sil_emb")
    else:
        pass
    if high_resolution:
        names.add(f"{modules}.subpixel_upsample.weight")
        names.add(f"{modules}.subpixel_upsample.bias")
    else:
        pass
    return names


def add_transformer_keys(
    names: set[str], prefix: str, layers: int, *, qk_norm: bool
) -> None:
    """Add one speech transformer's checkpoint names."""
    names.add(f"{prefix}.pre_encode.proj.weight")
    names.add(f"{prefix}.embed_norm.weight")
    names.add(f"{prefix}.embed_norm.bias")
    names.add(f"{prefix}.final_norm.weight")
    names.add(f"{prefix}.final_norm.bias")
    for index in range(layers):
        layer = f"{prefix}.layers.{index}"
        names.add(f"{layer}.attn.w_qkv.weight")
        names.add(f"{layer}.attn.out_proj.weight")
        names.add(f"{layer}.attn.out_proj.bias")
        names.add(f"{layer}.norm1.weight")
        names.add(f"{layer}.norm1.bias")
        names.add(f"{layer}.norm2.weight")
        names.add(f"{layer}.norm2.bias")
        names.add(f"{layer}.ffn.net.0.weight")
        names.add(f"{layer}.ffn.net.0.bias")
        names.add(f"{layer}.ffn.net.3.weight")
        names.add(f"{layer}.ffn.net.3.bias")
        if qk_norm:
            names.add(f"{layer}.attn.q_norm.weight")
            names.add(f"{layer}.attn.q_norm.bias")
            names.add(f"{layer}.attn.k_norm.weight")
            names.add(f"{layer}.attn.k_norm.bias")
        else:
            pass


FP32_PARAM_HOLDER_RENAMES = {
    ".mixer._fp32_params.A_log": ".mixer.A_log",
    ".mixer._fp32_params.dt_bias": ".mixer.dt_bias",
    ".mixer._fp32_params.D": ".mixer.D",
}
SKIPPED_CHECKPOINT_PREFIXES = ("llm.mtp.", "ctc_timestamp.")


def classify_checkpoint_name(name: str) -> tuple[CheckpointTensorKind, str]:
    """Classify one consolidated checkpoint tensor.

    Returns kind and payload. kind is skip, perception, rename, split_down,
    or split_up. perception payload is the name with the perception prefix
    removed. rename payload is the NemotronH name. split payloads are the
    module prefix in front of the fused expert tensor.
    """
    if "._extra_state" in name or name.startswith(SKIPPED_CHECKPOINT_PREFIXES):
        return "skip", ""
    else:
        pass
    if name.startswith("perception."):
        return "perception", name[len("perception.") :]
    else:
        pass
    hf_name = name.replace("llm.model.", "backbone.", 1)
    hf_name = hf_name.replace("llm.lm_head", "lm_head", 1)
    for holder, target in FP32_PARAM_HOLDER_RENAMES.items():
        if hf_name.endswith(holder):
            hf_name = hf_name[: -len(holder)] + target
            break
        else:
            pass
    if hf_name == "backbone.norm.weight":
        hf_name = "backbone.norm_f.weight"
    else:
        pass
    if hf_name.endswith(".experts.down_projs"):
        return "split_down", hf_name[: -len(".experts.down_projs")]
    elif hf_name.endswith(".experts.gate_and_up_projs"):
        return "split_up", hf_name[: -len(".experts.gate_and_up_projs")]
    else:
        return "rename", hf_name
