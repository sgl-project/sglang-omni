# SPDX-License-Identifier: Apache-2.0
"""Classify consolidated perception and language-model checkpoint tensors."""

from __future__ import annotations

from typing import Literal

CheckpointTensorKind = Literal["skip", "perception", "rename", "split_down", "split_up"]
MTPCheckpointTensorKind = Literal["skip", "rename", "split_down", "split_up"]

FP32_PARAM_HOLDER_RENAMES = {
    ".mixer._fp32_params.A_log": ".mixer.A_log",
    ".mixer._fp32_params.dt_bias": ".mixer.dt_bias",
    ".mixer._fp32_params.D": ".mixer.D",
}
MTP_CHECKPOINT_PREFIXES = ("llm.mtp.", "language_model.mtp.", "mtp.")
SKIPPED_CHECKPOINT_PREFIXES = (
    *MTP_CHECKPOINT_PREFIXES,
    "ctc_timestamp.",
    "perception.encoder.diarization_model.preprocessor.",
)


def classify_checkpoint_name(name: str) -> tuple[CheckpointTensorKind, str]:
    """Return the tensor kind and its target name or fused expert module prefix."""
    if "._extra_state" in name:
        return "skip", ""
    elif name.startswith(SKIPPED_CHECKPOINT_PREFIXES):
        return "skip", ""
    elif name.startswith("perception."):
        return "perception", name[len("perception.") :]
    else:
        hf_name = name.replace("llm.model.", "backbone.", 1)
        hf_name = hf_name.replace("llm.lm_head", "lm_head", 1)
    hf_name = rename_fp32_param_holder(hf_name)
    if hf_name == "backbone.norm.weight":
        hf_name = "backbone.norm_f.weight"
    else:
        pass
    return classify_expert_name(hf_name)


def classify_mtp_checkpoint_name(name: str) -> tuple[MTPCheckpointTensorKind, str]:
    """Map a bundled MTP tensor to the mtp. name SGLang's MTP loader expects, and skip the rest."""
    if "._extra_state" in name:
        return "skip", ""
    else:
        pass
    prefix = next(
        (prefix for prefix in MTP_CHECKPOINT_PREFIXES if name.startswith(prefix)),
        None,
    )
    if prefix is None:
        return "skip", ""
    else:
        return classify_expert_name("mtp." + name[len(prefix) :])


def rename_fp32_param_holder(name: str) -> str:
    for holder, target in FP32_PARAM_HOLDER_RENAMES.items():
        if name.endswith(holder):
            return name[: -len(holder)] + target
        else:
            pass
    return name


def classify_expert_name(
    name: str,
) -> tuple[Literal["rename", "split_down", "split_up"], str]:
    if name.endswith(".experts.down_projs"):
        return "split_down", name[: -len(".experts.down_projs")]
    elif name.endswith(".experts.gate_and_up_projs"):
        return "split_up", name[: -len(".experts.gate_and_up_projs")]
    else:
        return "rename", name
