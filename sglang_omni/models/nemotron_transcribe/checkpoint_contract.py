# SPDX-License-Identifier: Apache-2.0
"""Classify consolidated perception and language-model checkpoint tensors."""

from __future__ import annotations

from typing import Literal

CheckpointTensorKind = Literal["skip", "perception", "rename", "split_down", "split_up"]

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
