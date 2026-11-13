# SPDX-License-Identifier: Apache-2.0
"""Nemotron Transcribe adapter for SGLang's bundled NemotronH MTP model."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP
from sglang.srt.models.utils import WeightsMapper

from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_mtp_weights,
)


def expected_mtp_checkpoint_names(
    parameter_names: Iterable[str], *, num_experts: int
) -> set[str]:
    """Return source names that cover SGLang's physical MTP-layer parameters."""
    expected: set[str] = set()
    for parameter_name in parameter_names:
        if not parameter_name.startswith("model.layers."):
            continue
        else:
            pass
        checkpoint_name = parameter_name.replace("model.layers.", "mtp.layers.", 1)
        if ".qkv_proj." in parameter_name:
            for projection_name in ("q_proj", "k_proj", "v_proj"):
                expected.add(
                    checkpoint_name.replace(".qkv_proj.", f".{projection_name}.")
                )
            else:
                pass
        elif ".experts.w13_weight" in parameter_name:
            expert_prefix = checkpoint_name.split(".experts.w13_weight", 1)[0]
            for expert_index in range(num_experts):
                expected.add(f"{expert_prefix}.experts.{expert_index}.up_proj.weight")
            else:
                pass
        elif ".experts.w2_weight" in parameter_name:
            expert_prefix = checkpoint_name.split(".experts.w2_weight", 1)[0]
            for expert_index in range(num_experts):
                expected.add(f"{expert_prefix}.experts.{expert_index}.down_proj.weight")
            else:
                pass
        else:
            expected.add(checkpoint_name)
    return expected


def validate_mtp_checkpoint_inventory(
    expected: set[str],
    seen: set[str],
    unexpected: set[str],
    duplicates: set[str],
) -> None:
    """Fail when seen MTP tensors do not match the model's expected inventory."""
    missing = expected - seen
    if missing or unexpected or duplicates:
        raise ValueError(
            "Nemotron Transcribe MTP checkpoint layout mismatch; "
            "seen tensor names differ: "
            f"missing={sorted(missing)[:8]}, "
            f"unexpected={sorted(unexpected)[:8]}, "
            f"duplicates={sorted(duplicates)[:8]}"
        )
    else:
        pass


class NemotronTranscribeMTP(NemotronHForCausalLMMTP):
    """Reuse SGLang's MTP forward path with strict NeMo checkpoint routing."""

    hf_to_sglang_mapper = NemotronHForCausalLMMTP.hf_to_sglang_mapper | WeightsMapper(
        orig_to_new_prefix={"llm.mtp.": "mtp."}
    )

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        del is_mtp
        expected = expected_mtp_checkpoint_names(
            dict(self.named_parameters()),
            num_experts=self.config.max_n_routed_experts,
        )
        seen: set[str] = set()
        unexpected: set[str] = set()
        duplicates: set[str] = set()

        def checked_weights() -> Iterable[tuple[str, torch.Tensor]]:
            for name, tensor in checkpoint_to_mtp_weights(weights):
                if name not in expected:
                    unexpected.add(name)
                elif name in seen:
                    duplicates.add(name)
                else:
                    seen.add(name)
                    yield name, tensor

        super().load_weights(checked_weights())
        validate_mtp_checkpoint_inventory(expected, seen, unexpected, duplicates)


__all__ = [
    "NemotronTranscribeMTP",
    "expected_mtp_checkpoint_names",
    "validate_mtp_checkpoint_inventory",
]
