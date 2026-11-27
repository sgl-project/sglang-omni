# SPDX-License-Identifier: Apache-2.0
"""The checkpoint's native MTP layer as an SGLang draft for Nemotron Transcribe."""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch
from sglang.srt.models.nemotron_h_mtp import NemotronHForCausalLMMTP

from sglang_omni.models.nemotron_transcribe.decoder_weights import (
    checkpoint_to_mtp_weights,
)

MTP_LAYER_PARAMETER_PREFIX = "model.layers."
MAX_REPORTED_CHECKPOINT_NAMES = 8


def expected_mtp_checkpoint_names(
    parameter_names: Iterable[str], *, num_experts: int
) -> set[str]:
    """Return the converted mtp. names that fill every MTP layer parameter, one per expert."""
    expected_names: set[str] = set()
    for parameter_name in parameter_names:
        if not parameter_name.startswith(MTP_LAYER_PARAMETER_PREFIX):
            # note (scatyf3): the draft shares embed_tokens and lm_head with the target.
            continue
        else:
            pass
        checkpoint_name = "mtp." + parameter_name[len("model.") :]
        if ".qkv_proj." in checkpoint_name:
            expected_names.update(
                checkpoint_name.replace(".qkv_proj.", f".{projection}.")
                for projection in ("q_proj", "k_proj", "v_proj")
            )
        elif checkpoint_name.endswith(".experts.w13_weight"):
            experts_name = checkpoint_name[: -len(".w13_weight")]
            expected_names.update(
                f"{experts_name}.{index}.up_proj.weight" for index in range(num_experts)
            )
        elif checkpoint_name.endswith(".experts.w2_weight"):
            experts_name = checkpoint_name[: -len(".w2_weight")]
            expected_names.update(
                f"{experts_name}.{index}.down_proj.weight"
                for index in range(num_experts)
            )
        else:
            expected_names.add(checkpoint_name)
    return expected_names


def validate_mtp_checkpoint_inventory(
    expected: set[str],
    seen: set[str],
    unexpected: set[str],
    duplicates: set[str],
    *,
    layout: str,
) -> None:
    """Fail the draft load unless the converted MTP tensors match the expected inventory exactly."""
    missing = expected - seen
    if missing or unexpected or duplicates:
        raise ValueError(
            f"Nemotron Transcribe MTP checkpoint layout mismatch for the native "
            f"draft ({layout}): missing {len(missing)} tensors "
            f"{sorted(missing)[:MAX_REPORTED_CHECKPOINT_NAMES]}, unexpected "
            f"{len(unexpected)} tensors "
            f"{sorted(unexpected)[:MAX_REPORTED_CHECKPOINT_NAMES]}, duplicate "
            f"{len(duplicates)} tensors "
            f"{sorted(duplicates)[:MAX_REPORTED_CHECKPOINT_NAMES]}; serve a "
            "checkpoint with its complete llm.mtp. layer or turn off speculative "
            "decoding"
        )
    else:
        pass


class NemotronTranscribeMTPDraft(NemotronHForCausalLMMTP):

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]], is_mtp: bool = False
    ) -> None:
        """Load the bundled MTP layer and fail unless it fills every draft parameter exactly once."""
        del is_mtp
        num_experts = self.config.max_n_routed_experts
        expected = expected_mtp_checkpoint_names(
            (name for name, _ in self.named_parameters()), num_experts=num_experts
        )
        seen: set[str] = set()
        unexpected: set[str] = set()
        duplicates: set[str] = set()

        def checked_weights() -> Iterator[tuple[str, torch.Tensor]]:
            for name, tensor in checkpoint_to_mtp_weights(weights):
                if name not in expected:
                    unexpected.add(name)
                elif name in seen:
                    duplicates.add(name)
                else:
                    seen.add(name)
                    yield name, tensor

        super().load_weights(checked_weights(), is_mtp=True)
        validate_mtp_checkpoint_inventory(
            expected,
            seen,
            unexpected,
            duplicates,
            layout=f"{self.config.mtp_hybrid_override_pattern}, {num_experts} experts",
        )


EntryClass = NemotronTranscribeMTPDraft
