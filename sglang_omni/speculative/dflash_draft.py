# SPDX-License-Identifier: Apache-2.0
"""DFlash draft that proposes from its own reduced-vocabulary head.

SGLang's draft borrows the target head. A speculators checkpoint trains over a subset of
the target vocabulary instead, so proposals go through its head and its token mapping.
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from tempfile import TemporaryDirectory

import torch
from huggingface_hub import snapshot_download
from sglang.srt.environ import envs
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.dflash import DFlashDraftModel
from sglang.srt.runtime_context import get_exec
from sglang.srt.speculative.dflash_worker_v2 import DFlashWorkerV2
from torch import nn
from transformers import Qwen3Config

DRAFT_ARCHITECTURE = "ReducedVocabularyDFlashDraftModel"


def materialize_draft_checkpoint(
    draft_model_path: str, revision: str | None
) -> TemporaryDirectory[str]:
    """Write a loadable copy of a speculators-format DFlash draft.

    The copy has a flat qwen3 config that names the reduced-vocabulary draft class and
    links the original weights. The caller releases it once the engine has loaded.
    """
    source_dir = (
        draft_model_path
        if os.path.isdir(draft_model_path)
        else snapshot_download(draft_model_path, revision=revision)
    )
    with open(os.path.join(source_dir, "config.json")) as config_file:
        source_config = json.load(config_file)
    if source_config.get("speculators_model_type") != "dflash":
        raise ValueError(f"{draft_model_path} is not a speculators DFlash checkpoint")
    elif source_config["sample_from_anchor"]:
        raise ValueError(f"{draft_model_path} samples from the anchor token")
    elif source_config["sliding_window_non_causal"]:
        raise ValueError(f"{draft_model_path} uses non-causal sliding windows")
    else:
        pass
    draft_config = {
        **source_config["transformer_layer_config"],
        "model_type": "qwen3",
        "architectures": [DRAFT_ARCHITECTURE],
        "draft_vocab_size": source_config["draft_vocab_size"],
        "tie_word_embeddings": False,
        "dtype": source_config["dtype"],
        "dflash_config": {
            # note (wilsonzheng0327): speculators aux id i reads the residual stream
            # before decoder layer i, which SGLang captures for target layer id i - 1.
            "target_layer_ids": [
                int(layer_id) - 1
                for layer_id in source_config["aux_hidden_state_layer_ids"]
            ],
            "mask_token_id": source_config["mask_token_id"],
            "block_size": source_config["block_size"],
            "sample_from_anchor": False,
        },
    }
    draft_checkpoint = TemporaryDirectory(prefix="dflash-draft-")
    with open(os.path.join(draft_checkpoint.name, "config.json"), "w") as config_file:
        json.dump(draft_config, config_file, indent=2)
    for file_name in os.listdir(source_dir):
        if ".safetensors" in file_name:
            os.symlink(
                os.path.realpath(os.path.join(source_dir, file_name)),
                os.path.join(draft_checkpoint.name, file_name),
            )
        else:
            pass
    return draft_checkpoint


class ReducedVocabularyDFlashDraftModel(DFlashDraftModel):
    """DFlash backbone with its trained output head and target-token mapping."""

    draft_to_target_token_ids: torch.Tensor

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config, quant_config, prefix)
        self.lm_head: nn.Linear = nn.Linear(
            config.hidden_size, config.draft_vocab_size, bias=False
        )
        self.register_buffer(
            "draft_to_target_token_ids",
            torch.empty(config.draft_vocab_size, dtype=torch.long),
            persistent=False,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        checkpoint_weights = dict(weights)
        token_id_offsets = checkpoint_weights.pop("d2t")
        super().load_weights(checkpoint_weights.items())
        self.draft_to_target_token_ids.copy_(
            token_id_offsets
            + torch.arange(token_id_offsets.numel(), device=token_id_offsets.device)
        )

    def propose_token_ids(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        return self.draft_to_target_token_ids[logits.argmax(dim=-1)]


class DraftBlockSampler:
    """Fold the draft head into the draft CUDA graph, with its proposals in fixed storage."""

    def __init__(
        self,
        draft_model: ReducedVocabularyDFlashDraftModel,
        block_size: int,
        max_batch_size: int,
    ) -> None:
        self.draft_model: ReducedVocabularyDFlashDraftModel = draft_model
        self.block_size: int = block_size
        self.out: torch.Tensor = torch.empty(
            max_batch_size * (block_size - 1),
            dtype=torch.long,
            device=draft_model.lm_head.weight.device,
        )

    def __call__(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> None:
        # The capture hook hands every sampler input_ids; a block's first row is its
        # anchor, already committed, so it has no proposal.
        hidden_size = hidden_states.shape[-1]
        proposal_hidden_states = hidden_states.reshape(
            -1, self.block_size, hidden_size
        )[:, 1:].reshape(-1, hidden_size)
        self.out[: proposal_hidden_states.shape[0]].copy_(
            self.draft_model.propose_token_ids(proposal_hidden_states)
        )


class ReducedVocabularyDFlashWorker(DFlashWorkerV2):
    """Propose from the draft's trained head on both the captured and the eager path."""

    def _maybe_build_draft_sampler(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self,
    ) -> DraftBlockSampler | None:
        if envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER.get() or self.block_size <= 1:
            return None
        else:
            return DraftBlockSampler(
                self.draft_model,
                self.block_size,
                max(get_exec().graph.cuda_graph_config.decode.bs),
            )

    def _greedy_sample_from_vocab_parallel_head(  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        # The worker hands over the target head; this draft trained its own.
        return self.draft_model.propose_token_ids(hidden_states)
