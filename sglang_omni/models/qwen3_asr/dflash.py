# SPDX-License-Identifier: Apache-2.0
"""Load and sample the reduced-vocabulary Qwen3-ASR DFlash checkpoint."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
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

DRAFT_ARCHITECTURE = "Qwen3ASRDFlashDraftModel"


def materialize_draft_checkpoint(
    model_path: str, revision: str | None
) -> TemporaryDirectory[str]:
    """Flatten the draft configuration while retaining its original weight files."""
    source = Path(model_path)
    if not source.is_dir():
        source = Path(snapshot_download(model_path, revision=revision))
    else:
        pass
    source_config = json.loads((source / "config.json").read_text())
    if source_config.get("speculators_model_type") != "dflash":
        raise ValueError("Qwen3-ASR requires a Speculators DFlash checkpoint")
    elif source_config["sample_from_anchor"]:
        raise ValueError("Qwen3-ASR DFlash requires sample_from_anchor=False")
    else:
        pass
    draft_config = source_config["transformer_layer_config"].copy()
    draft_config.update(
        model_type="qwen3",
        architectures=[DRAFT_ARCHITECTURE],
        draft_vocab_size=source_config["draft_vocab_size"],
        tie_word_embeddings=False,
        dtype=source_config["dtype"],
        dflash_config={
            # note (xuanyili): Draft checkpoints capture before layer i; capture hooks name the preceding layer.
            "target_layer_ids": [
                layer_id - 1 for layer_id in source_config["aux_hidden_state_layer_ids"]
            ],
            "mask_token_id": source_config["mask_token_id"],
            "block_size": source_config["block_size"],
            "sample_from_anchor": source_config["sample_from_anchor"],
        },
    )
    if source_config["sliding_window_non_causal"]:
        draft_config["is_causal"] = False
    else:
        pass
    checkpoint = TemporaryDirectory(prefix="qwen3-asr-dflash-")
    destination = Path(checkpoint.name)
    (destination / "config.json").write_text(json.dumps(draft_config, indent=2) + "\n")
    for weight_file in source.glob("*.safetensors*"):
        (destination / weight_file.name).symlink_to(weight_file.resolve())
    return checkpoint


class Qwen3ASRDFlashDraftModel(DFlashDraftModel):
    """DFlash backbone with its trained output head and target-token mapping."""

    draft_to_target: torch.Tensor

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
            "draft_to_target",
            torch.empty(config.draft_vocab_size, dtype=torch.long),
            persistent=False,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        checkpoint_weights = dict(weights)
        vocabulary_offsets = checkpoint_weights.pop("d2t")
        super().load_weights(checkpoint_weights.items())
        self.draft_to_target.copy_(
            vocabulary_offsets
            + torch.arange(vocabulary_offsets.numel(), device=vocabulary_offsets.device)
        )

    def sample_tokens(self, hidden_states: torch.Tensor) -> torch.Tensor:
        logits = self.lm_head(hidden_states.to(self.lm_head.weight.dtype))
        return self.draft_to_target[logits.argmax(dim=-1)]


class DraftBlockSampler:
    """Keep mapped draft tokens in fixed storage for CUDA graph replay."""

    def __init__(
        self, model: Qwen3ASRDFlashDraftModel, block_size: int, max_batch_size: int
    ) -> None:
        self.model: Qwen3ASRDFlashDraftModel = model
        self.block_size: int = block_size
        self.out: torch.Tensor = torch.empty(
            max_batch_size * (block_size - 1),
            dtype=torch.long,
            device=model.lm_head.weight.device,
        )

    def __call__(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor | None = None
    ) -> None:
        # note (xuanyili): The capture hook passes input_ids for other samplers; DFlash skips the anchor row.
        proposed_hidden_states = hidden_states.reshape(
            -1, self.block_size, hidden_states.shape[-1]
        )[:, 1:].reshape(-1, hidden_states.shape[-1])
        self.out[: proposed_hidden_states.shape[0]].copy_(
            self.model.sample_tokens(proposed_hidden_states)
        )


class Qwen3ASRDFlashWorker(DFlashWorkerV2):
    """Use the trained draft head in both eager and captured proposal paths."""

    def _maybe_build_draft_sampler(
        self,
    ) -> DraftBlockSampler | None:  # noqa: leading-underscore  # SGLang override
        if envs.SGLANG_DFLASH_EAGER_DRAFT_SAMPLER.get() or self.block_size <= 1:
            return None
        else:
            return DraftBlockSampler(
                self.draft_model,
                self.block_size,
                max(get_exec().graph.cuda_graph_config.decode.bs),
            )

    def _greedy_sample_from_vocab_parallel_head(  # noqa: leading-underscore  # SGLang override
        self,
        *,
        hidden_states: torch.Tensor,
        lm_head: nn.Module,
        chunk_size: int = 256,
    ) -> torch.Tensor:
        # note (xuanyili): The worker supplies the target head; this checkpoint has its own trained head.
        return self.draft_model.sample_tokens(hidden_states)
