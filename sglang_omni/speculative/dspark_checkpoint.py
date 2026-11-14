# SPDX-License-Identifier: Apache-2.0
"""Adapt speculators DSpark checkpoint metadata to the serving runtime."""

from __future__ import annotations

import json
from pathlib import Path
from tempfile import TemporaryDirectory

from huggingface_hub import snapshot_download

DRAFT_ARCHITECTURE = "ReducedVocabularyDSparkDraftModel"


def materialize_draft_checkpoint(
    draft_model_path: str, revision: str | None
) -> TemporaryDirectory[str]:
    """Flatten a speculators DSpark config and retain its checkpoint-local heads."""
    source_directory = Path(draft_model_path)
    if not source_directory.is_dir():
        source_directory = Path(snapshot_download(draft_model_path, revision=revision))
    else:
        pass
    source_config = json.loads((source_directory / "config.json").read_text())
    if source_config.get("speculators_model_type") != "dspark":
        raise ValueError(f"{draft_model_path} is not a speculators DSpark checkpoint")
    elif source_config["markov_head_type"] != "vanilla":
        raise ValueError("Reduced-vocabulary DSpark requires a vanilla Markov head")
    elif source_config["sliding_window_non_causal"]:
        raise ValueError("Reduced-vocabulary DSpark requires causal attention")
    else:
        pass
    draft_config = {
        **source_config["transformer_layer_config"],
        "model_type": "qwen3",
        "architectures": [DRAFT_ARCHITECTURE],
        "tie_word_embeddings": False,
        "dtype": source_config["dtype"],
        "draft_vocab_size": source_config["draft_vocab_size"],
        "sample_from_anchor": source_config["sample_from_anchor"],
        "markov_rank": source_config["markov_rank"],
        "markov_head_type": source_config["markov_head_type"],
        "enable_confidence_head": source_config["enable_confidence_head"],
        "confidence_head_with_markov": source_config["confidence_head_with_markov"],
        "dflash_config": {
            # note (Codex): training captures layer inputs; the target exposes layer outputs.
            "target_layer_ids": [
                layer_id - 1 for layer_id in source_config["aux_hidden_state_layer_ids"]
            ],
            "mask_token_id": source_config["mask_token_id"],
            "block_size": source_config["block_size"],
        },
    }
    draft_checkpoint = TemporaryDirectory(prefix="dspark-draft-")
    draft_directory = Path(draft_checkpoint.name)
    (draft_directory / "config.json").write_text(json.dumps(draft_config, indent=2))
    for source_file in source_directory.iterdir():
        if ".safetensors" in source_file.name:
            (draft_directory / source_file.name).symlink_to(source_file.resolve())
        else:
            pass
    return draft_checkpoint
