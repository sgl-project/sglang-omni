# SPDX-License-Identifier: Apache-2.0
"""DSpark draft for the Fun-CosyVoice3 LLM, served in the stage's speech vocabulary.

The published draft (speculators format) indexes the unified text+speech vocabulary of
an HF export of the CosyVoice3 LLM, where speech id k is unified id k + 151924. The
stage samples only the 6761 speech and control ids, so the draft's embedding, head
and Markov projections are remapped onto those ids at load time.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Iterable

import torch
from huggingface_hub import snapshot_download
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.models.dspark import Qwen3DSparkModel, VanillaMarkov
from torch import nn
from transformers import Qwen3Config

from sglang_omni.models.fun_cosyvoice3.dspark_sampling import (
    DraftBlockSampler,
    DraftSamplingState,
    sample_draft_block,
)
from sglang_omni.models.fun_cosyvoice3.sglang_model import TOTAL_VOCAB_SIZE

DRAFT_ARCHITECTURE = "FunCosyVoice3DSparkDraftModel"
# Unified id of speech token 0 in the vocabulary the draft was trained on; the draft's
# mask token follows the stage's speech and control ids there.
UNIFIED_SPEECH_OFFSET = 151924
UNIFIED_MASK_TOKEN_ID = UNIFIED_SPEECH_OFFSET + TOTAL_VOCAB_SIZE
# note (wilsonzheng0327): the mask row needs an id inside the target vocabulary; the last
# control id stops a request, so it is never an anchor or a proposal.
MASK_TOKEN_ID = TOTAL_VOCAB_SIZE - 1
DRAFT_TOKENS_PER_STEP = 7
# The anchor and its drafts: one verify window, and the checkpoint's block size.
VERIFY_TOKENS_PER_STEP = DRAFT_TOKENS_PER_STEP + 1

DRAFT_CONFIG_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "head_dim",
    "hidden_act",
    "rms_norm_eps",
    "attention_bias",
    "attention_dropout",
    "rope_theta",
    "rope_parameters",
    "rope_scaling",
    "max_position_embeddings",
    "layer_types",
    "sliding_window",
    "use_sliding_window",
    "max_window_layers",
    "torch_dtype",
    "dtype",
)


def materialize_draft_checkpoint(draft_model_path: str, revision: str | None) -> str:
    """Write a loadable copy of a speculators-format CosyVoice3 DSpark draft.

    The copy has a flat qwen3 config that names the stage-vocabulary draft class
    and links the original weights, which the draft class remaps when it loads.
    """
    source_dir = (
        draft_model_path
        if os.path.isdir(draft_model_path)
        else snapshot_download(draft_model_path, revision=revision)
    )
    with open(os.path.join(source_dir, "config.json")) as config_file:
        source_config = json.load(config_file)
    if source_config.get("speculators_model_type") != "dspark":
        raise ValueError(f"{draft_model_path} is not a speculators DSpark checkpoint")
    elif source_config.get("markov_head_type") != "vanilla":
        raise ValueError("Fun-CosyVoice3 DSpark supports the vanilla Markov head")
    elif (
        source_config.get("mask_token_id") != UNIFIED_MASK_TOKEN_ID
        or source_config.get("block_size") != VERIFY_TOKENS_PER_STEP
    ):
        raise ValueError(
            f"{draft_model_path} does not use the CosyVoice3 DSpark vocabulary "
            f"and {VERIFY_TOKENS_PER_STEP}-token block"
        )
    else:
        pass
    layer_config = source_config["transformer_layer_config"]
    draft_config = {
        field_name: layer_config[field_name]
        for field_name in DRAFT_CONFIG_FIELDS
        if field_name in layer_config
    }
    # note (wilsonzheng0327): speculators aux id i reads the residual stream before
    # decoder layer i, which SGLang captures for target layer id i - 1.
    target_layer_ids = [
        int(layer_id) - 1 for layer_id in source_config["aux_hidden_state_layer_ids"]
    ]
    if "sliding_attention" in draft_config.get("layer_types", []):
        # A config without the flag would drop its window when loaded.
        draft_config["use_sliding_window"] = True
    else:
        pass
    draft_config.update(
        model_type="qwen3",
        architectures=[DRAFT_ARCHITECTURE],
        vocab_size=TOTAL_VOCAB_SIZE,
        tie_word_embeddings=False,
        block_size=DRAFT_TOKENS_PER_STEP,
        sample_from_anchor=False,
        mask_token_id=MASK_TOKEN_ID,
        markov_rank=int(source_config["markov_rank"]),
        markov_head_type="vanilla",
        enable_confidence_head=False,
        dflash_config={
            "target_layer_ids": target_layer_ids,
            "mask_token_id": MASK_TOKEN_ID,
            "block_size": DRAFT_TOKENS_PER_STEP,
        },
    )
    draft_dir = tempfile.mkdtemp(prefix="fun-cosyvoice3-dspark-")
    with open(os.path.join(draft_dir, "config.json"), "w") as config_file:
        json.dump(draft_config, config_file, indent=2)
    for file_name in os.listdir(source_dir):
        if file_name.startswith("model.safetensors"):
            os.symlink(
                os.path.realpath(os.path.join(source_dir, file_name)),
                os.path.join(draft_dir, file_name),
            )
        else:
            pass
    return draft_dir


class MirroredMarkovHead(VanillaMarkov):
    """Markov head that draws each draft step from the target's shaping of it.

    Rejection sampling divides by the proposal distribution, so the returned corrected
    logits are exactly the shaped logits the draft drew from.
    """

    def __init__(self, *, vocab_size: int, markov_rank: int) -> None:
        super().__init__(vocab_size=vocab_size, markov_rank=markov_rank)
        self.sampling_state: DraftSamplingState | None = None
        self.block_sampler: DraftBlockSampler = sample_draft_block

    def draw_block(
        self, base_logits: torch.Tensor, first_prev_tokens: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw the block from the stage-shaped logits and return those logits."""
        state = self.sampling_state
        assert state is not None, "a draft step needs its sampling state"
        return self.block_sampler(
            base_logits,
            first_prev_tokens.long(),
            torch.empty_like(base_logits).exponential_(),
            self.markov_w1.weight,
            self.markov_w2.weight,
            state.presence_mask,
            state.repetition_penalties,
            state.output_lengths,
            state.min_new_tokens,
            state.temperatures,
            state.greedy,
        )


class FunCosyVoice3DSparkDraftModel(Qwen3DSparkModel):
    """DSpark draft whose embedding, head and Markov head use the stage vocabulary."""

    uses_own_vocab_modules: bool = True
    unreachable_logits: torch.Tensor

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__(config=config, quant_config=quant_config, prefix=prefix)
        hidden_size = int(config.hidden_size)
        markov_rank = int(config.markov_rank)
        self.embed_tokens = nn.Embedding(TOTAL_VOCAB_SIZE, hidden_size)
        self.lm_head = nn.Linear(hidden_size, TOTAL_VOCAB_SIZE, bias=False)
        self.markov_head = MirroredMarkovHead(
            vocab_size=TOTAL_VOCAB_SIZE, markov_rank=markov_rank
        )
        # Stage ids outside the draft vocabulary are never proposed.
        self.register_buffer(
            "unreachable_logits",
            torch.zeros(TOTAL_VOCAB_SIZE, dtype=torch.float32),
            persistent=False,
        )

    def compute_base_logits(
        self, hidden: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = torch.mm(hidden, self.lm_head.weight.T, out_dtype=torch.float32)
        return logits + self.unreachable_logits, None

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> None:
        vocabulary_weights: dict[str, torch.Tensor] = {}
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        for name, loaded_weight in weights:
            normalized_name = name.removeprefix("model.")
            if normalized_name in (
                "embed_tokens.weight",
                "lm_head.weight",
                "d2t",
                "markov_head.markov_w1.weight",
                "markov_head.markov_w2.weight",
            ):
                vocabulary_weights[normalized_name] = loaded_weight
            elif normalized_name == "t2d" or normalized_name.startswith(
                "confidence_head."
            ):
                pass
            else:
                backbone_weights.append((name, loaded_weight))
        super().load_weights(backbone_weights)

        draft_to_unified = vocabulary_weights["d2t"].long() + torch.arange(
            vocabulary_weights["d2t"].numel()
        )
        draft_to_stage = draft_to_unified - UNIFIED_SPEECH_OFFSET
        # note (wilsonzheng0327): the mask slot is a control id, so a draft entry that
        # maps there must not survive the remap either.
        reachable = (draft_to_stage >= 0) & (draft_to_stage < MASK_TOKEN_ID)
        device = self.lm_head.weight.device
        dtype = self.lm_head.weight.dtype
        stage_ids = draft_to_stage[reachable].to(device)
        reachable = reachable.to(device)
        stage_rows = slice(
            UNIFIED_SPEECH_OFFSET, UNIFIED_SPEECH_OFFSET + TOTAL_VOCAB_SIZE
        )
        with torch.no_grad():
            embedding = vocabulary_weights["embed_tokens.weight"].to(device, dtype)
            self.embed_tokens.weight.copy_(embedding[stage_rows])
            self.embed_tokens.weight[MASK_TOKEN_ID].copy_(
                embedding[UNIFIED_MASK_TOKEN_ID]
            )
            self.lm_head.weight.zero_()
            self.lm_head.weight[stage_ids] = vocabulary_weights["lm_head.weight"].to(
                device, dtype
            )[reachable]
            markov_head = self.markov_head
            markov_head.markov_w1.weight.copy_(
                vocabulary_weights["markov_head.markov_w1.weight"][stage_rows]
            )
            markov_head.markov_w2.weight.zero_()
            markov_head.markov_w2.weight[stage_ids] = vocabulary_weights[
                "markov_head.markov_w2.weight"
            ].to(device, dtype)[reachable]
            unreachable = torch.ones(TOTAL_VOCAB_SIZE, dtype=torch.bool, device=device)
            unreachable[stage_ids] = False
            self.unreachable_logits.copy_(torch.where(unreachable, float("-inf"), 0.0))
