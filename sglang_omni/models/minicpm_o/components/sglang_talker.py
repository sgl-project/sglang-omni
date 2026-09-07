# SPDX-License-Identifier: Apache-2.0
"""Native sglang talker (TTS backbone) for MiniCPM-o.

Replaces the remote-code eager generate loop with a sglang Llama backbone so
the talker runs as a regular AR stage (continuous batching, CUDA graphs,
overlap schedule). The condition math mirrors the checkpoint's non-streaming
path: per thinker token ``emb_text(token) + l2_normalize(projector_semantic(
hidden))``, followed by the ``text_eos`` and ``audio_bos`` embeddings; decode
steps embed the previous codec token via ``emb_code``.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import torch
import torch.nn.functional as F
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.model_loader.weight_utils import default_weight_loader
from sglang.srt.models.llama import LlamaForCausalLM
from torch import nn
from transformers import LlamaConfig


def _restore_weight_norm_weight(
    weight_g: torch.Tensor, weight_v: torch.Tensor
) -> torch.Tensor:
    """Materialize ``weight_norm(..., dim=0)`` checkpoint parameters."""
    return torch._weight_norm(weight_v, weight_g, dim=0)


class _MiniCPMTTSProjector(nn.Module):
    """Checkpoint-compatible thinker-hidden → talker-hidden projector."""

    def __init__(self, input_size: int, hidden_size: int) -> None:
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size, bias=True)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, hidden_size, bias=True)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.relu(self.linear1(hidden_states)))


class MiniCPMOTalkerForCausalLM(nn.Module):
    """MiniCPM-o TTS llama backbone emitting s3tokenizer codec tokens."""

    def __init__(self, config: Any, quant_config: Any = None, prefix: str = "") -> None:
        super().__init__()
        tts_config = getattr(config, "tts_config", None)
        if tts_config is None:
            raise ValueError("MiniCPM-o talker requires config.tts_config")
        if not isinstance(tts_config, dict):
            tts_config = tts_config.to_dict()
        cfg = tts_config
        if int(cfg.get("num_vq", 1)) != 1:
            raise ValueError(
                f"MiniCPM-o talker requires num_vq=1, checkpoint reports "
                f"{cfg.get('num_vq')}"
            )
        self.config = config
        self.num_audio_tokens = int(cfg["num_audio_tokens"])
        self.codec_eos_id = self.num_audio_tokens - 1
        self.text_eos_token_id = int(cfg["text_eos_token_id"])
        self.audio_bos_token_id = int(cfg["audio_bos_token_id"])
        self.normalize_projected_hidden = bool(
            cfg.get("normalize_projected_hidden", True)
        )
        hidden_size = int(cfg["hidden_size"])

        # The checkpoint's remote code builds its backbone as
        # LlamaConfig(hidden_size, intermediate_size, heads, layers, kv_heads,
        # max_position_embeddings) with everything else at HF defaults
        # (rope_theta=10000, rms_norm_eps=1e-6, vocab_size=32000); replicate
        # that construction. The backbone vocab only sizes the (unused)
        # embed_tokens/lm_head — every forward feeds embeds (condition on
        # prefill, emb_code on decode) and logits come from head_code.
        llama_config = LlamaConfig(
            hidden_size=hidden_size,
            intermediate_size=int(cfg["intermediate_size"]),
            num_hidden_layers=int(cfg["num_hidden_layers"]),
            num_attention_heads=int(cfg["num_attention_heads"]),
            num_key_value_heads=int(cfg["num_key_value_heads"]),
            hidden_act=cfg.get("hidden_act", "silu"),
            max_position_embeddings=int(cfg["max_position_embeddings"]),
            tie_word_embeddings=False,
        )
        self.llama = LlamaForCausalLM(
            llama_config,
            quant_config=quant_config,
            prefix=f"{prefix}.llama" if prefix else "llama",
        )
        self.emb_text = nn.Embedding(int(cfg["num_text_tokens"]), hidden_size)
        self.projector_semantic = _MiniCPMTTSProjector(int(cfg["llm_dim"]), hidden_size)
        self.emb_code = nn.Embedding(self.num_audio_tokens, hidden_size)
        self.head_code = nn.Linear(hidden_size, self.num_audio_tokens, bias=False)

    def get_input_embeddings(self):
        return self.emb_code

    def build_condition_embeddings(
        self, tts_token_ids: torch.Tensor, tts_hidden: torch.Tensor
    ) -> torch.Tensor:
        """Per-token condition + text_eos + audio_bos, shape (T+2, hidden).

        An empty span (the thinker legally emits ``<|tts_bos|>`` immediately
        followed by the boundary) conditions on the boundary tokens alone.
        """
        device = self.emb_text.weight.device
        dtype = self.emb_text.weight.dtype
        boundary = self.emb_text(
            torch.tensor(
                [self.text_eos_token_id, self.audio_bos_token_id],
                device=device,
                dtype=torch.long,
            )
        )
        if tts_token_ids.numel() == 0:
            return boundary
        tokens = tts_token_ids.to(device=device, dtype=torch.long).reshape(-1)
        hidden = tts_hidden.to(device=device, dtype=dtype)
        if hidden.shape[0] != tokens.shape[0]:
            raise ValueError(
                f"talker condition length mismatch: token_ids={tokens.shape[0]} "
                f"hidden_states={hidden.shape[0]}"
            )
        hidden_embeds = self.projector_semantic(hidden)
        if self.normalize_projected_hidden:
            hidden_embeds = F.normalize(hidden_embeds, p=2, dim=-1)
        condition = self.emb_text(tokens) + hidden_embeds
        return torch.cat([condition, boundary], dim=0)

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        input_embeds_are_projected: bool = False,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected, kwargs
        if forward_batch.forward_mode.is_decode():
            # Decode embeds the previous codec token natively.
            input_embeds = self.emb_code(input_ids)
        elif input_embeds is None:
            input_embeds = forward_batch.input_embeds

        hidden_states = self.llama.model(
            input_ids=input_ids,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        if forward_batch.forward_mode.is_extend():
            extend_seq_lens = forward_batch.extend_seq_lens
            if extend_seq_lens is not None:
                last_indices = (
                    torch.cumsum(extend_seq_lens.to(device=hidden_states.device), dim=0)
                    - 1
                )
            else:
                last_indices = torch.tensor(
                    [hidden_states.shape[0] - 1], device=hidden_states.device
                )
            hidden_states = hidden_states[last_indices]

        logits = self.head_code(hidden_states)
        return LogitsProcessorOutput(
            next_token_logits=logits, hidden_states=hidden_states
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loaded: set[str] = set()
        backbone_weights: list[tuple[str, torch.Tensor]] = []
        direct_params = dict(self.named_parameters())
        head_g: torch.Tensor | None = None
        head_v: torch.Tensor | None = None

        for name, tensor in weights:
            if not name.startswith("tts."):
                continue
            stripped = name.removeprefix("tts.")
            if stripped.startswith("model."):
                backbone_weights.append((stripped, tensor))
                continue
            # The speaker projector only feeds the streaming condition path,
            # which this stage does not implement.
            if stripped.startswith("projector_spk."):
                continue
            if stripped == "emb_code.0.weight":
                stripped = "emb_code.weight"
            elif stripped == "head_code.0.parametrizations.weight.original0":
                head_g = tensor
                continue
            elif stripped == "head_code.0.parametrizations.weight.original1":
                head_v = tensor
                continue
            parameter = direct_params.get(stripped)
            assert (
                parameter is not None
            ), f"Unexpected MiniCPM-o tts checkpoint weight {name!r}"
            loader = getattr(parameter, "weight_loader", default_weight_loader)
            loader(parameter, tensor)
            loaded.add(stripped)

        self.llama.load_weights(backbone_weights)
        loaded.update(f"llama.{name}" for name, _ in backbone_weights)

        if head_g is None or head_v is None:
            raise ValueError(
                "MiniCPM-o checkpoint is missing weight-norm talker head "
                "parameters (tts.head_code.0.parametrizations.weight.*)"
            )
        restored = _restore_weight_norm_weight(head_g, head_v)
        self.head_code.weight.data.copy_(
            restored.to(
                device=self.head_code.weight.device,
                dtype=self.head_code.weight.dtype,
            )
        )
        loaded.add("head_code.weight")
        return loaded


EntryClass = MiniCPMOTalkerForCausalLM

__all__ = ["MiniCPMOTalkerForCausalLM", "EntryClass"]
