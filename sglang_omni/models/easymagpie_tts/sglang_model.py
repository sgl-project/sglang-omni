# SPDX-License-Identifier: Apache-2.0
"""EasyMagpie talker: SGLang's Nemotron-H backbone plus the TTS heads.

The model runner composes each step's input embeddings from request state;
this module turns backbone hidden states into acoustic codes, phoneme
feedback, and continue/stop logits over the dummy two-token vocabulary.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from sglang.srt.configs.nemotron_h import NemotronHConfig
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors
from sglang.srt.models.nemotron_h import NemotronHForCausalLM
from torch import nn

from sglang_omni.models.easymagpie_tts.hf_config import (
    EasyMagpieTTSConfig,
    adapt_backbone_config,
    partition_weights,
)
from sglang_omni.models.easymagpie_tts.local_transformer import EasyMagpieTTSHeads

STOP_LOGIT = 30.0


class SiluActivation(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.silu(values)


def patch_silu_shared_experts(backbone: nn.Module) -> int:
    """Give shared MoE experts the SiLU activation the checkpoint was trained with.

    SGLang's NemotronHMLP hardcodes ReLU2 for shared experts; the routed
    experts already follow config.mlp_hidden_act.
    """
    patched = 0
    for layer in backbone.model.layers:
        shared = getattr(layer.mixer, "shared_experts", None)
        if shared is not None:
            shared.act_fn = SiluActivation()
            patched += 1
        else:
            pass
    return patched


@dataclass
class EasyMagpieDecodeStep:
    """Per-row controls the runner installs before each decode forward."""

    audio_valid: torch.Tensor
    temperatures: torch.Tensor
    top_ks: torch.Tensor
    seeds: torch.Tensor
    positions: torch.Tensor
    max_top_k: int


class EasyMagpieTTSForConditionalGeneration(nn.Module):
    def __init__(
        self,
        config: NemotronHConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        adapted = adapt_backbone_config(config.to_dict())
        self.config = NemotronHConfig(**adapted)
        self.tts_config = EasyMagpieTTSConfig.from_dict(adapted)
        self.backbone = NemotronHForCausalLM(
            config=self.config, quant_config=quant_config, prefix=prefix
        )
        self.heads = EasyMagpieTTSHeads(self.tts_config)
        patch_silu_shared_experts(self.backbone)
        self.decode_step: EasyMagpieDecodeStep | None = None
        self.last_audio_codes: torch.Tensor | None = None
        self.last_phoneme_tokens: torch.Tensor | None = None
        self.last_audio_eos: torch.Tensor | None = None

    def get_input_embeddings(self) -> nn.Module:
        return self.backbone.get_input_embeddings()

    def compose_conditioning(
        self,
        *,
        text_tokens: torch.Tensor,
        text_valid: torch.Tensor,
        phoneme_tokens: torch.Tensor,
        phoneme_valid: torch.Tensor,
        previous_audio_codes: torch.Tensor,
        audio_valid: torch.Tensor,
    ) -> torch.Tensor:
        """Sum the current text, previous phoneme, and previous audio embeddings."""
        text = self.heads.text_embedding(text_tokens)
        phoneme = self.heads.embed_phonemes(phoneme_tokens)
        audio = self.heads.embed_audio_frame(previous_audio_codes)
        return (
            mask_rows(text, text_valid)
            + mask_rows(phoneme, phoneme_valid)
            + mask_rows(audio, audio_valid)
        )

    def decode_tts_heads(
        self, hidden_states: torch.Tensor, step: EasyMagpieDecodeStep
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample acoustic codes, predict phonemes, and flag acoustic EOS rows."""
        batch = hidden_states.shape[0]
        codes = self.heads.sample_codes(
            hidden_states,
            temperatures=step.temperatures[:batch],
            top_ks=step.top_ks[:batch],
            seeds=step.seeds[:batch],
            positions=step.positions[:batch],
            max_top_k=step.max_top_k,
        )
        phonemes = self.heads.predict_phonemes(hidden_states)
        eos = (codes == self.tts_config.audio_eos_id).any(dim=1) & step.audio_valid[
            :batch
        ]
        return codes, phonemes, eos

    def make_stop_logits(
        self, hidden_states: torch.Tensor, eos: torch.Tensor
    ) -> torch.Tensor:
        """Deterministic continue/stop logits for SGLang's greedy token sampler."""
        logits = hidden_states.new_zeros(
            hidden_states.shape[0], int(self.config.vocab_size)
        )
        logits[:, int(self.config.eos_token_id)] = torch.where(
            eos, logits.new_full((), STOP_LOGIT), logits.new_full((), -STOP_LOGIT)
        )
        return logits

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        pp_proxy_tensors: PPProxyTensors | None = None,
        input_embeds_are_projected: bool | None = None,
        omni_prefill_rids: list[str] | None = None,
    ) -> LogitsProcessorOutput:
        del input_embeds_are_projected, omni_prefill_rids
        if input_embeds is None:
            input_embeds = forward_batch.input_embeds
        else:
            pass
        hidden_states = self.backbone.model(
            input_ids,
            positions,
            forward_batch,
            pp_proxy_tensors=pp_proxy_tensors,
            inputs_embeds=input_embeds,
        )
        if forward_batch.forward_mode.is_extend():
            # Prefill folds the text lead-in; only the phoneme feedback from
            # the final row is needed before acoustic decoding starts.
            last_rows = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states = hidden_states[last_rows]
            self.last_phoneme_tokens = self.heads.predict_phonemes(hidden_states)
            eos = torch.zeros(
                hidden_states.shape[0], device=hidden_states.device, dtype=torch.bool
            )
        elif self.decode_step is not None:
            codes, phonemes, eos = self.decode_tts_heads(
                hidden_states, self.decode_step
            )
            self.last_audio_codes = codes
            self.last_phoneme_tokens = phonemes
            self.last_audio_eos = eos
        else:
            raise RuntimeError("EasyMagpie decode ran without per-row decode inputs")
        return LogitsProcessorOutput(
            next_token_logits=self.make_stop_logits(hidden_states, eos),
            hidden_states=hidden_states,
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        head_weights: list[tuple[str, torch.Tensor]] = []
        self.backbone.load_weights(partition_weights(weights, head_weights))
        self.heads.load_state_dict(dict(head_weights), strict=True)
        return set(dict(self.named_parameters()))


def mask_rows(values: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    return values * valid.to(dtype=values.dtype).unsqueeze(-1)


EntryClass = EasyMagpieTTSForConditionalGeneration

__all__ = [
    "EasyMagpieDecodeStep",
    "EasyMagpieTTSForConditionalGeneration",
    "EntryClass",
    "patch_silu_shared_experts",
]
