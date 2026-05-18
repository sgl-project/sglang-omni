# SPDX-License-Identifier: Apache-2.0
"""sglang-native Higgs Multimodal Qwen3 TTS model.

Composes sglang's built-in :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM`
as the text backbone with the fused multi-codebook embedding / head.
Registered in sglang's ``ModelRegistry`` under
``HiggsMultimodalQwen3ForConditionalGeneration`` by
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.qwen3 import Qwen3ForCausalLM
from torch import nn

from sglang_omni.models.higgs_tts.hf_config import HiggsMultimodalQwen3Config
from sglang_omni.models.higgs_tts.modeling import (
    HiggsFusedMultiTextEmbedding,
    HiggsFusedMultiTextHead,
)
from sglang_omni.models.higgs_tts.sampler import STOP_CODE, HiggsSamplerState
from sglang_omni.models.higgs_tts.sampler import step as sampler_step
from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper

# Higgs ckpt prefixes → sglang Qwen3ForCausalLM parameter tree (under ``backbone.``).
_BACKBONE_PREFIX_MAP: dict[str, str] = {
    "tied.embedding.text_embedding.": "backbone.model.embed_tokens.",
    "body.layers.": "backbone.model.layers.",
    "body.norm.": "backbone.model.norm.",
    "tied.head.text_head.": "backbone.lm_head.",
}


@dataclass
class HiggsGenParams:
    """Per-request decoding parameters consumed by :func:`sampler.step`."""

    temperature: float = 1.0
    top_p: float | None = None
    top_k: int | None = None


@dataclass
class _RequestSlot:
    """Per-request runtime bookkeeping inside :class:`HiggsTTSModel`."""

    sampler: HiggsSamplerState
    output_codes: list[torch.Tensor] = field(default_factory=list)


class _HiggsMultimodalEmbedding(nn.Module):
    """Container matching the Higgs checkpoint layout for straight prefix subst."""

    def __init__(self, num_codebooks: int, vocab_size: int, hidden_size: int):
        super().__init__()
        self.modality_embedding_0 = HiggsFusedMultiTextEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )


class HiggsTTSModel(nn.Module):
    """Higgs Multimodal Qwen3 model (discrete TTS path) adapted for sglang.

    Composition over :class:`sglang.srt.models.qwen3.Qwen3ForCausalLM` —
    the backbone handles paged attention, KV cache, logits processing and
    standard text weight loading. This wrapper adds:

    - ``multimodal_embedding.modality_embedding_0``: the fused
      :class:`HiggsFusedMultiTextEmbedding` (shape ``[N*V, D]``).
    - ``modality_head``: the fused :class:`HiggsFusedMultiTextHead`, tied
      to the embedding weight when ``audio_encoder_config.tie_word_embeddings``.
    - :meth:`load_weights` that remaps Higgs checkpoint names and splits
      the stream between the backbone and the multimodal modules.

    Multi-codebook input embedding overlay (the ``-100`` placeholder paste
    from the reference audio) is performed by the engine model_runner; this
    model just consumes the prepared ``input_embeds`` in its forward.
    """

    def __init__(
        self,
        config: HiggsMultimodalQwen3Config,
        quant_config=None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        text_config = config.get_text_config()
        self.backbone = Qwen3ForCausalLM(
            text_config,
            quant_config=quant_config,
            prefix=prefix + "backbone" if prefix else "backbone",
        )

        enc_cfg = config.audio_encoder_config or {}
        encoder_type = enc_cfg.get("encoder_type", "discrete")
        if encoder_type != "discrete":
            raise NotImplementedError(
                f"HiggsTTSModel currently supports only the discrete "
                f"TTS path; got encoder_type={encoder_type!r}. Whisper/Qwen3-AUT "
                f"(ASR) encoders are planned for a future PR."
            )

        num_codebooks: int = int(enc_cfg["num_codebooks"])
        vocab_size: int = int(enc_cfg["vocab_size"])
        hidden_size: int = int(enc_cfg.get("out_dim", text_config.hidden_size))
        self._num_codebooks = num_codebooks
        self._codebook_vocab_size = vocab_size
        self._tie_modality = bool(enc_cfg.get("tie_word_embeddings", True))

        self.multimodal_embedding = _HiggsMultimodalEmbedding(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        self.modality_head = HiggsFusedMultiTextHead(
            num_codebooks=num_codebooks,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
        )
        # Match backbone bf16 dtype; fp32 fused embed accumulates ~1 ULP per AR step.
        backbone_dtype = self.backbone.model.embed_tokens.weight.dtype
        self.multimodal_embedding.to(dtype=backbone_dtype)
        self.modality_head.to(dtype=backbone_dtype)
        if self._tie_modality:
            self.modality_head.weight = (
                self.multimodal_embedding.modality_embedding_0.weight
            )

        self._slots: dict[str, _RequestSlot] = {}

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def get_multimodal_embedding(self) -> HiggsFusedMultiTextEmbedding:
        return self.multimodal_embedding.modality_embedding_0

    def get_modality_head(self) -> HiggsFusedMultiTextHead:
        return self.modality_head

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_vocab_size(self) -> int:
        return self._codebook_vocab_size

    def get_slot(self, req_id: str) -> _RequestSlot:
        slot = self._slots.get(req_id)
        if slot is None:
            slot = _RequestSlot(
                sampler=HiggsSamplerState(num_codebooks=self._num_codebooks)
            )
            self._slots[req_id] = slot
        return slot

    def reset_request(self, req_id: str) -> None:
        self._slots.pop(req_id, None)

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        slot = self._slots.get(req_id)
        if slot is None or not slot.output_codes:
            return torch.empty(
                (0, self._num_codebooks),
                dtype=torch.long,
                device=self.multimodal_embedding.modality_embedding_0.weight.device,
            )
        return torch.stack(slot.output_codes, dim=0).to(torch.long)

    @torch.no_grad()
    def decode_codebooks_batch(
        self,
        hidden_states_BD: torch.Tensor,
        req_ids: list[str],
        gen_params: list[HiggsGenParams],
    ) -> torch.Tensor:
        """Sample multi-codebook tokens for one forward step.

        Real codes land in ``self._slots[req_id].output_codes``; the returned
        text-vocab logits are a structural placeholder that sglang's downstream
        sampler walks over but whose ``next_token_ids`` are discarded by
        :class:`HiggsTTSModelRunner`.
        """
        batch_size = hidden_states_BD.shape[0]
        if len(req_ids) != batch_size or len(gen_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"req_ids={len(req_ids)}, gen_params={len(gen_params)}"
            )

        # fp32 for softmax numerical stability.
        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)

        for b in range(batch_size):
            slot = self.get_slot(req_ids[b])
            params = gen_params[b]
            codes_N = sampler_step(
                logits_BNV[b],
                slot.sampler,
                temperature=params.temperature,
                top_p=params.top_p,
                top_k=params.top_k,
            )
            # STOP_CODE sentinel rows can arrive if a finished request is
            # accidentally re-stepped; guard so output_codes stays clean.
            if int(codes_N[0].item()) != STOP_CODE:
                slot.output_codes.append(codes_N.detach().to(torch.long))

        text_vocab_size = self.backbone.config.vocab_size
        return torch.zeros(
            (batch_size, text_vocab_size),
            device=hidden_states_BD.device,
            dtype=torch.float32,
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ):
        """Run the backbone then sample multi-codebook codes per request.

        Prefill: caller supplies ``input_embeds`` with the ref-audio overlay
        already pasted at ``-100`` positions (see
        :class:`HiggsTTSModelRunner._build_prefill_input_embeds`).
        Decode: input_embeds is rebuilt here from each slot's ``last_codes``.
        """
        req_ids, gen_params = self._extract_batch_metadata(forward_batch)

        if input_embeds is None and self._is_decode_step(forward_batch):
            input_embeds = self._decode_step_embeds(req_ids, input_ids)

        hidden_states = self.backbone.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        if (
            hasattr(forward_batch, "forward_mode")
            and forward_batch.forward_mode.is_extend()
            and hasattr(forward_batch, "extend_seq_lens")
        ):
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states_last = hidden_states[last_index]
        else:
            hidden_states_last = hidden_states
            if hidden_states_last.ndim == 3:
                hidden_states_last = hidden_states_last[:, -1, :]

        text_logits_BV = self.decode_codebooks_batch(
            hidden_states_last, req_ids, gen_params
        )

        return LogitsProcessorOutput(
            next_token_logits=text_logits_BV,
            hidden_states=hidden_states_last,
        )

    @staticmethod
    def _is_decode_step(forward_batch) -> bool:
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None:
            return False
        is_decode = getattr(mode, "is_decode", None)
        return bool(is_decode()) if callable(is_decode) else False

    def _extract_batch_metadata(
        self, forward_batch
    ) -> tuple[list[str], list[HiggsGenParams]]:
        req_ids_raw = getattr(forward_batch, "req_ids", None)
        batch_size = self._infer_batch_size(forward_batch)
        if req_ids_raw is None:
            req_ids = [f"req-{i}" for i in range(batch_size)]
        else:
            req_ids = [str(r) for r in req_ids_raw]

        sampling_info = getattr(forward_batch, "sampling_info", None)
        gen_params: list[HiggsGenParams] = []
        for b in range(batch_size):
            gen_params.append(self._gen_params_for_row(sampling_info, b))
        return req_ids, gen_params

    @staticmethod
    def _gen_params_for_row(sampling_info, row: int) -> HiggsGenParams:
        if sampling_info is None:
            return HiggsGenParams()

        def _pick(attr: str, default):
            val = getattr(sampling_info, attr, None)
            if val is None:
                return default
            return float(val[row].item() if hasattr(val[row], "item") else val[row])

        return HiggsGenParams(
            temperature=_pick("temperatures", 1.0),
            top_p=_pick("top_ps", None),
            top_k=int(_pick("top_ks", 0)) or None,
        )

    @staticmethod
    def _infer_batch_size(forward_batch) -> int:
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None and hasattr(seq_lens, "shape"):
            return int(seq_lens.shape[0])
        return int(getattr(forward_batch, "batch_size", 1))

    def _decode_step_embeds(
        self, req_ids: list[str], input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build per-step embeddings from each slot's ``last_codes``.

        Falls back to the text embedding for any request whose slot has no
        ``last_codes`` yet (scheduler may send us a token before we've decoded).
        """
        device = input_ids.device
        N = self._num_codebooks
        last_codes_stack: list[torch.Tensor] = []
        mask: list[bool] = []
        for rid in req_ids:
            slot = self._slots.get(rid)
            last = None if slot is None else slot.sampler.last_codes
            if last is None:
                last_codes_stack.append(torch.zeros(N, dtype=torch.long, device=device))
                mask.append(False)
            else:
                last_codes_stack.append(last.to(device=device, dtype=torch.long))
                mask.append(True)
        codes_BN = torch.stack(last_codes_stack, dim=0)
        fused_embeds = self.multimodal_embedding.modality_embedding_0(codes_BN)

        text_embeds = self.backbone.model.embed_tokens(input_ids)
        if text_embeds.ndim == 3:
            text_embeds = text_embeds[:, -1, :]

        mask_t = torch.tensor(mask, device=device).unsqueeze(-1)
        return torch.where(mask_t, fused_embeds.to(text_embeds.dtype), text_embeds)

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Remap Higgs ckpt names then split between backbone and own modules.

        Returns the set of *own* parameter names loaded (multimodal embedding +
        optionally the untied modality head). Text-backbone loading delegates
        to :meth:`Qwen3ForCausalLM.load_weights`, which does qkv / gate_up
        stacking and lm_head tying internally.
        """
        mapper = DiscreteWeightMapper(
            text_prefix_map=_BACKBONE_PREFIX_MAP,
            tie_modality=self._tie_modality,
        )

        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        self_weights: list[Tuple[str, torch.Tensor]] = []
        loaded: set[str] = set()
        own_names = self._own_param_names()

        for name, tensor in weights:
            mapped = mapper.map(name)
            if mapped is None:
                continue
            if mapped.startswith("backbone."):
                backbone_weights.append((mapped[len("backbone.") :], tensor))
            elif mapped in own_names:
                self_weights.append((mapped, tensor))

        self.backbone.load_weights(iter(backbone_weights))

        own_params = dict(self.named_parameters(remove_duplicate=False))
        for name, tensor in self_weights:
            param = own_params.get(name)
            if param is None:
                continue
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)

        return loaded

    def _own_param_names(self) -> set[str]:
        names: set[str] = set()
        for name, _ in self.named_parameters(remove_duplicate=False):
            if not name.startswith("backbone."):
                names.add(name)
        return names


__all__ = ["HiggsGenParams", "HiggsTTSModel"]
