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
from sglang_omni.models.higgs_tts.sampler import (
    HiggsBatchedSamplerState,
    HiggsSamplerState,
    batched_step,
)
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


_DEFAULT_MAX_BATCH_SIZE = 64


@dataclass
class _RequestSlot:
    """Per-request runtime bookkeeping inside :class:`HiggsTTSModel`.

    Stage 1 of CUDA Graph migration: the ``sampler`` field is now a
    *view* of one row from :attr:`HiggsTTSModel._sampler_pool`. Mutating
    it in-place still works because the per-row Python dataclass shares
    references with the pool tensors; an explicit writeback through
    :meth:`HiggsBatchedSamplerState.write_row` is what actually persists
    new state values back into the pool.
    """

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
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
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

        # Per-request sampler state lives in a fixed-size GPU pool (Stage 1
        # of the CUDA Graph migration). ``_rid_to_row`` maps request id →
        # row index; ``_free_rows`` tracks unused rows; ``_output_codes``
        # holds the variable-length per-request code log (cannot be
        # tensorised cleanly).
        self._max_batch_size = int(max_batch_size)
        self._sampler_pool = HiggsBatchedSamplerState(
            max_batch_size=self._max_batch_size,
            num_codebooks=num_codebooks,
            device=self.backbone.model.embed_tokens.weight.device,
        )
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(self._max_batch_size))
        self._output_codes: dict[str, list[torch.Tensor]] = {}

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

    def acquire_row(self, req_id: str) -> int:
        """Allocate or look up the sampler-pool row for ``req_id``.

        Idempotent: returns the existing row index when ``req_id`` is
        already mapped. On first call, pops a free row, resets its
        state, and registers the mapping.
        """
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            raise RuntimeError(
                f"HiggsTTSModel sampler pool exhausted (max_batch_size="
                f"{self._max_batch_size}); raise ``max_batch_size`` or limit "
                f"concurrent requests."
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._sampler_pool.reset_row(row)
        return row

    def release_row(self, req_id: str) -> None:
        """Return ``req_id``'s row to the free pool and drop its output
        codes. Idempotent: a no-op when the request isn't mapped."""
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)

    def get_slot(self, req_id: str) -> _RequestSlot:
        """Return a read/write view of one request's slot.

        The ``sampler`` field is a snapshot of the pool row at call time;
        mutating it and then re-reading without going through
        :meth:`HiggsBatchedSamplerState.write_row` will not persist back
        to the pool. Internal call sites that mutate state always end
        with an explicit writeback; external callers (model_runner /
        tests) typically only read.
        """
        row = self.acquire_row(req_id)
        return _RequestSlot(
            sampler=self._sampler_pool.view_row(row),
            output_codes=self._output_codes.setdefault(req_id, []),
        )

    def reset_request(self, req_id: str) -> None:
        """Drop all per-request state. Compat shim over :meth:`release_row`."""
        self.release_row(req_id)

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        codes = self._output_codes.get(req_id)
        if not codes:
            return torch.empty(
                (0, self._num_codebooks),
                dtype=torch.long,
                device=self.multimodal_embedding.modality_embedding_0.weight.device,
            )
        return torch.stack(codes, dim=0).to(torch.long)

    @torch.no_grad()
    def decode_codebooks_batch(
        self,
        hidden_states_BD: torch.Tensor,
        req_ids: list[str],
        gen_params: list[HiggsGenParams],
    ) -> torch.Tensor:
        """Sample multi-codebook tokens for one forward step.

        Real codes land in ``self._output_codes[req_id]``; the returned
        text-vocab logits are a structural placeholder that sglang's downstream
        sampler walks over but whose ``next_token_ids`` are discarded by
        :class:`HiggsTTSModelRunner`.

        Stage 2 of the CUDA Graph migration: the per-row Python sampler
        loop is replaced with a single :func:`batched_step` call. The
        per-row ``.item()`` sync inside the old loop is gone; the only
        remaining D2H copy per step is a single ``[B]``-shape transfer
        of the "was already done at entry" flags so we know which rows
        to skip appending to :attr:`_output_codes`.
        """
        batch_size = hidden_states_BD.shape[0]
        if len(req_ids) != batch_size or len(gen_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"req_ids={len(req_ids)}, gen_params={len(gen_params)}"
            )

        # fp32 for softmax numerical stability.
        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)
        device = logits_BNV.device

        # Allocate / look up pool rows. ``acquire_row`` resets new rows so
        # stale state from a previous owner can't leak in.
        row_indices = torch.tensor(
            [self.acquire_row(rid) for rid in req_ids],
            dtype=torch.long,
            device=device,
        )

        temperature = torch.tensor(
            [p.temperature for p in gen_params],
            dtype=torch.float32,
            device=device,
        )
        has_top_p = any(p.top_p is not None for p in gen_params)
        top_p = (
            torch.tensor(
                [p.top_p if p.top_p is not None else 1.0 for p in gen_params],
                dtype=torch.float32,
                device=device,
            )
            if has_top_p
            else None
        )
        # ``batched_step`` requires uniform top_k across the batch (matches
        # how every Higgs request is configured in practice). If callers
        # ever pass heterogeneous top_k we want to know — fail loudly.
        distinct_top_k = {p.top_k for p in gen_params}
        if len(distinct_top_k) > 1:
            raise ValueError(
                f"batched_step requires uniform top_k across the batch; "
                f"got {distinct_top_k}"
            )
        top_k = next(iter(distinct_top_k))

        was_done = self._sampler_pool.generation_done[row_indices].clone()

        codes_BN = batched_step(
            logits_BNV,
            self._sampler_pool,
            row_indices,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        # One D2H per step: which rows were already finished at entry.
        # Their codes are STOP_CODE sentinels; don't append to output_codes.
        was_done_cpu = was_done.cpu().tolist()
        codes_BN = codes_BN.detach().to(torch.long)
        for b in range(batch_size):
            if was_done_cpu[b]:
                continue
            self._output_codes.setdefault(req_ids[b], []).append(codes_BN[b])

        text_vocab_size = self.backbone.config.vocab_size
        return torch.zeros(
            (batch_size, text_vocab_size),
            device=device,
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
        gen_params = self._gen_params_for_batch(sampling_info, batch_size)
        return req_ids, gen_params

    @staticmethod
    def _gen_params_for_batch(
        sampling_info, batch_size: int
    ) -> list[HiggsGenParams]:
        """Pull per-row sampling params off ``sampling_info`` with at most
        one D2H per attribute (instead of one per row).
        """
        if sampling_info is None:
            return [HiggsGenParams() for _ in range(batch_size)]

        def _to_list(attr: str):
            val = getattr(sampling_info, attr, None)
            if val is None:
                return None
            if hasattr(val, "cpu"):
                # sglang stores some of these as [B, 1] — flatten first
                # so we always get a flat per-row list.
                return val.detach().cpu().flatten().tolist()
            return list(val)

        temps = _to_list("temperatures")
        top_ps = _to_list("top_ps")
        top_ks = _to_list("top_ks")

        params: list[HiggsGenParams] = []
        for b in range(batch_size):
            temp = float(temps[b]) if temps is not None else 1.0
            tp = float(top_ps[b]) if top_ps is not None else None
            tk_raw = int(top_ks[b]) if top_ks is not None else 0
            params.append(
                HiggsGenParams(
                    temperature=temp,
                    top_p=tp,
                    top_k=tk_raw or None,
                )
            )
        return params

    @staticmethod
    def _infer_batch_size(forward_batch) -> int:
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None and hasattr(seq_lens, "shape"):
            return int(seq_lens.shape[0])
        return int(getattr(forward_batch, "batch_size", 1))

    def _decode_step_embeds(
        self, req_ids: list[str], input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Build per-step embeddings from each request's last sampled codes.

        Reads ``last_codes`` directly from the GPU sampler pool. A row
        whose ``delay_count == 0`` has never sampled (the scheduler may
        send us a token before our first decode step for it), so the
        fused-codec embedding is masked out in favour of the text embed
        at those positions.

        Vectorised — zero per-row D2H sync: row mapping uses a single
        Python ``dict.get`` per request, but every GPU read is a single
        gather over ``[B]`` rows.
        """
        device = input_ids.device

        rows_py = [self._rid_to_row.get(rid, -1) for rid in req_ids]
        rows = torch.tensor(rows_py, dtype=torch.long, device=device)
        has_row = rows >= 0
        # ``safe_rows`` lets us gather without an OOB branch; the mask
        # below discards rows we shouldn't have read.
        safe_rows = torch.where(has_row, rows, torch.zeros_like(rows))
        delay_counts = self._sampler_pool.delay_count[safe_rows].to(torch.long)
        has_codes = has_row & (delay_counts > 0)

        last_codes_BN = self._sampler_pool.last_codes[safe_rows].to(torch.long)
        fused_embeds = self.multimodal_embedding.modality_embedding_0(last_codes_BN)

        text_embeds = self.backbone.model.embed_tokens(input_ids)
        if text_embeds.ndim == 3:
            text_embeds = text_embeds[:, -1, :]

        return torch.where(
            has_codes.unsqueeze(-1),
            fused_embeds.to(text_embeds.dtype),
            text_embeds,
        )

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
