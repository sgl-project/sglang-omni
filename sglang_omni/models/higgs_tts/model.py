# SPDX-License-Identifier: Apache-2.0
"""sglang-native Higgs Multimodal Qwen3 TTS model."""

from __future__ import annotations

import logging
from dataclasses import dataclass
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
    K_MAX,
    NO_SEED,
    HiggsBatchedSamplerState,
    batched_step,
    batched_step_direct,
)
from sglang_omni.models.higgs_tts.weight_loader import DiscreteWeightMapper
from sglang_omni.sampling.seed import resolve_row_seed

logger = logging.getLogger(__name__)

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


def _resolve_max_running_requests() -> int:
    try:
        from sglang.srt.server_args import get_global_server_args

        return int(get_global_server_args().max_running_requests)
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        fallback = 64
        logger.warning(
            f"Falling back to Higgs max_running_requests={fallback} because "
            f"SGLang global server args are unavailable: {exc}"
        )
        return fallback


def _flat_sampling_attr(sampling_info, attr: str) -> list | None:
    """Return ``sampling_info.<attr>`` as a flat Python list, or ``None``.

    One D2H per attribute (not per row).
    """
    val = getattr(sampling_info, attr, None)
    if val is None:
        return None
    if hasattr(val, "cpu"):
        return val.detach().cpu().flatten().tolist()
    return list(val)


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

        self._sampler_pool_max_running_requests = _resolve_max_running_requests()
        pool_size = self._sampler_pool_max_running_requests + 1
        self._sampler_pool = HiggsBatchedSamplerState(
            max_batch_size=pool_size,
            num_codebooks=num_codebooks,
            device=self.backbone.model.embed_tokens.weight.device,
        )
        self._padding_row = self._sampler_pool_max_running_requests
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(
            range(self._sampler_pool_max_running_requests)
        )
        self._output_codes: dict[str, list[torch.Tensor]] = {}
        cg_device = self.backbone.model.embed_tokens.weight.device
        self._cg_row_indices = torch.zeros(
            pool_size, dtype=torch.long, device=cg_device
        )
        self._cg_temperature = torch.ones(
            pool_size, dtype=torch.float32, device=cg_device
        )
        self._cg_top_p = torch.ones(pool_size, dtype=torch.float32, device=cg_device)
        self._cg_top_k_buf = torch.full(
            (pool_size,),
            K_MAX,
            dtype=torch.long,
            device=cg_device,
        )
        self._cg_codes_BN = torch.zeros(
            pool_size, num_codebooks, dtype=torch.long, device=cg_device
        )
        # Note(Jiaxin): Packs codes_BN | was_done | active_generation_done into one buffer.
        self._cg_collect_staging = torch.zeros(
            pool_size, num_codebooks + 2, dtype=torch.long, device=cg_device
        )
        self._cg_was_done = torch.zeros(pool_size, dtype=torch.bool, device=cg_device)

        self._cg_active_delay_count = torch.zeros(
            pool_size, dtype=torch.int32, device=cg_device
        )
        self._cg_active_eoc_countdown = torch.full(
            (pool_size,), -1, dtype=torch.int32, device=cg_device
        )
        self._cg_active_generation_done = torch.zeros(
            pool_size, dtype=torch.bool, device=cg_device
        )
        self._cg_active_last_codes = torch.zeros(
            pool_size, num_codebooks, dtype=torch.long, device=cg_device
        )
        self._cg_active_seeds = torch.full(
            (pool_size,), NO_SEED, dtype=torch.long, device=cg_device
        )
        self._cg_active_step_count = torch.zeros(
            pool_size, dtype=torch.long, device=cg_device
        )

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

    @property
    def sampler_pool_max_running_requests(self) -> int:
        return self._sampler_pool_max_running_requests

    def acquire_row(self, req_id: str) -> int:
        """Allocate or look up the sampler-pool row for ``req_id``. Idempotent."""
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            max_running_requests = self._sampler_pool_max_running_requests
            raise RuntimeError(
                f"HiggsTTSModel sampler pool exhausted "
                f"(max_running_requests={max_running_requests}); raise "
                f"``max_running_requests`` or limit concurrent requests."
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._sampler_pool.reset_row(row)
        return row

    def set_request_seed(self, req_id: str, seed: int | None) -> None:
        """Pin ``req_id``'s sampler seed (``None`` -> unseeded/random). Constant
        across the request's AR steps; consumed by ``multinomial_with_seed``."""
        row = self.acquire_row(req_id)
        self._sampler_pool.seeds[row] = (
            NO_SEED if seed is None else resolve_row_seed(seed)
        )

    def release_row(self, req_id: str) -> None:
        """Return ``req_id``'s row to the free pool and drop its output codes."""
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)

    def reset_request(self, req_id: str) -> None:
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
        """Sample multi-codebook tokens for one forward step."""
        batch_size = hidden_states_BD.shape[0]
        if len(req_ids) != batch_size or len(gen_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"req_ids={len(req_ids)}, gen_params={len(gen_params)}"
            )

        # fp32 for softmax numerical stability.
        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)
        device = logits_BNV.device

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
        top_k_buf = torch.tensor(
            [
                (p.top_k if (p.top_k is not None and p.top_k > 0) else K_MAX)
                for p in gen_params
            ],
            dtype=torch.long,
            device=device,
        )

        was_done = self._sampler_pool.generation_done[row_indices].clone()

        codes_BN = batched_step(
            logits_BNV,
            self._sampler_pool,
            row_indices,
            temperature=temperature,
            top_p=top_p,
            top_k_buf=top_k_buf,
        )

        # Note(yichi): One D2H per step to skip STOP-sentinel rows in the Python append loop.
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

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_states_BD: torch.Tensor) -> torch.Tensor:
        """CG-friendly variant of :meth:`decode_codebooks_batch`: reads/writes
        only preallocated ``_cg_*`` buffers, no Python control flow on
        tensor values, no D2H syncs.
        """
        batch_size = hidden_states_BD.shape[0]
        device = hidden_states_BD.device

        logits_BNV = self.modality_head.generate(hidden_states_BD).to(torch.float32)

        temperature = self._cg_temperature[:batch_size]
        top_p = self._cg_top_p[:batch_size]
        top_k_buf = self._cg_top_k_buf[:batch_size]

        delay_count_B = self._cg_active_delay_count[:batch_size].to(torch.long)
        eoc_countdown_B = self._cg_active_eoc_countdown[:batch_size].to(torch.long)
        generation_done_B = self._cg_active_generation_done[:batch_size]
        last_codes_BN_in = self._cg_active_last_codes[:batch_size]
        seeds_B = self._cg_active_seeds[:batch_size]
        step_count_B = self._cg_active_step_count[:batch_size]

        self._cg_was_done[:batch_size] = generation_done_B

        (
            codes_BN,
            new_delay_count_B,
            new_eoc_countdown_B,
            new_generation_done_B,
            new_last_codes_BN,
            new_step_count_B,
        ) = batched_step_direct(
            logits_BNV,
            delay_count_B,
            eoc_countdown_B,
            generation_done_B,
            last_codes_BN_in,
            temperature=temperature,
            top_p=top_p,
            top_k_buf=top_k_buf,
            seeds=seeds_B,
            step_count=step_count_B,
        )
        self._cg_active_step_count[:batch_size] = new_step_count_B
        self._cg_active_delay_count[:batch_size] = new_delay_count_B.to(
            self._cg_active_delay_count.dtype
        )
        self._cg_active_eoc_countdown[:batch_size] = new_eoc_countdown_B.to(
            self._cg_active_eoc_countdown.dtype
        )
        self._cg_active_generation_done[:batch_size] = new_generation_done_B
        self._cg_active_last_codes[:batch_size] = new_last_codes_BN
        self._cg_codes_BN[:batch_size] = codes_BN

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

        Prefill takes runner-supplied ``input_embeds`` (ref-audio pasted
        at ``-100``); decode reads embeds and sampling state from
        ``_cg_active_*`` shadow buffers populated by the runner.
        """
        is_decode = self._is_decode_step(forward_batch)

        if is_decode:
            input_embeds = self._decode_step_embeds_cg(
                input_ids, batch_size=input_ids.shape[0]
            )
        else:
            req_ids, gen_params = self._extract_batch_metadata(forward_batch)

        hidden_states = self.backbone.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )

        if (
            not is_decode
            and hasattr(forward_batch, "forward_mode")
            and forward_batch.forward_mode.is_extend()
            and hasattr(forward_batch, "extend_seq_lens")
        ):
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states_last = hidden_states[last_index]
        else:
            hidden_states_last = hidden_states
            if hidden_states_last.ndim == 3:
                hidden_states_last = hidden_states_last[:, -1, :]

        if is_decode:
            text_logits_BV = self.decode_codebooks_batch_cg(hidden_states_last)
        else:
            text_logits_BV = self.decode_codebooks_batch(
                hidden_states_last, req_ids, gen_params
            )

        return LogitsProcessorOutput(
            next_token_logits=text_logits_BV,
            hidden_states=hidden_states_last,
        )

    def _decode_step_embeds_cg(
        self, input_ids: torch.Tensor, batch_size: int
    ) -> torch.Tensor:
        """Graph-capture-friendly decode-step embedding lookup; reads from
        shadow `_cg_active_*[:bs]` populated by ``before_decode``.
        """
        delay_counts = self._cg_active_delay_count[:batch_size].to(torch.long)
        has_codes = (delay_counts > 0).unsqueeze(-1)

        last_codes_BN = self._cg_active_last_codes[:batch_size].to(torch.long)
        fused_embeds = self.multimodal_embedding.modality_embedding_0(last_codes_BN)

        text_embeds = self.backbone.model.embed_tokens(input_ids)
        if text_embeds.ndim == 3:
            text_embeds = text_embeds[:, -1, :]

        return torch.where(has_codes, fused_embeds.to(text_embeds.dtype), text_embeds)

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
    def _gen_params_for_batch(sampling_info, batch_size: int) -> list[HiggsGenParams]:
        """Pull per-row sampling params off ``sampling_info``."""
        if sampling_info is None:
            return [HiggsGenParams() for _ in range(batch_size)]

        temps = _flat_sampling_attr(sampling_info, "temperatures")
        top_ps = _flat_sampling_attr(sampling_info, "top_ps")
        top_ks = _flat_sampling_attr(sampling_info, "top_ks")

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
