# SPDX-License-Identifier: Apache-2.0
# CSM-1B (sesame/csm-1b): an SGLang `LlamaForCausalLM` backbone + the CSM depth decoder
# ported from HuggingFace Transformers `transformers/models/csm` (Apache-2.0), integrated via
# the sglang-omni TTS model pattern (cf. Higgs / Qwen3-TTS).
"""SGLang-integrated CSM-1B model: paged Llama backbone + cb0 head + dense
depth decoder, with the 31-step depth loop INSIDE the scheduler-visible
decode step (1 frame = 1 backbone position = 1 SGLang "token").

Registered as ``CsmForConditionalGeneration`` in
:meth:`sglang_omni.model_runner.sglang_model_runner.SGLModelRunner._register_omni_model`.

The backbone runs under SGLang paged/varlen attention (composed
``LlamaForCausalLM``) — no dense left-padded batch, which sidesteps the known
bf16 B>=2 left-pad KV-corruption trap by construction.


"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Tuple

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.csm_tts.hf_config import (
    CsmTtsHfConfig,
    build_backbone_llama_config,
    build_depth_decoder_config,
)
from sglang_omni.models.csm_tts.modeling import CsmDepthDecoder, CsmFrameEmbedding
from sglang_omni.models.csm_tts.sampler import (
    K_MAX,
    STAGING_WIDTH,
    CsmBatchedSamplerState,
    frame_finalize_direct,
    sample_codes_batched,
)
from sglang_omni.models.csm_tts.utils import NUM_CODEBOOKS
from sglang_omni.models.csm_tts.weight_loader import CsmWeightMapper

_DEFAULT_MAX_BATCH_SIZE = 64


@dataclass
class CsmGenParams:
    """Per-request generation params — TWO sampling sets:
    cb0 rides SGLang's ``sampling_info``; the depth set is CSM-private
    (defaults (0.9, 50), HF parity) and travels via ``req._omni_data``."""

    temperature: float = 0.9
    top_k: int = 50
    top_p: float | None = None
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    seed: int | None = None


def _flat_sampling_attr(sampling_info: Any, attr: str) -> list | None:
    """Read a flat per-row attribute off SGLang's ``sampling_info`` (one D2H
    per attribute; higgs pattern). Returns ``None`` when absent."""
    val = getattr(sampling_info, attr, None)
    if val is None:
        return None
    if hasattr(val, "cpu"):
        return val.detach().cpu().flatten().tolist()
    return list(val)


def _normalize_top_k(top_k: Any) -> int:
    """Map a raw top_k (incl. SGLang's TOP_K_ALL sentinel / -1 / 0 / huge) to
    a valid filter width: values outside ``(0, K_MAX]`` become ``K_MAX``
    (= no-op filter)."""
    try:
        k = int(top_k)
    except (TypeError, ValueError):
        return K_MAX
    return k if 0 < k <= K_MAX else K_MAX


class CsmTTSModel(nn.Module):
    """CSM-1B for SGLang: ``backbone`` (paged LlamaForCausalLM, 16L/2048/
    32h/8kv/hd64, ctx 2048, tie_word_embeddings=True so the never-used text
    lm_head is free), ``frame_embedding`` ([65632, 2048] fused audio table),
    ``codebook0_head`` (Linear 2048→2051, no bias — loaded from the ckpt's
    ``lm_head.weight``), and ``depth_decoder`` (4L dense, slot-indexed static
    KV ``[4, max_slots, 2, 33, 128]`` bf16).

    Sampler pool: :class:`CsmBatchedSamplerState` with
    ``pool_size = max_batch_size + 1`` (last row = padding row);
    ``_rid_to_row`` / ``_free_rows`` / ``acquire_row`` / ``release_row`` /
    ``reset_request`` follow higgs verbatim.

    CG shadow buffers (pool-sized, sliced ``[:bs]`` in-graph):
    ``_cg_row_indices`` (long), ``_cg_temperature`` / ``_cg_top_p`` (fp32),
    ``_cg_top_k_buf`` (long, = K_MAX neutral), ``_cg_depth_temperature``
    (fp32) + ``_cg_depth_top_k_buf`` (long) — the CSM-delta second param set,
    ``_cg_codes_BN`` (long [P, 32]), ``_cg_collect_staging`` (long [P, 34] =
    ``c0..c31 | was_done | generation_done``), ``_cg_was_done`` (bool),
    ``_cg_active_generation_done`` (bool), ``_cg_active_last_codes``
    (long [P, 32]).


    """

    def __init__(
        self,
        config: CsmTtsHfConfig,
        quant_config: Any | None = None,
        prefix: str = "",
        max_batch_size: int = _DEFAULT_MAX_BATCH_SIZE,
    ) -> None:
        super().__init__()
        self.config = config

        self.backbone = LlamaForCausalLM(
            build_backbone_llama_config(config),
            quant_config=quant_config,
            prefix=add_prefix("backbone", prefix),
        )

        depth_cfg = build_depth_decoder_config(config)
        self._num_codebooks = int(depth_cfg.num_codebooks)
        self._codebook_vocab = int(depth_cfg.vocab_size)
        backbone_hidden = int(self.backbone.config.hidden_size)

        self.frame_embedding = CsmFrameEmbedding(
            num_codebooks=self._num_codebooks,
            codebook_vocab=self._codebook_vocab,
            hidden_size=backbone_hidden,
        )
        self.codebook0_head = nn.Linear(
            backbone_hidden, self._codebook_vocab, bias=False
        )

        self._max_batch_size = int(max_batch_size)
        pool_size = self._max_batch_size + 1
        # frame_embedding is SHARED with the depth decoder (tied table);
        # depth KV is slot-indexed so max_slots = pool_size covers any batch.
        self.depth_decoder = CsmDepthDecoder(
            depth_cfg, self.frame_embedding, max_slots=pool_size
        )
        # Loader-visible alias: the remap routes the checkpoint's
        # ``depth_decoder.codebooks_head.weight`` to ``codebooks_head.weight``;
        # registering the same module under this name makes that resolve
        # without touching the mapper (same Parameter object).
        self.codebooks_head = self.depth_decoder.codebooks_head

        # Match backbone bf16 dtype (higgs "~1 ULP/step" note); buffers
        # (rope cos/sin, mask, depth static KV) cast along with parameters.
        backbone_dtype = self.backbone.model.embed_tokens.weight.dtype
        self.frame_embedding.to(dtype=backbone_dtype)
        self.codebook0_head.to(dtype=backbone_dtype)
        self.depth_decoder.to(dtype=backbone_dtype)

        device = self.backbone.model.embed_tokens.weight.device
        self._sampler_pool = CsmBatchedSamplerState(pool_size, device=device)
        self._padding_row = self._max_batch_size  # last row reserved
        self._rid_to_row: dict[str, int] = {}
        self._free_rows: list[int] = list(range(self._max_batch_size))
        self._output_codes: dict[str, list[torch.Tensor]] = {}

        self._cg_row_indices = torch.zeros(pool_size, dtype=torch.long, device=device)
        self._cg_temperature = torch.ones(pool_size, dtype=torch.float32, device=device)
        self._cg_top_p = torch.ones(pool_size, dtype=torch.float32, device=device)
        self._cg_top_k_buf = torch.full(
            (pool_size,), K_MAX, dtype=torch.long, device=device
        )
        # CSM delta vs higgs: second (depth-decoder-private) sampling set.
        self._cg_depth_temperature = torch.ones(
            pool_size, dtype=torch.float32, device=device
        )
        self._cg_depth_top_k_buf = torch.full(
            (pool_size,), K_MAX, dtype=torch.long, device=device
        )
        self._cg_codes_BN = torch.zeros(
            pool_size, NUM_CODEBOOKS, dtype=torch.long, device=device
        )
        # Packed D2H row: c0..c31 | was_done | generation_done.
        self._cg_collect_staging = torch.zeros(
            pool_size, STAGING_WIDTH, dtype=torch.long, device=device
        )
        self._cg_was_done = torch.zeros(pool_size, dtype=torch.bool, device=device)
        self._cg_active_generation_done = torch.zeros(
            pool_size, dtype=torch.bool, device=device
        )
        self._cg_active_last_codes = torch.zeros(
            pool_size, NUM_CODEBOOKS, dtype=torch.long, device=device
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    @property
    def num_codebooks(self) -> int:
        return self._num_codebooks

    @property
    def codebook_vocab_size(self) -> int:
        return self._codebook_vocab

    # --- sampler-pool row lifecycle (higgs verbatim) -------------------------

    def acquire_row(self, req_id: str) -> int:
        """Bind ``req_id`` to a free pool row (reset on acquire). Returns row."""
        row = self._rid_to_row.get(req_id)
        if row is not None:
            return row
        if not self._free_rows:
            raise RuntimeError(
                f"CsmTTSModel sampler pool exhausted (max_batch_size="
                f"{self._max_batch_size}); raise ``max_batch_size`` or limit "
                f"concurrent requests."
            )
        row = self._free_rows.pop()
        self._rid_to_row[req_id] = row
        self._sampler_pool.reset_row(row)
        return row

    def release_row(self, req_id: str) -> None:
        """Return the row bound to ``req_id`` to the free list (idempotent)."""
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)

    def reset_request(self, req_id: str) -> None:
        """Abort/result-adapter hook: release the sampler row + drop
        ``_output_codes[req_id]`` (wired as ``OmniScheduler.abort_callback``)."""
        self.release_row(req_id)

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        """All frames emitted so far for ``req_id``.

        Returns:
            int64 CPU ``[F, 32]`` (includes the EOS frame — HF ``sequences``
            parity; the vocoder never receives it).
        """
        codes = self._output_codes.get(req_id)
        if not codes:
            return torch.empty(0, self._num_codebooks, dtype=torch.long)
        return torch.stack(codes, dim=0).to(torch.long).cpu()

    # --- forward -------------------------------------------------------------

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        """One scheduler step.

        Decode: ``input_embeds = _decode_step_embeds_cg(bs)``; prefill: the
        runner supplies overlay embeds. Then ``hidden = self.backbone.model(
        input_ids, positions, forward_batch, input_embeds)`` — SGLang's
        LlamaModel applies the final RMSNorm (upstream llama.py:400), giving
        the POST-norm hidden CSM's depth conditioning requires. Last-token
        hidden via ``cumsum(extend_seq_lens) - 1`` on prefill. Then
        ``decode_codebooks_batch_cg(hidden)`` (decode) or
        ``decode_codebooks_batch(hidden, req_ids, gen_params)`` (prefill —
        frame 0 IS sampled at prefill, like higgs).

        Args:
            input_ids: int64 ``[T_total]`` (decode: ``[bs]``).
            positions: int64 ``[T_total]``.
            forward_batch: SGLang ``ForwardBatch``.
            input_embeds: optional ``[T_total, 2048]`` bf16 overlay.

        Returns:
            Dummy ``LogitsProcessorOutput(next_token_logits=zeros(bs,
            vocab_size, fp32))``: cb0 is published via the runner, logit
            processors run over zeros harmlessly.
        """
        is_decode = self._is_decode_step(forward_batch)

        if is_decode:
            input_embeds = self._decode_step_embeds_cg(int(input_ids.shape[0]))
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
            and getattr(forward_batch, "extend_seq_lens", None) is not None
        ):
            last_index = torch.cumsum(forward_batch.extend_seq_lens, dim=0) - 1
            hidden_states_last = hidden_states[last_index]
        else:
            hidden_states_last = hidden_states
            if hidden_states_last.ndim == 3:
                hidden_states_last = hidden_states_last[:, -1, :]

        if is_decode:
            self.decode_codebooks_batch_cg(hidden_states_last)
        else:
            self.decode_codebooks_batch(hidden_states_last, req_ids, gen_params)

        batch_size = hidden_states_last.shape[0]
        text_logits_BV = torch.zeros(
            (batch_size, self.backbone.config.vocab_size),
            device=hidden_states_last.device,
            dtype=torch.float32,
        )
        return LogitsProcessorOutput(
            next_token_logits=text_logits_BV,
            hidden_states=hidden_states_last,
        )

    def _decode_step_embeds_cg(self, bs: int) -> torch.Tensor:
        """UNCONDITIONAL ``frame_embedding(_cg_active_last_codes[:bs])`` —
        post-prefill a last frame always exists (no higgs ``delay_count>0``
        text fallback; padding rows embed zeros-garbage that the collect
        discards).

        Returns:
            bf16 ``[bs, 2048]``.
        """
        return self.frame_embedding(self._cg_active_last_codes[:bs])

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_BD: torch.Tensor) -> torch.Tensor:
        """CG decode tail: ``logits0 = codebook0_head(hidden).float()`` →
        ``cb0 = sample_codes_batched(logits0, _cg_temperature, _cg_top_k_buf)``
        → ``codes = depth_decoder.generate_frame(hidden, cb0,
        _cg_depth_temperature, _cg_depth_top_k_buf, bs=bs)`` →
        :func:`frame_finalize_direct` → write ``_cg_codes_BN`` /
        ``_cg_was_done`` / ``_cg_active_*``. No host control flow, no D2H.

        TODO: capture this whole tail (together with the backbone forward)
        as one CUDA graph per batch size — eager-only for now.

        Args:
            hidden_BD: bf16 ``[bs, 2048]`` post-norm backbone hidden.

        Returns:
            int64 ``[bs, 32]`` finalized frame codes (STOP_CODE on frozen rows).
        """
        bs = int(hidden_BD.shape[0])

        logits0 = self.codebook0_head(hidden_BD).to(torch.float32)
        cb0_B = sample_codes_batched(
            logits0,
            self._cg_temperature[:bs],
            self._cg_top_k_buf[:bs],
            self._cg_top_p[:bs],
        )
        codes_B32 = self.depth_decoder.generate_frame(
            hidden_BD,
            cb0_B,
            self._cg_depth_temperature,
            self._cg_depth_top_k_buf,
            bs=bs,
        )

        last_codes_in = self._cg_active_last_codes[:bs]
        out_codes, new_done, was_done = frame_finalize_direct(
            codes_B32, self._cg_active_generation_done[:bs]
        )
        self._cg_was_done[:bs] = was_done
        self._cg_active_generation_done[:bs] = new_done
        # Frozen rows keep their previous frame (raw sampled codes stay in
        # embed range; STOP_CODE=-1 must never reach frame_embedding).
        self._cg_active_last_codes[:bs] = torch.where(
            was_done.unsqueeze(-1), last_codes_in, codes_B32
        )
        self._cg_codes_BN[:bs] = out_codes
        return out_codes

    @torch.no_grad()
    def decode_codebooks_batch(
        self,
        hidden_BD: torch.Tensor,
        req_ids: list[str],
        gen_params: list[CsmGenParams],
    ) -> torch.Tensor:
        """Eager pool-indexed variant for PREFILL (rows via ``acquire_row``,
        params from request data). Samples frame 0 (cb0 + full depth loop on
        the last prompt position's hidden) and runs the SAME
        :func:`frame_finalize_direct` EOS machine — frame-0 EOS is legal HF
        behavior and empirically real for this checkpoint:
        the runner then marks the request finished at prefill, emits NOTHING
        to the vocoder, zero-decode lifecycle. Appends to
        ``_output_codes[rid]``.

        Args:
            hidden_BD: bf16 ``[n_req, 2048]`` last-prompt-position hiddens.
            req_ids: rids aligned with rows of ``hidden_BD``.
            gen_params: per-request dual sampling param sets.

        Returns:
            int64 ``[n_req, 32]`` frame-0 codes.
        """
        batch_size = int(hidden_BD.shape[0])
        if len(req_ids) != batch_size or len(gen_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"req_ids={len(req_ids)}, gen_params={len(gen_params)}"
            )
        device = hidden_BD.device

        logits0 = self.codebook0_head(hidden_BD).to(torch.float32)

        row_indices = torch.tensor(
            [self.acquire_row(rid) for rid in req_ids],
            dtype=torch.long,
            device=device,
        )

        temperature = torch.tensor(
            [p.temperature for p in gen_params], dtype=torch.float32, device=device
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
            [_normalize_top_k(p.top_k) for p in gen_params],
            dtype=torch.long,
            device=device,
        )
        depth_temperature = torch.tensor(
            [p.depth_temperature for p in gen_params],
            dtype=torch.float32,
            device=device,
        )
        depth_top_k_buf = torch.tensor(
            [_normalize_top_k(p.depth_top_k) for p in gen_params],
            dtype=torch.long,
            device=device,
        )

        cb0_B = sample_codes_batched(logits0, temperature, top_k_buf, top_p)
        codes_B32 = self.depth_decoder.generate_frame(
            hidden_BD, cb0_B, depth_temperature, depth_top_k_buf, bs=batch_size
        )

        pool = self._sampler_pool
        out_codes, new_done, was_done = frame_finalize_direct(
            codes_B32, pool.generation_done[row_indices]
        )
        pool.generation_done[row_indices] = new_done
        pool.last_codes[row_indices] = torch.where(
            was_done.unsqueeze(-1), pool.last_codes[row_indices], codes_B32
        )
        pool.frames_emitted[row_indices] += (~was_done).to(torch.int32)

        # Note: one D2H per step to skip STOP-sentinel rows in the append loop
        # (higgs pattern).
        was_done_cpu = was_done.cpu().tolist()
        out_codes = out_codes.detach().to(torch.long)
        for b in range(batch_size):
            if was_done_cpu[b]:
                continue
            self._output_codes.setdefault(req_ids[b], []).append(out_codes[b])

        return out_codes

    # --- batch metadata (prefill path) ----------------------------------------

    @staticmethod
    def _is_decode_step(forward_batch: Any) -> bool:
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None:
            return False
        is_decode = getattr(mode, "is_decode", None)
        return bool(is_decode()) if callable(is_decode) else False

    @staticmethod
    def _infer_batch_size(forward_batch: Any) -> int:
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None and hasattr(seq_lens, "shape"):
            return int(seq_lens.shape[0])
        return int(getattr(forward_batch, "batch_size", 1))

    def _extract_batch_metadata(
        self, forward_batch: Any
    ) -> tuple[list[str], list[CsmGenParams]]:
        req_ids_raw = getattr(forward_batch, "req_ids", None)
        batch_size = self._infer_batch_size(forward_batch)
        if req_ids_raw is None:
            req_ids = [f"req-{i}" for i in range(batch_size)]
        else:
            req_ids = [str(r) for r in req_ids_raw]

        sampling_info = getattr(forward_batch, "sampling_info", None)
        # Depth params never ride sampling_info — the runner stamps them from
        # ``req._omni_data`` in before_prefill.
        depth_params = getattr(forward_batch, "csm_depth_params", None)
        gen_params = self._gen_params_for_batch(sampling_info, depth_params, batch_size)
        return req_ids, gen_params

    @staticmethod
    def _gen_params_for_batch(
        sampling_info: Any,
        depth_params: list[tuple[float, int]] | None,
        batch_size: int,
    ) -> list[CsmGenParams]:
        """Per-row dual param sets: cb0 from ``sampling_info`` (one D2H per
        attribute), depth from the runner-stamped list; HF defaults
        (0.9, 50) for whichever side is absent."""
        temps = top_ps = top_ks = None
        if sampling_info is not None:
            temps = _flat_sampling_attr(sampling_info, "temperatures")
            top_ps = _flat_sampling_attr(sampling_info, "top_ps")
            top_ks = _flat_sampling_attr(sampling_info, "top_ks")

        params: list[CsmGenParams] = []
        for b in range(batch_size):
            p = CsmGenParams()
            if temps is not None:
                p.temperature = float(temps[b])
            if top_ps is not None:
                p.top_p = float(top_ps[b])
            if top_ks is not None:
                p.top_k = _normalize_top_k(top_ks[b])
            if depth_params is not None and b < len(depth_params):
                dt, dk = depth_params[b]
                p.depth_temperature = float(dt)
                p.depth_top_k = _normalize_top_k(dk)
            params.append(p)
        return params

    # --- weights -------------------------------------------------------------

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Split the checkpoint stream per :class:`CsmWeightMapper`:
        backbone names go through ``LlamaForCausalLM.load_weights`` (qkv /
        gate_up stacking + tied-lm_head skip), own params are copied manually
        with shape checks, everything cast to backbone bf16 on copy. The
        mapper raises on any tensor outside the known census (strict
        accounting; the depth tied-embed duplicate is a deliberate skip).

        Returns:
            Set of own (non-backbone) destination param names that received
            a tensor.
        """
        mapper = CsmWeightMapper(
            backbone_dtype=self.backbone.model.embed_tokens.weight.dtype
        )

        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        self_weights: list[Tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            mapped = mapper.map(name)  # raises ValueError on unexpected names
            if mapped is None:
                continue
            if mapped.startswith("backbone."):
                backbone_weights.append((mapped[len("backbone.") :], tensor))
            else:
                self_weights.append((mapped, tensor))

        self.backbone.load_weights(iter(backbone_weights))

        own_params = dict(self.named_parameters(remove_duplicate=False))
        loaded: set[str] = set()
        for name, tensor in self_weights:
            param = own_params.get(name)
            if param is None:
                raise ValueError(
                    f"CSM weight remap produced unknown destination {name!r}"
                )
            if param.shape != tensor.shape:
                raise ValueError(
                    f"Shape mismatch for {name}: expected {tuple(param.shape)}, "
                    f"got {tuple(tensor.shape)}"
                )
            param.data.copy_(tensor.to(param.dtype))
            loaded.add(name)

        return loaded


__all__ = ["CsmGenParams", "CsmTTSModel", "_flat_sampling_attr"]
