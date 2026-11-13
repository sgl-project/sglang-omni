# SPDX-License-Identifier: Apache-2.0
"""Paged Llama backbone + codebook-0 head + dense depth decoder.

One scheduler step is one audio frame. Registered as CsmForConditionalGeneration.
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
from sglang_omni.models.csm_tts.modeling import (
    CsmDepthDecoder,
    CsmFrameEmbedding,
)
from sglang_omni.models.csm_tts.sampler import (
    K_MAX,
    STAGING_WIDTH,
    CsmBatchedSamplerState,
    finalize_frame,
    sample_codes_batched,
)
from sglang_omni.models.csm_tts.utils import NUM_CODEBOOKS
from sglang_omni.models.csm_tts.weight_loader import CsmWeightMapper

_DEFAULT_MAX_BATCH_SIZE = 64


@dataclass
class CsmGenParams:
    """Per-request sampling for codebook-0 and the depth decoder."""

    temperature: float = 0.9
    top_k: int = 50
    top_p: float | None = None
    depth_temperature: float = 0.9
    depth_top_k: int = 50
    seed: int | None = None


def flat_sampling_attr(sampling_info: Any, attr: str) -> list | None:
    val = getattr(sampling_info, attr, None)
    if val is None:
        return None
    if hasattr(val, "cpu"):
        return val.detach().cpu().flatten().tolist()
    return list(val)


def normalize_top_k(top_k: Any) -> int:
    """Clamp top_k to (0, K_MAX]; anything else is a no-op filter."""
    try:
        k = int(top_k)
    except (TypeError, ValueError):
        return K_MAX
    return k if 0 < k <= K_MAX else K_MAX


class CsmTTSModel(nn.Module):
    """Paged Llama backbone + fused frame embedding + codebook-0 head + depth decoder."""

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

        self._max_batch_size = min(int(max_batch_size), 16)
        pool_size = self._max_batch_size + 1
        self.depth_decoder = CsmDepthDecoder(
            depth_cfg, self.frame_embedding, max_slots=pool_size
        )
        self.codebooks_head = self.depth_decoder.codebooks_head

        backbone_dtype = self.backbone.model.embed_tokens.weight.dtype
        self.frame_embedding.to(dtype=backbone_dtype)
        self.codebook0_head.to(dtype=backbone_dtype)
        self.depth_decoder.to(dtype=backbone_dtype)
        self.frame_embedding.requires_grad_(False)
        self.codebook0_head.requires_grad_(False)
        self.depth_decoder.requires_grad_(False)
        self.depth_decoder.eval()

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
        self._cg_depth_temperature = torch.ones(
            pool_size, dtype=torch.float32, device=device
        )
        self._cg_depth_top_k_buf = torch.full(
            (pool_size,), K_MAX, dtype=torch.long, device=device
        )
        self._cg_codes_BN = torch.zeros(
            pool_size, NUM_CODEBOOKS, dtype=torch.long, device=device
        )
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

    def acquire_row(self, req_id: str) -> int:
        """Bind req_id to a free pool row."""
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
        """Return the row bound to req_id to the free list."""
        row = self._rid_to_row.pop(req_id, None)
        if row is not None:
            self._free_rows.append(row)
        self._output_codes.pop(req_id, None)

    def reset_request(self, req_id: str) -> None:
        """Release the sampler row for an aborted or finished request."""
        self.release_row(req_id)

    @property
    def sampler_pool_max_running_requests(self) -> int:
        return self._max_batch_size

    def get_output_codes(self, req_id: str) -> torch.Tensor:
        """Frames emitted so far for req_id, including the EOS frame."""
        codes = self._output_codes.get(req_id)
        if not codes:
            return torch.empty(0, self._num_codebooks, dtype=torch.long)
        return torch.stack(codes, dim=0).to(torch.long).cpu()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: Any,
        input_embeds: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> LogitsProcessorOutput:
        """One scheduler step: prefill samples frame 0, decode samples the next."""
        is_decode = self.is_decode_step(forward_batch)
        omni_prefill_rids = kwargs.get("omni_prefill_rids")
        if is_decode:
            input_embeds = self.decode_step_embeds_cg(int(input_ids.shape[0]))
            req_ids = None
            gen_params = None
        else:
            req_ids, gen_params = self.extract_batch_metadata(
                forward_batch, omni_prefill_rids
            )

        hidden_states = self.backbone.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
        )
        hidden_states_last = self.last_token_hidden(
            hidden_states, forward_batch, is_decode=is_decode
        )

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

    @staticmethod
    def last_token_hidden(
        hidden_states: torch.Tensor,
        forward_batch: Any,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """Last hidden of each sequence in a packed prefill batch."""
        extend = getattr(forward_batch, "extend_seq_lens", None)
        mode = getattr(forward_batch, "forward_mode", None)
        is_extend = bool(
            mode is not None
            and callable(getattr(mode, "is_extend", None))
            and mode.is_extend()
        )
        if not is_decode and is_extend and extend is not None:
            last_index = torch.cumsum(extend, dim=0) - 1
            return hidden_states[last_index]
        if hidden_states.ndim == 3:
            return hidden_states[:, -1, :]
        return hidden_states

    def decode_step_embeds_cg(self, bs: int) -> torch.Tensor:
        """Embed the previous frame for every decode row. Padding rows are discarded."""
        return self.frame_embedding(self._cg_active_last_codes[:bs])

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_BD: torch.Tensor) -> torch.Tensor:
        """Sample one decode frame into the ``_cg_*`` buffers. Frozen rows emit STOP_CODE."""
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
        out_codes, new_done, was_done = finalize_frame(
            codes_B32, self._cg_active_generation_done[:bs]
        )
        self._cg_was_done[:bs] = was_done
        self._cg_active_generation_done[:bs] = new_done
        # STOP_CODE=-1 must never reach frame_embedding.
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
        """Sample frame 0 on prefill and record it in _output_codes."""
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
            [normalize_top_k(p.top_k) for p in gen_params],
            dtype=torch.long,
            device=device,
        )
        depth_temperature = torch.tensor(
            [p.depth_temperature for p in gen_params],
            dtype=torch.float32,
            device=device,
        )
        depth_top_k_buf = torch.tensor(
            [normalize_top_k(p.depth_top_k) for p in gen_params],
            dtype=torch.long,
            device=device,
        )

        cb0_B = sample_codes_batched(logits0, temperature, top_k_buf, top_p)
        codes_B32 = self.depth_decoder.generate_frame(
            hidden_BD, cb0_B, depth_temperature, depth_top_k_buf, bs=batch_size
        )

        pool = self._sampler_pool
        out_codes, new_done, was_done = finalize_frame(
            codes_B32, pool.generation_done[row_indices]
        )
        pool.generation_done[row_indices] = new_done
        pool.last_codes[row_indices] = torch.where(
            was_done.unsqueeze(-1), pool.last_codes[row_indices], codes_B32
        )
        pool.frames_emitted[row_indices] += (~was_done).to(torch.int32)

        was_done_cpu = was_done.cpu().tolist()
        out_codes = out_codes.detach().to(torch.long)
        for b in range(batch_size):
            if was_done_cpu[b]:
                continue
            self._output_codes.setdefault(req_ids[b], []).append(out_codes[b])

        return out_codes

    @staticmethod
    def is_decode_step(forward_batch: Any) -> bool:
        mode = getattr(forward_batch, "forward_mode", None)
        if mode is None:
            return False
        is_decode = getattr(mode, "is_decode", None)
        return bool(is_decode()) if callable(is_decode) else False

    @staticmethod
    def infer_batch_size(forward_batch: Any) -> int:
        seq_lens = getattr(forward_batch, "seq_lens", None)
        if seq_lens is not None and hasattr(seq_lens, "shape"):
            return int(seq_lens.shape[0])
        return int(getattr(forward_batch, "batch_size", 1))

    def extract_batch_metadata(
        self,
        forward_batch: Any,
        omni_prefill_rids: list[str] | None = None,
    ) -> tuple[list[str], list[CsmGenParams]]:
        req_ids_raw = (
            omni_prefill_rids
            or getattr(forward_batch, "rids", None)
            or getattr(forward_batch, "req_ids", None)
        )
        batch_size = self.infer_batch_size(forward_batch)
        if not req_ids_raw:
            raise RuntimeError(
                "CSM prefill requires ForwardBatch.rids / omni_prefill_rids "
                "or req_ids"
            )
        req_ids = [str(r) for r in req_ids_raw]

        sampling_info = getattr(forward_batch, "sampling_info", None)
        # Depth params are stamped onto the batch in before_prefill.
        depth_params = getattr(forward_batch, "csm_depth_params", None)
        gen_params = self.gen_params_for_batch(sampling_info, depth_params, batch_size)
        return req_ids, gen_params

    @staticmethod
    def gen_params_for_batch(
        sampling_info: Any,
        depth_params: list[tuple[float, int]] | None,
        batch_size: int,
    ) -> list[CsmGenParams]:
        """Build per-row params from sampling_info and runner-stamped depth values."""
        temps = top_ps = top_ks = None
        if sampling_info is not None:
            temps = flat_sampling_attr(sampling_info, "temperatures")
            top_ps = flat_sampling_attr(sampling_info, "top_ps")
            top_ks = flat_sampling_attr(sampling_info, "top_ks")

        params: list[CsmGenParams] = []
        for b in range(batch_size):
            p = CsmGenParams()
            if temps is not None:
                p.temperature = float(temps[b])
            if top_ps is not None:
                p.top_p = float(top_ps[b])
            if top_ks is not None:
                p.top_k = normalize_top_k(top_ks[b])
            if depth_params is not None and b < len(depth_params):
                dt, dk = depth_params[b]
                p.depth_temperature = float(dt)
                p.depth_top_k = normalize_top_k(dk)
            params.append(p)
        return params

    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]) -> set[str]:
        """Load checkpoint tensors through CsmWeightMapper."""
        mapper = CsmWeightMapper(
            backbone_dtype=self.backbone.model.embed_tokens.weight.dtype
        )

        backbone_weights: list[Tuple[str, torch.Tensor]] = []
        self_weights: list[Tuple[str, torch.Tensor]] = []
        for name, tensor in weights:
            mapped = mapper.map(name)
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


__all__ = ["CsmGenParams", "CsmTTSModel", "flat_sampling_attr"]
