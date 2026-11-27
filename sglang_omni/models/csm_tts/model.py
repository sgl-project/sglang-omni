# SPDX-License-Identifier: Apache-2.0
"""Paged Llama backbone + codebook-0 head + dense depth decoder.

One scheduler step is one audio frame. Registered as CsmForConditionalGeneration.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.layers.quantization.base_config import QuantizationConfig
from sglang.srt.model_executor.forward_batch_info import ForwardBatch
from sglang.srt.models.llama import LlamaForCausalLM
from sglang.srt.runtime_context import get_schedule
from sglang.srt.utils import add_prefix
from torch import nn

from sglang_omni.models.csm_tts.generation_params import (
    CsmGenerationParams,
    generation_params_for_batch,
    sampling_tensors_for_batch,
)
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
    sample_codebook0_batch,
    sample_independent_batched,
)
from sglang_omni.models.csm_tts.utils import NUM_CODEBOOKS
from sglang_omni.models.csm_tts.weight_loader import copy_into_params, split_csm_weights


class CsmTTSModel(nn.Module):
    """Paged Llama backbone + fused frame embedding + codebook-0 head + depth decoder."""

    def __init__(
        self,
        config: CsmTtsHfConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config

        self.backbone = LlamaForCausalLM(
            build_backbone_llama_config(config),
            quant_config=quant_config,
            prefix=add_prefix("backbone", prefix),
        )

        depth_config = build_depth_decoder_config(config)
        self.num_codebooks = int(depth_config.num_codebooks)
        self.codebook_vocab_size = int(depth_config.vocab_size)
        backbone_hidden_size = int(self.backbone.config.hidden_size)

        self.frame_embedding = CsmFrameEmbedding(
            num_codebooks=self.num_codebooks,
            codebook_vocab=self.codebook_vocab_size,
            hidden_size=backbone_hidden_size,
        )
        self.codebook0_head = nn.Linear(
            backbone_hidden_size,
            self.codebook_vocab_size,
            bias=False,
        )

        self.sampler_pool_max_running_requests = get_schedule().max_running_requests
        self.max_batch_size = self.sampler_pool_max_running_requests
        if self.max_batch_size < 1:
            raise ValueError(
                f"max_running_requests must be >= 1, got {self.max_batch_size}"
            )
        else:
            pass
        pool_size = self.max_batch_size + 1
        self.depth_decoder = CsmDepthDecoder(
            depth_config, self.frame_embedding, max_slots=pool_size
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
        self.sampler_pool = CsmBatchedSamplerState(pool_size, device=device)
        self.padding_sampler_slot = self.max_batch_size  # last slot reserved
        self.request_id_to_sampler_slot: dict[str, int] = {}
        self.free_sampler_slots: list[int] = list(range(self.max_batch_size))
        self.output_codes: dict[str, list[torch.Tensor]] = {}

        self.cg_row_indices = torch.zeros(pool_size, dtype=torch.long, device=device)
        self.cg_temperature = torch.ones(pool_size, dtype=torch.float32, device=device)
        self.cg_top_p = torch.ones(pool_size, dtype=torch.float32, device=device)
        self.cg_top_k_buf = torch.full(
            (pool_size,), K_MAX, dtype=torch.long, device=device
        )
        self.cg_depth_temperature = torch.ones(
            pool_size, dtype=torch.float32, device=device
        )
        self.cg_depth_top_k_buf = torch.full(
            (pool_size,), K_MAX, dtype=torch.long, device=device
        )
        self.cg_codes_BN = torch.zeros(
            pool_size, NUM_CODEBOOKS, dtype=torch.long, device=device
        )
        self.cg_collect_staging = torch.zeros(
            pool_size, STAGING_WIDTH, dtype=torch.long, device=device
        )
        self.cg_was_done = torch.zeros(pool_size, dtype=torch.bool, device=device)
        self.cg_active_generation_done = torch.zeros(
            pool_size, dtype=torch.bool, device=device
        )
        self.cg_active_last_codes = torch.zeros(
            pool_size, NUM_CODEBOOKS, dtype=torch.long, device=device
        )

    def get_input_embeddings(self) -> nn.Embedding:
        return self.backbone.get_input_embeddings()

    def acquire_sampler_slot(self, request_id: str) -> int:
        sampler_slot = self.request_id_to_sampler_slot.get(request_id)
        if sampler_slot is not None:
            return sampler_slot
        else:
            pass
        if not self.free_sampler_slots:
            raise RuntimeError(
                f"CsmTTSModel sampler pool exhausted (max_batch_size="
                f"{self.max_batch_size}); raise engine max_running_requests "
                f"or limit concurrent requests."
            )
        else:
            pass
        sampler_slot = self.free_sampler_slots.pop()
        self.request_id_to_sampler_slot[request_id] = sampler_slot
        self.sampler_pool.reset_sampler_slot(sampler_slot)
        return sampler_slot

    def release_sampler_slot(self, request_id: str) -> None:
        sampler_slot = self.request_id_to_sampler_slot.pop(request_id, None)
        if sampler_slot is not None:
            self.free_sampler_slots.append(sampler_slot)
        else:
            pass
        self.output_codes.pop(request_id, None)

    def reset_request(self, request_id: str) -> None:
        """Release the sampler slot for an aborted or finished request."""
        self.release_sampler_slot(request_id)

    def get_output_codes(self, request_id: str) -> torch.Tensor:
        """Frames emitted so far for request_id, including the EOS frame."""
        codes = self.output_codes.get(request_id)
        if not codes:
            return torch.empty(
                (0, self.num_codebooks),
                dtype=torch.long,
            )
        else:
            pass
        return torch.stack(codes, dim=0).to(torch.long).cpu()

    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor | None = None,
        **kwargs,
    ) -> LogitsProcessorOutput:
        """One scheduler step: prefill samples frame 0, decode samples the next."""
        is_decode = self.is_decode_step(forward_batch)
        if is_decode:
            input_embeds = self.decode_step_embeds_cg(int(input_ids.shape[0]))
            request_ids = None
            generation_params = None
        else:
            request_ids, generation_params = self.extract_batch_metadata(
                forward_batch, kwargs.get("omni_prefill_rids")
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
            self.decode_codebooks_batch(
                hidden_states_last, request_ids, generation_params
            )

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
        forward_batch: ForwardBatch,
        *,
        is_decode: bool,
    ) -> torch.Tensor:
        """Last hidden of each sequence in a packed prefill batch."""
        extend = forward_batch.extend_seq_lens
        is_extend = forward_batch.forward_mode.is_extend()
        if not is_decode and is_extend and extend is not None:
            last_index = torch.cumsum(extend, dim=0) - 1
            return hidden_states[last_index]
        else:
            pass
        if hidden_states.ndim == 3:
            return hidden_states[:, -1, :]
        else:
            pass
        return hidden_states

    def decode_step_embeds_cg(self, batch_size: int) -> torch.Tensor:
        """Embed the previous frame for every decode row. Padding rows are discarded."""
        return self.frame_embedding(self.cg_active_last_codes[:batch_size])

    @torch.no_grad()
    def decode_codebooks_batch_cg(self, hidden_BD: torch.Tensor) -> torch.Tensor:
        """Sample one decode frame into the cg_* buffers. Frozen rows emit STOP_CODE."""
        batch_size = int(hidden_BD.shape[0])

        logits0 = self.codebook0_head(hidden_BD).to(torch.float32)
        cb0_B = sample_independent_batched(
            logits0,
            self.cg_temperature[:batch_size],
            self.cg_top_k_buf[:batch_size],
            self.cg_top_p[:batch_size],
        )
        codes_B32 = self.depth_decoder.generate_frame(
            hidden_BD,
            cb0_B,
            self.cg_depth_temperature,
            self.cg_depth_top_k_buf,
            batch_size=batch_size,
        )

        last_codes_in = self.cg_active_last_codes[:batch_size]
        out_codes, new_done, was_done = sample_codebook0_batch(
            codes_B32,
            self.cg_active_generation_done[:batch_size],
        )
        self.cg_was_done[:batch_size] = was_done
        self.cg_active_generation_done[:batch_size] = new_done
        # STOP_CODE=-1 must never reach frame_embedding.
        self.cg_active_last_codes[:batch_size] = torch.where(
            was_done.unsqueeze(-1), last_codes_in, codes_B32
        )
        self.cg_codes_BN[:batch_size] = out_codes
        return out_codes

    @torch.no_grad()
    def decode_codebooks_batch(
        self,
        hidden_BD: torch.Tensor,
        request_ids: list[str],
        generation_params: list[CsmGenerationParams],
    ) -> torch.Tensor:
        """Sample frame 0 on prefill and record it in output_codes."""
        batch_size = int(hidden_BD.shape[0])
        if len(request_ids) != batch_size or len(generation_params) != batch_size:
            raise ValueError(
                f"batch size mismatch: hidden={batch_size}, "
                f"request_ids={len(request_ids)}, "
                f"generation_params={len(generation_params)}"
            )
        else:
            pass
        device = hidden_BD.device

        logits0 = self.codebook0_head(hidden_BD).to(torch.float32)

        sampler_slots = torch.tensor(
            [self.acquire_sampler_slot(request_id) for request_id in request_ids],
            dtype=torch.long,
            device=device,
        )
        sampling = sampling_tensors_for_batch(generation_params, device)

        cb0_B = sample_independent_batched(
            logits0, sampling.temperature, sampling.top_k, sampling.top_p
        )
        codes_B32 = self.depth_decoder.generate_frame(
            hidden_BD,
            cb0_B,
            sampling.depth_temperature,
            sampling.depth_top_k,
            batch_size=batch_size,
        )

        pool = self.sampler_pool
        out_codes, new_done, was_done = sample_codebook0_batch(
            codes_B32, pool.generation_done[sampler_slots]
        )
        pool.generation_done[sampler_slots] = new_done
        pool.last_codes[sampler_slots] = torch.where(
            was_done.unsqueeze(-1), pool.last_codes[sampler_slots], codes_B32
        )
        pool.frames_emitted[sampler_slots] += (~was_done).to(torch.int32)

        was_done_cpu = was_done.cpu().tolist()
        out_codes = out_codes.detach().to(torch.long)
        for row in range(batch_size):
            if was_done_cpu[row]:
                continue
            else:
                pass
            self.output_codes.setdefault(request_ids[row], []).append(out_codes[row])

        return out_codes

    @staticmethod
    def is_decode_step(forward_batch: ForwardBatch) -> bool:
        return bool(forward_batch.forward_mode.is_decode())

    @staticmethod
    def infer_batch_size(forward_batch: ForwardBatch) -> int:
        return int(forward_batch.batch_size)

    def extract_batch_metadata(
        self,
        forward_batch: ForwardBatch,
        omni_prefill_rids: list[str] | None = None,
    ) -> tuple[list[str], list[CsmGenerationParams]]:
        raw_request_ids = (
            omni_prefill_rids
            or forward_batch.rids
            or getattr(forward_batch, "request_ids", None)
        )
        batch_size = self.infer_batch_size(forward_batch)
        if not raw_request_ids:
            raise RuntimeError(
                "CSM prefill requires ForwardBatch.rids / omni_prefill_rids "
                "or request_ids"
            )
        else:
            pass
        request_ids = [str(value) for value in raw_request_ids]

        # before_prefill stamps csm_depth_params; ForwardBatch copies may drop it.
        generation_params = generation_params_for_batch(
            forward_batch.sampling_info,
            getattr(forward_batch, "csm_depth_params", None),
            batch_size,
        )
        return request_ids, generation_params

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        backbone_weights, csm_weights = split_csm_weights(weights)
        self.backbone.load_weights(iter(backbone_weights))
        return copy_into_params(
            dict(self.named_parameters(remove_duplicate=False)), csm_weights
        )


__all__ = ["CsmTTSModel"]
