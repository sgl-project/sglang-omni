# SPDX-License-Identifier: Apache-2.0
"""Tiny-specific startup compilation and request noise storage."""

from __future__ import annotations

import logging
import time

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.models.bailing_moe_v3 import BailingKDA

from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.models.ming3_tts.acoustic import FLOW_STEPS
from sglang_omni.models.ming3_tts.model_runner import (
    LaunchedAcousticStep,
    Ming3TTSModelRunner,
    StopPolicy,
)
from sglang_omni.modules.ming_audio.flow.normalization import HAS_TRITON
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest

if HAS_TRITON:
    from sglang.kernels.ops.attention.fla.kda import chunk_kda
else:
    chunk_kda = None

logger = logging.getLogger(__name__)
KDA_CHUNK_TOKEN_COUNT = 64
KDA_FUSED_GRID_BLOCK_LIMIT = 256


@torch.inference_mode()
def warmup_kda_prefill(
    *,
    head_count: int,
    key_channel_count: int,
    value_channel_count: int,
    lower_bound: float | None,
    activation_dtype: torch.dtype,
    state_dtype: torch.dtype,
    device: torch.device,
) -> None:
    """Compile both KDA prefill branches using state independent of requests."""
    if chunk_kda is None:
        raise RuntimeError("KDA prefill warmup requires Triton")
    else:
        pass
    start_seconds = time.monotonic()
    # note (Zhuofeng Li): Decode warmup misses the kernels above KDA's fused-grid cutoff.
    for sequence_count in (1, KDA_FUSED_GRID_BLOCK_LIMIT // head_count + 1):
        token_count = sequence_count * KDA_CHUNK_TOKEN_COUNT
        query = torch.zeros(
            (1, token_count, head_count, key_channel_count),
            dtype=activation_dtype,
            device=device,
        )
        value = torch.zeros(
            (1, token_count, head_count, value_channel_count),
            dtype=activation_dtype,
            device=device,
        )
        beta = torch.zeros(
            (1, token_count, head_count), dtype=activation_dtype, device=device
        )
        log_decay = torch.zeros(
            (1, 1, head_count, 1), dtype=torch.float32, device=device
        )
        time_bias = torch.zeros(
            (head_count * key_channel_count,), dtype=torch.float32, device=device
        )
        cumulative_sequence_lengths = torch.arange(
            0, token_count + 1, KDA_CHUNK_TOKEN_COUNT, dtype=torch.int32, device=device
        )
        state_indices = torch.arange(sequence_count, dtype=torch.int32, device=device)
        state = torch.zeros(
            (sequence_count, head_count, value_channel_count, key_channel_count),
            dtype=state_dtype,
            device=device,
        )
        chunk_kda(
            q=query,
            k=query,
            v=value,
            g=query,
            beta=beta,
            initial_state=state,
            initial_state_indices=state_indices,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=cumulative_sequence_lengths,
            A_log=log_decay,
            dt_bias=time_bias,
            lower_bound=lower_bound,
            beta_is_raw=True,
        )
    torch.cuda.synchronize(device)
    logger.info(
        f"Warmed Tiny KDA prefill branches in {time.monotonic() - start_seconds:.2f} seconds"
    )


class TinyTTSModelRunner(Ming3TTSModelRunner):
    def __init__(
        self,
        tp_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        stop_policy: StopPolicy,
        acoustic_graph_max_batch_size: int | None,
    ) -> None:
        super().__init__(
            tp_worker, output_processor, stop_policy, acoustic_graph_max_batch_size
        )
        if (
            self.device.type == "cuda"
            and torch.version.hip is None
            and HAS_TRITON
            and tp_worker.model_runner.linear_attn_backends.prefill.is_triton()
        ):
            for layer in self.model.modules():
                if isinstance(layer, BailingKDA):
                    state_dtype = (
                        tp_worker.model_runner.req_to_token_pool.mamba2_layer_cache(
                            layer.layer_idx
                        ).temporal.dtype
                    )
                    warmup_kda_prefill(
                        head_count=layer.local_num_heads,
                        key_channel_count=layer.head_k_dim,
                        value_channel_count=layer.head_v_dim,
                        lower_bound=layer.lower_bound,
                        activation_dtype=tp_worker.model_runner.dtype,
                        state_dtype=state_dtype,
                        device=self.device,
                    )
                    break
                else:
                    pass
        else:
            pass

    def launch_acoustic_step(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> LaunchedAcousticStep:
        for request in requests:
            request_state = request.data
            if request_state.req.inflight_middle_chunks > 0:
                pass
            elif request_state.initial_noise_buffer is None:
                latent_dim = request_state.latent_history.shape[-1]
                request_state.initial_noise_buffer = torch.empty(
                    (1, latent_dim, 1), device=self.device
                )
                request_state.solver_noise_buffer = torch.empty(
                    (FLOW_STEPS - 1, 1, 1, latent_dim), device=self.device
                )
            else:
                pass
        return super().launch_acoustic_step(result, requests)
