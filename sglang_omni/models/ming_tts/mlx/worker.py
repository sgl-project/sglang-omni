# SPDX-License-Identifier: Apache-2.0
"""Ming's continuous-feedback execution on the shared MLX bookkeeping worker."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
import torch

from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.models.ming_tts.engine_io import MingTTSLatentPatch

from .loading import load_ming_tts_model
from .runner import MingTTSMlxRunner


class MingTTSMlxBackend(MingTTSMlxRunner):
    def __init__(
        self,
        *,
        model_path: str,
        trust_remote_code: bool,
        disable_radix_cache: bool,
        mem_fraction_static: float,
        quantization: str | None,
        revision: str | None,
        enable_sampling: bool,
        sampling_rng_seed: int,
        deterministic_seeding: bool,
        pool_size: int | None = None,
    ) -> None:
        from sglang.srt.hardware_backend.mlx.remote_code_gate import (
            resolve_model_directory,
        )

        del trust_remote_code, mem_fraction_static, enable_sampling, deterministic_seeding
        if not disable_radix_cache:
            raise ValueError("Ming MLX requires disable_radix_cache=true")
        path = resolve_model_directory(model_path, revision=revision)
        model = load_ming_tts_model(path, quantization=quantization)
        self.pool_size = pool_size or model.config.llm_config.max_position_embeddings
        mx.random.seed(sampling_rng_seed)
        super().__init__(model, cache_factory=self._make_cache)

    def _make_cache(self) -> list[Any]:
        from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache

        return [
            ContiguousAttentionKVCache(max_seq_len=self.pool_size)
            for _ in self.model.model.layers
        ]


def to_mlx(value: torch.Tensor) -> mx.array:
    return mx.array(value.detach().cpu().float().numpy())


class MingTTSMlxModelRunner(MlxSchedulerModelRunner):
    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self.backend: MingTTSMlxBackend = tp_worker._mlx_runner
        self._generated: dict[str, list[torch.Tensor]] = {}

    def reset_request(self, request_id: str) -> None:
        with self._mlx_stream_context():
            self.backend.release(request_id)
            self._generated.pop(request_id, None)

    def lookahead_eligible(self, batch: Any) -> bool:
        return False

    def custom_prefill_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> Any:
        if len(requests) != 1:
            raise ValueError("Ming MLX supports one active request")
        request = requests[0]
        data = request.data
        state = data.state
        with self._mlx_stream_context():
            self.backend.start(
                request.request_id,
                mx.array(data.input_ids.tolist(), dtype=mx.int32),
                max_steps=data.max_new_tokens,
                reference_latents=(
                    to_mlx(state.prompt_latent).reshape(
                        -1, self.backend.model.config.latent_dim
                    )
                    if state.prompt_latent is not None
                    else None
                ),
                reference_start=state.prompt_latent_start_position,
                speaker_embedding=(
                    to_mlx(state.spk_emb) if state.spk_emb is not None else None
                ),
                speaker_positions=state.spk_injection_positions,
            )
            self._generated[request.request_id] = []
            return self._step(request)

    def custom_decode_forward(
        self, forward_batch: Any, schedule_batch: Any, requests: list[Any]
    ) -> Any:
        if len(requests) != 1:
            raise ValueError("Ming MLX supports one active request")
        with self._mlx_stream_context():
            return self._step(requests[0])

    def _step(self, request: Any) -> Any:
        from sglang.srt.managers.scheduler import GenerationBatchResult

        data = request.data
        try:
            step = self.backend.step(
                request.request_id,
                cfg=data.state.cfg,
                sigma=data.state.sigma,
                temperature=data.state.temperature,
            )
            patch = torch.from_numpy(np.array(step.latent.astype(mx.float32)))
            if data.is_streaming:
                data.pending_stream_patch = MingTTSLatentPatch(
                    patch, is_last=step.finish_reason is not None
                )
            else:
                self._generated[request.request_id].append(patch)
                if step.finish_reason is not None:
                    data.generated_latents = torch.stack(
                        self._generated[request.request_id]
                    )
            if step.finish_reason == "stop":
                data.stop_step = data.generation_steps
            token = (
                data.audio_eos_token_id
                if step.finish_reason == "stop"
                else data.audio_patch_token_id
            )
            return GenerationBatchResult(
                logits_output=None,
                next_token_ids=torch.tensor([token], dtype=torch.long),
                can_run_cuda_graph=False,
            )
        except Exception:
            self.reset_request(request.request_id)
            raise
