# SPDX-License-Identifier: Apache-2.0
"""Ming's continuous-feedback execution on the shared MLX bookkeeping worker."""

from __future__ import annotations

import mlx.core as mx
import numpy as np
import torch
from sglang.srt.hardware_backend.mlx.kv_cache import ContiguousAttentionKVCache
from sglang.srt.hardware_backend.mlx.remote_code_gate import resolve_model_directory
from sglang.srt.hardware_backend.mlx.tp_worker import MlxTpModelWorker
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.models.ming_tts.engine_io import MingTTSLatentPatch
from sglang_omni.models.ming_tts.mlx.loading import load_ming_tts_model
from sglang_omni.models.ming_tts.mlx.runner import MingTTSMlxRunner
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest


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
        # Note (altale): Unused options are required by the shared MLX worker interface.
        if not disable_radix_cache:
            raise ValueError("Ming MLX requires disable_radix_cache=true")
        else:
            pass
        path = resolve_model_directory(model_path, revision=revision)
        model = load_ming_tts_model(path, quantization=quantization)
        self.pool_size = pool_size or model.config.llm_config.max_position_embeddings
        mx.random.seed(sampling_rng_seed)
        super().__init__(model, cache_factory=self.make_cache)

    def make_cache(self) -> list[ContiguousAttentionKVCache]:
        return [
            ContiguousAttentionKVCache(max_seq_len=self.pool_size)
            for _ in self.model.model.layers
        ]


def to_mlx(value: torch.Tensor) -> mx.array:
    return mx.array(value.detach().cpu().float().numpy())


class MingTTSMlxModelRunner(MlxSchedulerModelRunner):
    def __init__(
        self, tp_worker: MlxTpModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        self.backend: MingTTSMlxBackend = tp_worker._mlx_runner  # noqa: leading-underscore - SGLang worker interface.
        self.generated_latents: dict[str, list[torch.Tensor]] = {}

    def reset_request(self, request_id: str) -> None:
        with self.mlx_stream_context():
            self.backend.release(request_id)
            self.generated_latents.pop(request_id, None)

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        return False

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult:
        if len(requests) != 1:
            raise ValueError("Ming MLX supports one active request")
        else:
            pass
        request = requests[0]
        data = request.data
        state = data.state
        with self.mlx_stream_context():
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
            self.generated_latents[request.request_id] = []
            return self.step(request)

    def custom_decode_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult:
        if len(requests) != 1:
            raise ValueError("Ming MLX supports one active request")
        else:
            pass
        with self.mlx_stream_context():
            return self.step(requests[0])

    def step(self, request: SchedulerRequest) -> GenerationBatchResult:
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
                self.generated_latents[request.request_id].append(patch)
                if step.finish_reason is not None:
                    data.generated_latents = torch.stack(
                        self.generated_latents[request.request_id]
                    )
                else:
                    pass
            if step.finish_reason == "stop":
                data.stop_step = data.generation_steps
            else:
                pass
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
