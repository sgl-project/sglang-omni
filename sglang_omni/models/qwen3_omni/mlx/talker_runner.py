"""Single-request speech steps on the shared Omni scheduler."""

import mlx.core as mx
import torch
from mlx_lm.sample_utils import make_sampler
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.qwen3_omni.mlx.runner import OmniMlxSchedulerModelRunner
from sglang_omni.models.qwen3_omni.mlx.tensors import (
    mlx_to_torch,
    request_positions,
    torch_to_mlx,
)
from sglang_omni.models.qwen3_omni.talker_model_runner import QwenTalkerModelRunner
from sglang_omni.scheduling.message import OutgoingMessage
from sglang_omni.scheduling.pending_text_queue import PendingTextTensorQueue
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest


class MlxTalkerModelRunner(OmniMlxSchedulerModelRunner):
    def is_decode_batch_ready(self, schedule_batch: ScheduleBatch) -> bool:
        return not schedule_batch.forward_mode.is_decode() or all(
            QwenTalkerModelRunner.data_has_next_decode_input(request.omni_data)
            for request in schedule_batch.reqs
        )

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult:
        if len(requests) != 1:
            raise ValueError("MLX talker supports one request per batch")
        else:
            pass
        request = requests[0]
        data = request.data
        if not data.input_embeds_are_projected or len(data.req.prefix_indices):
            raise ValueError("MLX talker requires a complete projected prompt")
        else:
            pass
        rows = QwenTalkerModelRunner.projected_prefill_slice(
            sched_req=request,
            prefix_len=0,
            extend_len=data.req.extend_range.length,
            device=torch.device("cpu"),
        )
        if rows is None or rows.shape[0] != data.req.extend_range.length:
            raise ValueError("MLX talker prompt does not cover its prefill window")
        else:
            pass
        runner = (
            self.tp_worker._mlx_runner
        )  # noqa: leading-underscore - native worker contract
        if runner.position_deltas:
            raise ValueError("MLX talker already has an active request")
        else:
            pass
        with self.mlx_stream_context():
            positions, delta = request_positions(data.req, rows.shape[0])
            runner.position_deltas[request.request_id] = delta
            runner._req_caches[request.request_id] = (
                runner._acquire_cache()
            )  # noqa: leading-underscore - native cache ownership
            seed = data.req.sampling_params.sampling_seed
            if seed is not None:
                mx.random.seed(int(seed))
            else:
                pass
            return self.speech_step(request, torch_to_mlx(rows)[None], positions)

    def custom_decode_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult:
        if len(requests) != 1:
            raise ValueError("MLX talker supports one request per batch")
        else:
            pass
        request = requests[0]
        rows = QwenTalkerModelRunner.take_next_decode_input_embed(
            sched_req=request, device=torch.device("cpu"), dtype=torch.float32
        )
        if rows is None:
            raise ValueError("MLX talker decode requires feedback and text")
        else:
            pass
        QwenTalkerModelRunner.append_decode_input_history(request.data, rows)
        runner = (
            self.tp_worker._mlx_runner
        )  # noqa: leading-underscore - native worker contract
        position = (
            runner._req_caches[request.request_id][
                0
            ].offset  # noqa: leading-underscore - native cache ownership
            + runner.position_deltas[request.request_id]
        )
        with self.mlx_stream_context():
            return self.speech_step(
                request,
                torch_to_mlx(rows)[None, None],
                mx.full((3, 1, 1), position, dtype=mx.int32),
            )

    def speech_step(
        self, request: SchedulerRequest, embeddings: mx.array, positions: mx.array
    ) -> GenerationBatchResult:
        runner = (
            self.tp_worker._mlx_runner
        )  # noqa: leading-underscore - native worker contract
        model = runner.model
        data = request.data
        logits, hidden = model(
            inputs_embeds=embeddings,
            position_ids=positions,
            past_key_values=runner._req_caches[
                request.request_id
            ],  # noqa: leading-underscore - native cache ownership
            use_cache=True,
        )
        logits = logits[:, -1, :]
        suppress = data.suppress_tokens
        if suppress:
            logits[:, mx.array(suppress)] = -float("inf")
        else:
            pass
        sampling = data.req.sampling_params
        if sampling.repetition_penalty != 1 and data.req.output_ids:
            history = mx.array(list(set(data.req.output_ids)), dtype=mx.int32)
            previous = logits[:, history]
            logits[:, history] = mx.where(
                previous < 0,
                previous * sampling.repetition_penalty,
                previous / sampling.repetition_penalty,
            )
        else:
            pass
        temperature = 0.0 if sampling.top_k == 1 else sampling.temperature
        top_k = sampling.top_k if 0 < sampling.top_k < logits.shape[-1] else 0
        logprobs = logits.astype(mx.float32)
        logprobs = logprobs - mx.logsumexp(logprobs, axis=-1, keepdims=True)
        token = make_sampler(
            temp=temperature,
            top_p=sampling.top_p,
            top_k=top_k,
        )(logprobs)
        padding = mx.zeros(
            (1, 1, model.config.text_config.hidden_size), dtype=hidden.dtype
        )
        feedback, codes = model.prepare_inputs_for_generation(
            token[:, None],
            hidden[:, -1:, :],
            trailing_text_hidden=padding[:, :0, :],
            tts_pad_embed=padding,
            generation_step=0,
            temperature=temperature,
            top_p=sampling.top_p,
        )
        codes_tensor = mlx_to_torch(codes).to(torch.long).reshape(-1)
        data.pending_feedback_queue.append(mlx_to_torch(feedback).reshape(-1))
        self.outbox.put(
            OutgoingMessage(
                request_id=request.request_id,
                type="stream",
                data=codes_tensor,
                target="code2wav",
                metadata={
                    "stream": bool(
                        data.stage_payload.request.params.get("stream", False)
                    )
                },
            )
        )
        return GenerationBatchResult(next_token_ids=codes_tensor[:1])

    def on_request_finished(
        self, request_id: str, req_data: SGLangARRequestData
    ) -> None:
        super().on_request_finished(request_id, req_data)
        req_data.pending_feedback_queue.clear()
        req_data.pending_text_queue = PendingTextTensorQueue()
        req_data.decode_input_embeds = []
