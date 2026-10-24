"""Batched acoustic fusion for Nemotron VoiceChat Thinker requests."""

from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.scheduler import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.nemotron_voicechat.request_builders import (
    NemotronVoiceChatRequestData,
)
from sglang_omni.scheduling.sglang_backend.output_processor import (
    SGLangOutputProcessor,
)
from sglang_omni.scheduling.types import SchedulerRequest


class NemotronVoiceChatModelRunner(ModelRunner[NemotronVoiceChatRequestData]):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        token_id_buffer = self.model.fusion_token_ids
        self.decode_token_ids_host: torch.Tensor = torch.empty(
            token_id_buffer.shape,
            dtype=token_id_buffer.dtype,
            device="cpu",
            pin_memory=token_id_buffer.device.type == "cuda",
        )

    @staticmethod
    def request_data(request: SchedulerRequest) -> NemotronVoiceChatRequestData:
        data = request.data
        if not isinstance(data, NemotronVoiceChatRequestData):
            raise TypeError(
                "Nemotron VoiceChat runner received incompatible request data"
            )
        else:
            pass
        return data

    def device_acoustic_frames(
        self, data: NemotronVoiceChatRequestData
    ) -> torch.Tensor:
        acoustic_frames = data.acoustic_frames
        if acoustic_frames is None:
            raise RuntimeError("Nemotron VoiceChat request has no acoustic frames")
        else:
            pass
        buffer = self.model.fusion_buffer
        if (
            acoustic_frames.device != buffer.device
            or acoustic_frames.dtype != buffer.dtype
        ):
            acoustic_frames = acoustic_frames.to(
                device=buffer.device, dtype=buffer.dtype, non_blocking=True
            )
            data.acoustic_frames = acoustic_frames
        else:
            pass
        return acoustic_frames

    def build_prefill_rows(
        self, requests: list[SchedulerRequest]
    ) -> torch.Tensor:
        channel_token_ids: list[list[int]] = []
        acoustic_positions: list[int] = []
        acoustic_rows: list[torch.Tensor] = []

        for request in requests:
            data = self.request_data(request)
            req = data.req
            if req is None:
                raise RuntimeError(
                    f"Nemotron VoiceChat request {request.request_id} has no prompt"
                )
            else:
                pass
            prompt_token_ids = [int(token_id) for token_id in req.origin_input_ids]
            prompt_length = len(prompt_token_ids)
            prefix_length = len(req.prefix_indices)
            extend_length = int(req.extend_range.length)
            extend_end = prefix_length + extend_length
            generated_length = len(req.output_ids)
            logical_length = prompt_length + generated_length
            if extend_end > logical_length:
                raise RuntimeError(
                    f"Nemotron VoiceChat request {request.request_id} needs "
                    f"{extend_end} prefill rows but owns {logical_length}"
                )
            else:
                pass
            if len(data.function_ids) < generated_length:
                raise RuntimeError(
                    f"Nemotron VoiceChat request {request.request_id} has "
                    f"{len(data.function_ids)} function ids for "
                    f"{generated_length} generated tokens"
                )
            else:
                pass

            acoustic_frames = self.device_acoustic_frames(data)
            pad_token_id = prompt_token_ids[-1]
            for position in range(prefix_length, extend_end):
                if position < prompt_length - 1:
                    channel_token_ids.append(
                        [prompt_token_ids[position], pad_token_id, pad_token_id]
                    )
                elif position == prompt_length - 1:
                    channel_token_ids.append(
                        [pad_token_id, pad_token_id, pad_token_id]
                    )
                    acoustic_positions.append(len(channel_token_ids) - 1)
                    acoustic_rows.append(acoustic_frames[0])
                else:
                    generated_index = position - prompt_length
                    channel_token_ids.append(
                        [
                            pad_token_id,
                            int(req.output_ids[generated_index]),
                            data.function_ids[generated_index],
                        ]
                    )
                    acoustic_positions.append(len(channel_token_ids) - 1)
                    acoustic_rows.append(acoustic_frames[generated_index + 1])

        embedding = self.model.llm.get_input_embeddings()
        packed_token_ids = torch.tensor(
            channel_token_ids,
            dtype=torch.long,
            device=embedding.weight.device,
        )
        channel_embeddings = embedding(packed_token_ids)
        if acoustic_rows:
            acoustic_indices = torch.tensor(
                acoustic_positions,
                dtype=torch.long,
                device=channel_embeddings.device,
            )
            channel_embeddings[acoustic_indices, 0] = torch.stack(
                acoustic_rows, dim=0
            )
        else:
            pass

        fusion = self.model.fusion
        channel_embeddings[:, 1].mul_(fusion.text_weight)
        channel_embeddings[:, 0].mul_(fusion.user_weight)
        fused_rows = channel_embeddings[:, 1].add_(channel_embeddings[:, 0])
        channel_embeddings[:, 2].mul_(fusion.function_weight)
        fused_rows.add_(channel_embeddings[:, 2])
        return fused_rows

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
        if forward_batch is None:
            raise RuntimeError("Nemotron VoiceChat prefill has no forward batch")
        else:
            pass
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=self.build_prefill_rows(requests),
                input_embeds_are_projected=True,
            ),
        )

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        del forward_batch, schedule_batch, is_lookahead
        batch_size = len(requests)
        host_token_ids = self.decode_token_ids_host
        acoustic_rows: list[torch.Tensor] = []
        for row_index, request in enumerate(requests):
            data = self.request_data(request)
            req = data.req
            if req is None or not req.output_ids or not data.function_ids:
                raise RuntimeError(
                    f"Nemotron VoiceChat request {request.request_id} has no "
                    "decode history"
                )
            else:
                pass
            frame_index = len(req.output_ids)
            acoustic_frames = self.device_acoustic_frames(data)
            if frame_index >= acoustic_frames.shape[0]:
                raise RuntimeError(
                    f"Nemotron VoiceChat request {request.request_id} needs acoustic "
                    f"frame {frame_index}, but owns {acoustic_frames.shape[0]} rows"
                )
            else:
                pass
            host_token_ids[row_index, 0] = int(req.output_ids[-1])
            host_token_ids[row_index, 1] = data.function_ids[-1]
            acoustic_rows.append(acoustic_frames[frame_index])

        model = self.model
        device_token_ids = model.fusion_token_ids[:batch_size]
        device_token_ids.copy_(host_token_ids[:batch_size], non_blocking=True)
        token_embeddings = model.llm.get_input_embeddings()(device_token_ids)
        fusion_buffer = model.fusion_buffer[:batch_size]
        torch.stack(acoustic_rows, dim=0, out=fusion_buffer)
        fusion_buffer.mul_(model.fusion.user_weight)
        token_embeddings[:, 0].mul_(model.fusion.text_weight)
        fusion_buffer.add_(token_embeddings[:, 0])
        token_embeddings[:, 1].mul_(model.fusion.function_weight)
        fusion_buffer.add_(token_embeddings[:, 1])
        model.has_staged_decode = True

    def record_function_ids(self, requests: list[SchedulerRequest]) -> None:
        sampled_function_ids = self.model.function_ids[: len(requests)].tolist()
        for request, function_id in zip(requests, sampled_function_ids, strict=True):
            self.request_data(request).function_ids.append(int(function_id))

    def record_stream_tokens(
        self, result: GenerationBatchResult, requests: list[SchedulerRequest]
    ) -> None:
        # Sampling has not happened yet in the post hooks; the Thinker is greedy,
        # so this argmax is the token that the sampler will record.
        logits = result.logits_output.next_token_logits
        sampled_token_ids = logits[: len(requests)].argmax(dim=-1).tolist()
        for request, token_id in zip(requests, sampled_token_ids, strict=True):
            self.request_data(request).pending_stream_tokens.append(int(token_id))

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        self.record_function_ids(requests)
        self.record_stream_tokens(result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del forward_batch, schedule_batch
        self.record_function_ids(requests)
        self.record_stream_tokens(result, requests)
