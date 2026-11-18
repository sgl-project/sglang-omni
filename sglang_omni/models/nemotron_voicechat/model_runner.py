from __future__ import annotations

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
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
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest


class VoiceChatModelRunner(ModelRunner):
    def record_function_ids(self, requests) -> None:
        sampled = self.model.function_ids[: len(requests)].tolist()
        for request, token in zip(requests, sampled):
            request.data.function_ids.append(token)

    @staticmethod
    def record_stream_tokens(result, requests) -> None:
        # note (Junnan Li): the post hooks run before sampling; the thinker decodes greedily, so the argmax is the sampled token.
        logits = result.logits_output.next_token_logits
        sampled = logits[: len(requests)].argmax(dim=-1).tolist()
        for request, token in zip(requests, sampled):
            request.data.pending_stream_tokens.append(int(token))

    def post_prefill(self, result, forward_batch, schedule_batch, requests) -> None:
        del forward_batch, schedule_batch
        self.record_function_ids(requests)
        self.record_stream_tokens(result, requests)

    def post_decode(self, result, forward_batch, schedule_batch, requests) -> None:
        del forward_batch, schedule_batch
        self.record_function_ids(requests)
        self.record_stream_tokens(result, requests)


class NemotronVoiceChatModelRunner(VoiceChatModelRunner):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        token_id_buffer = self.model.fusion_token_ids
        self.decode_token_ids_host = torch.empty(
            token_id_buffer.shape, dtype=token_id_buffer.dtype
        )

    def device_acoustic_frames(
        self, data: NemotronVoiceChatRequestData
    ) -> torch.Tensor:
        acoustic_frames = data.acoustic_frames
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

    def build_prefill_rows(self, requests: list[SchedulerRequest]) -> torch.Tensor:
        channel_token_ids: list[list[int]] = []
        acoustic_positions: list[int] = []
        acoustic_rows: list[torch.Tensor] = []

        for request in requests:
            data = request.data
            req = data.req
            prompt_token_ids = [int(token_id) for token_id in req.origin_input_ids]
            prompt_length = len(prompt_token_ids)
            prefix_length = int(req.prefix_len)
            extend_end = int(req.extend_end)
            acoustic_frames = self.device_acoustic_frames(data)
            pad_token_id = prompt_token_ids[-1]
            for position in range(prefix_length, extend_end):
                if position < prompt_length - 1:
                    channel_token_ids.append(
                        [prompt_token_ids[position], pad_token_id, pad_token_id]
                    )
                elif position == prompt_length - 1:
                    channel_token_ids.append([pad_token_id, pad_token_id, pad_token_id])
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
            channel_embeddings[acoustic_indices, 0] = torch.stack(acoustic_rows, dim=0)
        else:
            pass

        return self.model.fusion(
            channel_embeddings[:, 0], channel_embeddings[:, 1], channel_embeddings[:, 2]
        )

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        del schedule_batch
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
            data = request.data
            req = data.req
            frame_index = len(req.output_ids)
            acoustic_frames = self.device_acoustic_frames(data)
            host_token_ids[row_index, 0] = int(req.output_ids[-1])
            host_token_ids[row_index, 1] = data.function_ids[-1]
            acoustic_rows.append(acoustic_frames[frame_index])

        model = self.model
        device_token_ids = model.fusion_token_ids[:batch_size]
        device_token_ids.copy_(host_token_ids[:batch_size])
        token_embeddings = model.llm.get_input_embeddings()(device_token_ids)
        model.fusion_buffer[:batch_size] = model.fusion(
            torch.stack(acoustic_rows, dim=0),
            token_embeddings[:, 0],
            token_embeddings[:, 1],
        )
