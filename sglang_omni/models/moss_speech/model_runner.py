# SPDX-License-Identifier: Apache-2.0
"""Runner hooks composing dual-channel embeddings and committing sampled rows."""

from __future__ import annotations

import torch
from sglang.srt.layers.logits_processor import LogitsProcessorOutput
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.model_worker import ModelWorker
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.moss_speech.fsm import (
    channel_histories,
    finalize_row,
    next_mode,
    row_selected_token,
    sample_row,
    stop_hit,
)
from sglang_omni.models.moss_speech.request_builders import MossSpeechSGLangRequestData
from sglang_omni.models.moss_speech.sglang_model import (
    MossSpeechModelOutput,
    MossSpeechSGLangModel,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerRequest


def moss_request_data(request: SchedulerRequest) -> MossSpeechSGLangRequestData:
    assert isinstance(request.data, MossSpeechSGLangRequestData)
    return request.data


class MossSpeechModelRunner(ModelRunner[MossSpeechSGLangRequestData]):
    def __init__(
        self, tp_worker: ModelWorker, output_processor: SGLangOutputProcessor
    ) -> None:
        super().__init__(tp_worker, output_processor)
        assert isinstance(self.model, MossSpeechSGLangModel)

    def rows_to_embeddings(self, rows: torch.Tensor) -> torch.Tensor:
        return self.model.rows_to_embeddings(rows)

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        assert forward_batch is not None
        assert forward_batch.extend_seq_lens_cpu is not None
        assert forward_batch.extend_prefix_lens_cpu is not None
        assert len(requests) == len(forward_batch.extend_seq_lens_cpu)
        embeddings = []
        for request, prefix_length, extend_length in zip(
            requests,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
        ):
            request_data = moss_request_data(request)
            assert request_data.prompt_rows is not None
            rows = request_data.prompt_rows
            if request_data.output_rows:
                generated_rows = torch.tensor(
                    request_data.output_rows, dtype=torch.long
                )
                rows = torch.cat([rows, generated_rows], dim=0)
                request_data.pending_feedback_queue.clear()
            else:
                pass
            current_rows = rows[prefix_length : prefix_length + extend_length]
            if current_rows.shape[0] != extend_length:
                raise RuntimeError(
                    f"MOSS-Speech prefill row mismatch for {request.request_id}"
                )
            else:
                embeddings.append(self.rows_to_embeddings(current_rows))
        if embeddings:
            packed_embeddings = torch.cat(embeddings, dim=0)
        else:
            packed_embeddings = self.model.embed_tokens.weight.new_empty(
                (0, self.model.hidden_size)
            )
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=packed_embeddings.to(
                    device=forward_batch.input_ids.device, dtype=self.model.dtype
                )
            ),
        )
        return None

    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        assert forward_batch is not None
        batch_size = len(requests)
        if batch_size == 0:
            return
        else:
            pass
        if forward_batch.input_ids.numel() != batch_size:
            raise RuntimeError("MOSS-Speech decode requires one row per request")
        else:
            pass
        # note (BraveBeter): Pack both channels into SG's native graph input buffer.
        rows = [moss_request_data(request).output_rows[-1] for request in requests]
        forward_batch.input_ids.copy_(
            torch.tensor(
                [text * self.model.audio_vocab_size + audio for text, audio in rows],
                dtype=torch.long,
                device=forward_batch.input_ids.device,
            )
        )

    @staticmethod
    def generator_for(
        request_data: MossSpeechSGLangRequestData, device: torch.device
    ) -> torch.Generator:
        if request_data.rng_generator is None:
            request_data.rng_generator = torch.Generator(device=device)
            request_data.rng_generator.manual_seed(request_data.effective_seed)
        else:
            pass
        return request_data.rng_generator

    def sample_next_token_ids(
        self,
        logits_output: LogitsProcessorOutput,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        assert isinstance(logits_output, MossSpeechModelOutput)
        assert (
            logits_output.text_logits is not None
            and logits_output.audio_logits is not None
        )
        selected_tokens = []
        logits_device = logits_output.text_logits.device
        for index, request in enumerate(requests):
            request_data = moss_request_data(request)
            assert request_data.prompt_rows is not None
            text_logits = logits_output.text_logits[index]
            audio_logits = logits_output.audio_logits[index]
            last_row = (
                request_data.output_rows[-1]
                if request_data.output_rows
                else request_data.prompt_rows[-1]
            )
            request_data.mode = next_mode(last_row, request_data.mode)
            text_history, audio_history = channel_histories(
                request_data.prompt_rows, request_data.output_rows
            )
            generator = (
                self.generator_for(request_data, logits_device)
                if request_data.params.do_sample
                else None
            )
            text_token, audio_token = sample_row(
                text_logits,
                audio_logits,
                request_data.params,
                text_history.to(logits_device),
                audio_history.to(logits_device),
                generator,
            )
            row = finalize_row(text_token, audio_token, request_data.mode)
            request_data.output_rows.append(row)
            selected_tokens.append(row_selected_token(row))
            if stop_hit(row):
                request_data.finished = True
                request_data.stop_reason = "stop_token"
            else:
                pass
        return torch.tensor(selected_tokens, dtype=torch.long, device=logits_device)

    def lookahead_eligible(self, batch: ScheduleBatch) -> bool:
        """Dual-channel history must be committed before the following step."""
        return False
