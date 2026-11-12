# SPDX-License-Identifier: Apache-2.0
"""Zonos3 model runner: samples every codebook per step and stages the frame embedding as the next decode input.

Sampling state lives on the GPU per request pool row, so async decode launches a step before the host reads the previous step's rows.
"""

from __future__ import annotations

import secrets

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
from sglang_omni.models.zonos3.request_builders import (
    CONTINUE_TOKEN_ID,
    REPETITION_CODEBOOKS,
    REPETITION_WINDOW_ROWS,
    SEED_BITS,
    STOP_TOKEN_ID,
    Zonos3RequestData,
)
from sglang_omni.models.zonos3.sampling_kernels import (
    NO_CODE,
    sample_code_rows,
    store_frame_embeddings,
)
from sglang_omni.models.zonos3.sglang_model import Zonos3SGLangModel
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest


def device_tensor(
    values: list[int] | list[float] | list[list[float]], device: torch.device
) -> torch.Tensor:
    """Per-request host values copied to device on the current stream without waiting for the copy."""
    return torch.tensor(values, device="cpu").to(device, non_blocking=True)


class Zonos3ModelRunner(ModelRunner[Zonos3RequestData]):
    """The host appends rows and decides stops one step behind the GPU under async decode."""

    def __init__(
        self,
        model_worker: ModelWorker,
        output_processor: SGLangOutputProcessor,
        *,
        model: Zonos3SGLangModel,
    ) -> None:
        super().__init__(model_worker, output_processor)
        self.zonos3_model = model
        config = model.config
        self.codebook_count = config.codebook_count
        self.audio_pad_id = config.audio_pad_id
        self.codebook_size = config.codebook_size
        self.end_of_audio_id = config.end_of_audio_id
        self.text_pad_id = config.text_pad_id
        decode_buffer = model.decode_input_embedding.weight
        device = decode_buffer.device
        self.decode_capacity = decode_buffer.shape[0]
        # A request's req_pool_idx addresses its frame state for its whole life.
        request_pool = model_worker.model_runner.req_to_token_pool
        pool_row_count = request_pool.req_to_token.shape[0]
        self.generated_row_counts = torch.zeros(
            pool_row_count, dtype=torch.int64, device=device
        )
        self.row_increments = torch.ones(
            self.decode_capacity, dtype=torch.int64, device=device
        )
        # note (zhaochenyang20): the penalty reaches only generated rows, never the prompt's reference codes or silence prefill, so a request's window starts empty.
        self.repetition_windows = torch.full(
            (pool_row_count, REPETITION_CODEBOOKS, REPETITION_WINDOW_ROWS),
            NO_CODE,
            dtype=torch.int64,
            device=device,
        )
        self.repetition_window_counts = torch.zeros(
            (pool_row_count, REPETITION_CODEBOOKS, self.codebook_size),
            dtype=torch.int32,
            device=device,
        )
        self.feedback_embeddings = torch.zeros(
            (pool_row_count, decode_buffer.shape[1]),
            dtype=decode_buffer.dtype,
            device=device,
        )
        self.decode_row_ids = torch.arange(
            self.decode_capacity, dtype=torch.int64, device=device
        )
        # The next forward reads frame embeddings, so the token rail only carries continue.
        self.continue_token_ids = torch.full(
            (self.decode_capacity,), CONTINUE_TOKEN_ID, dtype=torch.int64, device=device
        )
        # Each request's temperature, top-p and repetition penalty, its top-k and its seed, written once at prefill.
        self.sampling_knobs = torch.ones(
            (3, pool_row_count), dtype=torch.float32, device=device
        )
        self.top_ks = torch.ones(pool_row_count, dtype=torch.int64, device=device)
        self.seeds = torch.zeros(pool_row_count, dtype=torch.int64, device=device)
        self.text_pad_embedding = model.embedders[self.codebook_count].weight[
            self.text_pad_id
        ]
        self.is_fused_embedding = (
            device.type == "cuda"
            and all(
                embedder.weight.dtype == torch.bfloat16 for embedder in model.embedders
            )
            and decode_buffer.dtype == torch.bfloat16
        )
        # note (0xtoward): The fused frame-embedding kernel reads the bfloat16 embedding tables by address and writes bfloat16 decode inputs.
        # An XPU device pointer occupies all 64 bits, which no int64 tensor holds, so the table is built only for the kernel that reads it.
        self.codebook_table_addresses = (
            torch.tensor(
                [
                    embedder.weight.data_ptr()
                    for embedder in model.embedders[: self.codebook_count]
                ],
                dtype=torch.int64,
                device=device,
            )
            if self.is_fused_embedding
            else None
        )
        # A prefill's first rows on the GPU until a lookahead launch stages them or a synchronous step records them.
        self.prefill_rows: torch.Tensor | None = None

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        assert forward_batch is not None
        model = self.zonos3_model
        device = model.multi_output.weight.device
        host_rows = []
        for scheduler_request in requests:
            request_data = scheduler_request.data
            generation_input = request_data.generation_input
            assert generation_input is not None
            prefix_length = request_data.req.prefix_len
            host_rows.append(
                generation_input.prompt_rows[
                    prefix_length : prefix_length + request_data.req.extend_len
                ]
            )
        embeddings = model.embed_rows(
            torch.cat(host_rows).to(device, non_blocking=True)
        )
        token_offset = 0
        for scheduler_request in requests:
            request_data = scheduler_request.data
            generation_input = request_data.generation_input
            prefix_length = request_data.req.prefix_len
            extend_length = request_data.req.extend_len
            # A condition row's embedding is its table's bucket entry, not the pad-row embedding.
            for table_index, bucket in enumerate(generation_input.condition_buckets):
                condition_offset = (
                    generation_input.condition_row_index + table_index - prefix_length
                )
                if 0 <= condition_offset < extend_length:
                    embeddings[token_offset + condition_offset] = (
                        model.condition_embeddings[table_index].weight[bucket].float()
                    )
                else:
                    pass
            token_offset += extend_length
        # note (zhaochenyang20): the prefill CUDA graph refuses a batch carrying input_embeds, so the embeddings travel as model inputs.
        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(
                input_embeds=embeddings.to(model.multi_output.weight.dtype)
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
        """A lookahead step reads the same GPU frame state, which the previous step's sampling already wrote."""
        assert forward_batch is not None
        batch_size = int(forward_batch.batch_size)
        torch.index_select(
            self.feedback_embeddings,
            0,
            forward_batch.req_pool_indices[:batch_size],
            out=self.zonos3_model.decode_input_embedding.weight[:batch_size],
        )
        forward_batch.input_ids = self.decode_row_ids[:batch_size]
        forward_batch.input_embeds = None

    def next_input_token_ids(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor | None:
        return self.continue_token_ids[: len(requests)]

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        assert forward_batch is not None
        hidden_states = result.logits_output.hidden_states
        last_token_indices = (
            torch.cumsum(forward_batch.extend_seq_lens.to(hidden_states.device), dim=0)
            - 1
        )
        pool_rows = forward_batch.req_pool_indices[: len(requests)]
        # Indexed assignment of a Python scalar copies it to the device and synchronizes the stream; index_fill_ does not.
        self.generated_row_counts.index_fill_(0, pool_rows, 0)
        self.repetition_windows.index_fill_(0, pool_rows, NO_CODE)
        self.repetition_window_counts.index_fill_(0, pool_rows, 0)
        generation_inputs = [
            scheduler_request.data.generation_input for scheduler_request in requests
        ]
        self.sampling_knobs[:, pool_rows] = device_tensor(
            [
                [
                    generation_input.temperature
                    for generation_input in generation_inputs
                ],
                [generation_input.top_p for generation_input in generation_inputs],
                [
                    generation_input.repetition_penalty
                    for generation_input in generation_inputs
                ],
            ],
            self.sampling_knobs.device,
        )
        self.top_ks[pool_rows] = device_tensor(
            [generation_input.top_k for generation_input in generation_inputs],
            self.top_ks.device,
        )
        self.seeds[pool_rows] = device_tensor(
            [
                (
                    secrets.randbits(SEED_BITS)
                    if generation_input.seed is None
                    else generation_input.seed
                )
                for generation_input in generation_inputs
            ],
            self.seeds.device,
        )
        self.prefill_rows = self.sample_frames(
            hidden_states[last_token_indices], pool_rows, requests
        )

    def ensure_next_token_ids(
        self,
        batch_result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """A synchronous prefill records its first rows here; a lookahead prefill's launch already staged them."""
        if self.prefill_rows is None:
            super().ensure_next_token_ids(
                batch_result, forward_batch, schedule_batch, scheduler_output
            )
        else:
            rows = self.prefill_rows
            self.prefill_rows = None
            self.record_frames(rows.cpu(), batch_result, scheduler_output.requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        assert forward_batch is not None
        request_count = len(requests)
        rows = self.sample_frames(
            result.logits_output.hidden_states[:request_count],
            forward_batch.req_pool_indices[:request_count],
            requests,
        )
        self.record_frames(rows.cpu(), result, requests)

    def post_decode_launch(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        """Sample and stage the rows into pinned host memory; the launch event marks the copy done.

        A prefill sampled its first rows in post_prefill, so its launch stages those.
        """
        assert forward_batch is not None
        request_count = len(requests)
        if self.prefill_rows is None:
            rows = self.sample_frames(
                result.logits_output.hidden_states[:request_count],
                forward_batch.req_pool_indices[:request_count],
                requests,
            )
        else:
            rows = self.prefill_rows
            self.prefill_rows = None
        staged_rows = self.next_host_staging(
            (self.decode_capacity, self.codebook_count), torch.int64
        )[:request_count]
        staged_rows.copy_(rows, non_blocking=True)
        result.next_token_ids = self.continue_token_ids[:request_count]
        return staged_rows

    def post_decode_resolve(
        self,
        launch_buf: torch.Tensor,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        """Copy out of the staging buffer, which the launch after the next one overwrites."""
        self.record_frames(launch_buf.clone(), result, requests)

    def sample_frames(
        self,
        hidden_states: torch.Tensor,
        pool_rows: torch.Tensor,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        """Delayed rows [requests, codebooks] on the GPU; advances each request's sampling state without a host read.

        Every codebook of every row is the sampled code; the positions before a codebook starts or after the end of audio are never read as audio, and forcing them to the training layout tripled short unintelligible outputs.
        """
        model = self.zonos3_model
        request_count = len(requests)
        rows = sample_code_rows(
            model.compute_logits(hidden_states),
            pool_rows,
            self.sampling_knobs,
            self.top_ks,
            self.seeds,
            self.generated_row_counts,
            self.repetition_windows,
            self.repetition_window_counts,
            max_top_k=max(
                scheduler_request.data.generation_input.top_k
                for scheduler_request in requests
            ),
            pad_id=self.audio_pad_id,
            codebook_size=self.codebook_size,
            softcap=model.config.logit_softcap,
        )
        # note (zhaochenyang20): the frame-embedding kernel advances the generated row counts, which saved one launch per step.
        if self.is_fused_embedding:
            store_frame_embeddings(
                rows,
                self.codebook_table_addresses,
                self.text_pad_embedding,
                pool_rows,
                self.generated_row_counts,
                self.feedback_embeddings,
            )
        else:
            self.generated_row_counts.index_add_(
                0, pool_rows, self.row_increments[:request_count]
            )
            frame_rows = torch.cat(
                [rows, torch.full_like(rows[:, :1], self.text_pad_id)], dim=1
            )
            self.feedback_embeddings[pool_rows] = model.embed_rows(frame_rows).to(
                self.feedback_embeddings.dtype
            )
        return rows

    def record_frames(
        self,
        host_rows: torch.Tensor,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
    ) -> None:
        """Append each row on the host; a request stops after the row that carries the last codebook's end-of-audio.

        End-of-audio may be sampled in any codebook: codebook c ending at row g ends spoken frame g - c, so the highest codebook that ended names the earliest frame, and only the first row that ends sets it.
        """
        # note (zhaochenyang20): the end-of-audio scan reads the rows as ints; as four CPU tensor ops, each a point where the engine thread gives up the GIL, it held about 3% of that thread's time (py-spy, SeedTTS EN at concurrency 16, one H100).
        row_codes = host_rows.tolist()
        row_views = host_rows.unbind(0)
        next_token_ids = []
        for row_index, scheduler_request in enumerate(requests):
            request_data = scheduler_request.data
            if request_data.req.finished() or self.req_is_retracted(request_data.req):
                # note (zhaochenyang20): A step launched before the host read the stop still carries this request; its row is past the end.
                next_token_ids.append(STOP_TOKEN_ID)
            else:
                generation_step = len(request_data.generated_rows)
                request_data.generated_rows.append(row_views[row_index])
                ended_codebooks = [
                    codebook
                    for codebook, code in enumerate(row_codes[row_index])
                    if code == self.end_of_audio_id
                ]
                if request_data.end_of_audio_row is None and ended_codebooks:
                    request_data.end_of_audio_row = max(
                        generation_step - ended_codebooks[-1], 0
                    )
                else:
                    pass
                is_finished = (
                    request_data.end_of_audio_row is not None
                    and generation_step
                    >= request_data.end_of_audio_row + self.codebook_count - 1
                )
                next_token_ids.append(
                    STOP_TOKEN_ID if is_finished else CONTINUE_TOKEN_ID
                )
        result.next_token_ids = torch.tensor(next_token_ids, dtype=torch.int64)
