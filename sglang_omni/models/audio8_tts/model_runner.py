# SPDX-License-Identifier: Apache-2.0
"""Run Audio8's semantic and residual codebook sampling in the same AR step."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import torch

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
)
from sglang_omni.models.audio8_tts.sglang_model import Audio8SGLangModel
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.types import SchedulerRequest
from sglang_omni.vendor.sglang.core import (
    ForwardBatch,
    GenerationBatchResult,
    ScheduleBatch,
)

MIN_SAMPLING_TEMPERATURE = 1e-5


@dataclass(kw_only=True)
class Audio8RequestData(SGLangARRequestData):
    """State owned by a request across prefill, decode, and retraction."""

    prompt_codes: torch.Tensor
    random_generator: torch.Generator | None = None
    semantic_history_token_ids: deque[int]
    generated_codec_frames: list[torch.Tensor] = field(default_factory=list)


def sample_audio8_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    random_generator: torch.Generator | None,
) -> torch.Tensor:
    """Filter candidates in logit order before applying the temperature."""
    if temperature == 0:
        return logits.argmax()
    else:
        sorted_logits, sorted_token_ids = logits.sort(descending=True)
        removed = sorted_logits.softmax(-1).cumsum(-1) > top_p
        if top_k > 0:
            removed[top_k:] = True
        else:
            pass
        removed[0] = False
        filtered_logits = sorted_logits.masked_fill(removed, -torch.inf)
        probabilities = (
            filtered_logits / max(temperature, MIN_SAMPLING_TEMPERATURE)
        ).softmax(-1)
        uniform = torch.rand(
            probabilities.shape,
            device=probabilities.device,
            dtype=probabilities.dtype,
            generator=random_generator,
        )
        scores = probabilities / (
            -uniform.clamp_min(torch.finfo(uniform.dtype).tiny).log()
        )
        return sorted_token_ids.gather(0, scores.argmax().reshape(1)).squeeze(0)


class Audio8ModelRunner(ModelRunner[Audio8RequestData]):
    """Pair each sampled semantic token with its current slow-AR hidden state."""

    model: Audio8SGLangModel

    @torch.no_grad()
    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        assert forward_batch is not None
        assert forward_batch.extend_prefix_lens_cpu is not None
        assert forward_batch.extend_seq_lens_cpu is not None
        prefill_codebook_ids: list[torch.Tensor] = []
        for request, prefix_token_count, extend_token_count in zip(
            requests,
            forward_batch.extend_prefix_lens_cpu,
            forward_batch.extend_seq_lens_cpu,
            strict=True,
        ):
            request_data = request.data
            assert isinstance(request_data, Audio8RequestData)
            codebook_ids = request_data.prompt_codes[:, 1:]
            end_token_index = prefix_token_count + extend_token_count
            if end_token_index > len(codebook_ids):
                # note (Beining): KV retraction replays previously emitted codec frames.
                codebook_ids = torch.cat(
                    (codebook_ids, torch.stack(request_data.generated_codec_frames))
                )
            else:
                pass
            assert end_token_index <= len(codebook_ids)
            prefill_codebook_ids.append(
                codebook_ids[prefix_token_count:end_token_index]
            )
        input_embeddings = self.model.embed_tokens(
            forward_batch.input_ids, torch.cat(prefill_codebook_ids)
        )
        attach_omni_prefill_inputs(
            forward_batch, OmniPrefillInputs(input_embeds=input_embeddings)
        )

    @torch.no_grad()
    def before_decode(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
        *,
        is_lookahead: bool = False,
    ) -> None:
        assert forward_batch is not None and not is_lookahead
        previous_codec_frames: list[torch.Tensor] = []
        for request in requests:
            request_data = request.data
            assert isinstance(request_data, Audio8RequestData)
            previous_codec_frames.append(request_data.generated_codec_frames[-1])
        self.model.decode_input_embeds = self.model.embed_tokens(
            forward_batch.input_ids, torch.stack(previous_codec_frames)
        )

    def post_prefill(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        assert schedule_batch is not None
        if schedule_batch.is_prefill_only:
            return
        else:
            self.collect_codec_frames(result, requests)

    def post_decode(
        self,
        result: GenerationBatchResult,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        self.collect_codec_frames(result, requests)

    @torch.no_grad()
    def collect_codec_frames(
        self,
        result: GenerationBatchResult,
        requests: list[SchedulerRequest],
    ) -> None:
        logits_output = result.logits_output
        assert logits_output is not None
        logits = logits_output.next_token_logits
        hidden_states = logits_output.hidden_states
        assert logits is not None and hidden_states is not None
        assert logits.shape[0] == hidden_states.shape[0] == len(requests)
        config = self.model.config
        semantic_start_id = config.semantic_begin_id
        semantic_stop_id = config.semantic_end_id + 1
        eos_token_id = config.eos_token_id
        candidate_logits = torch.cat(
            (
                logits[:, semantic_start_id:semantic_stop_id],
                logits[:, eos_token_id : eos_token_id + 1],
            ),
            dim=1,
        )
        candidate_token_ids = torch.cat(
            (
                torch.arange(semantic_start_id, semantic_stop_id, device=logits.device),
                torch.full((1,), eos_token_id, device=logits.device, dtype=torch.long),
            )
        )
        emitted_token_ids: list[torch.Tensor] = []
        active_request_data: dict[int, Audio8RequestData] = {}
        for row_index, request in enumerate(requests):
            request_data = request.data
            assert isinstance(request_data, Audio8RequestData)
            assert request_data.req is not None
            if request_data.req.inflight_middle_chunks > 0:
                emitted_token_ids.append(candidate_token_ids[-1])
            else:
                emitted_token_ids.append(
                    self.sample_semantic(
                        candidate_logits[row_index], request_data, candidate_token_ids
                    )
                )
                active_request_data[row_index] = request_data
        result.next_token_ids = torch.stack(emitted_token_ids)
        semantic_token_ids = result.next_token_ids.tolist()
        codec_row_indices = [
            row_index
            for row_index in active_request_data
            if semantic_token_ids[row_index] != eos_token_id
        ]
        if codec_row_indices:
            codec_request_data = [
                active_request_data[row_index] for row_index in codec_row_indices
            ]

            def sample_codebook(codebook_logits: torch.Tensor) -> torch.Tensor:
                return torch.stack(
                    [
                        sample_audio8_token(
                            row_logits,
                            temperature=request_data.temperature,
                            top_k=request_data.top_k,
                            top_p=request_data.top_p,
                            random_generator=request_data.random_generator,
                        )
                        for row_logits, request_data in zip(
                            codebook_logits, codec_request_data, strict=True
                        )
                    ]
                )

            codec_frames = self.model.generate_codebooks(
                hidden_states[codec_row_indices],
                result.next_token_ids[codec_row_indices],
                sample_codebook,
            )
            for request_data, row_index, codec_frame in zip(
                codec_request_data, codec_row_indices, codec_frames, strict=True
            ):
                if request_data.generated_codec_frames:
                    request_data.semantic_history_token_ids.append(
                        semantic_token_ids[row_index]
                    )
                else:
                    pass
                request_data.generated_codec_frames.append(codec_frame)
        else:
            pass

    def sample_semantic(
        self,
        candidate_logits: torch.Tensor,
        request_data: Audio8RequestData,
        candidate_token_ids: torch.Tensor,
    ) -> torch.Tensor:
        sampled_index = sample_audio8_token(
            candidate_logits,
            temperature=request_data.temperature,
            top_k=request_data.top_k,
            top_p=request_data.top_p,
            random_generator=request_data.random_generator,
        )
        sampled_token_id = candidate_token_ids.gather(
            0, sampled_index.reshape(1)
        ).squeeze(0)
        if request_data.temperature == 0:
            return sampled_token_id
        else:
            pass
        resampled_index = sample_audio8_token(
            candidate_logits,
            temperature=self.model.config.ras_temperature,
            top_k=request_data.top_k,
            top_p=self.model.config.ras_top_p,
            random_generator=request_data.random_generator,
        )
        if request_data.semantic_history_token_ids:
            history = torch.tensor(
                list(request_data.semantic_history_token_ids),
                device=candidate_logits.device,
            )
            repeated = (history == sampled_token_id).any()
            return torch.where(
                repeated,
                candidate_token_ids.gather(0, resampled_index.reshape(1)).squeeze(0),
                sampled_token_id,
            )
        else:
            return sampled_token_id
