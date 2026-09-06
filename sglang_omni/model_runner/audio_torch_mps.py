# SPDX-License-Identifier: Apache-2.0
"""Shared Torch/MPS audio-language runner."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.base import ModelRunner


class AudioTorchMpsModelRunner(ModelRunner):
    """Single-request audio prefill and cached Hugging Face Torch decoding."""

    model_name = "Audio ASR"
    prefill_chunk_size: int | None = None
    requires_contiguous_audio_positions = True

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)
        self.past_key_values: dict[str, Any] = {}

    def lookahead_eligible(self, batch: Any) -> bool:
        del batch
        return False

    def one_request(self, requests: list[Any]) -> Any:
        if len(requests) != 1:
            raise RuntimeError(
                f"{self.model_name} Torch MPS currently requires max_running_requests=1"
            )
        else:
            pass
        return requests[0]

    def next_token_result(self, next_token_ids: torch.Tensor) -> Any:
        from sglang.srt.managers.scheduler import GenerationBatchResult

        return GenerationBatchResult(
            logits_output=None,
            next_token_ids=next_token_ids,
            can_run_cuda_graph=False,
        )

    def get_audio_feature(self, item: Any, forward_batch: Any) -> torch.Tensor:
        """Encode all audio windows in one model call."""
        return self.model.get_audio_feature([item])

    def assign_audio_features(
        self,
        input_embeddings: torch.Tensor,
        audio_features: torch.Tensor,
        audio_positions: list[int],
    ) -> None:
        """Assign embeddings to one contiguous placeholder span."""
        audio_start = audio_positions[0]
        input_embeddings[0, audio_start : audio_start + len(audio_positions), :] = (
            audio_features[0]
        )

    @torch.inference_mode()
    def custom_prefill_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        scheduler_request = self.one_request(requests)
        req = scheduler_request.data.req
        mm_inputs = req.multimodal_inputs
        if mm_inputs is None or len(mm_inputs.mm_items) != 1:
            raise ValueError(
                f"{self.model_name} Torch MPS requires exactly one audio item"
            )
        else:
            pass
        item = mm_inputs.mm_items[0]
        if item.feature is None or item.pad_value is None:
            raise ValueError(
                f"{self.model_name} Torch MPS requires audio features and pad value"
            )
        else:
            pass
        if mm_inputs.audio_token_id is None:
            raise ValueError(
                f"{self.model_name} Torch MPS is missing its audio token ID"
            )
        else:
            pass

        token_ids = [int(token_id) for token_id in schedule_batch.input_ids.tolist()]
        pad_value = int(item.pad_value)
        audio_token_id = int(mm_inputs.audio_token_id)
        normalized_ids = [
            audio_token_id if token_id == pad_value else token_id
            for token_id in token_ids
        ]
        audio_positions = [
            index
            for index, token_id in enumerate(normalized_ids)
            if token_id == audio_token_id
        ]
        if not audio_positions:
            raise ValueError(
                f"{self.model_name} Torch MPS prefill has no audio placeholders"
            )
        else:
            pass
        if self.requires_contiguous_audio_positions:
            audio_start = audio_positions[0]
            if audio_positions != list(
                range(audio_start, audio_start + len(audio_positions))
            ):
                raise ValueError(
                    f"{self.model_name} Torch MPS audio placeholders must be contiguous"
                )
            else:
                pass
        else:
            pass

        language_model = self.model.language_model
        input_ids = torch.tensor(
            [normalized_ids],
            dtype=torch.long,
            device=self.device,
        )
        # Run the audio tower before the token embedding. Its masked gather is
        # data dependent, so it forces a device read, and on MPS an embedding
        # kernel left pending across that read is dispatched short: leading rows
        # of the prompt come back unwritten, or the gather itself selects zero
        # frames and the encoder fails on an empty reshape.
        audio_features = self.get_audio_feature(item, forward_batch)
        input_embeddings = language_model.model.embed_tokens(input_ids)
        audio_features = audio_features.to(
            device=self.device,
            dtype=input_embeddings.dtype,
        )
        # Audio families return either [tokens, hidden] or [1, tokens, hidden].
        if audio_features.ndim == 2:
            audio_features = audio_features.unsqueeze(0)
        else:
            pass
        if audio_features.shape != (
            1,
            len(audio_positions),
            input_embeddings.shape[-1],
        ):
            raise ValueError(
                f"{self.model_name} Torch MPS audio embedding shape does not match its "
                f"placeholder span: {tuple(audio_features.shape)}"
            )
        else:
            pass
        self.assign_audio_features(input_embeddings, audio_features, audio_positions)

        chunk_size = self.prefill_chunk_size or input_embeddings.shape[1]
        if self.prefill_chunk_size is not None:
            from transformers import StaticCache

            past_key_values = StaticCache(
                language_model.config,
                max_cache_len=(
                    input_embeddings.shape[1] + int(req.sampling_params.max_new_tokens)
                ),
            )
        else:
            past_key_values = None
        output = None
        for start in range(0, input_embeddings.shape[1], chunk_size):
            output = language_model(
                inputs_embeds=input_embeddings[:, start : start + chunk_size],
                past_key_values=past_key_values,
                use_cache=True,
                logits_to_keep=1,
            )
            past_key_values = output.past_key_values
        assert output is not None
        self.past_key_values[scheduler_request.request_id] = output.past_key_values
        return self.next_token_result(output.logits[:, -1, :].argmax(dim=-1))

    @torch.inference_mode()
    def custom_decode_forward(
        self,
        forward_batch: Any,
        schedule_batch: Any,
        requests: list[Any],
    ) -> Any:
        del forward_batch
        scheduler_request = self.one_request(requests)
        request_id = scheduler_request.request_id
        try:
            past_key_values = self.past_key_values[request_id]
        except KeyError as exc:
            raise RuntimeError(
                f"{self.model_name} Torch MPS decode has no cache for {request_id}"
            ) from exc

        input_ids = schedule_batch.input_ids.reshape(1, 1).to(
            device=self.device,
            dtype=torch.long,
        )
        output = self.model.language_model(
            input_ids=input_ids,
            past_key_values=past_key_values,
            use_cache=True,
            logits_to_keep=1,
        )
        self.past_key_values[request_id] = output.past_key_values
        return self.next_token_result(output.logits[:, -1, :].argmax(dim=-1))

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        del req_data
        self.past_key_values.pop(request_id, None)

    def abort_request(self, request_id: str) -> None:
        self.past_key_values.pop(request_id, None)
