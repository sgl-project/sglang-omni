# SPDX-License-Identifier: Apache-2.0
"""V1 multimodal prefill injection."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.scheduler import GenerationBatchResult

from sglang_omni.model_runner.base import ModelRunner


class MingThinkerModelRunner(ModelRunner):
    """Inject Ming image/audio embeddings into thinker prefill requests."""

    def __init__(self, tp_worker: Any, output_processor: Any):
        super().__init__(tp_worker, output_processor)

        self._outer_model = self.model
        self._text_model = getattr(self._outer_model, "model", self._outer_model)
        self._embed_tokens = self._get_embed_tokens(self._text_model)

        hf_config = tp_worker.model_runner.model_config.hf_config
        llm_config = getattr(hf_config, "llm_config", hf_config)
        self._image_token_id = self._token_id(
            hf_config,
            "image_token_id",
            fallback=getattr(llm_config, "image_patch_token", None),
        )
        self._video_token_id = self._token_id(
            hf_config,
            "video_token_id",
            fallback=getattr(llm_config, "video_patch_token", None),
        )
        self._audio_token_id = self._token_id(hf_config, "audio_token_id")

    @staticmethod
    def _get_embed_tokens(text_model: Any) -> Any:
        embed_tokens = getattr(text_model, "embed_tokens", None)
        if embed_tokens is not None:
            return embed_tokens
        get_input_embeddings = getattr(text_model, "get_input_embeddings", None)
        if callable(get_input_embeddings):
            embed_tokens = get_input_embeddings()
        if embed_tokens is None:
            raise AttributeError("Ming thinker model does not expose embed_tokens")
        return embed_tokens

    @staticmethod
    def _token_id(config: Any, name: str, *, fallback: Any = None) -> int | None:
        value = getattr(config, name, None)
        if value is None:
            value = fallback
        return int(value) if value is not None else None

    def prepare_prefill(self, forward_batch: Any, schedule_batch: Any, requests: list):
        """Custom prefill for multimodal inputs."""
        del requests
        if not schedule_batch.forward_mode.is_extend():
            return None

        input_embeds = self._inject_multimodal_embeds(forward_batch, schedule_batch)
        if input_embeds is None:
            return None
        return self._forward_with_omni_embeds(forward_batch, input_embeds)

    def _inject_multimodal_embeds(
        self, forward_batch: Any, schedule_batch: Any
    ) -> torch.Tensor | None:
        if not any(req.omni_model_inputs is not None for req in schedule_batch.reqs):
            return None

        device = forward_batch.input_ids.device
        embed_input_ids = forward_batch.input_ids.clamp(
            0, self._embed_tokens.num_embeddings - 1
        )
        input_embeds = self._embed_tokens(embed_input_ids)

        extend_lens = forward_batch.extend_seq_lens_cpu
        offsets = []
        pos = 0
        for length in extend_lens:
            offsets.append(pos)
            pos += length

        for i, req in enumerate(schedule_batch.reqs):
            omni_inputs = req.omni_model_inputs
            if omni_inputs is None:
                continue

            start = offsets[i]
            end = start + extend_lens[i]
            req_input_ids = forward_batch.input_ids[start:end]
            consumed = getattr(req, "_omni_consumed", None) or {}
            pad_values = omni_inputs.get("pad_values", {})
            is_final_chunk = getattr(req, "is_chunked", 0) == 0
            req_id = self._request_id(req)

            for modality, embed_key, token_id in [
                ("image", "image_embeds", self._image_token_id),
                ("video", "video_embeds", self._video_token_id),
                ("audio", "audio_embeds", self._audio_token_id),
            ]:
                embeds = omni_inputs.get(embed_key)
                if embeds is None:
                    continue

                total_rows = self._num_embed_rows(embeds)
                offset = int(consumed.get(modality, 0))
                match_id = self._resolve_match_id(pad_values, modality, token_id)
                if match_id is None:
                    if is_final_chunk and offset < total_rows:
                        raise self._missing_match_id_error(
                            modality, req_id, remaining=total_rows - offset
                        )
                    continue

                mask = req_input_ids == match_id
                if not mask.any():
                    # Cache hit may have absorbed all placeholder positions for
                    # this modality; aligning with qwen3 thinker_model_runner
                    # which silently continues here.
                    continue

                n_tokens = int(mask.sum().item())
                available = total_rows - offset
                if available < n_tokens:
                    raise self._short_embeds_error(
                        modality,
                        req_id,
                        needed=n_tokens,
                        available=available,
                    )
                chunk_embeds = embeds[offset : offset + n_tokens].to(
                    device=device, dtype=input_embeds.dtype
                )
                input_embeds[torch.where(mask)[0] + start] = chunk_embeds
                consumed[modality] = offset + n_tokens

            req._omni_consumed = consumed

            if is_final_chunk:
                # Skip strict consumed==total validation: radix prefix cache may
                # have absorbed placeholder positions. Mirrors qwen3 path.
                req.omni_model_inputs = None
                req._omni_consumed = None

        return input_embeds

    @staticmethod
    def _resolve_match_id(
        pad_values: dict[str, Any], modality: str, token_id: int | None
    ) -> int | None:
        if modality in pad_values:
            return int(pad_values[modality])
        return token_id

    @staticmethod
    def _num_embed_rows(embeds: Any) -> int:
        shape = getattr(embeds, "shape", None)
        if shape is not None and len(shape) > 0:
            return int(shape[0])
        return len(embeds)

    @staticmethod
    def _request_id(req: Any) -> str:
        return str(getattr(req, "rid", getattr(req, "request_id", "<unknown>")))

    def _validate_final_consumption(
        self, req: Any, omni_inputs: dict[str, Any], consumed: dict[str, int]
    ) -> None:
        req_id = self._request_id(req)
        for modality, embed_key in [
            ("image", "image_embeds"),
            ("video", "video_embeds"),
            ("audio", "audio_embeds"),
        ]:
            embeds = omni_inputs.get(embed_key)
            if embeds is None:
                continue
            total_rows = self._num_embed_rows(embeds)
            consumed_rows = int(consumed.get(modality, 0))
            if consumed_rows != total_rows:
                raise ValueError(
                    "Ming thinker multimodal embed count mismatch for "
                    f"{modality} request_id={req_id}: "
                    f"consumed={consumed_rows}, total={total_rows}"
                )

    @staticmethod
    def _missing_match_id_error(
        modality: str, req_id: str, *, remaining: int
    ) -> ValueError:
        return ValueError(
            "Ming thinker multimodal embeds could not be injected for "
            f"{modality} request_id={req_id}: no usable match id "
            f"for remaining={remaining} embed rows"
        )

    @staticmethod
    def _missing_placeholder_error(
        modality: str, req_id: str, *, remaining: int
    ) -> ValueError:
        return ValueError(
            "Ming thinker multimodal embeds could not be injected for "
            f"{modality} request_id={req_id}: no placeholders found "
            f"for remaining={remaining} embed rows"
        )

    @staticmethod
    def _short_embeds_error(
        modality: str, req_id: str, *, needed: int, available: int
    ) -> ValueError:
        return ValueError(
            "Ming thinker multimodal embeds are shorter than placeholders for "
            f"{modality} request_id={req_id}: "
            f"needed={needed}, available={available}"
        )

    def _forward_with_omni_embeds(
        self, forward_batch: Any, input_embeds: torch.Tensor
    ) -> Any:
        model_runner = self.tp_worker.model_runner
        outer = self._outer_model

        model_runner.attn_backend.init_forward_metadata(forward_batch)

        positions = forward_batch.positions
        if forward_batch.mrope_positions is not None:
            positions = forward_batch.mrope_positions

        hidden_states = outer.model(
            input_ids=None,
            positions=positions,
            forward_batch=forward_batch,
            input_embeds=input_embeds,
        )

        logits_output = outer.logits_processor(
            forward_batch.input_ids,
            hidden_states,
            outer.lm_head,
            forward_batch,
        )

        return GenerationBatchResult(
            logits_output=logits_output,
            can_run_cuda_graph=False,
        )
