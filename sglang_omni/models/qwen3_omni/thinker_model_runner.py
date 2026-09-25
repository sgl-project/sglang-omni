# SPDX-License-Identifier: Apache-2.0
"""Qwen3-Omni's model-local adopter for the shared prefill sidecar."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from numbers import Integral
from typing import TYPE_CHECKING, TypeGuard, TypeVar

import torch

from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.thinker_model_runner import ThinkerModelRunner

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req, ScheduleBatch
    from sglang.srt.managers.scheduler import GenerationBatchResult
    from sglang.srt.model_executor.forward_batch_info import ForwardBatch

    from sglang_omni.scheduling.types import SchedulerRequest
else:
    pass

_PREFILL_AUDIO_INPUT_KEYS = frozenset(
    {
        "audio_embeds",
        "audio_feature_lengths",
        "feature_attention_mask",
        "pad_values",
    }
)

_SIDECAR = "sidecar"
_UNSUPPORTED = "unsupported"
PadValueT = TypeVar("PadValueT")


@dataclass(frozen=True)
class PrefillDisposition:
    kind: str
    has_audio: bool = False


class Qwen3OmniThinkerModelRunner(ThinkerModelRunner):
    """Adopt the shared prefill sidecar for text/audio-to-text prefills.

    This class only qualifies Qwen request payloads. SGLang continues to own
    graph admission, bucket selection, padding, replay metadata, and eager
    fallback. Unsupported requests deliberately delegate to the inherited
    eager multimodal path.
    """

    @staticmethod
    def origin_num_tokens(value: object) -> int | None:
        if isinstance(value, torch.Tensor):
            return int(value.numel()) if value.ndim == 1 else None
        else:
            pass
        if isinstance(value, (array, list, tuple)):
            return len(value)
        else:
            pass
        return None

    @staticmethod
    def valid_positions(value: object) -> TypeGuard[torch.Tensor]:
        if not (
            isinstance(value, torch.Tensor)
            and value.ndim == 1
            and value.device.type == "cpu"
            and value.dtype != torch.bool
            and not torch.is_floating_point(value)
            and not torch.is_complex(value)
        ):
            return False
        else:
            pass
        if value.numel() and int(value[0]) < 0:
            return False
        else:
            pass
        return bool(torch.all(value[1:] > value[:-1]))

    @staticmethod
    def cpu_int_sequence(value: object) -> list[int] | None:
        if isinstance(value, torch.Tensor):
            if value.ndim != 1 or value.device.type != "cpu":
                return None
            else:
                pass
            value = value.tolist()
        else:
            pass
        if not isinstance(value, (list, tuple)):
            return None
        else:
            pass
        if any(
            not isinstance(item, Integral) or isinstance(item, bool) for item in value
        ):
            return None
        else:
            pass
        return [int(item) for item in value]

    def mm_positions(
        self, req: "Req", pad_values: dict[str, PadValueT]
    ) -> dict[str, torch.Tensor] | None:
        try:
            positions = self.req_mm_token_positions(req, pad_values)
        except (AttributeError, KeyError, TypeError, ValueError, RuntimeError):
            return None
        if not isinstance(positions, dict):
            return None
        else:
            pass

        origin_num_tokens = self.origin_num_tokens(
            getattr(req, "origin_input_ids", None)
        )
        if origin_num_tokens is None:
            return None
        else:
            pass

        validated: dict[str, torch.Tensor] = {}
        for modality in ("image", "video", "audio"):
            value = positions.get(modality)
            if not self.valid_positions(value) or (
                value.numel() and int(value[-1]) >= origin_num_tokens
            ):
                return None
            else:
                pass
            validated[modality] = value.to(dtype=torch.long)
        return validated

    @classmethod
    def batch_chunk_spans(
        cls, forward_batch: object, expected_batch_size: int
    ) -> list[tuple[int, int]] | None:
        extend_lens = cls.cpu_int_sequence(
            getattr(forward_batch, "extend_seq_lens_cpu", None)
        )
        if extend_lens is None or len(extend_lens) != expected_batch_size:
            return None
        else:
            pass

        prefix_lens = getattr(forward_batch, "extend_prefix_lens_cpu", None)
        if prefix_lens is None:
            prefix_values = [0] * expected_batch_size
        else:
            prefix_values = cls.cpu_int_sequence(prefix_lens)
            if prefix_values is None or len(prefix_values) != expected_batch_size:
                return None
            else:
                pass

        spans: list[tuple[int, int]] = []
        for prefix, length in zip(prefix_values, extend_lens):
            if prefix < 0 or length <= 0:
                return None
            else:
                pass
            spans.append((prefix, length))
        return spans

    def audio_inputs_are_supported(
        self,
        req: "Req",
        model_inputs: object,
        chunk_span: tuple[int, int],
    ) -> bool:
        if not isinstance(model_inputs, dict) or not model_inputs:
            return False
        else:
            pass
        if set(model_inputs) - _PREFILL_AUDIO_INPUT_KEYS:
            return False
        else:
            pass

        audio_embeds = model_inputs.get("audio_embeds")
        if (
            not isinstance(audio_embeds, torch.Tensor)
            or audio_embeds.ndim != 2
            or audio_embeds.shape[0] <= 0
            or audio_embeds.shape[1] <= 0
        ):
            return False
        else:
            pass
        embedding_dim = getattr(self.embed_tokens, "embedding_dim", None)
        if embedding_dim is not None and audio_embeds.shape[1] != embedding_dim:
            return False
        else:
            pass

        feature_lengths = model_inputs.get("audio_feature_lengths")
        if feature_lengths is not None and (
            not isinstance(feature_lengths, torch.Tensor)
            or feature_lengths.ndim != 1
            or feature_lengths.numel() == 0
            or feature_lengths.dtype == torch.bool
            or torch.is_floating_point(feature_lengths)
            or torch.is_complex(feature_lengths)
        ):
            return False
        else:
            pass
        feature_mask = model_inputs.get("feature_attention_mask")
        if feature_mask is not None and (
            not isinstance(feature_mask, torch.Tensor) or feature_mask.ndim != 2
        ):
            return False
        else:
            pass

        pad_values = model_inputs.get("pad_values", {})
        if not isinstance(pad_values, dict) or set(pad_values) - {"audio"}:
            return False
        else:
            pass
        if "audio" in pad_values and (
            not isinstance(pad_values["audio"], Integral)
            or isinstance(pad_values["audio"], bool)
        ):
            return False
        else:
            pass

        positions = self.mm_positions(req, pad_values)
        if positions is None:
            return False
        else:
            pass
        if positions["image"].numel() or positions["video"].numel():
            return False
        else:
            pass
        if positions["audio"].numel() != audio_embeds.shape[0]:
            return False
        else:
            pass

        prefix, length = chunk_span
        consumed = getattr(
            req, "_omni_consumed", None
        )  # noqa: leading-underscore  # upstream spelling, or the public name is already taken
        if consumed is None:
            cached_audio = positions["audio"][positions["audio"] < prefix]
            future_audio = positions["audio"][positions["audio"] >= prefix]
            # note(chenye): A fresh radix prefix can hide prior audio rows without
            # advancing the shared multimodal cursor; keep that state outside the
            # Qwen sidecar.
            if cached_audio.numel() and future_audio.numel():
                return False
            else:
                pass
            chunk_consumed = {}
        elif isinstance(consumed, dict) and set(consumed) <= {"audio"}:
            chunk_consumed = consumed
        else:
            return False
        _, audio_offset, live_audio_count = self.plan_modality_chunk(
            positions["audio"], chunk_consumed, "audio", prefix, length
        )
        if (
            not isinstance(audio_offset, Integral)
            or isinstance(audio_offset, bool)
            or audio_offset < 0
            or audio_offset + live_audio_count > audio_embeds.shape[0]
        ):
            return False
        else:
            pass

        middle_chunks = getattr(req, "inflight_middle_chunks", None)
        return (
            isinstance(middle_chunks, Integral)
            and not isinstance(middle_chunks, bool)
            and middle_chunks >= 0
        )

    def classify_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> PrefillDisposition:
        if len(requests) != getattr(forward_batch, "batch_size", None):
            return PrefillDisposition(_UNSUPPORTED)
        else:
            pass
        schedule_reqs = getattr(schedule_batch, "reqs", None)
        if schedule_reqs is None or len(schedule_reqs) != len(requests):
            return PrefillDisposition(_UNSUPPORTED)
        else:
            pass
        if (
            getattr(forward_batch, "input_embeds", None) is not None
            or getattr(forward_batch, "replace_embeds", None) is not None
            or get_omni_prefill_inputs(forward_batch) is not None
        ):
            return PrefillDisposition(_UNSUPPORTED)
        else:
            pass

        model_inputs_by_request = [
            getattr(req, "omni_model_inputs", None) for req in schedule_reqs
        ]
        has_model_inputs = any(
            model_inputs is not None
            and not (isinstance(model_inputs, dict) and not model_inputs)
            for model_inputs in model_inputs_by_request
        )
        if not has_model_inputs:
            return PrefillDisposition(_SIDECAR)
        else:
            pass

        chunk_spans = self.batch_chunk_spans(forward_batch, len(schedule_reqs))
        if chunk_spans is None:
            return PrefillDisposition(_UNSUPPORTED)
        else:
            pass
        has_audio = False
        for request_index, (req, model_inputs) in enumerate(
            zip(schedule_reqs, model_inputs_by_request)
        ):
            if model_inputs is None or (
                isinstance(model_inputs, dict) and not model_inputs
            ):
                continue
            else:
                pass
            if not self.audio_inputs_are_supported(
                req, model_inputs, chunk_spans[request_index]
            ):
                return PrefillDisposition(_UNSUPPORTED)
            else:
                pass
            has_audio = True

        return PrefillDisposition(_SIDECAR, has_audio=has_audio)

    def text_input_embeds(self, forward_batch: ForwardBatch) -> torch.Tensor:
        return self.embed_tokens(forward_batch.input_ids)

    def before_prefill(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> None:
        disposition = self.classify_prefill(forward_batch, schedule_batch, requests)
        if disposition.kind != _SIDECAR:
            return
        else:
            pass

        if disposition.has_audio:
            omni_result = self.inject_multimodal_embeds(forward_batch, schedule_batch)
            if omni_result is None:
                raise RuntimeError(
                    "Qwen audio prefill was classified as sidecar-compatible, "
                    "but multimodal embedding composition returned no result"
                )
            else:
                pass
            input_embeds, deepstack_embeds, visual_masks = omni_result
            if input_embeds is None:
                raise RuntimeError(
                    "Qwen audio prefill composition returned no input embeddings"
                )
            else:
                pass
            if deepstack_embeds is not None or visual_masks is not None:
                raise RuntimeError(
                    "Qwen text-output sidecar cannot carry visual deepstack embeddings"
                )
            else:
                pass
        else:
            input_embeds = self.text_input_embeds(forward_batch)

        attach_omni_prefill_inputs(
            forward_batch,
            OmniPrefillInputs(input_embeds=input_embeds),
        )

    def custom_prefill_forward(
        self,
        forward_batch: ForwardBatch | None,
        schedule_batch: ScheduleBatch | None,
        requests: list[SchedulerRequest],
    ) -> GenerationBatchResult | None:
        if get_omni_prefill_inputs(forward_batch) is not None:
            return None
        else:
            pass

        disposition = self.classify_prefill(forward_batch, schedule_batch, requests)
        if disposition.kind == _SIDECAR:
            raise RuntimeError("Qwen prefill sidecar was not attached before forward")
        else:
            pass

        result = super().custom_prefill_forward(
            forward_batch,
            schedule_batch,
            requests,
        )
        if result is not None:
            self.tp_worker.record_custom_prefill_eager()
        else:
            pass
        return result


__all__ = ["Qwen3OmniThinkerModelRunner"]
