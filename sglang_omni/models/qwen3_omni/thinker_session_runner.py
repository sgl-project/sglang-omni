# SPDX-License-Identifier: Apache-2.0
"""Restore retained multimodal rows before each Qwen session prefill."""

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.model_executor.forward_batch_info import ForwardBatch

from sglang_omni.models.qwen3_omni.thinker_model_runner import (
    Qwen3OmniThinkerModelRunner,
)
from sglang_omni.scheduling.types import SchedulerRequest


class QwenSessionThinkerModelRunner(Qwen3OmniThinkerModelRunner):
    def before_prefill(
        self,
        forward_batch: ForwardBatch,
        schedule_batch: ScheduleBatch,
        requests: list[SchedulerRequest],
    ) -> None:
        for request_index, request in enumerate(schedule_batch.reqs):
            model_inputs = request.omni_data.model_inputs
            if not model_inputs:
                continue
            else:
                pass
            # Note (Junnan Li): A decode retraction rebuilds KV after the ordinary runner has cleared its prefill inputs.
            device = forward_batch.input_ids.device
            request.omni_model_inputs = {
                name: (
                    value.to(device)
                    if isinstance(value, torch.Tensor)
                    else (
                        [layer.to(device) for layer in value]
                        if isinstance(value, list)
                        else value
                    )
                )
                for name, value in model_inputs.items()
            }
            token_ids = torch.tensor(request.origin_input_ids, dtype=torch.long)
            positions = {
                modality: (token_ids == token_id).nonzero(as_tuple=True)[0]
                for modality, token_id in (
                    ("image", self.image_token_id),
                    ("video", self.video_token_id),
                    ("audio", self.audio_token_id),
                )
            }
            cached_tokens = forward_batch.extend_prefix_lens_cpu[request_index]
            consumed = {
                modality: int((token_positions < cached_tokens).sum())
                for modality, token_positions in positions.items()
            }
            if positions["image"].numel() or positions["video"].numel():
                consumed["_visual"] = consumed["image"] + consumed["video"]
            elif consumed["audio"]:
                # Note (Junnan Li): The audio sidecar takes an audio-only cursor; a visual cursor sends the prefill down the generic path.
                consumed = {"audio": consumed["audio"]}
            else:
                consumed = None
            request._omni_mm_positions = (
                positions  # noqa: leading-underscore - model runner contract
            )
            request._omni_consumed = (
                consumed  # noqa: leading-underscore - model runner contract
            )
        super().before_prefill(forward_batch, schedule_batch, requests)
