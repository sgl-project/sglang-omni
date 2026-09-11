# SPDX-License-Identifier: Apache-2.0
"""VoxCPM2 AR runner: drives the diffusion half of every decode step."""

from __future__ import annotations

from typing import Any

import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.models.voxcpm2.request_builders import VoxCPM2SGLangRequestData


class VoxCPM2ModelRunner(ModelRunner):
    """Runs the local DiT after each AR forward and feeds the result back."""

    def requested_capture_hidden_mode_prefill(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        return CaptureHiddenMode.FULL

    def requested_capture_hidden_mode_decode(
        self, schedule_batch: Any, requests: list
    ) -> Any:
        del schedule_batch, requests
        from sglang.srt.model_executor.forward_batch_info import CaptureHiddenMode

        if self.model.graph_feedback_buffer is not None:
            return CaptureHiddenMode.FULL
        return CaptureHiddenMode.LAST

    def post_prefill(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch
        if bool(getattr(schedule_batch, "is_prefill_only", False)) or not requests:
            return
        self._advance(requests, rows=self._prefill_rows(requests), is_prefill=True)

    def post_decode(
        self, result: Any, forward_batch: Any, schedule_batch: Any, requests: list
    ) -> None:
        del result, forward_batch, schedule_batch
        if requests:
            self._advance(requests, rows=None, is_prefill=False)

    @staticmethod
    def _prefill_rows(requests: list) -> torch.Tensor:
        """Index of each request's final prompt position in the packed batch."""
        indices: list[int] = []
        offset = 0
        for request in requests:
            offset += int(request.data.req.extend_range.length)
            indices.append(offset - 1)
        return torch.tensor(indices, dtype=torch.long)

    def _advance(
        self, requests: list, *, rows: torch.Tensor | None, is_prefill: bool
    ) -> None:
        """Sample one latent patch per request and stage the next step's input."""
        rows_data = [request.data for request in requests]
        timesteps, cfg_value = _shared_sampling(rows_data)

        patches, embeddings = self.model.decode_patch(
            self._batch_cond(rows_data),
            inference_timesteps=timesteps,
            cfg_value=cfg_value,
            rows=rows,
        )
        stop_flags = self.model.stop_flags(rows)

        for index, data in enumerate(rows_data):
            patch = patches[index : index + 1]
            data.cond = patch
            data.latent_patches.append(patch.squeeze(0).detach().cpu())
            if is_prefill:
                continue
            state = data.state
            steps = len(data.latent_patches)
            if steps > state.min_len and bool(stop_flags[index]):
                data.finish_reason = "stop"
            elif steps >= state.max_len:
                data.finish_reason = "length"
            if data.finish_reason is not None:
                data.req.finished_reason = FINISH_MATCHED_TOKEN(0)

        if self.model.graph_feedback_buffer is not None:
            self.model.write_feedback(embeddings)

    def _batch_cond(self, rows_data: list[VoxCPM2SGLangRequestData]) -> torch.Tensor:
        """Stack each request's previous patch into the DiT's condition batch."""
        parameter = next(self.model.parameters())
        zeros = torch.zeros(
            (1, self.model.patch_size, self.model.feat_dim),
            device=parameter.device,
            dtype=parameter.dtype,
        )
        return torch.cat(
            [
                zeros if data.cond is None else data.cond.to(parameter.device)
                for data in rows_data
            ],
            dim=0,
        )


def _shared_sampling(rows_data: list[VoxCPM2SGLangRequestData]) -> tuple[int, float]:
    """One batch runs one sampling recipe; reject a batch that mixes them.

    The DiT samples the whole batch in lockstep, so a differing step count or
    guidance scale cannot be honored per row. Refusing is the only option that
    does not silently synthesize some requests with another request's recipe.
    """
    first = rows_data[0].state
    timesteps = int(first.inference_timesteps)
    cfg_value = float(first.cfg_value)
    for data in rows_data[1:]:
        if int(data.state.inference_timesteps) != timesteps:
            raise ValueError(
                "VoxCPM2 cannot batch requests with different inference_timesteps: "
                f"{timesteps} and {data.state.inference_timesteps}"
            )
        if float(data.state.cfg_value) != cfg_value:
            raise ValueError(
                "VoxCPM2 cannot batch requests with different cfg_value: "
                f"{cfg_value} and {data.state.cfg_value}"
            )
    return timesteps, cfg_value


__all__ = ["VoxCPM2ModelRunner"]
