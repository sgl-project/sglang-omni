# SPDX-License-Identifier: Apache-2.0
"""Staged frame inputs and fixed buffers shared by the model and its runner."""

from __future__ import annotations

import dataclasses

import torch

from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3StepInputs:
    """What one prefill forward needs besides the batch, staged by the model runner.

    frames holds the input frame of every forward row and next_frames each row's
    next frame, the tower prefill's label; the other tensors hold one entry per
    request, in batch order.
    """

    request_ids: list[str]
    frames: torch.Tensor
    next_frames: torch.Tensor
    ras_windows: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3SamplingRows:
    """One row per request or batch slot: its newest frame, RAS window and sampling knobs."""

    frames: torch.Tensor
    ras_windows: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    epsilon_cutoff: torch.Tensor

    @staticmethod
    def greedy_rows(
        row_count: int, frame_width: int, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Rows that sample like a greedy request with an empty window."""
        return {
            "frames": torch.zeros(
                (row_count, frame_width), dtype=torch.int64, device=device
            ),
            "ras_windows": torch.full(
                (row_count, RAS_WINDOW_FRAMES),
                EMPTY_RAS_SLOT,
                dtype=torch.int64,
                device=device,
            ),
            "temperature": torch.ones(row_count, dtype=torch.float32, device=device),
            "top_k": torch.ones(row_count, dtype=torch.int64, device=device),
            "top_p": torch.ones(row_count, dtype=torch.float32, device=device),
            "epsilon_cutoff": torch.zeros(
                row_count, dtype=torch.float32, device=device
            ),
        }


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3DecodeBuffers(Drama3SamplingRows):
    """Decode inputs and sampled frames at fixed addresses, one row per batch slot.

    A decode forward gathers its rows here from the state pool, so a padded graph
    row holds the pool's dummy row 0. A prefill stages its requests' epsilon_cutoff
    rows here too, where the first-frame step reads them.
    """

    sampled_frames: torch.Tensor

    @classmethod
    def allocate(
        cls, *, row_count: int, frame_width: int, device: torch.device
    ) -> Drama3DecodeBuffers:
        rows = cls.greedy_rows(row_count, frame_width, device)
        return cls(**rows, sampled_frames=torch.zeros_like(rows["frames"]))


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3DecodeStatePool(Drama3SamplingRows):
    """Each request slot's newest frame, RAS window and sampling knobs, indexed by
    req_pool_indices; row 0 is the dummy row every padded graph row points at.

    A decode forward gathers and publishes its rows inside the decode CUDA graph.
    """

    @classmethod
    def allocate(
        cls, *, request_slot_count: int, frame_width: int, device: torch.device
    ) -> Drama3DecodeStatePool:
        return cls(**cls.greedy_rows(request_slot_count + 1, frame_width, device))

    def gather(self, request_slots: torch.Tensor, buffers: Drama3DecodeBuffers) -> None:
        """Copy the slots' rows into the first rows of the decode buffers, in order."""
        row_count = request_slots.shape[0]
        for pooled, staged in (
            (self.frames, buffers.frames),
            (self.ras_windows, buffers.ras_windows),
            (self.temperature, buffers.temperature),
            (self.top_k, buffers.top_k),
            (self.top_p, buffers.top_p),
            (self.epsilon_cutoff, buffers.epsilon_cutoff),
        ):
            torch.index_select(pooled, 0, request_slots, out=staged[:row_count])

    def publish(self, request_slots: torch.Tensor, frames: torch.Tensor) -> None:
        """Make frames the slots' newest frames and slide their RAS windows.

        The windows slide from the pool, not from the staged copy a forward read.
        """
        ras_windows = self.ras_windows.index_select(0, request_slots)
        self.frames.index_copy_(0, request_slots, frames)
        self.ras_windows.index_copy_(
            0, request_slots, torch.cat((ras_windows[:, 1:], frames[:, :1]), dim=1)
        )


class StepExchange:
    """Hands each forward its staged inputs, and the runner that forward's frames.

    Both are handed over once: a forward clears what it did not produce, so a
    failed step leaves nothing for the next one to pick up.
    """

    def __init__(self) -> None:
        self.staged_inputs: Drama3StepInputs | None = None
        self.sampled_frames: torch.Tensor | None = None

    def stage(self, step_inputs: Drama3StepInputs) -> None:
        self.staged_inputs = step_inputs

    def clear(self) -> None:
        self.staged_inputs = None

    def take_inputs(self, request_ids: list[str], row_count: int) -> Drama3StepInputs:
        """The inputs staged for a batch of these requests and forward rows."""
        step_inputs = self.staged_inputs
        self.staged_inputs = None
        self.sampled_frames = None
        if step_inputs is None:
            raise RuntimeError("the model runner staged no inputs for this forward")
        elif step_inputs.request_ids != request_ids:
            raise RuntimeError(
                f"staged requests {step_inputs.request_ids} are not the batch's "
                f"{request_ids}"
            )
        elif step_inputs.frames.shape[0] != row_count:
            raise RuntimeError(
                f"staged {step_inputs.frames.shape[0]} frames for {row_count} rows"
            )
        else:
            return step_inputs

    def publish(self, frames: torch.Tensor) -> None:
        self.sampled_frames = frames

    def take_frames(self) -> torch.Tensor:
        frames = self.sampled_frames
        self.sampled_frames = None
        if frames is None:
            raise RuntimeError("no forward has sampled frames since the last take")
        else:
            return frames


def next_frame_labels(
    frames: torch.Tensor, last_rows: torch.Tensor, eos_token_id: int
) -> torch.Tensor:
    """Each extend row's next frame; a request's last row gets [eos, 0, ...].

    The placeholder only fills that row's slot until the same forward's decode
    step rewrites it, before any row attends to it.
    """
    next_frames = frames.roll(-1, dims=0)
    next_frames[last_rows] = 0
    next_frames[last_rows, 0] = eos_token_id
    return next_frames
