# SPDX-License-Identifier: Apache-2.0
"""Persistent request rows and frame histories used to stage Drama 3 decode."""

from __future__ import annotations

import dataclasses

import torch

from sglang_omni.models.fishaudio_drama3.request_builders import (
    MINIMUM_PROMPT_ROWS,
    Drama3RequestData,
)
from sglang_omni.models.fishaudio_drama3.sampling import (
    EMPTY_RAS_SLOT,
    RAS_WINDOW_FRAMES,
)
from sglang_omni.scheduling.types import SchedulerRequest


@dataclasses.dataclass(frozen=True, kw_only=True)
class Drama3DecodeStatePool:
    frames: torch.Tensor
    ras_windows: torch.Tensor
    temperature: torch.Tensor
    top_k: torch.Tensor
    top_p: torch.Tensor
    epsilon_cutoff: torch.Tensor
    row_indices: torch.Tensor

    @classmethod
    def allocate(
        cls, *, request_slot_count: int, frame_width: int, device: torch.device
    ) -> Drama3DecodeStatePool:
        row_count = request_slot_count + 1
        return cls(
            frames=torch.zeros(
                (row_count, frame_width), dtype=torch.int64, device=device
            ),
            ras_windows=torch.full(
                (row_count, RAS_WINDOW_FRAMES),
                EMPTY_RAS_SLOT,
                dtype=torch.int64,
                device=device,
            ),
            temperature=torch.ones(row_count, dtype=torch.float32, device=device),
            top_k=torch.ones(row_count, dtype=torch.int64, device=device),
            top_p=torch.ones(row_count, dtype=torch.float32, device=device),
            epsilon_cutoff=torch.zeros(row_count, dtype=torch.float32, device=device),
            row_indices=torch.zeros(
                request_slot_count, dtype=torch.int64, device=device
            ),
        )


def ras_window(frame_rows: list[list[int]]) -> list[int]:
    """Main tokens of the last RAS_WINDOW_FRAMES frames, oldest first, left-padded with
    EMPTY_RAS_SLOT."""
    main_tokens = [frame_row[0] for frame_row in frame_rows[-RAS_WINDOW_FRAMES:]]
    return [EMPTY_RAS_SLOT] * (RAS_WINDOW_FRAMES - len(main_tokens)) + main_tokens


def extend_frame_rows(request_data: Drama3RequestData) -> list[list[int]]:
    """The rows of a prefill, which must cover the request's whole frame history."""
    sglang_request = request_data.req
    assert sglang_request is not None
    sequence_ids = list(sglang_request.full_untruncated_fill_ids)
    if (
        sglang_request.extend_end is None
        or sglang_request.prefix_len != 0
        or sglang_request.extend_end != len(sequence_ids)
    ):
        raise RuntimeError(
            f"{sglang_request.rid}: a Drama 3 prefill must cover the whole sequence of "
            f"{len(sequence_ids)} rows in one extend; got "
            f"[{sglang_request.prefix_len}, {sglang_request.extend_end})"
        )
    elif len(sequence_ids) < MINIMUM_PROMPT_ROWS:
        raise RuntimeError(
            f"{sglang_request.rid}: an extend needs at least {MINIMUM_PROMPT_ROWS} rows"
        )
    elif [frame_row[0] for frame_row in request_data.frame_rows] != sequence_ids:
        raise RuntimeError(
            f"{sglang_request.rid}: the frame history is not the sequence"
        )
    else:
        return request_data.frame_rows


def request_data_of(request: SchedulerRequest) -> Drama3RequestData:
    request_data = request.data
    assert isinstance(request_data, Drama3RequestData)
    return request_data


def request_ids_of(batch_request_data: list[Drama3RequestData]) -> list[str]:
    request_ids = []
    for request_data in batch_request_data:
        request = request_data.req
        assert request is not None
        request_ids.append(request.rid)
    return request_ids
