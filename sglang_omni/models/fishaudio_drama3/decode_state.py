# SPDX-License-Identifier: Apache-2.0
"""Frame histories and request lookups that stage Drama 3 forwards."""

from __future__ import annotations

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


def extend_frame_history(request_data: Drama3RequestData) -> torch.Tensor:
    """extend_frame_rows as an int64 CPU tensor [rows, frame width]: the prompt
    tensor the request builder kept, then any frames sampled before a retraction."""
    frame_rows = extend_frame_rows(request_data)
    prompt_frames = request_data.prompt_frames
    assert prompt_frames is not None
    sampled_rows = frame_rows[request_data.prompt_row_count :]
    if not sampled_rows:
        return prompt_frames
    else:
        return torch.cat((prompt_frames, torch.tensor(sampled_rows, dtype=torch.int64)))


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
