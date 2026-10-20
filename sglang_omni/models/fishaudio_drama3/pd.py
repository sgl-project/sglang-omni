# SPDX-License-Identifier: Apache-2.0
"""Drama 3 request continuation; attention storage is transferred separately."""

from __future__ import annotations

from dataclasses import fields, replace

from sglang.srt.managers.schedule_batch import Req

from sglang_omni.models.fishaudio_drama3.request_builders import (
    MINIMUM_PROMPT_ROWS,
    Drama3RequestData,
)
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData

RESUME_SCHEMA = "drama3-v1"


def validate_frame_history(
    request: Req, frame_rows: object, prompt_row_count: object
) -> list[list[int]]:
    """Require the complete prompt and exactly one sampled audio frame."""
    if type(prompt_row_count) is not int or prompt_row_count < MINIMUM_PROMPT_ROWS:
        raise ValueError("Drama 3 PD requires a valid prompt row count")
    elif not isinstance(frame_rows, list) or len(frame_rows) != prompt_row_count + 1:
        raise ValueError("Drama 3 PD requires prompt rows and one sampled frame")
    elif len(request.output_ids) != 1:
        raise ValueError("Drama 3 PD requires exactly one sampled main token")
    elif prompt_row_count != len(request.origin_input_ids):
        raise ValueError("Drama 3 PD prompt length differs from the request")
    else:
        pass
    frame_width = len(frame_rows[0]) if isinstance(frame_rows[0], list) else 0
    if frame_width < 2 or any(
        not isinstance(row, list)
        or len(row) != frame_width
        or any(type(token) is not int or token < 0 for token in row)
        for row in frame_rows
    ):
        raise ValueError("Drama 3 PD requires rectangular nonnegative integer frames")
    elif [row[0] for row in frame_rows] != [
        *request.origin_input_ids,
        *request.output_ids,
    ]:
        raise ValueError("Drama 3 PD frame history differs from the main tokens")
    else:
        return [list(row) for row in frame_rows]


def build_state(request: Req) -> tuple[dict, dict, list[int]]:
    request_data = request.omni_data
    if not isinstance(request_data, Drama3RequestData):
        raise TypeError("Drama 3 PD requires Drama3RequestData")
    elif request_data.stage_payload is None:
        raise ValueError("Drama 3 PD requires a stage payload")
    elif request_data.streamed_frame_count != 0:
        raise ValueError("Drama 3 PD Prefill must not emit audio frames")
    else:
        pass
    frame_rows = validate_frame_history(
        request, request_data.frame_rows, request_data.prompt_row_count
    )
    payload = request_data.stage_payload
    continuation_payload = replace(
        payload, request=replace(payload.request, inputs=None), data=None
    )
    return (
        continuation_payload.to_dict(),
        {
            "schema": RESUME_SCHEMA,
            "frame_rows": frame_rows,
            "prompt_row_count": request_data.prompt_row_count,
        },
        list(request.origin_input_ids),
    )


def restore_state(
    request: Req,
    request_data: SGLangARRequestData,
    resume: dict[str, object] | None,
    *,
    frame_width: int | None = None,
    codebook_size: int | None = None,
) -> None:
    if not isinstance(resume, dict) or resume.get("schema") != RESUME_SCHEMA:
        raise ValueError("unsupported Drama 3 PD resume schema")
    elif set(resume) != {"schema", "frame_rows", "prompt_row_count"}:
        raise ValueError("invalid Drama 3 PD resume fields")
    else:
        pass
    frame_rows = validate_frame_history(
        request, resume["frame_rows"], resume["prompt_row_count"]
    )
    if frame_width is not None and any(len(row) != frame_width for row in frame_rows):
        raise ValueError("Drama 3 PD frame width differs from the model")
    elif any(row[0] >= request.vocab_size for row in frame_rows):
        raise ValueError("Drama 3 PD main token exceeds the model vocabulary")
    elif codebook_size is not None and any(
        token >= codebook_size for row in frame_rows for token in row[1:]
    ):
        raise ValueError("Drama 3 PD codebook token exceeds the model vocabulary")
    else:
        pass
    request.omni_data = Drama3RequestData(
        **{
            field.name: getattr(request_data, field.name)
            for field in fields(request_data)
        },
        frame_rows=frame_rows,
        prompt_row_count=resume["prompt_row_count"],
        streamed_frame_count=0,
        pd_state_pending=True,
    )
