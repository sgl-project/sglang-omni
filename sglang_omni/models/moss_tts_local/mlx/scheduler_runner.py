# SPDX-License-Identifier: Apache-2.0
"""Omni scheduler adapter for MOSS-TTS Local's MLX frame rows."""

from __future__ import annotations

from typing import Any

import torch

from sglang_omni.model_runner.mlx_model_worker import MlxSchedulerModelRunner
from sglang_omni.models.moss_tts_local.request_builders import (
    MOSS_STREAM_TRANSPORT_BATCH_FRAMES,
)
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.types import RequestOutput


class MossTTSLocalMlxSchedulerModelRunner(MlxSchedulerModelRunner):
    """Copies completed MLX code rows into the existing MOSS result adapter."""

    def __init__(self, tp_worker: Any, output_processor: Any) -> None:
        super().__init__(tp_worker, output_processor)
        self._outbox: Any | None = None

    def set_stream_outbox(self, outbox: Any) -> None:
        self._outbox = outbox

    def _flush_stream_rows(self, request_id: str, data: Any, *, force: bool) -> None:
        if data.stream_metadata is None or self._outbox is None:
            return
        pending = data.stream_pending_rows
        threshold = (
            1
            if not data.stream_first_batch_sent
            else MOSS_STREAM_TRANSPORT_BATCH_FRAMES
        )
        if not pending or (not force and len(pending) < threshold):
            return
        rows = pending[0] if len(pending) == 1 else torch.stack(pending)
        pending.clear()
        data.stream_first_batch_sent = True
        self._outbox.put(
            OutgoingMessage(
                request_id=request_id,
                type="stream",
                target="vocoder",
                data=rows,
                metadata=data.stream_metadata,
            )
        )

    def on_request_finished(self, request_id: str, req_data: Any) -> None:
        self._flush_stream_rows(request_id, req_data, force=True)

    def post_process_outputs(
        self,
        result: Any,
        scheduler_output: Any,
        outputs: dict[str, RequestOutput],
    ) -> None:
        del result
        native_runner = self.tp_worker._mlx_runner
        end_id = int(native_runner.model.config.audio_end_token_id)
        for request in scheduler_output.requests:
            rows = native_runner.pop_completed_rows(request.request_id)
            if len(rows) != 1:
                raise RuntimeError(
                    "MOSS-TTS Local MLX expected one completed row for "
                    f"{request.request_id}, got {len(rows)}"
                )
            output = outputs[request.request_id]
            if output.data is None or int(output.data) == end_id:
                self._flush_stream_rows(request.request_id, request.data, force=True)
                continue
            row = torch.tensor(rows[0], dtype=torch.long)
            request.data.output_rows.append(row)
            if request.data.stream_metadata is not None:
                request.data.stream_pending_rows.append(row)
                self._flush_stream_rows(request.request_id, request.data, force=False)
