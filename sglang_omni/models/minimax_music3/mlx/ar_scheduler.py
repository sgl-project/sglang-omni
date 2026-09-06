# SPDX-License-Identifier: Apache-2.0
"""Whole-request MLX AR scheduler with the CUDA pipeline's stream contract."""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from transformers import AutoTokenizer

from sglang_omni.proto import StagePayload
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.pipeline_state import store_state
from sglang_omni.scheduling.simple_scheduler import SimpleScheduler

from ..chunking import chunk_windows
from ..payload_types import MiniMaxMusic3State
from ..prompt import validate_tokenizer_ids
from .ar import generate_frame_hiddens
from .loader import MiniMaxMusic3MlxARModel, load_mlx_ar_model

logger = logging.getLogger(__name__)


def _build_text_pair(
    prompt: str,
    model: MiniMaxMusic3MlxARModel,
    tokenizer: object,
) -> mx.array:
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    if input_ids.shape[1] > 5_000:
        raise ValueError(
            f"MiniMax Music 3 prompt has {input_ids.shape[1]} tokens; "
            "the maximum is 5000"
        )
    conditional = mx.array(input_ids.astype("int32"))
    unconditional = conditional
    if unconditional.shape[1] > 3:
        middle = mx.full(
            (1, unconditional.shape[1] - 3),
            model.config.audio_cfg_token_id,
            dtype=mx.int32,
        )
        unconditional = mx.concatenate(
            [unconditional[:, :1], middle, unconditional[:, -2:]], axis=1
        )
    return mx.concatenate([conditional, unconditional], axis=0)


class MiniMaxMusic3MlxARScheduler(SimpleScheduler):
    """Generate MLX frame hiddens and stream transport-safe CPU chunks."""

    def __init__(
        self,
        model_path: str,
        *,
        revision: str | None = None,
    ) -> None:
        self.model = load_mlx_ar_model(model_path, revision)
        tokenizer_dir = Path(self.model.config.model_path) / "tokenizer"
        if not tokenizer_dir.is_dir():
            raise FileNotFoundError(
                "MiniMax Music 3 MLX artifact must include its tokenizer directory"
            )
        self.tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_dir,
            trust_remote_code=False,
        )
        validate_tokenizer_ids(self.tokenizer)
        self._abort_events: dict[str, threading.Event] = {}
        self._events_lock = threading.Lock()
        self._mlx_thread_stream = mx.new_thread_local_stream(mx.gpu)
        super().__init__(self._generate, max_concurrency=1)

    def _abort_event(self, request_id: str) -> threading.Event:
        with self._events_lock:
            return self._abort_events.setdefault(request_id, threading.Event())

    def _generate(self, payload: StagePayload) -> StagePayload:
        state = MiniMaxMusic3State.from_dict(payload.data)
        if state.prompt is None:
            raise ValueError("MiniMax Music 3 preprocessing did not build a prompt")
        abort_event = self._abort_event(payload.request_id)
        started = time.perf_counter()
        try:
            with mx.stream(self._mlx_thread_stream):
                text_ids = _build_text_pair(state.prompt, self.model, self.tokenizer)
                hidden = generate_frame_hiddens(
                    self.model.language_model,
                    self.model.rvq_depth_decoder,
                    self.model.config,
                    text_ids,
                    max_frames=state.max_audio_frames,
                    seed=state.seed,
                    should_abort=abort_event.is_set,
                )
                mx.eval(hidden)
                generated_frames = int(hidden.shape[1])
                for window in chunk_windows(generated_frames):
                    if abort_event.is_set():
                        raise InterruptedError("MiniMax Music 3 MLX generation aborted")
                    chunk = hidden[:, window.start : window.end].astype(mx.float16)
                    mx.eval(chunk)
                    transport = torch.from_numpy(
                        np.ascontiguousarray(np.asarray(chunk, dtype=np.float16))
                    )
                    self.outbox.put(
                        OutgoingMessage(
                            request_id=payload.request_id,
                            type="stream",
                            data=transport,
                            metadata={
                                "stream": True,
                                "modality": "ttm_hidden",
                                "chunk_idx": window.index,
                                "start_frame": window.start,
                                "end_frame": window.end,
                                "is_final": window.is_last,
                                "seed": state.seed,
                            },
                        )
                    )
        finally:
            with self._events_lock:
                self._abort_events.pop(payload.request_id, None)

        state.generated_frames = generated_frames
        state.finish_reason = (
            "length" if generated_frames >= state.max_audio_frames else "stop"
        )
        state.prompt = None
        state.caption = ""
        state.lyrics = ""
        logger.info(
            "MiniMax Music 3 MLX AR done request=%s frames=%d elapsed=%.1fs",
            payload.request_id,
            generated_frames,
            time.perf_counter() - started,
        )
        return store_state(payload, state)

    def abort(self, request_id: str) -> None:
        with self._events_lock:
            event = self._abort_events.get(request_id)
            if event is not None:
                event.set()
        super().abort(request_id)


__all__ = ["MiniMaxMusic3MlxARScheduler", "_build_text_pair"]
