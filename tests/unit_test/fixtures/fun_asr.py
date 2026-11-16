# SPDX-License-Identifier: Apache-2.0
"""Deterministic text and encoder boundaries around the real CPU frontend."""

from __future__ import annotations

import io
import threading
import wave
from collections import deque
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from sglang.srt.managers.schedule_batch import MultimodalDataItem

from sglang_omni.models.fun_asr.configuration_fun_asr import FunAsrNanoFeatureExtractor
from sglang_omni.models.fun_asr.request_builders import (
    FunASRRequestData,
    make_fun_asr_scheduler_adapters,
)
from sglang_omni.models.fun_asr.scheduler import FunASRScheduler
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

AUDIO_PAD = "<|object_ref_start|>"


class Tokenizer:
    eos_token_id = 2
    vocab_size = 200000

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == AUDIO_PAD
        return 42

    def __call__(
        self, text: str, *, add_special_tokens: bool = False
    ) -> SimpleNamespace:
        assert not add_special_tokens
        pieces = text.split(AUDIO_PAD)
        tokens = []
        for index, piece in enumerate(pieces):
            if index:
                tokens.append(42)
            tokens.extend(ord(character) + 100 for character in piece)
        return SimpleNamespace(input_ids=tokens)


class Frontend:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.extractor = FunAsrNanoFeatureExtractor()
        self.calls = 0

    def __call__(
        self,
        audio: np.ndarray,
        *,
        sampling_rate: int,
        return_tensors: str,
        return_attention_mask: bool,
        padding: str,
    ) -> dict[str, torch.Tensor]:
        self.calls += 1
        self.started.set()
        assert self.release.wait(10), "test did not release CPU preparation"
        return self.extractor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors=return_tensors,
            return_attention_mask=return_attention_mask,
            padding=padding,
        )


class Encoder:
    def __init__(self) -> None:
        self.started = threading.Event()
        self.release = threading.Event()
        self.release.set()
        self.calls = 0
        self.closed = threading.Event()
        self.inputs: list[torch.Tensor] = []

    def encode_item(self, item: MultimodalDataItem) -> None:
        assert not self.closed.is_set()
        self.calls += 1
        self.inputs.append(item.feature.clone())
        self.started.set()
        assert self.release.wait(10), "test did not release encoder"
        assert not self.closed.is_set(), "encoder closed before its builder finished"
        item.precomputed_embeddings = torch.zeros(item.num_audio_tokens, 4)
        item.feature = None

    def close(self) -> None:
        self.closed.set()


def payload(request_id: str = "audio") -> StagePayload:
    samples = (np.sin(np.arange(2000) / 10) * 12000).astype("<i2")
    output = io.BytesIO()
    with wave.open(output, "wb") as writer:
        writer.setnchannels(1)
        writer.setsampwidth(2)
        writer.setframerate(16000)
        writer.writeframes(samples.tobytes())
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(
            inputs={"audio_bytes": output.getvalue()}, params={"stream": True}
        ),
        data={},
    )


@dataclass
class Harness:
    scheduler: FunASRScheduler
    frontend: Frontend
    encoder: Encoder
    admitted: list[FunASRRequestData]
    admission_threads: list[int]


@pytest.fixture
def harness() -> Iterator[Harness]:
    frontend, encoder = Frontend(), Encoder()
    builder, _ = make_fun_asr_scheduler_adapters(
        tokenizer=Tokenizer(),
        feature_extractor=frontend,
        audio_encoder_service=encoder,
        max_new_tokens=200,
    )
    scheduler = FunASRScheduler.__new__(FunASRScheduler)
    scheduler.cpu_input_builder = builder
    scheduler.request_builder = builder
    scheduler.cpu_preparation = None
    scheduler.cpu_preparation_closed = False
    scheduler.is_processing_inputs = False
    scheduler.has_preparation_slot_credit = False
    scheduler.preparation_shutdown_lock = threading.Lock()
    scheduler.shutdown_lock = threading.Lock()
    scheduler.shutdown_callback = encoder.close
    scheduler.async_d2h_worker = None
    scheduler.request_admission_lock = threading.RLock()
    scheduler.request_build_executor = ThreadPoolExecutor(max_workers=2)
    scheduler.request_build_max_workers = 2
    scheduler.request_build_max_pending = 4
    scheduler.request_build_backlog_limit = 4
    scheduler.request_build_max_pending_observed = 0
    scheduler.pending_request_builds = {}
    scheduler.pending_request_admissions = {}
    scheduler.backlogged_request_build_payloads = deque()
    scheduler.deferred_request_payloads = {}
    scheduler.dirty_deferred_request_ids = set()
    scheduler.pending_stream_ingress = {}
    scheduler.waiting_queue = []
    scheduler.max_queued_requests = None
    scheduler.aborted_request_ids = set()
    scheduler.aborted_request_id_order = deque()
    scheduler.completed_request_ids = {}
    scheduler.inbox = Queue()
    scheduler.outbox = Queue()
    scheduler.idle_wait_message = None
    scheduler.tp_size = 1
    scheduler.is_entry_rank = True
    scheduler.enable_overlap = False
    scheduler.enable_async_decode = False
    scheduler.async_prefill = False
    scheduler.request_build_admission_wait_s = 0
    scheduler.session_bridge = None
    scheduler.stream_chunk_handler = None
    scheduler.stream_done_handler = None
    scheduler.stop_requested = False
    scheduler.scheduler_thread_id = None
    scheduler.running_batch = None
    scheduler.cur_batch = None
    scheduler.last_batch = None
    scheduler.previous_pending_decode = None
    scheduler.async_pending = None
    scheduler.result_queue = deque()
    scheduler.abort_callback = None
    scheduler.first_emit_done = set()
    scheduler.prefill_start_done = set()
    scheduler.prefill_end_done = set()
    admitted, admission_threads = [], []

    def admit(
        payload, pending_stream_done, result, *, request_admission_lock_held=False
    ):
        assert request_admission_lock_held
        assert not pending_stream_done
        admitted.append(result)
        admission_threads.append(threading.get_ident())

    scheduler.enqueue_built_request = admit
    try:
        yield Harness(scheduler, frontend, encoder, admitted, admission_threads)
    finally:
        frontend.release.set()
        encoder.release.set()
        scheduler.shutdown_resources()


def enqueue(scheduler: FunASRScheduler, request: StagePayload) -> None:
    scheduler.enqueue(
        IncomingMessage(request_id=request.request_id, type="new_request", data=request)
    )


def process(scheduler: FunASRScheduler) -> None:
    scheduler.process_input_requests(scheduler.recv_requests())


def await_build(scheduler: FunASRScheduler, request_id: str) -> None:
    with scheduler.request_admission_lock:
        pending = scheduler.pending_request_builds.get(request_id)
    if pending is not None:
        pending[2].result(timeout=10)
        scheduler.drain_request_build_results()
