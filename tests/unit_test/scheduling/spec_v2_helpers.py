# SPDX-License-Identifier: Apache-2.0
"""Native SpecV2 ownership with real batches, results and CPU event-order fakes."""
from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.managers.schedule_batch import ScheduleBatch
from sglang.srt.managers.utils import GenerationBatchResult
from sglang.srt.model_executor.forward_batch_info import ForwardMode
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from tests.unit_test.scheduling.test_speculative_output import make_request


class Event:
    def __init__(self, device, name):
        self.device, self.name = device, name
        self.fail = False

    def record(self, stream=None):
        self.device.log.append(
            ("record", self.name, (stream or self.device.current).name)
        )

    def synchronize(self):
        self.device.log.append(("wait", self.name))
        if self.fail:
            raise RuntimeError("device wait failed")


class Stream:
    def __init__(self, device, name):
        self.device, self.name = device, name

    def wait_stream(self, other):
        self.device.log.append(("stream", self.name, other.name))

    def wait_event(self, event):
        self.device.log.append(("barrier", self.name, event.name))

    def synchronize(self):
        self.device.log.append(("drain", self.name))


class Device:
    def __init__(self):
        self.log = []
        self.current = Stream(self, "schedule")
        self.event_count = 0
        self.stream_count = 0

    def Event(self):
        self.event_count += 1
        return Event(self, f"event-{self.event_count}")

    def Stream(self):
        self.stream_count += 1
        return Stream(self, f"new-stream-{self.stream_count}")

    def current_stream(self):
        return self.current

    @contextmanager
    def stream(self, stream):
        previous, self.current = self.current, stream
        try:
            yield
        finally:
            self.current = previous


def make_owner():
    scheduler = OmniScheduler.__new__(OmniScheduler)
    device = Device()
    scheduler.device_module = device
    scheduler.forward_stream = Stream(device, "forward")
    scheduler.schedule_stream = device.current
    scheduler.copy_stream = Stream(device, "copy")
    scheduler.forward_ct = 0
    scheduler._sched_idled = (
        False  # noqa: leading-underscore  # Native scheduler state.
    )
    scheduler.processed_tokens_counter = 0
    scheduler.enable_spec_v2_overlap = True
    scheduler.enable_overlap = True
    scheduler.spec_algorithm = SpeculativeAlgorithm.DFLASH
    scheduler.pending_spec_v2 = None
    scheduler.spec_v2_draft_tokens = 8
    scheduler.batch_record_buf = [None, None]
    scheduler.batch_record_ct = 0
    scheduler.req_to_token_pool = SimpleNamespace(
        req_generation=torch.ones(16, dtype=torch.int64)
    )
    scheduler.beam_coordinator = Mock()
    scheduler.chunked_req = None
    scheduler.future_map = Mock(spec_algo=SpeculativeAlgorithm.DFLASH)
    scheduler.future_map.resolve_seq_lens_cpu.side_effect = (
        lambda batch: device.log.append(("resolve_lengths",))
    )
    scheduler.future_map._resolve_spec_extras.side_effect = lambda batch: device.log.append(  # noqa: leading-underscore  # Native relay method.
        ("resolve_inputs",)
    )
    scheduler.future_map.publish.side_effect = (
        lambda indices, lengths: device.log.append(
            ("publish_lengths", indices.tolist())
        )
    )
    scheduler.future_map.stash.side_effect = lambda indices, payload: device.log.append(
        ("stash", indices.tolist())
    )
    scheduler.stored_results = []
    scheduler.process_batch_result = (
        lambda batch, result: scheduler.stored_results.append((batch, result))
    )
    scheduler.handle_batch_failure = Mock()
    scheduler.emit_prefill_start_for_batch = Mock()
    scheduler.emit_prefill_end_for_batch = Mock()
    scheduler.update_cache_from_scheduler = Mock()
    scheduler.wait_prefill_publication = Mock()
    requests = [
        make_request("a", 7, 20),
        make_request("b", 2, 20),
        make_request("c", 12, 20),
    ]
    batch = ScheduleBatch(
        reqs=requests,
        device="cpu",
        enable_overlap=True,
        spec_algorithm=SpeculativeAlgorithm.DFLASH,
        forward_mode=ForwardMode.DECODE,
        req_pool_indices=torch.tensor([7, 2, 12]),
        seq_lens=torch.tensor([2, 2, 2]),
        seq_lens_cpu=torch.tensor([2, 2, 2]),
        sampling_info=Mock(),
    )
    batch.spec_info = SimpleNamespace(future_indices=batch.req_pool_indices)
    extra = torch.tensor([71])
    scheduler.extra_reference = extra

    def forward(live_batch, *, on_publish):
        device.log.append(("native_forward", live_batch.forward_mode))
        assert live_batch.sampling_info is not batch.original_sampling
        lengths = torch.tensor([4, 5, 3])
        on_publish(lengths)
        draft = SimpleNamespace(
            new_seq_lens=lengths,
            bonus_tokens=torch.tensor([10, 20, 30]),
            topk_p=None,
            topk_index=None,
            hidden_states=None,
            dsa_topk_indices=None,
            future_indices=None,
        )
        live_batch.forward_mode = ForwardMode.TARGET_VERIFY
        live_batch.seq_lens = lengths
        live_batch.input_ids = torch.tensor([99])
        return GenerationBatchResult(
            next_token_ids=torch.arange(24),
            accept_lens=torch.tensor([2, 3, 1]),
            speculative_num_draft_tokens=8,
            speculative_output_stride=8,
            next_draft_input=draft,
            new_seq_lens=lengths,
            extra_keep_alive_refs=[extra],
        )

    batch.original_sampling = batch.sampling_info
    scheduler.draft_worker = SimpleNamespace(forward_batch_generation=forward)
    scheduler.tp_worker = SimpleNamespace(
        model_runner=SimpleNamespace(forward_stream=scheduler.forward_stream)
    )
    return scheduler, batch, device
