# SPDX-License-Identifier: Apache-2.0
from array import array
from queue import Queue
from threading import RLock
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.scheduling import omni_scheduler as module
from sglang_omni.scheduling.messages import OutgoingMessage
from sglang_omni.scheduling.omni_scheduler import OmniScheduler


def scheduler(monkeypatch):
    result = object.__new__(OmniScheduler)
    result._native_speculative = True
    result._request_admission_lock = RLock()
    result._aborted_request_ids = set()
    result._first_emit_done = set()
    result._prefill_start_done = set()
    result._prefill_end_done = set()
    result._completed_request_ids = {}
    result._pending_stream_ingress = {}
    result._request_finished_callback = None
    result._model_runner = None
    result.is_entry_rank = True
    result.outbox = Queue()
    result._stream_output_builder = lambda rid, data, output: [
        OutgoingMessage(request_id=rid, type="stream", data=output.data)
    ]
    result._result_adapter = lambda data: list(data.output_ids)
    monkeypatch.setattr(
        module, "get_serving", lambda: SimpleNamespace(weight_version=None)
    )
    return result


def request(rid, max_tokens=32):
    sampling = SamplingParams(temperature=0, max_new_tokens=max_tokens)
    sampling.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [1]),
        sampling_params=sampling,
        eos_token_ids={99},
        vocab_size=128,
    )
    req._omni_terminal_claimed = False
    req._omni_data = SimpleNamespace(
        req=req,
        prefill_input_embeds=None,
        decode_input_embeds=None,
        extra_model_outputs={},
    )
    return req


def drain(s):
    messages = []
    while not s.outbox.empty():
        msg = s.outbox.get_nowait()
        messages.append((msg.request_id, msg.type, msg.data))
    return messages


def test_accepted_blocks_preserve_per_request_order_without_duplicates(monkeypatch):
    s = scheduler(monkeypatch)
    a, b = request("a"), request("b")
    a.output_ids.extend([10, 11, 12])
    b.output_ids.extend([20])
    s.stream_output([a, b])
    s.stream_output([a, b])
    a.output_ids.extend([13])
    b.output_ids.extend([21, 22, 23])
    s.stream_output([a, b])
    assert drain(s) == [
        ("a", "stream", 10),
        ("a", "stream", 11),
        ("a", "stream", 12),
        ("b", "stream", 20),
        ("a", "stream", 13),
        ("b", "stream", 21),
        ("b", "stream", 22),
        ("b", "stream", 23),
    ]


@pytest.mark.parametrize(
    ("accepted", "limit", "expected"),
    [
        ([10, 99, 11, 12], 32, [10, 99]),
        ([10, 11, 12, 13], 2, [10, 11]),
        ([10, 99, 11, 12], 3, [10, 99]),
    ],
)
def test_eos_and_length_trim_stream_and_terminal(
    monkeypatch, accepted, limit, expected
):
    s = scheduler(monkeypatch)
    req = request("r", limit)
    req.output_ids.extend(accepted)
    req.update_finish_state(len(accepted))
    assert req.finished()
    s.stream_output([req])
    assert drain(s) == [("r", "stream", i) for i in expected] + [
        ("r", "result", expected)
    ]
    assert req._omni_data is None


def test_aborted_and_middle_chunk_requests_emit_no_tokens(monkeypatch):
    s = scheduler(monkeypatch)
    aborted, middle, skipped = request("a"), request("m"), request("s")
    for req in (aborted, middle, skipped):
        req.output_ids.extend([10, 11])
    s._aborted_request_ids.add("a")
    middle.inflight_middle_chunks = 1
    s.stream_output([aborted, middle, skipped], skip_req=skipped)
    assert drain(s) == []


def test_upstream_acceptance_unpack_excludes_rejected_padding():
    from sglang.srt.managers.scheduler_components.batch_result_processor import (
        SchedulerBatchResultProcessor,
    )

    a, b = request("a"), request("b")
    a.kv_committed_len = b.kv_committed_len = 3
    observations = []
    processor = SimpleNamespace(
        model_worker=SimpleNamespace(
            on_verify_complete_cpu=lambda counts, **kwargs: observations.append(counts)
        ),
        advance_grammar_fsm=lambda result, batch: None,
    )
    result = SimpleNamespace(
        next_token_ids=torch.tensor([10, 11, 12, 127, 20, 127, 127, 127]),
        accept_lens=torch.tensor([3, 1]),
        speculative_num_draft_tokens=4,
        block_accept_lens=None,
        cap_lens=None,
    )
    assert SchedulerBatchResultProcessor._resolve_spec_v2_tokens(
        processor, result, SimpleNamespace(reqs=[a, b])
    ) == [[10, 11, 12], [20]]
    assert [a.kv_committed_len, b.kv_committed_len] == [6, 4]
    assert [a.spec_verify_ct, b.spec_verify_ct] == [1, 1]
    assert observations == [[2, 0]]


def test_native_forward_retains_verified_lengths_and_restores_batch(monkeypatch):
    from sglang.srt.managers import overlap_utils
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

    batch = ScheduleBatch(reqs=[], spec_algorithm=SpeculativeAlgorithm.DFLASH)
    batch.input_ids = torch.tensor([1])
    batch.seq_lens = torch.tensor([3])
    batch.seq_lens_cpu = torch.tensor([3])
    batch.seq_lens_sum = 3
    original_sampling = SimpleNamespace(copy_for_forward=lambda: "forward-only")
    batch.sampling_info = original_sampling
    original_spec = object()
    batch.spec_info = original_spec
    next_spec = object()
    events = []
    result = SimpleNamespace(
        next_draft_input=next_spec,
        new_seq_lens=torch.tensor([6]),
        copy_to_cpu=lambda **kwargs: events.append("copy"),
    )

    def forward(sb):
        assert sb.sampling_info == "forward-only"
        assert events == ["resolve"]
        sb.seq_lens = torch.tensor([999])
        sb.spec_info = "temporary"
        return result

    s = object.__new__(OmniScheduler)
    s._emit_prefill_start_for_batch = lambda sb: None
    s._emit_prefill_end_for_batch = lambda sb: None
    s._stamp_batch_launch = lambda sb: None
    s.future_map = object()
    s.model_worker = SimpleNamespace(forward_batch_generation=forward)
    s.device_module = SimpleNamespace(Event=object)
    s.update_cache_from_scheduler = lambda sb, r: events.append("cache")
    monkeypatch.setattr(
        overlap_utils,
        "resolve_forward_inputs",
        lambda sb, fm: events.append("resolve"),
    )
    assert s._run_speculative_batch(batch) is result
    assert batch.sampling_info is original_sampling
    assert batch.spec_info is next_spec
    assert batch.seq_lens.tolist() == [6]
    assert batch.seq_lens_cpu.tolist() == [6]
    assert batch.seq_lens_sum == 6
    assert batch.input_ids is None
    assert events == ["resolve", "cache", "copy"]
