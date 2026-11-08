# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.runtime_context import get_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.srt.speculative.spec_info import SpeculativeAlgorithm

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.personaplex.lm_session import PersonaPlexSessionAdapter
from sglang_omni.models.personaplex.model_runner import PersonaPlexModelRunner
from sglang_omni.models.personaplex.payload_types import PersonaPlexState
from sglang_omni.proto.request import OmniRequest
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionBridge,
    BridgeSession,
)

from .test_lm_session import (
    CALL,
    CONTEXT_LENGTH,
    PieceTokenizer,
    caller_codes,
    run_offline,
    unit_chunk,
    unit_payload,
)
from .test_model_runner import DeferredHostRows, FakeModel, make_runner


def open_bridge():
    adapter = PersonaPlexSessionAdapter(
        vocab_size=32000, context_length=CONTEXT_LENGTH, tokenizer=PieceTokenizer()
    )
    adapter.open(CALL, OmniRequest(inputs=None))
    engine = Mock()
    engine.session_controller.get.return_value = None
    engine.tree_cache.session_records.return_value = {}
    bridge = ARSessionBridge(engine, adapter, steps_in_place=True)
    bridge.sessions[CALL.id] = BridgeSession(session_identity=CALL)
    return bridge, engine


def accept(bridge, index, codes):
    payload = unit_payload(index, codes)
    unit = bridge.accept(
        payload,
        SimpleNamespace(chunk=unit_chunk(index), session_identity=CALL, stages=("lm",)),
    )
    return unit, payload


def test_steady_units_step_one_request_as_the_offline_request():
    frames_per_unit = [1, 1, 2, 1]
    num_frames = sum(frames_per_unit)
    tokens = [77 + index for index in range(num_frames + 1)]
    offline_inputs, offline_frames, _, _ = run_offline(num_frames + 1, tokens)

    bridge, _ = open_bridge()
    runner = make_runner(FakeModel())
    model = runner.model
    codes = caller_codes(num_frames + 1)
    inputs, results = [], []
    frame = token = 0
    for index, count in enumerate(frames_per_unit):
        unit, payload = accept(bridge, index, codes[frame : frame + count])
        frame += count
        if index == 0:
            data = bridge.adapter.build(CALL, unit.chunk, payload)
            bridge.create_session_request(payload, data)
            request = SimpleNamespace(data=data)
            req = data.req
            assert bridge.is_stepping(CALL)
            assert req.sampling_params.max_new_tokens == count + 2
            origin = list(req.origin_input_ids)
            forward_batch = SimpleNamespace(
                replace_embeds=None, input_ids=torch.zeros(len(origin))
            )
            runner.before_prefill(
                forward_batch,
                SimpleNamespace(
                    reqs=[
                        SimpleNamespace(
                            origin_input_ids=origin, output_ids=[], prefix_len=0
                        )
                    ]
                ),
                [request],
            )
            inputs.append(get_omni_prefill_inputs(forward_batch).input_embeds.long())
            runner.post_prefill(
                SimpleNamespace(next_token_ids=torch.tensor(tokens[:1])),
                None,
                None,
                [request],
            )
            req.output_ids.append(tokens[0])
            token = 1
        else:
            bridge.feed(unit, payload)
            assert bridge.take_resumable() == [req]
        while not bridge.should_park(req):
            runner.before_decode(None, None, [request])
            inputs.append(model.fusion_buffer[:1].long().clone())
            runner.post_decode(
                SimpleNamespace(next_token_ids=torch.tensor(tokens[token : token + 1])),
                None,
                None,
                [request],
            )
            req.output_ids.append(tokens[token])
            token += 1
        bridge.park(req)
        [(request_id, result)] = bridge.take_stepped_results(bridge.sessions[CALL.id])
        assert request_id == payload.request_id
        assert CALL.id not in [
            u.session_identity.id for u in bridge.units_by_request_id.values()
        ]
        results.append(PersonaPlexState.from_dict(result.data))

    assert token == len(tokens)
    for rows, offline_rows in zip(inputs, offline_inputs, strict=True):
        assert torch.equal(rows, offline_rows)
    assert [state.codes.shape[0] for state in results] == [2, 1, 2, 1]
    assert torch.equal(torch.cat([state.codes for state in results]), offline_frames)


def test_closing_a_parked_session_frees_its_request_and_a_running_one_aborts_it():
    for parked in (True, False):
        bridge, engine = open_bridge()
        unit, payload = accept(bridge, 0, caller_codes(1))
        data = bridge.adapter.build(CALL, unit.chunk, payload)
        bridge.create_stepped_request(payload, data)
        session = bridge.sessions[CALL.id]
        if parked:
            bridge.park(data.req)
        else:
            pass
        bridge.close_streaming_session(session)
        assert session.stepped.is_ended and bridge.stepped_session(data.req.rid) is None
        if parked:
            engine.release_request_kv_cache.assert_called_once_with(data.req)
        else:
            engine.abort.assert_called_once_with(data.req.rid)


def test_a_parked_request_resumes_into_a_running_batch_that_prefill_built():
    req_to_token_pool = ReqToTokenPool(
        size=4, max_context_len=64, device="cpu", enable_memory_saver=False
    )
    running_req, parked_req = (
        Req(rid, "", [1, 2, 3], SamplingParams(max_new_tokens=8))
        for rid in ("running", "parked")
    )
    running_req.output_ids = [5]
    parked_req.output_ids = [6, 7]
    req_to_token_pool.alloc([running_req, parked_req])
    scheduler = object.__new__(OmniScheduler)
    scheduler.req_to_token_pool = req_to_token_pool
    scheduler.token_to_kv_pool_allocator = None
    scheduler.tree_cache = None
    scheduler.model_config = SimpleNamespace(vocab_size=32, is_encoder_decoder=False)
    scheduler.enable_overlap = False
    scheduler.spec_algorithm = SpeculativeAlgorithm.NONE
    scheduler.device = "cpu"
    scheduler.session_bridge = SimpleNamespace(take_resumable=lambda: [parked_req])
    with get_context().override_server_args():
        running_batch = scheduler.build_resumed_decode_batch([running_req])
        running_batch.multimodal_inputs = [None]

        batch = scheduler.resume_parked_requests(running_batch)

    assert batch.reqs == [running_req, parked_req]
    assert batch.multimodal_inputs == [None, None]
    assert batch.seq_lens_cpu.tolist() == [3, 4]
    assert batch.req_pool_indices_cpu.tolist() == [
        running_req.kv.req_pool_idx,
        parked_req.kv.req_pool_idx,
    ]


def run_first_unit(bridge, runner):
    unit, payload = accept(bridge, 0, caller_codes(1))
    data = bridge.adapter.build(CALL, unit.chunk, payload)
    bridge.create_session_request(payload, data)
    request, req = SimpleNamespace(data=data), data.req
    req.omni_data = data
    origin = list(req.origin_input_ids)
    runner.before_prefill(
        SimpleNamespace(replace_embeds=None, input_ids=torch.zeros(len(origin))),
        SimpleNamespace(
            reqs=[SimpleNamespace(origin_input_ids=origin, output_ids=[], prefix_len=0)]
        ),
        [request],
    )
    runner.post_prefill(
        SimpleNamespace(next_token_ids=torch.tensor([70])), None, None, [request]
    )
    req.output_ids.append(70)
    runner.before_decode(None, None, [request])
    runner.post_decode(
        SimpleNamespace(next_token_ids=torch.tensor([71])), None, None, [request]
    )
    req.output_ids.append(71)
    assert bridge.should_park(req)
    bridge.park(req)
    bridge.take_stepped_results(bridge.sessions[CALL.id])
    return request, req


def test_a_two_frame_unit_takes_two_steps_and_none_past_its_frames(monkeypatch):
    bridge, _ = open_bridge()
    runner = make_runner(FakeModel())
    request, req = run_first_unit(bridge, runner)
    data = request.data

    def lookahead_eligible() -> bool:
        runner.decode_feedback(data)
        with monkeypatch.context() as patched:
            patched.setattr(
                PersonaPlexModelRunner,
                "model_device",
                property(lambda self: torch.device("cuda")),
            )
            return runner.lookahead_eligible(SimpleNamespace(reqs=[req]))

    unit, payload = accept(bridge, 1, caller_codes(3)[1:])
    bridge.feed(unit, payload)
    assert bridge.take_resumable() == [req]
    timeline = data.talker_model_inputs["timeline"]
    feedback = data.talker_model_inputs["decode_feedback"]
    rows = timeline.user_rows.shape[0]

    assert lookahead_eligible()
    assert timeline.input_position(feedback.launched_frames) <= rows - 1
    runner.before_decode(None, None, [request], is_lookahead=True)
    host_rows = DeferredHostRows()
    runner.next_host_staging = lambda shape, dtype: host_rows
    frames = runner.post_decode_launch(
        SimpleNamespace(next_token_ids=torch.tensor([72])), None, [request]
    )
    assert not bridge.should_park(req)
    assert not lookahead_eligible()
    host_rows.synchronize()
    runner.post_decode_resolve(
        frames,
        SimpleNamespace(next_token_ids=torch.tensor([72])),
        None,
        None,
        [request],
    )
    req.output_ids.append(72)
    assert timeline.input_position(feedback.launched_frames) == rows - 1
    runner.before_decode(None, None, [request])
    runner.post_decode(
        SimpleNamespace(next_token_ids=torch.tensor([73])), None, None, [request]
    )
    req.output_ids.append(73)
    assert timeline.input_position(feedback.launched_frames) == rows
    assert bridge.should_park(req)
    [(_, result)] = bridge.take_stepped_results(bridge.sessions[CALL.id])
    assert PersonaPlexState.from_dict(result.data).codes.shape[0] == 2


def test_closing_a_session_whose_request_took_every_step_frees_it_before_another():
    bridge, engine = open_bridge()
    runner = make_runner(FakeModel())
    request, req = run_first_unit(bridge, runner)
    session = bridge.sessions[CALL.id]
    unit, payload = accept(bridge, 1, caller_codes(2)[1:])
    bridge.feed(unit, payload)
    assert bridge.take_resumable() == [req]
    runner.before_decode(None, None, [request])
    runner.post_decode(
        SimpleNamespace(next_token_ids=torch.tensor([72])), None, None, [request]
    )
    req.output_ids.append(72)
    assert bridge.should_park(req)

    bridge.close_streaming_session(session)

    engine.resolve_pending_async.assert_called()
    engine.release_request_kv_cache.assert_called_once_with(req)
    engine.abort.assert_not_called()
    assert req.finished()
