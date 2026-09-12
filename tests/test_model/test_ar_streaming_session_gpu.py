# SPDX-License-Identifier: Apache-2.0
"""Verify physical KV reuse, continuation parity, and allocator cleanup."""
from __future__ import annotations

import os
from array import array
from dataclasses import asdict

import pytest
import torch

pytestmark = [pytest.mark.accelerator, pytest.mark.benchmark]


def test_real_streaming_session_gpu():
    """Requires one CUDA GPU and a local small causal LM for KV/parity checks."""
    model = os.environ.get("OMNI_SESSION_TEST_MODEL")
    if not model:
        pytest.skip("set OMNI_SESSION_TEST_MODEL to a local small causal LM")
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")

    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.sampling.sampling_params import SamplingParams
    from sglang.srt.server_args import ServerArgs

    from sglang_omni.model_runner.base import ModelRunner
    from sglang_omni.proto import OmniRequest, StagePayload
    from sglang_omni.proto.session import SessionRef, TimedChunk
    from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
    from sglang_omni.scheduling.omni_scheduler import OmniScheduler
    from sglang_omni.scheduling.sglang_backend import (
        SGLangARRequestData,
        SGLangOutputProcessor,
    )

    args = ServerArgs(
        model_path=model,
        tokenizer_path=model,
        dtype="bfloat16",
        device="cuda",
        tp_size=1,
        max_running_requests=8,
        max_total_tokens=1024,
        context_length=512,
        mem_fraction_static=0.1,
        disable_cuda_graph=True,
        disable_radix_cache=True,
        enable_streaming_session=True,
        chunked_prefill_size=-1,
        attention_backend="triton",
        sampling_backend="pytorch",
        disable_overlap_schedule=True,
    )
    worker, cache, req_pool, kv_pool, config = create_sglang_infrastructure(args, 0)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    serial = 0

    def make_data(p):
        params = SamplingParams(
            temperature=0, max_new_tokens=p.request.params.get("n", 8), ignore_eos=True
        )
        params.normalize(tokenizer)
        req = Req(
            p.request_id, None, array("q", p.data), params, vocab_size=config.vocab_size
        )
        req.tokenizer = tokenizer
        return SGLangARRequestData(req=req, stage_payload=p)

    def result(d):
        d.stage_payload.data = list(d.output_ids)
        return d.stage_payload

    from sglang_omni.scheduling.sglang_backend.ar_session import ARSessionAdapter

    class Adapter(ARSessionAdapter):
        def build(self, ref, chunk, p):
            return make_data(p)

        def result(self, ref, d):
            return result(d)

        def stream(self, ref, d, output):
            return ()

    scheduler = OmniScheduler(
        worker,
        cache,
        req_pool,
        kv_pool,
        args,
        config,
        model_runner=ModelRunner(worker, SGLangOutputProcessor()),
        request_builder=make_data,
        result_adapter=result,
        session_adapter=Adapter(),
        enable_async_decode=True,
    )
    b = scheduler._session_bridge

    def p(op=None, sid="s", ids=None, n=8, epoch=0):
        nonlocal serial
        serial += 1
        meta = (
            {}
            if op is None
            else {
                "omni_session": {
                    "op": op,
                    "ref": asdict(SessionRef(sid, epoch=epoch)),
                    "stages": ["ar"],
                    "chunk": asdict(TimedChunk("text", 0, 0, serial, ids)),
                    "output_limits": {"chunks": 64, "bytes": 4096},
                }
            }
        )
        return StagePayload(
            str(serial), OmniRequest(inputs=None, params={"n": n}, metadata=meta), ids
        )

    def submit(payload):
        scheduler.process_input_requests([payload])
        return payload.request_id

    def step():
        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if batch:
            output = scheduler.run_batch(batch)
            scheduler.process_batch_result(batch, output)
        scheduler.last_batch = batch
        return batch

    def collect(rids):
        outputs = {}
        for _ in range(1000):
            while not scheduler.outbox.empty():
                msg = scheduler.outbox.get_nowait()
                if msg.type == "error":
                    raise AssertionError(f"{msg.request_id}: {msg.data}")
                if msg.type == "result" and msg.request_id in rids:
                    outputs[msg.request_id] = msg.data.data
            if all(rid in outputs for rid in rids):
                return outputs
            step()
        raise AssertionError("scheduler did not finish")

    def lifecycle(op, sid="s", epoch=0):
        rid = submit(p(op, sid, epoch=epoch))
        return collect([rid])[rid]

    def generate(ids, sid=None, n=8, epoch=0):
        rid = submit(p("append" if sid else None, sid or "s", ids, n, epoch))
        return collect([rid])[rid]

    def free():
        torch.cuda.synchronize()
        return len(req_pool.free_slots), kv_pool.available_size()

    baseline = free()
    print(
        "CONFIG",
        model,
        args.dtype,
        args.attention_backend,
        "radix_disabled",
        args.disable_radix_cache,
        "BASELINE",
        baseline,
        flush=True,
    )
    prompt = tokenizer.encode("The capital of France is", add_special_tokens=False)
    tail = tokenizer.encode(". The capital of Italy is", add_special_tokens=False)
    try:
        lifecycle("open")
        first = generate(prompt, "s")
        slot = scheduler.tree_cache.slots[b.native_id(SessionRef("s"))]
        first_slot, retained = slot.req_pool_idx, slot.kv_committed_len
        assert first_slot is not None and retained >= len(prompt)
        assert b.usage(SessionRef("s")).bytes > 0
        prior = free()
        previous_limit = scheduler.max_req_len
        scheduler.max_req_len = len(prompt)
        rejected = p("append", "s", tail)
        submit(rejected)
        error = scheduler.outbox.get_nowait()
        assert error.request_id == rejected.request_id and error.type == "error"
        assert "KV cache can hold" in str(error.data)
        assert rejected.request_id not in b.requests
        assert not scheduler.session_controller.get(
            b.native_id(SessionRef("s"))
        )._inflight
        assert scheduler.tree_cache.slots[b.native_id(SessionRef("s"))] is slot
        assert slot.req_pool_idx == first_slot and slot.kv_committed_len == retained
        assert free() == prior
        scheduler.max_req_len = previous_limit
        print(
            "PRE_ENQUEUE_REJECT_RETAINS_PREFIX", first_slot, retained, prior, flush=True
        )
        second_req = p("append", "s", tail)
        submit(second_req)
        req = b.requests[second_req.request_id].req
        assert list(req.origin_input_ids) == prompt + first + tail
        batch = scheduler.get_next_batch_to_run()
        assert req.req_pool_idx == first_slot and len(req.prefix_indices) >= retained
        scheduler.cur_batch = batch
        scheduler.process_batch_result(batch, scheduler.run_batch(batch))
        scheduler.last_batch = batch
        second = collect([second_req.request_id])[second_req.request_id]
        oracle = generate(prompt + first + tail)
        assert second == oracle
        print("RETAIN_REUSE_EXACT", first_slot, retained, second, free(), flush=True)

        prior = free()
        queued = p("append", "s", tail)
        submit(queued)
        scheduler.abort(queued.request_id)
        assert free() == prior
        assert (
            scheduler.tree_cache.slots[b.native_id(SessionRef("s"))].req_pool_idx
            == first_slot
        )
        history = prompt + first + tail + second
        assert generate(tail, "s") == generate(history + tail)
        history += tail + generate(history + tail)

        lifecycle("open", "t")
        tfirst = generate(prompt, "t")
        left = p("append", "t", tail, 12)
        middle = p("append", "s", tail, 12)
        right = p(None, ids=prompt, n=12)
        for item in (left, middle, right):
            submit(item)
        prefill = step()
        assert [r.rid for r in prefill.reqs] == [
            left.request_id,
            middle.request_id,
            right.request_id,
        ]
        decode = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = decode
        assert decode.forward_mode.is_decode() and len(decode.reqs) == 3
        out, pending = scheduler._run_batch_launch(decode)
        scheduler._async_pending = (decode.copy(), out, pending)
        scheduler.last_batch = decode
        print("GPU_LAUNCHED_CANCEL_MIDDLE", [r.rid for r in decode.reqs], flush=True)
        # Note (Junnan Li): Session cancellation waits for unit completion;
        # request abort would exercise a different KV lifetime.
        scheduler._resolve_pending_async()
        survivors = collect([left.request_id, middle.request_id, right.request_id])
        committed_slot = scheduler.tree_cache.slots["s"]
        committed_length = committed_slot.kv_committed_len
        committed_pool = committed_slot.req_pool_idx
        lifecycle("abort", "s", epoch=1)
        assert scheduler._async_pending is None
        assert scheduler.tree_cache.slots["s"] is committed_slot
        assert committed_slot.kv_committed_len == committed_length
        assert committed_slot.req_pool_idx == committed_pool
        left_oracle = generate(prompt + tfirst + tail, n=12)
        right_oracle = generate(prompt, n=12)
        assert survivors[left.request_id] == left_oracle
        assert survivors[right.request_id] == right_oracle
        history += tail + survivors[middle.request_id]
        continuation = p("append", "s", tail, epoch=1)
        submit(continuation)
        req = b.requests[continuation.request_id].req
        batch = scheduler.get_next_batch_to_run()
        assert len(req.prefix_indices) >= committed_length
        assert req.req_pool_idx == committed_pool
        matched = len(req.prefix_indices)
        scheduler.cur_batch = batch
        scheduler.process_batch_result(batch, scheduler.run_batch(batch))
        scheduler.last_batch = batch
        answer = collect([continuation.request_id])[continuation.request_id]
        assert answer == generate(history + tail)
        print(
            "CANCEL_BOUNDARY_RETAINS_KV_EXACT",
            committed_length,
            matched,
            committed_pool,
            flush=True,
        )

        lifecycle("close", "t")
        active = p("append", "s", tail, 20, 1)
        submit(active)
        step()
        lifecycle("close", "s", epoch=1)
        assert free() == baseline
        # Note (Junnan Li): Native completion can precede collection of an
        # already-launched overrun row, leaving device work for close or append.
        for finish_action in ("close", "append"):
            lifecycle("open")
            short = p("append", "s", prompt, 3)
            long = p(None, ids=prompt, n=12)
            submit(short)
            submit(long)
            step()
            for _ in range(3):
                current = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = current
                out, pending = scheduler._run_batch_launch(current)
                previous = scheduler._async_pending
                scheduler._async_pending = (current.copy(), out, pending)
                if previous is not None:
                    scheduler._resolve_and_process(*previous)
                scheduler.last_batch = current
            owner = b.owners[b.native_id(SessionRef("s"))]
            assert owner.active is None and scheduler._async_pending is not None
            assert any(
                r.rid == short.request_id for r in scheduler._async_pending[0].reqs
            )
            if finish_action == "close":
                lifecycle("close")
            else:
                next_unit = p("append", "s", tail)
                submit(next_unit)
                assert scheduler._async_pending is None
                collect([next_unit.request_id])
                lifecycle("close")
            assert scheduler._async_pending is None
            collect([long.request_id])
            assert free() == baseline
        print("COMPLETED_LOOKAHEAD_CLOSE_APPEND_SAFE", free(), flush=True)
        for _ in range(5):
            lifecycle("open")
            generate(prompt, "s")
            lifecycle("close")
            assert free() == baseline
        print("CLOSE_CYCLES_ZERO_HELD", free(), flush=True)
        for i in range(8):
            lifecycle("open", str(i))
        rejected = submit(p("open", "overflow"))
        msg = scheduler.outbox.get_nowait()
        assert msg.request_id == rejected and msg.type == "error"
        for i in range(8):
            lifecycle("close", str(i))
        assert free() == baseline
        for i in range(8):
            lifecycle("open", str(i))
        for i in range(7):
            generate(prompt, str(i))
        held_free = free()
        rejected = submit(p("append", "7", prompt))
        msg = scheduler.outbox.get_nowait()
        assert msg.request_id == rejected and msg.type == "error"
        assert free() == held_free
        lifecycle("close", "0")
        generate(prompt, "7")
        for i in range(1, 8):
            lifecycle("close", str(i))
        assert free() == baseline
        print("PHYSICAL_SLOT_EXHAUSTION_RETURN", held_free, free(), flush=True)

        for i in range(4):
            lifecycle("open", str(i))
        for i in range(3):
            generate([1] * 300, str(i))
        held_free = free()
        rejected = submit(p("append", "3", [1] * 300))
        msg = scheduler.outbox.get_nowait()
        assert msg.request_id == rejected and msg.type == "error"
        assert free() == held_free
        for i in range(4):
            lifecycle("close", str(i))
        assert free() == baseline
        print("PHYSICAL_KV_EXHAUSTION_RETURN", held_free, free(), flush=True)
        lifecycle("open", "idle")
        generate(prompt, "idle")
        lifecycle("open", "active")
        active = p("append", "active", prompt, 20)
        submit(active)
        step()
        current = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = current
        out, pending = scheduler._run_batch_launch(current)
        scheduler._async_pending = (current.copy(), out, pending)
        scheduler.last_batch = current
        scheduler.stop()
        assert scheduler._async_pending is None
        assert not b.owners and not scheduler.session_controller.sessions
        assert not scheduler.tree_cache.slots and free() == baseline
        print("SHUTDOWN_IDLE_AND_GPU_INFLIGHT_ZERO_HELD", free(), flush=True)
        print("GPU_GATE_PASS", free(), flush=True)
    finally:
        b.shutdown()
        torch.distributed.destroy_process_group()
