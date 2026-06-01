# SPDX-License-Identifier: Apache-2.0
"""Unit tests for _CodePredictorWrapper and _CodePredictorStreamingExecutor."""

from __future__ import annotations

import asyncio
import contextlib
from dataclasses import dataclass
from typing import Any

import pytest
import torch
from torch import nn
from transformers.models.qwen3_omni_moe.configuration_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorConfig,
)
from transformers.models.qwen3_omni_moe.modeling_qwen3_omni_moe import (
    Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration,
)

from sglang_omni.models.qwen3_omni.components.code_predictor_executor import (
    _CodePredictorStreamingExecutor,
    _CodePredictorWrapper,
    _LightweightTalkerShell,
    _PredictRequest,
)
from sglang_omni.models.qwen3_omni.pipeline.next_stage import (
    CODE2WAV_STAGE,
    TALKER_AR_STAGE,
)

_TINY_HIDDEN_SIZE = 64
_TINY_NUM_HEADS = 4
_TINY_NUM_KV_HEADS = 2
_TINY_NUM_LAYERS = 2
_TINY_INTERMEDIATE = 128
_TINY_VOCAB = 256
_TINY_NUM_CODE_GROUPS = 4


@pytest.fixture(scope="module")
def tiny_predictor_config() -> Qwen3OmniMoeTalkerCodePredictorConfig:
    return Qwen3OmniMoeTalkerCodePredictorConfig(
        vocab_size=_TINY_VOCAB,
        hidden_size=_TINY_HIDDEN_SIZE,
        intermediate_size=_TINY_INTERMEDIATE,
        num_hidden_layers=_TINY_NUM_LAYERS,
        num_attention_heads=_TINY_NUM_HEADS,
        num_key_value_heads=_TINY_NUM_KV_HEADS,
        num_code_groups=_TINY_NUM_CODE_GROUPS,
        max_position_embeddings=32,
        rms_norm_eps=1e-6,
    )


@pytest.fixture(scope="module")
def tiny_talker_shell(
    tiny_predictor_config: Qwen3OmniMoeTalkerCodePredictorConfig,
) -> _LightweightTalkerShell:
    torch.manual_seed(0)

    predictor = (
        Qwen3OmniMoeTalkerCodePredictorModelForConditionalGeneration._from_config(
            tiny_predictor_config,
        )
    )
    predictor = predictor.to(dtype=torch.float32).eval()

    codec_embedding = nn.Embedding(_TINY_VOCAB, _TINY_HIDDEN_SIZE).eval()

    @dataclass
    class _TalkerCfgStub:
        num_code_groups: int

    shell = _LightweightTalkerShell(
        config=_TalkerCfgStub(num_code_groups=_TINY_NUM_CODE_GROUPS),
        code_predictor=predictor,
        codec_embedding=codec_embedding,
    )
    shell.eval()
    return shell


@pytest.fixture(scope="module")
def tiny_wrapper(tiny_talker_shell: _LightweightTalkerShell) -> _CodePredictorWrapper:
    return _CodePredictorWrapper(tiny_talker_shell)


def _random_chunk_inputs(
    num_requests: int,
    hidden_size: int,
    vocab: int,
    seed: int = 42,
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    g = torch.Generator().manual_seed(seed)
    inputs: list[tuple[torch.Tensor, torch.Tensor]] = []
    for _ in range(num_requests):
        h = torch.randn(hidden_size, generator=g)
        c = torch.randint(0, vocab, (), dtype=torch.long, generator=g)
        inputs.append((h, c))
    return inputs


@pytest.mark.parametrize("num_requests", [1, 2, 4, 8])
def test_batched_wrapper_matches_single_request_path(
    tiny_wrapper: _CodePredictorWrapper,
    num_requests: int,
) -> None:
    inputs = _random_chunk_inputs(
        num_requests,
        _TINY_HIDDEN_SIZE,
        _TINY_VOCAB,
    )

    singles: list[dict[str, torch.Tensor]] = []
    for h, c in inputs:
        with torch.no_grad():
            out = tiny_wrapper(
                talker_hidden=h.unsqueeze(0),
                layer0_code=c.unsqueeze(0),
            )
        singles.append(
            {
                "codes": out["codes"][0].clone(),
                "summed_embeddings": out["summed_embeddings"][0].clone(),
            }
        )

    talker_hidden = torch.stack([h for h, _ in inputs], dim=0)
    layer0_code = torch.stack([c for _, c in inputs], dim=0)
    with torch.no_grad():
        batched = tiny_wrapper(
            talker_hidden=talker_hidden,
            layer0_code=layer0_code,
        )

    for i in range(num_requests):
        assert torch.equal(
            batched["codes"][i], singles[i]["codes"]
        ), f"request {i}: codes mismatch"
        assert torch.allclose(
            batched["summed_embeddings"][i],
            singles[i]["summed_embeddings"],
            atol=1e-5,
            rtol=1e-5,
        ), f"request {i}: summed_embeddings mismatch"


def test_wrapper_output_shapes_and_dtypes(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    h = torch.randn(3, _TINY_HIDDEN_SIZE)
    c = torch.randint(0, _TINY_VOCAB, (3,), dtype=torch.long)
    with torch.no_grad():
        out = tiny_wrapper(talker_hidden=h, layer0_code=c)

    assert out["codes"].shape == (3, _TINY_NUM_CODE_GROUPS)
    assert out["codes"].dtype == torch.long
    assert out["summed_embeddings"].shape == (3, _TINY_HIDDEN_SIZE)
    assert out["summed_embeddings"].dtype == torch.float32


def test_wrapper_first_code_is_layer0_code(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    h = torch.randn(5, _TINY_HIDDEN_SIZE)
    c = torch.randint(0, _TINY_VOCAB, (5,), dtype=torch.long)
    with torch.no_grad():
        out = tiny_wrapper(talker_hidden=h, layer0_code=c)
    assert torch.equal(out["codes"][:, 0], c)


class _FakeStreamItem:
    def __init__(self, data: torch.Tensor, codec_code: int) -> None:
        self.data = data
        self.metadata = {"codec_code": codec_code}


class _FakeStreamQueue:
    def __init__(self) -> None:
        self._queues: dict[str, asyncio.Queue[_FakeStreamItem | None]] = {}

    def open(self, request_id: str) -> None:
        self._queues.setdefault(request_id, asyncio.Queue())

    def put(self, request_id: str, item: _FakeStreamItem) -> None:
        self._queues.setdefault(request_id, asyncio.Queue()).put_nowait(item)

    def put_done(self, request_id: str) -> None:
        self._queues.setdefault(request_id, asyncio.Queue()).put_nowait(None)

    async def get(self, request_id: str) -> _FakeStreamItem | None:
        return await self._queues[request_id].get()

    def close(self, request_id: str) -> None:
        q = self._queues.get(request_id)
        if q is not None:
            q.put_nowait(None)


@pytest.mark.asyncio
async def test_abort_stops_dispatch(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    exe = _CodePredictorStreamingExecutor(
        model=tiny_wrapper,
        device="cpu",
        max_batch_size=4,
    )
    stream_queue = _FakeStreamQueue()
    exe._stream_queue = stream_queue

    dispatched: list[tuple[str, Any]] = []
    first_chunk_done = asyncio.Event()

    def capture(request_id: str, tensor: torch.Tensor, *, target_stage: str) -> None:
        dispatched.append((request_id, target_stage))
        if len(dispatched) >= 2:
            first_chunk_done.set()

    exe.set_stream_fn(capture)

    rid = "req-abort"
    stream_queue.open(rid)

    from sglang_omni.proto import OmniRequest, StagePayload

    payload = StagePayload(
        request_id=rid,
        request=OmniRequest(inputs=""),
        data=None,
    )

    task = asyncio.create_task(exe.add_request(payload))

    stream_queue.put(
        rid, _FakeStreamItem(data=torch.randn(_TINY_HIDDEN_SIZE), codec_code=1)
    )

    await asyncio.wait_for(first_chunk_done.wait(), timeout=2.0)
    assert len(dispatched) == 2, (
        f"expected 2 dispatches (TALKER_AR + CODE2WAV) before abort, got "
        f"{len(dispatched)}"
    )
    assert {ts for _, ts in dispatched} == {TALKER_AR_STAGE, CODE2WAV_STAGE}

    count_pre_abort = len(dispatched)

    await exe.abort(rid)

    stream_queue.put(
        rid, _FakeStreamItem(data=torch.randn(_TINY_HIDDEN_SIZE), codec_code=2)
    )
    stream_queue.put_done(rid)

    try:
        await asyncio.wait_for(task, timeout=2.0)
    except asyncio.TimeoutError:
        task.cancel()
        raise AssertionError("add_request did not return after abort")
    except asyncio.CancelledError:
        pass

    assert len(dispatched) == count_pre_abort, (
        f"post-abort dispatches leaked: {len(dispatched) - count_pre_abort} "
        f"new entries after abort"
    )
    assert all(r == rid for r, _ in dispatched)


async def _drain_batch_loop(exe: _CodePredictorStreamingExecutor) -> None:
    if exe._batch_loop_task is None:
        return
    exe._batch_loop_task.cancel()
    try:
        await exe._batch_loop_task
    except asyncio.CancelledError:
        pass


@pytest.mark.asyncio
async def test_mixed_batch_live_plus_aborted(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    exe = _CodePredictorStreamingExecutor(
        model=tiny_wrapper,
        device="cpu",
        max_batch_size=4,
    )
    try:
        exe._ensure_batch_loop()
        loop = asyncio.get_running_loop()
        live = _PredictRequest(
            request_id="live",
            talker_hidden=torch.randn(_TINY_HIDDEN_SIZE),
            layer0_code=torch.tensor(1, dtype=torch.long),
            result_future=loop.create_future(),
        )
        dead = _PredictRequest(
            request_id="dead",
            talker_hidden=torch.randn(_TINY_HIDDEN_SIZE),
            layer0_code=torch.tensor(2, dtype=torch.long),
            result_future=loop.create_future(),
        )
        exe._aborted.add("dead")
        exe._pending.put_nowait(live)
        exe._pending.put_nowait(dead)

        out = await asyncio.wait_for(live.result_future, timeout=5.0)
        with pytest.raises(asyncio.CancelledError):
            await dead.result_future

        assert out["codes"].shape == (_TINY_NUM_CODE_GROUPS,)
        assert out["summed_embeddings"].shape == (_TINY_HIDDEN_SIZE,)
    finally:
        await _drain_batch_loop(exe)


@pytest.mark.asyncio
async def test_abort_during_in_flight_forward(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    exe = _CodePredictorStreamingExecutor(
        model=tiny_wrapper,
        device="cpu",
        max_batch_size=4,
    )
    stream_queue = _FakeStreamQueue()
    exe._stream_queue = stream_queue

    dispatched: list[tuple[str, Any]] = []

    def capture(request_id: str, tensor: torch.Tensor, *, target_stage: str) -> None:
        dispatched.append((request_id, target_stage))

    exe.set_stream_fn(capture)

    rid = "in-flight-abort"
    stream_queue.open(rid)
    payload = type("P", (), {"request_id": rid, "request": None, "data": None})()

    exe._aborted.add(rid)
    exe._dispatch_outputs(
        rid,
        {
            "codes": torch.zeros(_TINY_NUM_CODE_GROUPS, dtype=torch.long),
            "summed_embeddings": torch.zeros(_TINY_HIDDEN_SIZE),
        },
    )

    assert dispatched == [], f"_dispatch_outputs leaked after abort: {dispatched}"
    assert rid in exe._aborted, (
        "_aborted is write-once per abort(); handlers only read, cleanup "
        "happens in the add_request done_callback"
    )


@pytest.mark.asyncio
async def test_stop_cancels_pending_and_tasks(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    """stop() must cancel pending futures, registered tasks, and the batch loop."""
    exe = _CodePredictorStreamingExecutor(
        model=tiny_wrapper,
        device="cpu",
        max_batch_size=4,
    )
    exe._ensure_batch_loop()
    batch_loop_task = exe._batch_loop_task
    assert batch_loop_task is not None

    loop = asyncio.get_running_loop()
    pending_req = _PredictRequest(
        request_id="pending",
        talker_hidden=torch.randn(_TINY_HIDDEN_SIZE),
        layer0_code=torch.tensor(0, dtype=torch.long),
        result_future=loop.create_future(),
    )

    async def fake_run(payload: Any) -> None:
        await asyncio.sleep(60)

    fake_payload = type("P", (), {"request_id": "inflight"})()
    inflight_task = asyncio.create_task(fake_run(fake_payload))
    exe._request_tasks["inflight"] = inflight_task
    exe._pending.put_nowait(pending_req)

    await exe.stop()

    with contextlib.suppress(asyncio.CancelledError):
        await inflight_task

    assert (
        pending_req.result_future.cancelled()
    ), "pending request's future should be cancelled by stop()"
    assert inflight_task.cancelled(), "tracked run_task should be cancelled by stop()"
    assert exe._request_tasks == {}
    assert batch_loop_task.done(), "batch loop task should be done after stop()"


@pytest.mark.asyncio
async def test_forward_exception_propagates_and_loop_survives(
    tiny_wrapper: _CodePredictorWrapper,
) -> None:
    """If _run_batch raises, all live futures get the exception and the loop recovers."""
    exe = _CodePredictorStreamingExecutor(
        model=tiny_wrapper,
        device="cpu",
        max_batch_size=4,
    )
    try:
        exe._ensure_batch_loop()

        original_run_batch = exe._run_batch
        call_count = {"n": 0}

        def flaky_run_batch(batch: list[_PredictRequest]):
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise RuntimeError("simulated forward failure")
            return original_run_batch(batch)

        exe._run_batch = flaky_run_batch

        loop = asyncio.get_running_loop()
        first = _PredictRequest(
            request_id="first",
            talker_hidden=torch.randn(_TINY_HIDDEN_SIZE),
            layer0_code=torch.tensor(1, dtype=torch.long),
            result_future=loop.create_future(),
        )
        exe._pending.put_nowait(first)

        with pytest.raises(RuntimeError, match="simulated forward failure"):
            await asyncio.wait_for(first.result_future, timeout=5.0)

        assert exe._batch_loop_task is not None
        assert (
            not exe._batch_loop_task.done()
        ), "batch loop should survive a single forward exception"

        second = _PredictRequest(
            request_id="second",
            talker_hidden=torch.randn(_TINY_HIDDEN_SIZE),
            layer0_code=torch.tensor(2, dtype=torch.long),
            result_future=loop.create_future(),
        )
        exe._pending.put_nowait(second)

        out = await asyncio.wait_for(second.result_future, timeout=5.0)
        assert out["codes"].shape == (_TINY_NUM_CODE_GROUPS,)
    finally:
        await _drain_batch_loop(exe)
