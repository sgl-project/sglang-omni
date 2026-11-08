# SPDX-License-Identifier: Apache-2.0
"""A real tiny Qwen3 engine for the speculative session integration test."""

from collections.abc import Iterable

import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.profiler.event_recorder_types import (
    reset_active_stage,
    set_active_stage,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import (
    SESSION_METADATA_KEY,
    SessionIdentity,
    SessionOperation,
    TimedChunk,
)
from sglang_omni.scheduling.bootstrap import create_sglang_infrastructure
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.ar_session import (
    ARSessionAdapter,
    ARSessionPreparation,
)
from sglang_omni.scheduling.sglang_backend.history_cut import DraftSessionKV, HistoryCut
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData
from sglang_omni.scheduling.sglang_backend.server_args_builder import (
    build_sglang_server_args,
)
from sglang_omni.scheduling.types import RequestOutput


class TinyTextAdapter(ARSessionAdapter):
    def __init__(self, target_kv: DraftSessionKV) -> None:
        self.target_kv = target_kv
        self.next_cut: HistoryCut | None = None
        self.next_preparation = ARSessionPreparation()

    def build(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> SGLangARRequestData:
        sampling = SamplingParams(
            temperature=0,
            top_k=1,
            max_new_tokens=payload.request.params["max_new_tokens"],
            ignore_eos=True,
        )
        sampling.normalize(None)
        request = Req(
            payload.request_id, "", payload.data["token_ids"], sampling, vocab_size=32
        )
        request.tokenizer = None
        return SGLangARRequestData(req=request, stage_payload=payload)

    def prepare_unit(
        self,
        session_identity: SessionIdentity,
        chunk: TimedChunk,
        payload: StagePayload,
    ) -> ARSessionPreparation:
        preparation = self.next_preparation
        self.next_preparation = ARSessionPreparation()
        return preparation

    def plan_history_cut(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        history_length: int,
        history_capacity: int,
    ) -> HistoryCut | None:
        cut = self.next_cut
        self.next_cut = None
        return cut

    def shift_key_positions(
        self, keys: torch.Tensor, distances: torch.Tensor
    ) -> torch.Tensor:
        return self.target_kv.shift_key_positions(keys, distances)

    def result(
        self, session_identity: SessionIdentity, request_data: SGLangARRequestData
    ) -> StagePayload:
        payload = request_data.stage_payload
        return StagePayload(
            payload.request_id,
            payload.request,
            {"output_ids": list(request_data.output_ids)},
        )

    def stream(
        self,
        session_identity: SessionIdentity,
        request_data: SGLangARRequestData,
        output: RequestOutput,
    ) -> Iterable[TimedChunk]:
        return (
            TimedChunk(
                "text", 0, 0, 0, {"text": str(output.data)}, format="text_delta"
            ),
        )


def session_payload(
    session_id: str, request_id: str, token_ids: list[int], output_tokens: int
) -> StagePayload:
    operation = SessionOperation(
        "append",
        SessionIdentity(session_id),
        ("tiny",),
        TimedChunk("text", 0, 0, 0, {}, format="text_delta"),
    )
    return StagePayload(
        request_id,
        OmniRequest(
            None,
            params={"max_new_tokens": output_tokens},
            metadata={SESSION_METADATA_KEY: operation.to_dict()},
        ),
        {"token_ids": token_ids},
    )


def run_tiny_sessions(
    target_path: str, draft_path: str | None, captured: bool
) -> dict[str, list[int] | int]:
    server_args = build_sglang_server_args(
        target_path,
        context_length=128,
        max_running_requests=4,
        max_total_tokens=256,
        max_prefill_tokens=128,
        chunked_prefill_size=-1,
        mem_fraction_static=0.1,
        dtype="bfloat16",
        attention_backend="fa3",
        sampling_backend="pytorch",
        disable_radix_cache=True,
        enable_streaming_session=True,
        enable_mixed_chunk=False,
        enable_unified_memory=False,
        page_size=1,
        skip_tokenizer_init=True,
        disable_cuda_graph=not captured,
        cuda_graph_bs=[1, 2],
        cuda_graph_max_bs=2,
        enable_torch_compile=False,
        speculative_algorithm="DFLASH" if draft_path else None,
        speculative_draft_model_path=draft_path,
        speculative_num_draft_tokens=4 if draft_path else None,
    )
    worker, cache, request_pool, allocator, model_config = create_sglang_infrastructure(
        server_args, 0, supports_speculative_decoding=True
    )
    adapter = TinyTextAdapter(
        DraftSessionKV(
            allocator.get_kvcache(),
            worker.model_runner.model.model.layers[
                0
            ].self_attn.rotary_emb.cos_sin_cache,
        )
    )
    scheduler = OmniScheduler(
        worker,
        cache,
        request_pool,
        allocator,
        server_args,
        model_config,
        model_runner=ModelRunner(worker, SGLangOutputProcessor()),
        session_adapter=adapter,
        enable_async_decode=False,
    )
    initial_free = allocator.available_size()
    initial_free_request_rows = request_pool.available_size()
    outputs: dict[str, list[int] | int] = {}
    streamed: dict[str, list[int]] = {}
    graph_replays = 0

    def step() -> None:
        nonlocal graph_replays
        scheduler.process_input_requests([])
        batch = scheduler.get_next_batch_to_run()
        scheduler.cur_batch = batch
        if batch is not None:
            result = scheduler.run_batch(batch)
            graph_replays += int(result.can_run_cuda_graph)
            scheduler.process_batch_result(batch, result)
        else:
            pass
        scheduler.last_batch = batch

    def drain() -> None:
        while not scheduler.outbox.empty():
            message = scheduler.outbox.get_nowait()
            if message.type == "error":
                raise AssertionError(message.data)
            elif message.type == "result":
                outputs[message.request_id] = message.data.data["output_ids"]
            elif message.type == "stream":
                streamed.setdefault(message.request_id, []).append(
                    int(message.data["payload"]["text"])
                )
            else:
                pass

    def finish(payloads: list[StagePayload]) -> None:
        scheduler.process_input_requests(payloads)
        for _ in range(100):
            step()
            drain()
            if all(payload.request_id in outputs for payload in payloads):
                return
            else:
                pass
        raise AssertionError("session append did not complete")

    stage_token = set_active_stage("tiny")
    with torch.inference_mode():
        for session_id in ("a", "b"):
            scheduler.session_bridge.apply_operation(
                StagePayload("open", OmniRequest(None), {}),
                SessionOperation("open", SessionIdentity(session_id), ("tiny",)),
            )
        finish(
            [
                session_payload("a", "a0", [1, 2, 3, 4], 7),
                session_payload("b", "b0", [5, 6, 7], 7),
            ]
        )
        finish(
            [
                session_payload("a", "a1", [8, 9], 5),
                session_payload("b", "b1", [10, 11], 5),
            ]
        )
        adapter.next_cut = HistoryCut(((1, 3),))
        finish([session_payload("a", "cut", [12, 13], 5)])
        adapter.next_preparation = ARSessionPreparation(kept_token_count=4)
        finish([session_payload("a", "rewind", [14], 5)])
        scheduler.process_input_requests([session_payload("b", "cancel", [15, 16], 30)])
        step()
        scheduler.session_bridge.cancel("cancel")
        drain()
        finish([session_payload("b", "after_cancel", [17], 5)])
        adapter.next_preparation = ARSessionPreparation(reset_history=True)
        finish([session_payload("a", "reset", [18, 19], 5)])
        for session in list(scheduler.session_bridge.sessions.values()):
            scheduler.session_bridge.close_session(session)
        for request_id, token_ids in outputs.items():
            assert streamed[request_id] == token_ids
        assert not cache.session_records()
        assert allocator.available_size() == initial_free
        assert request_pool.available_size() == initial_free_request_rows
        if captured:
            assert graph_replays > 0
        else:
            pass
    scheduler.stop()
    reset_active_stage(stage_token)
    outputs["graph_replays"] = graph_replays
    return outputs
