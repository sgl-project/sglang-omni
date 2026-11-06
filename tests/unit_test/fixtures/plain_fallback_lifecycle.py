# SPDX-License-Identifier: Apache-2.0
"""Model-backed probes for plain decoding and draft cache isolation."""

from __future__ import annotations

import json
import re
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Literal, TypedDict
from unittest.mock import patch

import pytest
import torch
from sglang.srt.runtime_context import get_spec

from sglang_omni.proto.request import OmniRequest, StagePayload
from tests.unit_test.fixtures.plain_fallback import create_plain_fallback_scheduler


class BatchProbeMeasurement(TypedDict):
    decode_path: Literal["plain", "spec"]
    requests: str
    mode: str
    forced_eager: int
    cached_tokens: int
    sampler: int
    draft_forward: int
    draft_kv_write: int
    verify: int
    plain_graph_replay: int
    verify_graph_replay: int


def probe_plain_fallback(
    model_family: Literal["qwen3_asr", "moss_td"],
    *,
    enable_cuda_graph: bool,
    enable_radix_cache: bool,
) -> None:
    """Fallback requests must never forward the draft or write its KV pool."""
    scheduler = create_plain_fallback_scheduler(
        model_family,
        enable_cuda_graph=enable_cuda_graph,
        enable_radix_cache=enable_radix_cache,
        max_running_requests=4,
    )
    target_runner = scheduler.tp_worker.model_runner
    draft_runner = scheduler.draft_worker.draft_model_runner
    target_worker = scheduler.tp_worker.speculative_target_worker
    draft_cache_pool = draft_runner.token_to_kv_pool
    target_model_identity = id(target_runner.model)
    target_spec_algorithm = target_runner.spec_algorithm
    runtime_speculation_config = get_spec()
    initial_request_slot_count = scheduler.req_to_token_pool.available_size()
    initial_cache_slot_count = scheduler.token_to_kv_pool_allocator.available_size()
    audio_path = Path(__file__).parents[2] / "data" / "query_to_draw.wav"

    def build_payload(request_id: str, use_plain: bool) -> StagePayload:
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs={"audio_path": str(audio_path)},
                params={
                    "repetition_penalty": 1.2 if use_plain else 1.0,
                    "stream": True,
                },
            ),
            data=None,
        )

    batch_measurements: list[BatchProbeMeasurement] = []
    completed_request_ids: set[str] = set()
    streamed_request_ids: set[str] = set()
    transcript_texts: dict[str, str] = {}
    has_cancelled_request = False
    has_retracted_requests = False
    has_exercised_eager_fallback = False
    try:
        with (
            ExitStack() as graph_patches,
            torch.inference_mode(),
            patch.object(
                target_runner.sampler, "forward", wraps=target_runner.sampler.forward
            ) as sampler,
            patch.object(
                draft_runner, "forward", wraps=draft_runner.forward
            ) as draft_forward,
            patch.object(
                target_worker,
                "forward_batch_generation",
                wraps=target_worker.forward_batch_generation,
            ) as target_forward,
            patch.object(
                draft_cache_pool, "set_kv_buffer", wraps=draft_cache_pool.set_kv_buffer
            ) as draft_kv_write,
            patch.object(
                draft_cache_pool,
                "set_kv_buffer_prefix_valid",
                wraps=draft_cache_pool.set_kv_buffer_prefix_valid,
            ) as draft_prefix_write,
        ):
            if enable_cuda_graph:
                plain_decode_runner = target_runner.plain_decode_runner
                assert plain_decode_runner.model is target_runner.model
                assert (
                    plain_decode_runner.req_to_token_pool
                    is target_runner.req_to_token_pool
                )
                assert (
                    plain_decode_runner.token_to_kv_pool
                    is target_runner.token_to_kv_pool
                )
                assert (
                    plain_decode_runner.attn_backend is not target_runner.attn_backend
                )
                plain_graph_replay = graph_patches.enter_context(
                    patch.object(
                        plain_decode_runner.decode_cuda_graph_runner,
                        "execute",
                        wraps=plain_decode_runner.decode_cuda_graph_runner.execute,
                    )
                )
                verify_graph_replay = graph_patches.enter_context(
                    patch.object(
                        target_runner.decode_cuda_graph_runner,
                        "execute",
                        wraps=target_runner.decode_cuda_graph_runner.execute,
                    )
                )
            else:
                plain_graph_replay = None
                verify_graph_replay = None
            initial_requests = [
                build_payload("greedy", False),
                build_payload("sampled", True),
                build_payload("sampled-second", True),
                build_payload("cancel", True),
            ]
            expected_request_ids = {
                "greedy",
                "sampled",
                "sampled-second",
                "replacement",
            }
            has_finished_seed_request = not enable_radix_cache
            has_submitted_repeated_requests = False
            has_seen_first_speculative_prefill = False
            scheduler.process_input_requests(
                [build_payload("plain-seed", True)]
                if enable_radix_cache
                else initial_requests
            )
            deadline_seconds = time.monotonic() + 180
            while time.monotonic() < deadline_seconds:
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                if batch is not None:
                    decode_path = "plain" if batch.spec_algorithm.is_none() else "spec"
                    assert all(
                        request.omni_data.decode_path == decode_path
                        for request in batch.reqs
                    )
                    if enable_radix_cache and batch.forward_mode.is_extend():
                        for request in batch.reqs:
                            if (
                                request.rid == "greedy"
                                and not has_seen_first_speculative_prefill
                            ):
                                assert request.prefix_len == 0
                                has_seen_first_speculative_prefill = True
                            elif request.rid in {"plain-repeat", "spec-repeat"}:
                                assert request.prefix_len > 0
                            else:
                                pass
                    else:
                        pass
                    forward_counts_before_batch = (
                        sampler.call_count,
                        draft_forward.call_count,
                        draft_kv_write.call_count + draft_prefix_write.call_count,
                    )
                    target_forward.reset_mock()
                    if plain_graph_replay is not None:
                        plain_graph_replay.reset_mock()
                        verify_graph_replay.reset_mock()
                    else:
                        pass
                    should_force_eager = (
                        enable_cuda_graph
                        and decode_path == "plain"
                        and batch.forward_mode.is_decode()
                        and not has_exercised_eager_fallback
                    )
                    if should_force_eager:
                        with patch.object(
                            plain_decode_runner.decode_cuda_graph_runner,
                            "can_run_graph",
                            return_value=False,
                        ):
                            result = scheduler.run_batch(batch)
                        has_exercised_eager_fallback = True
                    else:
                        result = scheduler.run_batch(batch)
                    verification_count = sum(
                        bool(forward_call.kwargs.get("is_verify"))
                        for forward_call in target_forward.call_args_list
                    )
                    batch_measurement: BatchProbeMeasurement = {
                        "decode_path": decode_path,
                        "requests": ",".join(request.rid for request in batch.reqs),
                        "mode": batch.forward_mode.name,
                        "forced_eager": int(should_force_eager),
                        "cached_tokens": (
                            sum(request.prefix_len for request in batch.reqs)
                            if batch.forward_mode.is_extend()
                            else 0
                        ),
                        "sampler": sampler.call_count - forward_counts_before_batch[0],
                        "draft_forward": draft_forward.call_count
                        - forward_counts_before_batch[1],
                        "draft_kv_write": draft_kv_write.call_count
                        + draft_prefix_write.call_count
                        - forward_counts_before_batch[2],
                        "verify": verification_count,
                        "plain_graph_replay": (
                            0
                            if plain_graph_replay is None
                            else plain_graph_replay.call_count
                        ),
                        "verify_graph_replay": (
                            0
                            if verify_graph_replay is None
                            else verify_graph_replay.call_count
                        ),
                    }
                    batch_measurements.append(batch_measurement)
                    if decode_path == "plain":
                        assert batch_measurement["sampler"] > 0
                        assert all(
                            request.sampling_params.repetition_penalty == 1.2
                            for request in batch.reqs
                        )
                        assert (
                            batch_measurement["draft_forward"]
                            == batch_measurement["draft_kv_write"]
                            == batch_measurement["verify"]
                            == 0
                        )
                    else:
                        pass
                    if (
                        enable_cuda_graph
                        and batch.forward_mode.is_decode()
                        and not should_force_eager
                    ):
                        assert (
                            batch_measurement[
                                (
                                    "plain_graph_replay"
                                    if decode_path == "plain"
                                    else "verify_graph_replay"
                                )
                            ]
                            > 0
                        )
                    else:
                        pass
                    scheduler.process_batch_result(batch, result)
                    if (
                        not has_cancelled_request
                        and decode_path == "plain"
                        and batch.forward_mode.is_decode()
                        and any(request.rid == "cancel" for request in batch.reqs)
                    ):
                        scheduler.abort("cancel", defer_running_cleanup=False)
                        scheduler.process_input_requests(
                            [build_payload("replacement", True)]
                        )
                        has_cancelled_request = True
                    else:
                        pass
                else:
                    pass
                scheduler.last_batch = batch
                if has_cancelled_request and not has_retracted_requests:
                    request_paths = {
                        request.rid: request.omni_data.decode_path
                        for held_batch in (
                            scheduler.running_batch,
                            *scheduler.pending_batches(),
                        )
                        for request in held_batch.reqs
                        if not request.finished()
                    }
                    assert scheduler.retract_running_requests() > 0
                    for request in scheduler.waiting_queue:
                        if request.rid in request_paths:
                            assert (
                                request.omni_data.decode_path
                                == request_paths[request.rid]
                            )
                        else:
                            pass
                    has_retracted_requests = True
                else:
                    pass
                while not scheduler.outbox.empty():
                    message = scheduler.outbox.get_nowait()
                    assert message.type != "error", str(message.data)
                    if message.type == "result":
                        completed_request_ids.add(message.request_id)
                        transcript_texts[message.request_id] = message.data.data["text"]
                    elif message.type == "stream":
                        streamed_request_ids.add(message.request_id)
                    else:
                        pass
                if (
                    enable_radix_cache
                    and not has_finished_seed_request
                    and "plain-seed" in completed_request_ids
                ):
                    has_finished_seed_request = True
                    scheduler.process_input_requests(initial_requests)
                elif (
                    enable_radix_cache
                    and not has_submitted_repeated_requests
                    and expected_request_ids <= completed_request_ids
                ):
                    has_submitted_repeated_requests = True
                    expected_request_ids.update({"plain-repeat", "spec-repeat"})
                    scheduler.process_input_requests(
                        [
                            build_payload("plain-repeat", True),
                            build_payload("spec-repeat", False),
                        ]
                    )
                elif (
                    expected_request_ids <= completed_request_ids
                    and not scheduler.active_request_ids()
                ):
                    break
                else:
                    pass
            else:
                pytest.fail("Mixed-path scheduler did not finish before the deadline")
        assert has_cancelled_request
        assert has_retracted_requests
        if enable_cuda_graph:
            assert has_exercised_eager_fallback
        else:
            pass
        assert "cancel" not in completed_request_ids
        assert expected_request_ids <= streamed_request_ids
        assert id(target_runner.model) == target_model_identity
        assert target_runner.spec_algorithm is target_spec_algorithm
        assert get_spec() is runtime_speculation_config
        for request_id in (
            {"greedy", "spec-repeat"} if enable_radix_cache else {"greedy"}
        ):
            transcript = re.sub(r"\[[^\]]*\]", "", transcript_texts[request_id])
            assert (
                re.sub(r"[^a-z0-9]+", " ", transcript.casefold()).strip()
                == "what is happening in this video answer in ten words or fewer"
            )
        assert (
            scheduler.req_to_token_pool.available_size() == initial_request_slot_count
        )
        if enable_radix_cache:
            assert has_seen_first_speculative_prefill
            assert has_submitted_repeated_requests
            for _ in range(4):
                assert scheduler.get_next_batch_to_run() is None
                scheduler.last_batch = None
            assert scheduler.flush_cache()
        else:
            pass
        assert (
            scheduler.token_to_kv_pool_allocator.available_size()
            == initial_cache_slot_count
        )
        assert any(
            batch_measurement["draft_forward"] > 0 and batch_measurement["verify"] > 0
            for batch_measurement in batch_measurements
        )
        assert any(
            batch_measurement["decode_path"] == "plain"
            and "DECODE" in batch_measurement["mode"]
            for batch_measurement in batch_measurements
        )
        print(
            json.dumps(
                {
                    "model_family": model_family,
                    "enable_cuda_graph": enable_cuda_graph,
                    "enable_radix_cache": enable_radix_cache,
                    "batches": batch_measurements,
                    "completed": sorted(completed_request_ids),
                    "streamed": sorted(streamed_request_ids),
                    "transcripts": transcript_texts,
                }
            )
        )
    finally:
        scheduler.shutdown_request_build_executor()
        scheduler.shutdown_resources()
