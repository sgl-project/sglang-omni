# SPDX-License-Identifier: Apache-2.0
"""Single-worker eager routing, sampler, cancellation, and KV lifecycle probe."""

from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import pytest
import torch
from sglang.srt.runtime_context import get_spec

from sglang_omni.models.moss_transcribe_diarize.stages import (
    create_sglang_moss_transcribe_diarize_executor,
)
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.proto import OmniRequest, StagePayload


@pytest.mark.accelerator
@pytest.mark.parametrize("model_family", ["qwen3_asr", "moss_td"])
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
@pytest.mark.parametrize("enable_radix_cache", [False, True])
def test_plain_fallback_uses_one_target_worker(
    model_family: Literal["qwen3_asr", "moss_td"],
    enable_cuda_graph: bool,
    enable_radix_cache: bool,
) -> None:
    """Each engine owns its process and keeps one target through both paths."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.test_model.test_plain_fallback import probe_plain_fallback; probe_plain_fallback({model_family!r}, enable_cuda_graph={enable_cuda_graph!r}, enable_radix_cache={enable_radix_cache!r})",
        ],
        check=True,
        timeout=300,
    )


def probe_plain_fallback(
    model_family: Literal["qwen3_asr", "moss_td"],
    *,
    enable_cuda_graph: bool = False,
    enable_radix_cache: bool = False,
) -> None:
    """Fallback requests must never forward the draft or write its KV pool."""
    server_overrides = {
        "speculative_algorithm": "DFLASH",
        "speculative_num_draft_tokens": 8,
        "disable_cuda_graph": not enable_cuda_graph,
        "cuda_graph_backend_prefill": "disabled",
        "max_running_requests": 4,
        "cuda_graph_bs": [1, 2, 4],
        "cuda_graph_max_bs": 4,
        "disable_radix_cache": not enable_radix_cache,
        "enable_torch_compile": False,
        "max_total_tokens": 32768,
    }
    if model_family == "qwen3_asr":
        server_overrides.update(
            {
                "speculative_draft_model_path": "wilsonzheng0327/Qwen3-ASR-1.7B-dflash",
                "speculative_draft_model_revision": "d62bd4051e68282e02f93de774036a35f22f1373",
            }
        )
        scheduler = create_sglang_qwen3_asr_executor(
            "Qwen/Qwen3-ASR-1.7B",
            device="cuda",
            gpu_id=0,
            max_running_requests=4,
            enable_plain_fallback=True,
            enable_async_decode=False,
            enable_encoder_cuda_graph=False,
            enable_pre_lm_encoder=False,
            request_build_max_workers=1,
            prefill_coalesce_requests=0,
            server_args_overrides=server_overrides,
        )
    else:
        server_overrides.update(
            {
                "speculative_draft_model_path": "wilsonzheng0327/MOSS-Transcribe-Diarize-dflash",
                "speculative_draft_model_revision": "2e3f00b0932112e4f92a4ef67a4cd60ee855056c",
            }
        )
        scheduler = create_sglang_moss_transcribe_diarize_executor(
            "OpenMOSS-Team/MOSS-Transcribe-Diarize",
            device="cuda",
            gpu_id=0,
            max_running_requests=4,
            enable_plain_fallback=True,
            enable_async_decode=False,
            max_new_tokens=128,
            request_build_max_workers=1,
            prefill_coalesce_requests=0,
            server_args_overrides=server_overrides,
        )
    target_runner = scheduler.tp_worker.model_runner
    draft_runner = scheduler.draft_worker.draft_model_runner
    target_worker = scheduler.tp_worker.speculative_target_worker
    draft_pool = draft_runner.token_to_kv_pool
    target_identity = id(target_runner.model)
    target_spec_algorithm = target_runner.spec_algorithm
    runtime_spec = get_spec()
    initial_request_slots = scheduler.req_to_token_pool.available_size()
    initial_kv_slots = scheduler.token_to_kv_pool_allocator.available_size()
    audio_path = Path(__file__).parents[1] / "data" / "query_to_draw.wav"

    def payload(request_id: str, temperature: float) -> StagePayload:
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs={"audio_path": str(audio_path)},
                params=(
                    {"temperature": temperature, "stream": True}
                    if model_family == "qwen3_asr"
                    else {
                        "repetition_penalty": 1.2 if temperature > 0 else 1.0,
                        "stream": True,
                    }
                ),
            ),
            data=None,
        )

    records: list[dict[str, str | int]] = []
    completed_requests: set[str] = set()
    streamed_requests: set[str] = set()
    transcript_texts: dict[str, str] = {}
    cancelled = False
    retracted = False
    eager_fallback_exercised = False
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
                draft_pool, "set_kv_buffer", wraps=draft_pool.set_kv_buffer
            ) as draft_kv_write,
            patch.object(
                draft_pool,
                "set_kv_buffer_prefix_valid",
                wraps=draft_pool.set_kv_buffer_prefix_valid,
            ) as draft_prefix_write,
        ):
            if enable_cuda_graph:
                plain_runner = target_runner.plain_decode_runner
                assert plain_runner.model is target_runner.model
                assert plain_runner.req_to_token_pool is target_runner.req_to_token_pool
                assert plain_runner.token_to_kv_pool is target_runner.token_to_kv_pool
                assert plain_runner.attn_backend is not target_runner.attn_backend
                plain_graph_replay = graph_patches.enter_context(
                    patch.object(
                        plain_runner.decode_cuda_graph_runner,
                        "execute",
                        wraps=plain_runner.decode_cuda_graph_runner.execute,
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
                payload("greedy", 0.0),
                payload("sampled", 0.7),
                payload("sampled-second", 0.7),
                payload("cancel", 0.7),
            ]
            expected_requests = {"greedy", "sampled", "sampled-second", "replacement"}
            seed_finished = not enable_radix_cache
            repeats_submitted = False
            first_spec_prefill_seen = False
            scheduler.process_input_requests(
                [payload("plain-seed", 0.7)] if enable_radix_cache else initial_requests
            )
            deadline = time.monotonic() + 180
            while time.monotonic() < deadline:
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
                            if request.rid == "greedy" and not first_spec_prefill_seen:
                                assert request.prefix_len == 0
                                first_spec_prefill_seen = True
                            elif request.rid in {"plain-repeat", "spec-repeat"}:
                                assert request.prefix_len > 0
                            else:
                                pass
                    else:
                        pass
                    before = (
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
                    force_eager = (
                        enable_cuda_graph
                        and decode_path == "plain"
                        and batch.forward_mode.is_decode()
                        and not eager_fallback_exercised
                    )
                    if force_eager:
                        with patch.object(
                            plain_runner.decode_cuda_graph_runner,
                            "can_run_graph",
                            return_value=False,
                        ):
                            result = scheduler.run_batch(batch)
                        eager_fallback_exercised = True
                    else:
                        result = scheduler.run_batch(batch)
                    verification_count = sum(
                        bool(call.kwargs.get("is_verify"))
                        for call in target_forward.call_args_list
                    )
                    record = {
                        "decode_path": decode_path,
                        "requests": ",".join(request.rid for request in batch.reqs),
                        "mode": batch.forward_mode.name,
                        "forced_eager": int(force_eager),
                        "cached_tokens": (
                            sum(request.prefix_len for request in batch.reqs)
                            if batch.forward_mode.is_extend()
                            else 0
                        ),
                        "sampler": sampler.call_count - before[0],
                        "draft_forward": draft_forward.call_count - before[1],
                        "draft_kv_write": draft_kv_write.call_count
                        + draft_prefix_write.call_count
                        - before[2],
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
                    records.append(record)
                    if decode_path == "plain":
                        assert record["sampler"] > 0
                        if model_family == "qwen3_asr":
                            assert not sampler.call_args.args[1].is_all_greedy
                        else:
                            assert all(
                                request.sampling_params.repetition_penalty == 1.2
                                for request in batch.reqs
                            )
                        assert (
                            record["draft_forward"]
                            == record["draft_kv_write"]
                            == record["verify"]
                            == 0
                        )
                    else:
                        pass
                    if (
                        enable_cuda_graph
                        and batch.forward_mode.is_decode()
                        and not force_eager
                    ):
                        assert (
                            record[
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
                        not cancelled
                        and decode_path == "plain"
                        and batch.forward_mode.is_decode()
                        and any(request.rid == "cancel" for request in batch.reqs)
                    ):
                        scheduler.abort("cancel", defer_running_cleanup=False)
                        scheduler.process_input_requests([payload("replacement", 0.7)])
                        cancelled = True
                    else:
                        pass
                else:
                    pass
                scheduler.last_batch = batch
                if cancelled and not retracted:
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
                    retracted = True
                else:
                    pass
                while not scheduler.outbox.empty():
                    message = scheduler.outbox.get_nowait()
                    assert message.type != "error", str(message.data)
                    if message.type == "result":
                        completed_requests.add(message.request_id)
                        transcript_texts[message.request_id] = message.data.data["text"]
                    elif message.type == "stream":
                        streamed_requests.add(message.request_id)
                    else:
                        pass
                if (
                    enable_radix_cache
                    and not seed_finished
                    and "plain-seed" in completed_requests
                ):
                    seed_finished = True
                    scheduler.process_input_requests(initial_requests)
                elif (
                    enable_radix_cache
                    and not repeats_submitted
                    and expected_requests <= completed_requests
                ):
                    repeats_submitted = True
                    expected_requests.update({"plain-repeat", "spec-repeat"})
                    scheduler.process_input_requests(
                        [payload("plain-repeat", 0.7), payload("spec-repeat", 0.0)]
                    )
                elif (
                    expected_requests <= completed_requests
                    and not scheduler.active_request_ids()
                ):
                    break
                else:
                    pass
            else:
                pytest.fail("Mixed-path scheduler did not finish before the deadline")
        assert cancelled
        assert retracted
        if enable_cuda_graph:
            assert eager_fallback_exercised
        else:
            pass
        assert "cancel" not in completed_requests
        assert expected_requests <= streamed_requests
        assert id(target_runner.model) == target_identity
        assert target_runner.spec_algorithm is target_spec_algorithm
        assert get_spec() is runtime_spec
        for request_id in (
            {"greedy", "spec-repeat"} if enable_radix_cache else {"greedy"}
        ):
            transcript = re.sub(r"\[[^\]]*\]", "", transcript_texts[request_id])
            assert (
                re.sub(r"[^a-z0-9]+", " ", transcript.casefold()).strip()
                == "what is happening in this video answer in ten words or fewer"
            )
        assert scheduler.req_to_token_pool.available_size() == initial_request_slots
        if enable_radix_cache:
            assert first_spec_prefill_seen
            assert repeats_submitted
            for _ in range(4):
                assert scheduler.get_next_batch_to_run() is None
                scheduler.last_batch = None
            assert scheduler.flush_cache()
        else:
            pass
        assert scheduler.token_to_kv_pool_allocator.available_size() == initial_kv_slots
        assert any(
            record["draft_forward"] > 0 and record["verify"] > 0 for record in records
        )
        assert any(
            record["decode_path"] == "plain" and "DECODE" in record["mode"]
            for record in records
        )
        print(
            json.dumps(
                {
                    "model_family": model_family,
                    "enable_cuda_graph": enable_cuda_graph,
                    "enable_radix_cache": enable_radix_cache,
                    "batches": records,
                    "completed": sorted(completed_requests),
                    "streamed": sorted(streamed_requests),
                    "transcripts": transcript_texts,
                }
            )
        )
    finally:
        scheduler.shutdown_request_build_executor()
        scheduler.shutdown_resources()
