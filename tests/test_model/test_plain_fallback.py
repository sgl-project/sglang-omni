# SPDX-License-Identifier: Apache-2.0
"""Single-worker eager routing, sampler, cancellation, and KV lifecycle probe."""

from __future__ import annotations

import json
import subprocess
import sys
import time
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
def test_plain_fallback_uses_one_target_worker(
    model_family: Literal["qwen3_asr", "moss_td"]
) -> None:
    """Each engine owns its process and keeps one target through both paths."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            f"from tests.test_model.test_plain_fallback import probe_plain_fallback; probe_plain_fallback({model_family!r})",
        ],
        check=True,
        timeout=300,
    )


def probe_plain_fallback(model_family: Literal["qwen3_asr", "moss_td"]) -> None:
    """Fallback requests must never forward the draft or write its KV pool."""
    server_overrides = {
        "speculative_algorithm": "DFLASH",
        "speculative_num_draft_tokens": 8,
        "disable_cuda_graph": True,
        "disable_radix_cache": True,
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
    cancelled = False
    retracted = False
    try:
        with (
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
            scheduler.process_input_requests(
                [
                    payload("greedy", 0.0),
                    payload("sampled", 0.7),
                    payload("cancel", 0.7),
                ]
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
                    before = (
                        sampler.call_count,
                        draft_forward.call_count,
                        draft_kv_write.call_count + draft_prefix_write.call_count,
                    )
                    target_forward.reset_mock()
                    result = scheduler.run_batch(batch)
                    verification_count = sum(
                        bool(call.kwargs.get("is_verify"))
                        for call in target_forward.call_args_list
                    )
                    record = {
                        "decode_path": decode_path,
                        "mode": batch.forward_mode.name,
                        "sampler": sampler.call_count - before[0],
                        "draft_forward": draft_forward.call_count - before[1],
                        "draft_kv_write": draft_kv_write.call_count
                        + draft_prefix_write.call_count
                        - before[2],
                        "verify": verification_count,
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
                    scheduler.process_batch_result(batch, result)
                    if (
                        not cancelled
                        and decode_path == "plain"
                        and batch.forward_mode.is_decode()
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
                    elif message.type == "stream":
                        streamed_requests.add(message.request_id)
                    else:
                        pass
                if {
                    "greedy",
                    "sampled",
                    "replacement",
                } <= completed_requests and not scheduler.active_request_ids():
                    break
                else:
                    pass
            else:
                pytest.fail("Mixed-path scheduler did not finish before the deadline")
        assert cancelled
        assert retracted
        assert "cancel" not in completed_requests
        assert {"greedy", "sampled", "replacement"} <= streamed_requests
        assert id(target_runner.model) == target_identity
        assert get_spec() is runtime_spec
        assert scheduler.req_to_token_pool.available_size() == initial_request_slots
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
                    "batches": records,
                    "completed": sorted(completed_requests),
                    "streamed": sorted(streamed_requests),
                }
            )
        )
    finally:
        scheduler.shutdown_request_build_executor()
        scheduler.shutdown_resources()
