# SPDX-License-Identifier: Apache-2.0
"""Model-backed fixtures for shared plain/spec scheduling regressions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Literal
from unittest.mock import patch

import torch

from sglang_omni.models.moss_transcribe_diarize.request_builders import (
    MossTranscribeDiarizeRequestData,
)
from sglang_omni.models.moss_transcribe_diarize.stages import (
    create_sglang_moss_transcribe_diarize_executor,
)
from sglang_omni.models.qwen3_asr.request_builders import Qwen3ASRRequestData
from sglang_omni.models.qwen3_asr.stages import create_sglang_qwen3_asr_executor
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.scheduling.omni_scheduler import OmniScheduler
from sglang_omni.scheduling.sglang_backend.request_data import SGLangARRequestData


def create_plain_fallback_scheduler(
    model_family: Literal["qwen3_asr", "moss_td"],
    *,
    enable_cuda_graph: bool,
    enable_radix_cache: bool,
    max_running_requests: int,
    max_total_tokens: int | None = None,
) -> (
    OmniScheduler[Qwen3ASRRequestData] | OmniScheduler[MossTranscribeDiarizeRequestData]
):
    """Build the two supported model integrations with one shared target."""
    server_overrides = {
        "speculative_algorithm": "DFLASH",
        "speculative_num_draft_tokens": 8,
        "disable_cuda_graph": not enable_cuda_graph,
        "cuda_graph_backend_prefill": "disabled",
        "max_running_requests": max_running_requests,
        "cuda_graph_bs": [1] if max_running_requests == 1 else [1, 2, 4],
        "cuda_graph_max_bs": max_running_requests,
        "disable_radix_cache": not enable_radix_cache,
        "enable_torch_compile": False,
    }
    if max_total_tokens is not None:
        server_overrides["max_total_tokens"] = max_total_tokens
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
            max_running_requests=max_running_requests,
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
            max_running_requests=max_running_requests,
            enable_plain_fallback=True,
            enable_async_decode=False,
            max_new_tokens=128,
            request_build_max_workers=1,
            prefill_coalesce_requests=0,
            server_args_overrides=server_overrides,
        )
    return scheduler


def probe_plain_fallback_progress(
    model_family: Literal["qwen3_asr", "moss_td"],
    scenario: Literal["kv_pressure", "blocked_path"],
    *,
    enable_cuda_graph: bool,
) -> None:
    """Exercise native KV exhaustion and slot-blocked admission with real forwards."""
    if scenario == "kv_pressure":
        # note (AkazaAkane): the MOSS fixture reaches EOS early, so a shorter budget and smaller pool sustain pressure.
        max_running_requests = 4
        max_total_tokens = 2048 if model_family == "qwen3_asr" else 352
        max_new_tokens = 1700 if model_family == "qwen3_asr" else 32
    else:
        max_running_requests = 1
        max_total_tokens = 32768
        max_new_tokens = 128
    scheduler = create_plain_fallback_scheduler(
        model_family,
        enable_cuda_graph=enable_cuda_graph,
        enable_radix_cache=False,
        max_running_requests=max_running_requests,
        max_total_tokens=max_total_tokens,
    )
    initial_request_slot_count = scheduler.req_to_token_pool.available_size()
    initial_cache_slot_count = scheduler.token_to_kv_pool_allocator.available_size()
    request_builder = scheduler.request_builder
    request_data_by_id: dict[str, SGLangARRequestData] = {}
    cache_salts: dict[str, str] = {}

    def build_request(payload: StagePayload) -> SGLangARRequestData:
        request_data = request_builder(payload)
        assert isinstance(request_data, SGLangARRequestData)
        request_data.req.sampling_params.stop_token_ids = set()
        request_data.req.sampling_params.stop_strs = []
        request_data.req.sampling_params.stop_regex_strs = []
        request_data_by_id[payload.request_id] = request_data
        return request_data

    audio_path = Path(__file__).parents[2] / "data" / "query_to_draw.wav"

    def build_payload(request_id: str) -> StagePayload:
        return StagePayload(
            request_id=request_id,
            request=OmniRequest(
                inputs={"audio_path": str(audio_path)},
                params={
                    "repetition_penalty": 1.2 if request_id == "plain" else 1.0,
                    "max_new_tokens": max_new_tokens,
                    "stream": True,
                },
            ),
            data=None,
        )

    completed_request_ids: set[str] = set()
    has_submitted_speculative_request = False
    false_idle_count = 0
    try:
        with (
            torch.inference_mode(),
            patch.object(scheduler, "request_builder", side_effect=build_request),
            patch.object(
                scheduler,
                "retract_parked_decode_requests",
                wraps=scheduler.retract_parked_decode_requests,
            ) as retractions,
        ):
            scheduler.new_token_ratio_tracker.current = (
                scheduler.new_token_ratio_tracker.min
            )
            scheduler.process_input_requests([build_payload("plain")])
            deadline_seconds = time.monotonic() + 240
            while time.monotonic() < deadline_seconds:
                batch = scheduler.get_next_batch_to_run()
                scheduler.cur_batch = batch
                if batch is not None:
                    for request in batch.reqs:
                        assert request.omni_data.decode_path == request.rid
                        assert (
                            cache_salts.setdefault(request.rid, request.cache_salt)
                            == request.cache_salt
                        )
                    result = scheduler.run_batch(batch)
                    scheduler.process_batch_result(batch, result)
                    if not has_submitted_speculative_request and len(
                        request_data_by_id["plain"].req.output_ids
                    ) >= (64 if model_family == "qwen3_asr" else 8):
                        scheduler.process_input_requests([build_payload("spec")])
                        has_submitted_speculative_request = True
                    else:
                        pass
                else:
                    if any(
                        request_data.req.kv.holds_kv and not request_data.req.finished()
                        for request_data in request_data_by_id.values()
                    ):
                        false_idle_count += 1
                    else:
                        pass
                    scheduler.self_check_during_idle()
                    scheduler.sleep_during_idle()
                scheduler.last_batch = batch
                while not scheduler.outbox.empty():
                    message = scheduler.outbox.get_nowait()
                    assert message.type != "error", str(message.data)
                    if message.type == "result":
                        completed_request_ids.add(message.request_id)
                    else:
                        pass
                if (
                    completed_request_ids == {"plain", "spec"}
                    and not scheduler.active_request_ids()
                ):
                    break
                else:
                    pass
            else:
                raise AssertionError(
                    f"Mixed paths stalled: completed={sorted(completed_request_ids)}, "
                    f"waiting={[request.rid for request in scheduler.waiting_queue]}"
                )
            assert has_submitted_speculative_request
            assert false_idle_count == 0
            if scenario == "kv_pressure":
                assert retractions.call_count > 0
            else:
                assert retractions.call_count == 0
            for _ in range(4):
                assert scheduler.get_next_batch_to_run() is None
                scheduler.last_batch = None
            assert scheduler.flush_cache()
            assert (
                scheduler.req_to_token_pool.available_size()
                == initial_request_slot_count
            )
            assert (
                scheduler.token_to_kv_pool_allocator.available_size()
                == initial_cache_slot_count
            )
            print(
                json.dumps(
                    {
                        "model_family": model_family,
                        "scenario": scenario,
                        "enable_cuda_graph": enable_cuda_graph,
                        "completed": sorted(completed_request_ids),
                        "retractions": retractions.call_count,
                        "idle_with_running_requests": false_idle_count,
                        "recovered_kv_slots": initial_cache_slot_count,
                        "recovered_request_slots": initial_request_slot_count,
                    }
                )
            )
    finally:
        scheduler.shutdown_request_build_executor()
        scheduler.shutdown_resources()
