# SPDX-License-Identifier: Apache-2.0
"""Text sampling, audio transitions, and request stop ownership on real CPU flow."""

from __future__ import annotations

from typing import Literal
from unittest.mock import patch

import pytest
import torch
from sglang.srt.managers.utils import GenerationBatchResult

from sglang_omni.models.firered_tts3.model_runner import (
    AUDIO_SOS_ID,
    TEXT_EOT_ID,
    FireRedTTS3ModelRunner,
)
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    TEXT_MODE_AUDIO,
    TEXT_MODE_FINALIZE,
    TEXT_MODE_TEXT,
)
from sglang_omni.scheduling.sglang_backend.output_processor import SGLangOutputProcessor
from sglang_omni.scheduling.types import SchedulerOutput, SchedulerRequest
from tests.unit_test.fixtures.firered_tts3 import (
    SmallFlowModel,
    make_flow_request,
    make_flow_result,
    make_flow_runner,
)


def make_runner() -> FireRedTTS3ModelRunner:
    runner = make_flow_runner(SmallFlowModel())
    runner.output_processor = SGLangOutputProcessor()
    return runner


def make_request(
    task_type: Literal["voice_design", "tts"] = "voice_design"
) -> SchedulerRequest:
    request = make_flow_request()
    request.data.state.variant = "instruct"
    request.data.state.task_type = task_type
    request.data.text_ar_mode = TEXT_MODE_TEXT
    request.data.req.prefix_len = 0
    request.data.req.extend_end = 1
    return request


def make_result(next_ids: list[int]) -> GenerationBatchResult:
    return make_flow_result(torch.zeros(len(next_ids), 8), next_ids)


def decode_step(
    runner: FireRedTTS3ModelRunner, request: SchedulerRequest, sampled: int
) -> int:
    result = make_result([sampled])
    runner.post_decode(result, None, None, [request])
    return int(result.next_token_ids[0])


@pytest.mark.parametrize(
    "text_count,has_eot", [(0, True), (1, True), (199, True), (200, False)]
)
def test_text_transition_preserves_tokens_and_double_audio_start(
    text_count: int, has_eot: bool
) -> None:
    """Empty, short, boundary-EOT, and exhausted plans must enter audio through two start feeds."""
    runner, request = make_runner(), make_request()
    request_data = request.data
    for index in range(text_count):
        assert request_data.text_ar_mode == TEXT_MODE_TEXT
        assert (
            runner.record_text_sample(request_data, torch.tensor(1000 + index))
            == 1000 + index
        )
        assert request_data.text_fed == index + 1
    if has_eot:
        assert (
            runner.record_text_sample(request_data, torch.tensor(TEXT_EOT_ID))
            == TEXT_EOT_ID
        )
    else:
        pass
    assert request_data.text_ar_mode == TEXT_MODE_FINALIZE
    assert request_data.text_token_ids == list(range(1000, 1000 + text_count))
    assert request_data.text_fed == text_count + int(has_eot)
    assert request_data.eot_sampled is has_eot
    tokens, patch_counts = [], []
    for _ in range(4):
        tokens.append(decode_step(runner, request, 0))
        patch_counts.append(len(request_data.latent_patches))
    assert tokens == [AUDIO_SOS_ID, AUDIO_SOS_ID, CONTROL_TOKEN_ID, CONTROL_TOKEN_ID]
    assert patch_counts == [0, 0, 1, 2]
    assert request_data.text_ar_mode == TEXT_MODE_AUDIO


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("prefill", [False, True])
def test_speaker_known_ids_survive_finalize(batch_size: int, prefill: bool) -> None:
    runner = make_runner()
    requests = []
    for index in range(batch_size):
        request = make_request("tts")
        request.request_id = f"speaker-{index}"
        request.data.state.variant = "speaker"
        request.data.text_ar_mode = TEXT_MODE_AUDIO
        requests.append(request)
    result = make_result([123] * batch_size)
    result.can_run_cuda_graph = False
    if prefill:
        runner.post_prefill(result, None, None, requests)
    else:
        runner.post_decode(result, None, None, requests)
    assert result.next_token_ids.tolist() == [CONTROL_TOKEN_ID] * batch_size
    result.next_token_ids.fill_(123)
    output = runner.finalize(
        result,
        None,
        None,
        SchedulerOutput(requests=requests, batch_data=None, step_id=0),
    )
    assert output.host_token_ids.device.type == "cpu"
    assert output.host_token_ids.tolist() == [CONTROL_TOKEN_ID] * batch_size
    assert [output.outputs[r.request_id].data for r in requests] == [
        CONTROL_TOKEN_ID
    ] * batch_size
    assert all(r.data.generation_steps == 1 for r in requests)


@pytest.mark.parametrize("prefill", [False, True])
def test_mixed_speaker_and_text_batch_preserves_sampled_output(prefill: bool) -> None:
    runner = make_runner()
    for reverse_order in (False, True):
        requests = [
            make_request("tts"),
            make_request(),
            make_request(),
            make_request("tts"),
            make_request(),
        ]
        sampled_ids = [123, 123, TEXT_EOT_ID, 123, CONTROL_TOKEN_ID]
        for index, request in enumerate(requests):
            request.request_id = str(index)
            if index in (0, 3):
                request.data.state.variant = "speaker"
                request.data.text_ar_mode = TEXT_MODE_AUDIO
            else:
                pass
        if reverse_order:
            requests.reverse()
            sampled_ids.reverse()
        else:
            pass
        result = make_result(sampled_ids)
        if prefill:
            runner.post_prefill(result, None, None, requests)
        else:
            runner.post_decode(result, None, None, requests)
        output = runner.finalize(
            result,
            None,
            None,
            SchedulerOutput(requests=requests, batch_data=None, step_id=0),
        )
        assert [output.outputs[str(index)].data for index in range(5)] == [
            CONTROL_TOKEN_ID,
            123,
            TEXT_EOT_ID,
            CONTROL_TOKEN_ID,
            CONTROL_TOKEN_ID,
        ]
        by_id = {request.request_id: request.data for request in requests}
        assert by_id["1"].text_token_ids == [123]
        assert by_id["2"].text_token_ids == [] and by_id["2"].eot_sampled
        assert by_id["4"].text_token_ids == [CONTROL_TOKEN_ID]
        assert not by_id["1"].eot_sampled and not by_id["4"].eot_sampled
        assert output.host_token_ids is None
        assert [len(by_id[str(index)].latent_patches) for index in range(5)] == [
            1,
            0,
            0,
            1,
            0,
        ]
        assert all(request.data.generation_steps == 1 for request in requests)


@pytest.mark.parametrize(
    "enabled,async_enabled,variant,dual_stream,mode,ids",
    [
        (False, False, "speaker", False, TEXT_MODE_AUDIO, [0]),
        (True, True, "speaker", False, TEXT_MODE_AUDIO, [0]),
        (True, False, "base", False, TEXT_MODE_AUDIO, [0]),
        (True, False, "instruct", False, TEXT_MODE_AUDIO, [0]),
        (True, False, "speaker", True, TEXT_MODE_AUDIO, [0]),
        (True, False, "speaker", False, TEXT_MODE_TEXT, [0]),
        (True, False, "speaker", False, TEXT_MODE_AUDIO, [123]),
    ],
)
def test_control_fastpath_preserves_other_token_paths(
    enabled: bool,
    async_enabled: bool,
    variant: str,
    dual_stream: bool,
    mode: str,
    ids: list[int],
) -> None:
    runner = make_runner()
    runner.control_token_fastpath = enabled
    runner.async_enabled = async_enabled
    request = make_request("tts")
    request.data.state.variant = variant
    request.data.dual_stream = dual_stream
    request.data.text_ar_mode = mode
    device_ids, host_ids = runner.make_next_token_ids(
        ids, torch.device("cpu"), [request]
    )
    assert device_ids.tolist() == ids
    assert host_ids is None


@pytest.mark.parametrize(
    "skip,variant,dual_stream,async_enabled,min_patches,patch_count,logit,expected_stop,expected_calls",
    [
        (None, "speaker", False, False, 6, 5, 2.0, False, 0),
        (False, "speaker", False, False, 6, 5, 2.0, False, 1),
        (True, "speaker", False, False, 6, 5, 2.0, False, 0),
        (True, "speaker", False, False, 0, 0, 0.0, True, 1),
        (True, "speaker", False, False, 6, 6, -2.0, False, 1),
        (True, "speaker", False, False, 6, 6, 0.0, True, 1),
        (True, "speaker", False, False, 6, 7, 2.0, True, 1),
        (True, "speaker", False, True, 6, 5, 2.0, False, 1),
        (True, "speaker", True, False, 6, 5, 2.0, False, 1),
        (True, "base", False, False, 6, 5, 2.0, False, 1),
        (True, "instruct", False, False, 6, 5, 2.0, False, 1),
    ],
)
def test_min_patch_stop_boundary_is_unchanged(
    skip: bool | None,
    variant: str,
    dual_stream: bool,
    async_enabled: bool,
    min_patches: int,
    patch_count: int,
    logit: float,
    expected_stop: bool,
    expected_calls: int,
) -> None:
    runner = make_runner()
    if skip is not None:
        runner.skip_min_patch_stop = skip
    else:
        pass
    runner.async_enabled = async_enabled
    request = make_request("tts")
    data = request.data
    data.state.variant = variant
    data.dual_stream = dual_stream
    data.state.min_patches = min_patches
    data.state.stop_threshold = 0.5
    data.text_ar_mode = TEXT_MODE_AUDIO
    data.flow_state.patch_count = patch_count
    with torch.no_grad():
        runner.model.stop_head.weight.zero_()
        runner.model.stop_head.bias.fill_(logit)
    with patch.object(
        runner.model.stop_head, "forward", wraps=runner.model.stop_head.forward
    ) as forward:
        assert runner.stop_predicted(data, torch.zeros(1, 8)) is expected_stop
    assert forward.call_count == expected_calls


@pytest.mark.parametrize("control_token_fastpath", [False, True])
@pytest.mark.parametrize("skip_min_patch_stop", [False, True])
def test_speaker_control_flags_preserve_independent_request_stops(
    control_token_fastpath: bool, skip_min_patch_stop: bool
) -> None:
    runner = make_runner()
    runner.control_token_fastpath = control_token_fastpath
    runner.skip_min_patch_stop = skip_min_patch_stop
    requests = [make_request("tts"), make_request("tts")]
    for index, request in enumerate(requests):
        request.request_id = f"speaker-{index}"
        request.data.state.variant = "speaker"
        request.data.state.min_patches = 6
        request.data.state.stop_threshold = 0.5
        request.data.text_ar_mode = TEXT_MODE_AUDIO
        request.data.flow_state.patch_count = 5 + index
    with torch.no_grad():
        runner.model.stop_head.weight.zero_()
        runner.model.stop_head.bias.fill_(2.0)
    with patch.object(
        runner.model.stop_head, "forward", wraps=runner.model.stop_head.forward
    ) as forward:
        for active in (requests, requests[:1]):
            result = make_result([123] * len(active))
            runner.post_decode(result, None, None, active)
            output = runner.finalize(
                result,
                None,
                None,
                SchedulerOutput(requests=active, batch_data=None, step_id=0),
            )
            assert [output.outputs[request.request_id].data for request in active] == [
                CONTROL_TOKEN_ID
            ] * len(active)
            assert (output.host_token_ids is not None) is control_token_fastpath
            assert requests[1].data.req.finished_reason is not None
            assert (requests[0].data.req.finished_reason is not None) is (
                len(active) == 1
            )
    assert [len(request.data.latent_patches) for request in requests] == [1, 0]
    assert [request.data.flow_state.patch_count for request in requests] == [6, 6]
    assert [request.data.generation_steps for request in requests] == [2, 1]
    assert forward.call_count == (2 if skip_min_patch_stop else 3)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_control_token_device_snapshots_are_independent() -> None:
    runner = make_runner()
    request = make_request("tts")
    request.data.state.variant = "speaker"
    request.data.text_ar_mode = TEXT_MODE_AUDIO
    first, host = runner.make_next_token_ids([0], torch.device("cuda"), [request])
    second, next_host = runner.make_next_token_ids([0], first.device, [request])
    second.fill_(123)
    assert first.is_cuda and host.device.type == "cpu"
    assert first.tolist() == [0]
    assert host.tolist() == next_host.tolist() == [0]
