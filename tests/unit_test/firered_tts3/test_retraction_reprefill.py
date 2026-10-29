# SPDX-License-Identifier: Apache-2.0
"""Retraction resumes the pending feed's post-work and defers prefill stops."""

from __future__ import annotations

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from sglang.srt.managers.schedule_batch import FINISH_MATCHED_TOKEN
from sglang.srt.managers.utils import GenerationBatchResult

from sglang_omni.models.firered_tts3.model_runner import (
    AUDIO_SOS_ID,
    TEXT_EOT_ID,
    FireRedTTS3ModelRunner,
)
from sglang_omni.models.firered_tts3.payload_types import TEXT_TOKEN_BUDGET
from sglang_omni.models.firered_tts3.prefill import text_output_token_count
from sglang_omni.models.firered_tts3.request_builders import (
    CONTROL_TOKEN_ID,
    DS_STAGE_PATCH,
    TEXT_MODE_AUDIO,
    TEXT_MODE_AUDIO_ZERO,
    TEXT_MODE_FINALIZE,
    TEXT_MODE_SOSP,
    TEXT_MODE_TEXT,
)
from sglang_omni.scheduling.types import SchedulerRequest
from tests.unit_test.fixtures.firered_tts3 import (
    SmallFlowModel,
    make_flow_request,
    make_flow_result,
    make_flow_runner,
)


def make_runner(*, batching: bool = False) -> FireRedTTS3ModelRunner:
    runner = make_flow_runner(SmallFlowModel(), batching=batching)
    runner.stop_predicted = Mock(wraps=runner.stop_predicted)
    runner.prepare_flow_row = Mock(wraps=runner.prepare_flow_row)
    runner.run_flow_rows = Mock(wraps=runner.run_flow_rows)
    runner.model.patch_encoder.forward = Mock(wraps=runner.model.patch_encoder.forward)
    return runner


def make_request(
    *, dual: bool = False, task_type: str = "voice_design"
) -> SchedulerRequest:
    request = make_flow_request(dual_stream=dual, ratio=(3, 3), variant="instruct")
    request.data.state.task_type = task_type
    request.data.state.text_stream_ids = list(range(101, 131))
    request.data.text_ar_mode = (
        TEXT_MODE_AUDIO if dual or task_type == "tts" else TEXT_MODE_TEXT
    )
    return request


def make_result(rows: int, sampled: int = 123) -> GenerationBatchResult:
    return make_flow_result(
        torch.arange(rows, dtype=torch.float32).unsqueeze(-1).expand(-1, 8), [sampled]
    )


def prefill(
    runner: FireRedTTS3ModelRunner, request: SchedulerRequest, sampled: int = 123
) -> int:
    request_data = request.data
    request_data.req.prefix_len = 0
    request_data.req.extend_end = request_data.prefill_len + len(
        request_data.req.output_ids
    )
    result = make_result(request_data.req.extend_len, sampled)
    runner.post_prefill(result, None, None, [request])
    return int(result.next_token_ids[0])


def replay(runner: FireRedTTS3ModelRunner, request: SchedulerRequest) -> torch.Tensor:
    request_data = request.data
    request_data.req.prefix_len = 0
    request_data.req.extend_end = request_data.prefill_len + len(
        request_data.req.output_ids
    )
    batch = SimpleNamespace(
        input_ids=torch.zeros(request_data.req.extend_len, dtype=torch.long)
    )
    runner.before_prefill(batch, None, [request])
    assert batch.input_embeds.shape == (request_data.req.extend_len, 8)
    return batch.input_embeds


@pytest.mark.parametrize(
    "samples", [[TEXT_EOT_ID], [101, TEXT_EOT_ID], [101] * TEXT_TOKEN_BUDGET]
)
def test_single_history_invariant_at_every_pause(samples: list[int]) -> None:
    runner, request = make_runner(), make_request()
    request_data = request.data
    for sampled in [*samples, 0, 0, 0, 0]:
        replay(runner, request)
        generated = request_data.req.extend_len - request_data.prefill_len
        assert (
            generated
            == len(request_data.req.output_ids)
            == len(request_data.latent_patches) + text_output_token_count(request_data)
        )
        previous_mode = request_data.text_ar_mode
        previous_cond = request_data.flow_state.backbone_conditioning
        runner.prepare_flow_row.reset_mock()
        token = prefill(runner, request, sampled)
        if previous_mode in (TEXT_MODE_FINALIZE, TEXT_MODE_SOSP):
            assert token == AUDIO_SOS_ID
            assert request_data.text_ar_mode == (
                TEXT_MODE_SOSP
                if previous_mode == TEXT_MODE_FINALIZE
                else TEXT_MODE_AUDIO_ZERO
            )
            assert request_data.flow_state.backbone_conditioning is previous_cond
            runner.prepare_flow_row.assert_not_called()
        elif previous_mode == TEXT_MODE_AUDIO_ZERO:
            assert (
                token == CONTROL_TOKEN_ID
                and request_data.text_ar_mode == TEXT_MODE_AUDIO
            )
            assert runner.prepare_flow_row.call_count == 1
        else:
            pass
        request_data.req.output_ids.append(token)
    assert request_data.text_ar_mode == TEXT_MODE_AUDIO
    assert len(request_data.latent_patches) == 2
    replay(runner, request)


@pytest.mark.parametrize("batching", [False, True])
@pytest.mark.parametrize("ratio", [(2, 1), (3, 3)])
def test_dual_reprefill_matches_uninterrupted_decode_at_every_pause(
    batching: bool, ratio: tuple[int, int]
) -> None:
    """Re-prefill must resume each interrupted feed once, rebuilding its conditioning history in order."""
    runner, request = make_runner(batching=batching), make_request(dual=True)
    request_data = request.data
    request_data.interleave_ratio = ratio
    request_data.req.output_ids.append(prefill(runner, request))
    for _ in range(3 * sum(ratio)):
        assert request_data.req.output_ids.count(CONTROL_TOKEN_ID) == len(
            request_data.latent_patches
        )
        continuous = deepcopy(request)
        continuous_runner = make_runner(batching=batching)
        result = make_result(1)
        result.logits_output.hidden_states.fill_(
            request_data.prefill_len + len(request_data.req.output_ids) - 1
        )
        continuous_runner.post_decode(result, None, None, [continuous])
        produced = len(continuous.data.latent_patches) - len(
            request_data.latent_patches
        )
        runner.model.patch_encoder.forward.reset_mock()
        replay(runner, request)
        assert runner.model.patch_encoder.forward.call_count == len(
            request_data.latent_patches
        )
        for encoded, patch in zip(
            [
                call.args[0]
                for call in runner.model.patch_encoder.forward.call_args_list
            ],
            request_data.latent_patches,
            strict=True,
        ):
            assert torch.equal(encoded, patch)
        assert not request_data.pending_feedback_queue
        runner.prepare_flow_row.reset_mock()
        runner.stop_predicted.reset_mock()
        token = prefill(runner, request)
        assert token == int(result.next_token_ids[0])
        assert request_data.text_pos == continuous.data.text_pos
        assert len(request_data.pending_feedback_queue) == produced
        assert runner.prepare_flow_row.call_count == produced
        assert (
            runner.stop_predicted.call_count
            == continuous_runner.stop_predicted.call_count
        )
        for actual, expected in zip(
            request_data.latent_patches, continuous.data.latent_patches, strict=True
        ):
            assert torch.equal(actual, expected)
        torch.testing.assert_close(
            request_data.flow_state.backbone_conditioning,
            continuous.data.flow_state.backbone_conditioning,
        )
        request_data.req.output_ids.append(token)
    assert len(request_data.latent_patches) == 3 * ratio[1]
    assert request_data.text_pos == 3 * ratio[0] + 1


@pytest.mark.parametrize("dual", [False, True])
def test_prefill_stop_finishes_on_next_sync_decode(dual: bool) -> None:
    runner, request = make_runner(), make_request(dual=dual, task_type="tts")
    request_data = request.data
    if dual:
        request_data.req.output_ids.append(prefill(runner, request))
        for _ in range(5):
            result = make_result(1)
            runner.post_decode(result, None, None, [request])
            request_data.req.output_ids.append(int(result.next_token_ids[0]))
        assert (request_data.ds_stage, request_data.ds_index) == (DS_STAGE_PATCH, 2)
        replay(runner, request)
    else:
        request_data.state.variant = "base"
    request_data.state.min_patches = 0
    request_data.state.stop_threshold = 0.0
    runner.prepare_flow_row.reset_mock()
    assert prefill(runner, request) == CONTROL_TOKEN_ID
    assert request_data.stop_pending and request_data.req.finished_reason is None
    forward = SimpleNamespace(input_ids=torch.tensor([CONTROL_TOKEN_ID]))
    runner.before_decode(forward, None, [request])
    assert forward.input_embeds.shape == (1, 8)
    runner.post_decode(make_result(1), None, None, [request])
    assert isinstance(request_data.req.finished_reason, FINISH_MATCHED_TOKEN)
    assert not request_data.stop_pending
    runner.prepare_flow_row.assert_not_called()


@pytest.mark.parametrize("defer_stop", [False, True])
@pytest.mark.parametrize("status", ["active", "retracted", "finished"])
def test_pending_stop_resolve_ownership(defer_stop: bool, status: str) -> None:
    runner, request = make_runner(), make_request(task_type="tts")
    request_data = request.data
    request_data.stop_pending = True
    request_data.req.is_retracted = status == "retracted"
    previous_finish = FINISH_MATCHED_TOKEN(99) if status == "finished" else None
    request_data.req.finished_reason = previous_finish
    result = make_result(1)
    runner.run_post_decode(result, [request], defer_stop=defer_stop)
    assert request_data.stop_pending is defer_stop
    if defer_stop:
        assert request_data.req.finished_reason is previous_finish
        runner.post_decode_resolve(None, result, None, None, [request])
    else:
        pass
    assert not request_data.stop_pending
    if status == "active":
        assert isinstance(request_data.req.finished_reason, FINISH_MATCHED_TOKEN)
    else:
        assert request_data.req.finished_reason is previous_finish
    assert int(result.next_token_ids[0]) == CONTROL_TOKEN_ID
    runner.prepare_flow_row.assert_not_called()
    runner.stop_predicted.assert_not_called()


def make_stopped_request(
    *, dual: bool, batching: bool = False
) -> tuple[FireRedTTS3ModelRunner, SchedulerRequest]:
    runner, request = make_runner(batching=batching), make_request(
        dual=dual, task_type="tts"
    )
    request_data = request.data
    if dual:
        request_data.req.output_ids.append(prefill(runner, request))
        for _ in range(5):
            result = make_result(1)
            runner.post_decode(result, None, None, [request])
            request_data.req.output_ids.append(int(result.next_token_ids[0]))
        assert (request_data.ds_stage, request_data.ds_index) == (DS_STAGE_PATCH, 2)
    else:
        pass
    replay(runner, request)
    request_data.state.min_patches = 0
    request_data.state.stop_threshold = 0.0
    request_data.req.output_ids.append(prefill(runner, request))
    assert request_data.stop_pending and request_data.req.finished_reason is None
    assert len(request_data.latent_patches) == (3 if dual else 0)
    runner.prepare_flow_row.reset_mock()
    runner.run_flow_rows.reset_mock()
    runner.stop_predicted.reset_mock()
    return runner, request


@pytest.mark.parametrize("dual", [False, True])
@pytest.mark.parametrize("batching", [False, True])
def test_deferred_stop_survives_repeated_retraction_until_decode(
    dual: bool, batching: bool
) -> None:
    """Retraction after a prefill stop must not generate patches or consume control feeds as audio."""
    runner, request = make_stopped_request(dual=dual, batching=batching)
    request_data = request.data
    machine = (
        request_data.text_ar_mode,
        request_data.ds_stage,
        request_data.ds_index,
        request_data.text_pos,
    )
    patches = tuple(request_data.latent_patches)
    for excess in range(1, 5):
        output_ids = list(request_data.req.output_ids)
        rows = replay(runner, request)[request_data.prefill_len :]
        expected = runner.model.get_input_embeddings()(
            torch.full((excess,), CONTROL_TOKEN_ID, dtype=torch.long)
        )
        torch.testing.assert_close(rows[-excess:], expected)
        cond = request_data.flow_state.backbone_conditioning
        assert prefill(runner, request) == CONTROL_TOKEN_ID
        assert request_data.flow_state.backbone_conditioning is cond
        assert list(request_data.req.output_ids) == output_ids
        assert (
            request_data.text_ar_mode,
            request_data.ds_stage,
            request_data.ds_index,
            request_data.text_pos,
        ) == machine
        assert len(request_data.latent_patches) == len(patches)
        for actual, original in zip(request_data.latent_patches, patches, strict=True):
            assert torch.equal(actual, original)
        assert not request_data.pending_feedback_queue
        assert request_data.stop_pending and request_data.req.finished_reason is None
        request_data.req.output_ids.append(CONTROL_TOKEN_ID)
    replay(runner, request)
    runner.prepare_flow_row.assert_not_called()
    runner.run_flow_rows.assert_not_called()
    runner.stop_predicted.assert_not_called()
    forward = SimpleNamespace(input_ids=torch.tensor([CONTROL_TOKEN_ID]))
    runner.before_decode(forward, None, [request])
    assert forward.input_embeds.shape == (1, 8)
    runner.post_decode(make_result(1), None, None, [request])
    assert isinstance(request_data.req.finished_reason, FINISH_MATCHED_TOKEN)
    assert not request_data.stop_pending
    runner.prepare_flow_row.assert_not_called()
