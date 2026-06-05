# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import importlib
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Iterator

import pytest
import torch

from sglang_omni.models.moss_tts.request_builders import MossTTSSGLangRequestData
from sglang_omni.scheduling.types import RequestOutput

_MISSING = object()


def _multinomial_with_seed(
    probs: torch.Tensor, seeds: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    del seeds, positions
    return torch.multinomial(probs, num_samples=1)


@contextmanager
def _patched_moss_runner_cls() -> Iterator[type]:
    sampler_module_name = "sglang.srt.layers.sampler"
    runner_module_name = "sglang_omni.models.moss_tts.model_runner"
    parent_module_name = "sglang_omni.models.moss_tts"
    runner_attr = "model_runner"

    previous_sampler = sys.modules.get(sampler_module_name, _MISSING)
    previous_runner = sys.modules.get(runner_module_name, _MISSING)
    parent_module = sys.modules.get(parent_module_name)
    previous_parent_attr = (
        getattr(parent_module, runner_attr, _MISSING)
        if parent_module is not None
        else _MISSING
    )

    sampler_stub = types.ModuleType(sampler_module_name)
    sampler_stub.multinomial_with_seed = _multinomial_with_seed
    sys.modules[sampler_module_name] = sampler_stub
    sys.modules.pop(runner_module_name, None)

    try:
        module = importlib.import_module(runner_module_name)
        yield module.MossTTSModelRunner
    finally:
        sys.modules.pop(runner_module_name, None)
        if previous_runner is not _MISSING:
            sys.modules[runner_module_name] = previous_runner

        if previous_sampler is _MISSING:
            sys.modules.pop(sampler_module_name, None)
        else:
            sys.modules[sampler_module_name] = previous_sampler

        if parent_module is not None:
            if previous_parent_attr is _MISSING:
                try:
                    delattr(parent_module, runner_attr)
                except AttributeError:
                    pass
            else:
                setattr(parent_module, runner_attr, previous_parent_attr)


@pytest.fixture()
def moss_runner_cls():
    with _patched_moss_runner_cls() as runner_cls:
        yield runner_cls


def _cfg() -> SimpleNamespace:
    return SimpleNamespace(
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=4,
        im_end_token_id=14,
    )


def _runner_for_commit_tests(runner_cls: type):
    runner = runner_cls.__new__(runner_cls)
    runner.model = SimpleNamespace(
        config=SimpleNamespace(im_end_token_id=14),
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32)[:, :3],
    )
    runner._pending_rows = None
    runner._pending_embeds = None
    return runner


def _unfinished_req(*, max_new_tokens: int = 8):
    req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=max_new_tokens),
        finished_reason=None,
    )
    req.finished = lambda: req.finished_reason is not None
    return req


def _eos_collect_fixture(runner_cls: type):
    cfg = _cfg()
    runner = runner_cls.__new__(runner_cls)
    runner.model = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32),
    )
    req = _unfinished_req()
    data = MossTTSSGLangRequestData(
        req=req,
        generation_steps=0,
        moss_sampling_steps=3,
        sampling_seed=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
    )
    text_logits = torch.full((1, 20), -100.0)
    text_logits[0, cfg.im_end_token_id] = 10.0
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            customized_info={
                "moss_tts_channel_logits": [
                    text_logits,
                    torch.zeros((1, 5)),
                    torch.zeros((1, 5)),
                ]
            }
        )
    )
    sched_req = SimpleNamespace(request_id="req-eos", data=data)
    return runner, req, data, result, sched_req, cfg


def test_moss_runner_patch_restores_modules_in_process() -> None:
    sampler_module_name = "sglang.srt.layers.sampler"
    runner_module_name = "sglang_omni.models.moss_tts.model_runner"
    parent_module_name = "sglang_omni.models.moss_tts"
    runner_attr = "model_runner"

    previous_sampler = sys.modules.get(sampler_module_name, _MISSING)
    previous_runner = sys.modules.get(runner_module_name, _MISSING)
    parent_module = sys.modules.get(parent_module_name)
    previous_parent_attr = (
        getattr(parent_module, runner_attr, _MISSING)
        if parent_module is not None
        else _MISSING
    )

    with _patched_moss_runner_cls() as runner_cls:
        assert runner_cls.__name__ == "MossTTSModelRunner"
        assert sys.modules.get(sampler_module_name) is not previous_sampler

    if previous_sampler is _MISSING:
        assert sampler_module_name not in sys.modules
    else:
        assert sys.modules.get(sampler_module_name) is previous_sampler
    if previous_runner is _MISSING:
        assert runner_module_name not in sys.modules
    else:
        assert sys.modules.get(runner_module_name) is previous_runner
    if parent_module is not None:
        if previous_parent_attr is _MISSING:
            assert not hasattr(parent_module, runner_attr)
        else:
            assert getattr(parent_module, runner_attr) is previous_parent_attr


def test_moss_async_state_defaults_follow_sync_generation_step() -> None:
    data = MossTTSSGLangRequestData(generation_steps=3)

    assert data.moss_sampling_steps == 3
    assert data.moss_stop_pending is False


def test_moss_sampling_steps_falls_back_to_zero_when_generation_steps_is_none(
    moss_runner_cls,
) -> None:
    data = SimpleNamespace(generation_steps=None)

    assert moss_runner_cls._moss_sampling_steps(data) == 0
    assert data.moss_sampling_steps == 0


def test_moss_sampling_steps_uses_generation_steps_as_floor(
    moss_runner_cls,
) -> None:
    data = SimpleNamespace(generation_steps=4, moss_sampling_steps=0)

    assert moss_runner_cls._moss_sampling_steps(data) == 4
    assert data.moss_sampling_steps == 4


def test_sample_rows_uses_moss_sampling_steps_not_finalize_generation_steps(
    moss_runner_cls,
) -> None:
    cfg = _cfg()
    runner = moss_runner_cls.__new__(moss_runner_cls)
    runner.model = SimpleNamespace(config=cfg)
    data = SimpleNamespace(
        generation_steps=0,
        moss_sampling_steps=3,
        audio_length=0,
        delayed_length=-1,
        is_audio=False,
        delay_state=None,
        sampling_seed=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
    )
    text_logits = torch.full((1, 20), -100.0)
    text_logits[0, cfg.im_end_token_id] = 10.0
    audio0_logits = torch.zeros((1, 5))
    audio1_logits = torch.zeros((1, 5))

    rows = runner._sample_rows(
        [text_logits, audio0_logits, audio1_logits], [data], n_vq=2
    )

    assert int(rows[0, 0]) == cfg.im_end_token_id


def test_collect_moss_step_advances_moss_sampling_steps(
    moss_runner_cls,
) -> None:
    cfg = _cfg()
    runner = moss_runner_cls.__new__(moss_runner_cls)
    runner.model = SimpleNamespace(
        config=cfg,
        device=torch.device("cpu"),
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32),
    )
    data = SimpleNamespace(
        generation_steps=0,
        moss_sampling_steps=0,
        audio_length=0,
        delayed_length=-1,
        is_audio=False,
        delay_state=None,
        sampling_seed=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
        pending_feedback_queue=[],
        req=SimpleNamespace(output_ids=[], finished=lambda: False),
        max_new_tokens=4,
    )
    text_logits = torch.zeros((1, 20))
    audio0_logits = torch.zeros((1, 5))
    audio1_logits = torch.zeros((1, 5))
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            customized_info={
                "moss_tts_channel_logits": [text_logits, audio0_logits, audio1_logits]
            }
        )
    )
    schedule_batch = SimpleNamespace(output_ids=None)

    runner._collect_moss_step(
        result,
        forward_batch=SimpleNamespace(),
        schedule_batch=schedule_batch,
        requests=[SimpleNamespace(data=data)],
    )

    assert data.moss_sampling_steps == 1


def test_sync_eos_collect_marks_sglang_request_finished(moss_runner_cls) -> None:
    runner, req, data, result, sched_req, cfg = _eos_collect_fixture(moss_runner_cls)

    runner.post_decode(
        result,
        forward_batch=SimpleNamespace(),
        schedule_batch=SimpleNamespace(output_ids=None),
        requests=[sched_req],
    )

    assert data.moss_stop_pending is True
    assert req.finished()
    assert req.finished_reason.to_json() == {
        "type": "stop",
        "matched": cfg.im_end_token_id,
    }


def test_async_eos_finish_is_deferred_until_resolve(moss_runner_cls) -> None:
    runner, req, data, result, sched_req, cfg = _eos_collect_fixture(moss_runner_cls)

    runner.post_decode_launch(
        result,
        forward_batch=SimpleNamespace(),
        requests=[sched_req],
    )

    assert result.next_token_ids.tolist() == [cfg.im_end_token_id]
    assert data.moss_stop_pending is True
    assert not req.finished()

    runner.post_decode_resolve(
        None,
        result,
        object(),
        object(),
        [sched_req],
    )

    assert req.finished()
    assert req.finished_reason.to_json() == {
        "type": "stop",
        "matched": cfg.im_end_token_id,
    }


def test_sync_length_boundary_marks_sglang_request_finished(moss_runner_cls) -> None:
    runner = _runner_for_commit_tests(moss_runner_cls)
    req = _unfinished_req(max_new_tokens=4)
    req.output_ids = [1, 2, 3]
    data = MossTTSSGLangRequestData(
        req=req,
        generation_steps=0,
        moss_sampling_steps=0,
    )
    result = SimpleNamespace()

    runner._commit_moss_rows(
        result,
        [SimpleNamespace(data=data)],
        torch.tensor([[12, 2, 4]], dtype=torch.long),
    )

    assert data.moss_stop_pending is True
    assert req.finished()
    assert req.finished_reason.to_json() == {"type": "length", "length": 4}


def test_commit_rows_appends_feedback_without_post_process_outputs(
    moss_runner_cls,
) -> None:
    runner = _runner_for_commit_tests(moss_runner_cls)
    req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=4),
        finished=lambda: False,
    )
    data = MossTTSSGLangRequestData(
        req=req,
        generation_steps=0,
        moss_sampling_steps=0,
    )
    sched_req = SimpleNamespace(data=data)
    result = SimpleNamespace()
    rows = torch.tensor([[12, 2, 4]], dtype=torch.long)

    runner._commit_moss_rows(result, [sched_req], rows)

    assert result.next_token_ids.tolist() == [12]
    assert [row.tolist() for row in data.output_rows] == [[12, 2, 4]]
    assert len(data.pending_feedback_queue) == 1
    assert torch.equal(
        data.pending_feedback_queue[0],
        torch.tensor([12.0, 2.0, 4.0]),
    )
    assert data.moss_sampling_steps == 1


def test_post_process_outputs_no_longer_appends_rows_or_feedback(
    moss_runner_cls,
) -> None:
    runner = _runner_for_commit_tests(moss_runner_cls)
    runner._pending_rows = torch.tensor([[12, 2, 4]], dtype=torch.long)
    runner._pending_embeds = torch.ones((1, 3))
    data = SimpleNamespace(output_rows=[], pending_feedback_queue=[])
    requests = [SimpleNamespace(request_id="active", data=data)]

    runner.post_process_outputs(
        object(),
        SimpleNamespace(requests=requests),
        {"active": RequestOutput("active", data=12)},
    )

    assert data.output_rows == []
    assert data.pending_feedback_queue == []
    assert runner._pending_rows is None
    assert runner._pending_embeds is None


def test_stop_pending_before_decode_uses_dummy_embedding_without_consuming_feedback(
    moss_runner_cls,
) -> None:
    runner = moss_runner_cls.__new__(moss_runner_cls)
    embedding = torch.nn.Embedding(2, 3)
    runner.model = SimpleNamespace(hidden_size=3, _decode_input_embedding=embedding)
    feedback = torch.ones(3)
    data = SimpleNamespace(
        moss_stop_pending=True,
        pending_feedback_queue=[feedback],
    )
    forward_batch = SimpleNamespace(input_ids=torch.full((1,), 99, dtype=torch.long))

    runner._write_decode_input_embedding(forward_batch, [SimpleNamespace(data=data)])

    assert forward_batch.input_ids.tolist() == [0]
    assert torch.equal(embedding.weight[0].detach(), torch.zeros(3))
    assert data.pending_feedback_queue == [feedback]


def test_stop_pending_collect_skips_overrun_row_and_counter(
    moss_runner_cls,
) -> None:
    runner = _runner_for_commit_tests(moss_runner_cls)
    data = MossTTSSGLangRequestData(
        output_rows=[],
        moss_sampling_steps=5,
        moss_stop_pending=True,
        req=SimpleNamespace(
            output_ids=[1, 2],
            sampling_params=SimpleNamespace(max_new_tokens=8),
            finished=lambda: False,
        ),
    )
    result = SimpleNamespace()

    runner._commit_moss_rows(
        result,
        [SimpleNamespace(data=data)],
        torch.tensor([[12, 2, 4]], dtype=torch.long),
    )

    assert result.next_token_ids.tolist() == [12]
    assert data.output_rows == []
    assert len(data.pending_feedback_queue) == 0
    assert data.moss_sampling_steps == 5


def test_post_decode_launch_collects_rows_and_publishes_token_ids(
    moss_runner_cls,
) -> None:
    runner = _runner_for_commit_tests(moss_runner_cls)

    def fake_collect(result, forward_batch, schedule_batch, requests, **kwargs):
        del forward_batch, schedule_batch
        rows = torch.tensor([[12, 2, 4]], dtype=torch.long)
        runner._commit_moss_rows(result, requests, rows, **kwargs)

    runner._collect_moss_step = fake_collect
    req = SimpleNamespace(
        output_ids=[],
        sampling_params=SimpleNamespace(max_new_tokens=8),
        finished=lambda: False,
    )
    data = MossTTSSGLangRequestData(
        output_rows=[],
        req=req,
        generation_steps=0,
        moss_sampling_steps=0,
    )
    result = SimpleNamespace(next_token_ids=None)

    host_buf = runner.post_decode_launch(
        result,
        object(),
        [SimpleNamespace(data=data)],
    )

    assert host_buf is None
    assert result.next_token_ids.tolist() == [12]
    assert [row.tolist() for row in data.output_rows] == [[12, 2, 4]]


def test_post_decode_resolve_is_noop_for_moss_state(
    moss_runner_cls,
) -> None:
    runner = _runner_for_commit_tests(moss_runner_cls)
    data = SimpleNamespace(output_rows=[], pending_feedback_queue=[])
    result = SimpleNamespace(next_token_ids=torch.tensor([12]))

    runner.post_decode_resolve(
        None,
        result,
        object(),
        object(),
        [SimpleNamespace(data=data)],
    )

    assert data.output_rows == []
    assert data.pending_feedback_queue == []
    assert result.next_token_ids.tolist() == [12]
