# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
import types
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.payload_types import MossTTSState
from sglang_omni.models.moss_tts.request_builders import (
    make_moss_tts_scheduler_adapters,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.types import RequestOutput


@pytest.fixture(autouse=True)
def cleanup_model_runner_import():
    yield
    sys.modules.pop("sglang_omni.models.moss_tts.model_runner", None)


def install_fake_sglang_sampler(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delitem(
        sys.modules, "sglang_omni.models.moss_tts.model_runner", raising=False
    )
    modules = {
        "sglang": types.ModuleType("sglang"),
        "sglang.srt": types.ModuleType("sglang.srt"),
        "sglang.srt.layers": types.ModuleType("sglang.srt.layers"),
        "sglang.srt.layers.sampler": types.ModuleType("sglang.srt.layers.sampler"),
    }
    for name in ("sglang", "sglang.srt", "sglang.srt.layers"):
        modules[name].__path__ = []
    modules["sglang"].srt = modules["sglang.srt"]
    modules["sglang.srt"].layers = modules["sglang.srt.layers"]
    modules["sglang.srt.layers"].sampler = modules["sglang.srt.layers.sampler"]
    modules["sglang.srt.layers.sampler"].multinomial_with_seed = (
        lambda probs, seeds, positions: torch.argmax(probs, dim=-1, keepdim=True)
    )
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)


def test_state_pool_acquire_release_and_exhaustion() -> None:
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=5,
        device="cpu",
        dtype=torch.float32,
    )

    assert pool.padding_row == 1

    row = pool.acquire_row("req-a")
    assert row == 0
    assert pool.acquire_row("req-a") == row
    assert pool.row_for("req-a") == row

    with pytest.raises(RuntimeError, match="MOSS decode state pool exhausted"):
        pool.acquire_row("req-b")

    pool.release_row("req-a")
    assert pool.row_for("req-a") is None
    assert pool.acquire_row("req-b") == row


def test_state_pool_release_resets_stale_request_state() -> None:
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=5,
        device="cpu",
        dtype=torch.float32,
    )

    row = pool.acquire_row("req-a")
    pool.write_feedback(row, torch.tensor([1.0, 2.0, 3.0]))
    pool.append_generated_row(row, torch.tensor([10, 11, 12, 13, 14]))
    pool.mark_stop(row, kind=1, value=99)

    assert torch.equal(pool.feedback_or_zero(row), torch.tensor([1.0, 2.0, 3.0]))
    assert pool.generated_history(row).shape == (1, 5)
    assert pool.stop_pending[row].item()

    pool.release_row("req-a")
    reused = pool.acquire_row("req-b")
    assert reused == row

    assert torch.equal(pool.feedback_or_zero(reused), torch.zeros(3))
    assert pool.generated_history(reused).shape == (0, 5)
    assert not pool.stop_pending[reused].item()


def test_state_pool_generated_history_is_bounded() -> None:
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=2,
        max_new_tokens=2,
        num_channels=3,
        device="cpu",
        dtype=torch.float32,
    )

    row = pool.acquire_row("req-a")
    first = torch.tensor([1, 2, 3])
    second = torch.tensor([4, 5, 6])
    pool.append_generated_row(row, first)
    pool.append_generated_row(row, second)

    assert torch.equal(pool.generated_history(row), torch.stack([first, second]))

    with pytest.raises(RuntimeError, match="generated history is full"):
        pool.append_generated_row(row, torch.tensor([7, 8, 9]))


def test_moss_result_adapter_releases_decode_state_row() -> None:
    released: list[str] = []
    model = SimpleNamespace(
        reset_request=lambda request_id: released.append(str(request_id))
    )
    _, result_adapter = make_moss_tts_scheduler_adapters(model=model)
    payload = StagePayload(
        request_id="req-a",
        request=OmniRequest(inputs="hello"),
        data=MossTTSState(text="hello").to_dict(),
    )
    data = SimpleNamespace(
        stage_payload=payload,
        state=MossTTSState(text="hello"),
        assistant_prefix_rows=None,
        output_rows=[],
        prompt_rows=torch.empty((0, 3), dtype=torch.long),
        input_ids=torch.tensor([1, 2, 3], dtype=torch.long),
        engine_start_s=0.0,
    )

    result = result_adapter(data)

    assert result.request_id == "req-a"
    assert released == ["req-a"]


def test_model_reset_request_releases_decode_state_row() -> None:
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=3,
        device="cpu",
        dtype=torch.float32,
    )
    model = SimpleNamespace(_decode_state_pool=pool)
    model.reset_request = lambda request_id: model._decode_state_pool.release_row(
        request_id
    )
    row = pool.acquire_row("req-a")
    pool.write_feedback(row, torch.tensor([1.0, 2.0, 3.0]))
    pool.delay_initialized[row] = True
    pool.append_generated_row(row, torch.tensor([7, 8, 9]))
    pool.mark_stop(row, kind=2, value=4)

    model.reset_request("req-a")
    reused = pool.acquire_row("req-b")

    assert reused == row
    assert torch.equal(pool.feedback_or_zero(reused), torch.zeros(3))
    assert not pool.delay_initialized[reused].item()
    assert pool.generated_history(reused).shape == (0, 3)
    assert not pool.stop_pending[reused].item()


def test_decode_input_embedding_reads_feedback_from_state_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=2,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=5,
        device="cpu",
        dtype=torch.float32,
    )
    row_a = pool.acquire_row("req-a")
    row_b = pool.acquire_row("req-b")
    pool.write_feedback(row_a, torch.tensor([1.0, 2.0, 3.0]))
    pool.write_feedback(row_b, torch.tensor([4.0, 5.0, 6.0]))

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    embedding = torch.nn.Embedding(4, 3)
    runner.model = SimpleNamespace(
        hidden_size=3,
        _decode_input_embedding=embedding,
        _decode_state_pool=pool,
    )
    forward_batch = SimpleNamespace(input_ids=torch.full((2,), 99, dtype=torch.long))
    requests = [
        SimpleNamespace(
            request_id="req-a",
            data=SimpleNamespace(pending_feedback_queue=[]),
        ),
        SimpleNamespace(
            request_id="req-b",
            data=SimpleNamespace(pending_feedback_queue=[]),
        ),
    ]

    runner._write_decode_input_embedding(forward_batch, requests)

    assert forward_batch.input_ids.tolist() == [0, 1]
    assert torch.equal(embedding.weight[0].detach(), torch.tensor([1.0, 2.0, 3.0]))
    assert torch.equal(embedding.weight[1].detach(), torch.tensor([4.0, 5.0, 6.0]))
    assert requests[0].data.pending_feedback_queue == []
    assert requests[1].data.pending_feedback_queue == []


def test_decode_input_embedding_survives_batch_reorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=2,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=5,
        device="cpu",
        dtype=torch.float32,
    )
    row_a = pool.acquire_row("req-a")
    row_b = pool.acquire_row("req-b")
    pool.write_feedback(row_a, torch.tensor([1.0, 2.0, 3.0]))
    pool.write_feedback(row_b, torch.tensor([4.0, 5.0, 6.0]))

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    embedding = torch.nn.Embedding(4, 3)
    runner.model = SimpleNamespace(
        hidden_size=3,
        _decode_input_embedding=embedding,
        _decode_state_pool=pool,
    )
    forward_batch = SimpleNamespace(input_ids=torch.full((2,), 99, dtype=torch.long))
    requests = [
        SimpleNamespace(request_id="req-b", data=SimpleNamespace()),
        SimpleNamespace(request_id="req-a", data=SimpleNamespace()),
    ]

    runner._write_decode_input_embedding(forward_batch, requests)

    assert torch.equal(embedding.weight[0].detach(), torch.tensor([4.0, 5.0, 6.0]))
    assert torch.equal(embedding.weight[1].detach(), torch.tensor([1.0, 2.0, 3.0]))


def test_collect_moss_step_writes_next_feedback_to_state_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=3,
        device="cpu",
        dtype=torch.float32,
    )
    row = pool.acquire_row("req-a")
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner._pending_rows = None
    runner._pending_embeds = None
    runner.model = SimpleNamespace(
        device=torch.device("cpu"),
        hidden_size=3,
        _decode_state_pool=pool,
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32) + 0.5,
    )
    runner._channel_logits_from_result = lambda result, forward_batch: [
        torch.empty(1, 1),
        torch.empty(1, 1),
    ]
    runner._sample_rows = lambda channel_logits, datas, n_vq, **kwargs: torch.tensor(
        [[7, 8, 9]], dtype=torch.long
    )
    result = SimpleNamespace(next_token_ids=None)
    schedule_batch = SimpleNamespace(output_ids=None)
    requests = [
        SimpleNamespace(
            request_id="req-a",
            data=SimpleNamespace(pending_feedback_queue=[]),
        )
    ]

    runner._collect_moss_step(result, object(), schedule_batch, requests)

    assert torch.equal(pool.feedback_or_zero(row), torch.tensor([7.5, 8.5, 9.5]))
    assert torch.equal(pool.generated_history(row), torch.tensor([[7, 8, 9]]))
    assert torch.equal(result.next_token_ids, torch.tensor([7]))
    assert torch.equal(schedule_batch.output_ids, torch.tensor([7]))
    assert int(pool.sampling_steps[row].item()) == 1


def test_collect_moss_step_does_not_write_feedback_for_im_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=3,
        device="cpu",
        dtype=torch.float32,
    )
    row = pool.acquire_row("req-a")
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner._pending_rows = None
    runner._pending_embeds = None
    runner.model = SimpleNamespace(
        config=SimpleNamespace(im_end_token_id=5),
        device=torch.device("cpu"),
        hidden_size=3,
        _decode_state_pool=pool,
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32) + 0.5,
    )
    runner._channel_logits_from_result = lambda result, forward_batch: [
        torch.empty(1, 1),
        torch.empty(1, 1),
    ]
    runner._sample_rows = lambda channel_logits, datas, n_vq, **kwargs: torch.tensor(
        [[5, 8, 9]], dtype=torch.long
    )
    result = SimpleNamespace(next_token_ids=None)
    schedule_batch = SimpleNamespace(output_ids=None)
    requests = [
        SimpleNamespace(
            request_id="req-a",
            data=SimpleNamespace(pending_feedback_queue=[]),
        )
    ]

    runner._collect_moss_step(result, object(), schedule_batch, requests)

    assert torch.equal(pool.feedback_or_zero(row), torch.zeros(3))
    assert pool.generated_history(row).shape == (0, 3)
    assert pool.stop_pending[row].item()
    assert runner._pending_journals is not None
    assert runner._pending_journals[0].emit_output_row is False


def test_collect_moss_step_freezes_feedback_at_length_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=3,
        device="cpu",
        dtype=torch.float32,
    )
    row = pool.acquire_row("req-a")
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner._pending_rows = None
    runner._pending_embeds = None
    runner.model = SimpleNamespace(
        config=SimpleNamespace(im_end_token_id=5),
        device=torch.device("cpu"),
        hidden_size=3,
        _decode_state_pool=pool,
        _prepare_multi_modal_inputs=lambda rows: rows.to(torch.float32) + 0.5,
    )
    runner._channel_logits_from_result = lambda result, forward_batch: [
        torch.empty(1, 1),
        torch.empty(1, 1),
    ]
    runner._sample_rows = lambda channel_logits, datas, n_vq, **kwargs: torch.tensor(
        [[7, 8, 9]], dtype=torch.long
    )
    result = SimpleNamespace(next_token_ids=None)
    schedule_batch = SimpleNamespace(output_ids=None)
    req = SimpleNamespace(
        output_ids=[101],
        sampling_params=SimpleNamespace(max_new_tokens=2),
    )
    data = SimpleNamespace(req=req, pending_feedback_queue=[])
    requests = [SimpleNamespace(request_id="req-a", data=data)]

    runner._collect_moss_step(result, object(), schedule_batch, requests)

    assert torch.equal(pool.feedback_or_zero(row), torch.zeros(3))
    assert torch.equal(pool.generated_history(row), torch.tensor([[7, 8, 9]]))
    assert pool.stop_pending[row].item()
    assert runner._pending_journals is not None
    journal = runner._pending_journals[0]
    assert journal.emit_output_row is True
    assert journal.finish_kind == "length"
    assert journal.finish_value == 2


def test_sample_rows_uses_pool_sampling_steps_for_early_im_end_mask(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import (
        _INT64_MAX,
        MossTTSModelRunner,
    )
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    cfg = SimpleNamespace(
        pad_token_id=0,
        audio_assistant_gen_slot_token_id=1,
        audio_assistant_delay_slot_token_id=2,
        audio_start_token_id=3,
        audio_end_token_id=4,
        im_end_token_id=5,
        audio_pad_code=6,
    )
    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=2,
        device="cpu",
        dtype=torch.float32,
    )
    row = pool.acquire_row("req-a")
    pool.delay_state[row] = torch.tensor([0, _INT64_MAX, 0], dtype=torch.long)
    pool.sampling_steps[row] = 0

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=cfg, _decode_state_pool=pool)
    data = SimpleNamespace(
        generation_steps=99,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        sampling_seed=123,
        prompt_rows=None,
        output_rows=[],
        delay_state=torch.tensor([0, _INT64_MAX, 0], dtype=torch.long),
        audio_length=0,
        delayed_length=_INT64_MAX,
        is_audio=False,
    )
    text_logits = torch.zeros(1, 7)
    text_logits[0, cfg.im_end_token_id] = 100.0
    text_logits[0, cfg.audio_start_token_id] = 90.0
    audio_logits = torch.zeros(1, 7)

    rows = runner._sample_rows(
        [text_logits, audio_logits],
        [data],
        n_vq=1,
        pool_rows=torch.tensor([row], dtype=torch.long),
    )

    assert int(rows[0, 0].item()) == cfg.audio_start_token_id


def test_sample_rows_initializes_pool_delay_state_from_request_data(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import (
        _INT64_MAX,
        MossTTSModelRunner,
    )
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    cfg = SimpleNamespace(
        pad_token_id=0,
        audio_assistant_gen_slot_token_id=1,
        audio_assistant_delay_slot_token_id=2,
        audio_start_token_id=3,
        audio_end_token_id=4,
        im_end_token_id=5,
        audio_pad_code=6,
    )
    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=2,
        device="cpu",
        dtype=torch.float32,
    )
    row = pool.acquire_row("req-a")
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=cfg, _decode_state_pool=pool)
    data = SimpleNamespace(
        generation_steps=0,
        text_temperature=0.0,
        text_top_p=1.0,
        text_top_k=-1,
        audio_temperature=0.0,
        audio_top_p=1.0,
        audio_top_k=-1,
        audio_repetition_penalty=1.0,
        sampling_seed=123,
        prompt_rows=None,
        output_rows=[],
        delay_state=torch.tensor([0, _INT64_MAX, 0], dtype=torch.long),
        audio_length=0,
        delayed_length=_INT64_MAX,
        is_audio=False,
    )
    text_logits = torch.zeros(1, 7)
    text_logits[0, cfg.im_end_token_id] = 100.0
    text_logits[0, cfg.audio_start_token_id] = 90.0
    audio_logits = torch.zeros(1, 7)

    rows = runner._sample_rows(
        [text_logits, audio_logits],
        [data],
        n_vq=1,
        pool_rows=torch.tensor([row], dtype=torch.long),
    )

    assert int(rows[0, 0].item()) == cfg.audio_start_token_id


def test_repetition_penalty_reads_generated_history_from_state_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeStatePool

    pool = MossDecodeStatePool(
        max_batch_size=1,
        hidden_size=3,
        max_new_tokens=4,
        num_channels=2,
        device="cpu",
        dtype=torch.float32,
    )
    row = pool.acquire_row("req-a")
    pool.append_generated_row(row, torch.tensor([10, 2]))
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(_decode_state_pool=pool)
    audio_logits = torch.zeros(1, 1, 8)
    audio_logits[0, 0, 2] = 10.0
    data = SimpleNamespace(
        audio_repetition_penalty=2.0,
        prompt_rows=None,
        output_rows=[],
    )

    runner._apply_audio_repetition_penalty(
        audio_logits,
        [data],
        n_vq=1,
        pool_rows=torch.tensor([row], dtype=torch.long),
    )

    assert float(audio_logits[0, 0, 2].item()) == 5.0


def test_post_process_outputs_applies_decode_journals_without_pending_embeds(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeJournal

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(im_end_token_id=5))
    runner._pending_rows = None
    runner._pending_embeds = None
    runner._pending_journals = [
        MossDecodeJournal(
            request_id="req-a",
            row=0,
            sampled_row=torch.tensor([7, 8, 9]),
            next_token_id=7,
            emit_output_row=True,
        )
    ]
    data = SimpleNamespace(output_rows=[], pending_feedback_queue=[])
    scheduler_output = SimpleNamespace(
        requests=[SimpleNamespace(request_id="req-a", data=data)]
    )
    outputs = {"req-a": RequestOutput(request_id="req-a", data=7)}

    runner.post_process_outputs(object(), scheduler_output, outputs)

    assert len(data.output_rows) == 1
    assert torch.equal(data.output_rows[0], torch.tensor([7, 8, 9]))
    assert data.pending_feedback_queue == []
    assert runner._pending_journals is None


def test_post_process_outputs_applies_journals_by_request_id_after_reorder(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    install_fake_sglang_sampler(monkeypatch)

    from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
    from sglang_omni.models.moss_tts.state_pool import MossDecodeJournal

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=SimpleNamespace(im_end_token_id=5))
    runner._pending_rows = None
    runner._pending_embeds = None
    runner._pending_journals = [
        MossDecodeJournal(
            request_id="req-a",
            row=0,
            sampled_row=torch.tensor([7, 8, 9]),
            next_token_id=7,
            emit_output_row=True,
        ),
        MossDecodeJournal(
            request_id="req-b",
            row=1,
            sampled_row=torch.tensor([10, 11, 12]),
            next_token_id=10,
            emit_output_row=True,
        ),
    ]
    data_a = SimpleNamespace(output_rows=[], pending_feedback_queue=[])
    data_b = SimpleNamespace(output_rows=[], pending_feedback_queue=[])
    scheduler_output = SimpleNamespace(
        requests=[
            SimpleNamespace(request_id="req-b", data=data_b),
            SimpleNamespace(request_id="req-a", data=data_a),
        ]
    )
    outputs = {
        "req-a": RequestOutput(request_id="req-a", data=7),
        "req-b": RequestOutput(request_id="req-b", data=10),
    }

    runner.post_process_outputs(object(), scheduler_output, outputs)

    assert torch.equal(data_a.output_rows[0], torch.tensor([7, 8, 9]))
    assert torch.equal(data_b.output_rows[0], torch.tensor([10, 11, 12]))
