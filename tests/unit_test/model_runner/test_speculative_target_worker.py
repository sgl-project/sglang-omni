# SPDX-License-Identifier: Apache-2.0
"""CPU contracts for Omni's native DFlash target adapter."""

from __future__ import annotations

from types import MethodType, SimpleNamespace

import pytest
import torch
from sglang.srt.model_executor.forward_batch_info import (
    CaptureHiddenMode,
    ForwardBatch,
    ForwardMode,
)
from sglang.srt.speculative import dflash_worker_v2

from sglang_omni.model_runner.base import ModelRunner
from sglang_omni.model_runner.prefill_inputs import (
    OmniPrefillInputs,
    attach_omni_prefill_inputs,
    get_omni_prefill_inputs,
)
from sglang_omni.model_runner.speculative_target_worker import SpeculativeTargetWorker
from sglang_omni.scheduling.types import SchedulerRequest


def make_speculative_target_probe(
    *, is_prefill_only: bool = False
) -> tuple[SpeculativeTargetWorker, SimpleNamespace, list[str]]:
    calls: list[str] = []
    forward_batch = SimpleNamespace(
        input_ids=torch.tensor([1, 2, 3]),
        seq_lens=torch.tensor([3]),
        capture_hidden_mode=CaptureHiddenMode.FULL,
        forward_mode=ForwardMode.EXTEND,
        sampling_info=SimpleNamespace(is_all_greedy=True),
        is_prefill_only=is_prefill_only,
        apply_deprecated_skip_attn_backend_init=lambda skip: calls.append(
            f"attention:{skip}"
        ),
    )
    logits_output = SimpleNamespace(hidden_states=torch.ones(3, 20))

    def native_forward(*, forward_batch: SimpleNamespace) -> SimpleNamespace:
        calls.append("verify")
        return SimpleNamespace(
            logits_output=logits_output,
            can_run_graph=True,
            expert_distribution_metrics=None,
            routed_experts_output=None,
            indexer_topk_output=None,
        )

    worker = SimpleNamespace(
        device="cpu",
        model_runner=SimpleNamespace(
            model_config=SimpleNamespace(hidden_size=4, vocab_size=19),
            forward=native_forward,
        ),
    )

    def prepare_and_forward(
        prepared_batch: SimpleNamespace,
        schedule_batch: SimpleNamespace,
        requests: list[SchedulerRequest],
        *,
        is_prefill: bool,
    ) -> SimpleNamespace:
        assert prepared_batch is forward_batch
        assert is_prefill
        assert requests[0].data is schedule_batch.reqs[0].omni_data
        calls.append("omni_prefill")
        return SimpleNamespace(
            logits_output=logits_output, next_token_ids=None, can_run_cuda_graph=False
        )

    def sample_next_token_ids(
        logits_output: SimpleNamespace,
        prepared_batch: SimpleNamespace,
        schedule_batch: SimpleNamespace,
        requests: list[SchedulerRequest],
    ) -> torch.Tensor:
        assert prepared_batch is forward_batch
        calls.append("sample")
        return torch.tensor([7])

    thinker_runner = SimpleNamespace(
        tp_worker=worker,
        outer_model=SimpleNamespace(capture_aux_hidden_states=True),
        text_model=SimpleNamespace(layers_to_capture=[2, 13, 24, 35, 46]),
        prepare_and_forward=prepare_and_forward,
        sample_next_token_ids=sample_next_token_ids,
    )
    adapter = SpeculativeTargetWorker(worker)
    adapter.attach_runner(thinker_runner)
    return adapter, forward_batch, calls


def test_native_sampling_prewarm_can_resolve_target_model_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    adapter, _, _ = make_speculative_target_probe()
    draft_worker = dflash_worker_v2.DFlashWorkerV2.__new__(
        dflash_worker_v2.DFlashWorkerV2
    )
    draft_worker._target_worker = adapter  # noqa: leading-underscore
    draft_worker.block_size = 8
    monkeypatch.setattr(dflash_worker_v2, "is_cuda", lambda: True)
    monkeypatch.setattr(
        dflash_worker_v2, "is_dflash_sampling_verify_available", lambda: True
    )

    def reach_allocation_boundary(block_size: int) -> int:
        assert block_size == 8
        assert draft_worker.model_config is adapter.model_runner.model_config
        raise RuntimeError("CPU probe reached prewarm allocation boundary")

    monkeypatch.setattr(draft_worker, "_prewarm_batch_size", reach_allocation_boundary)
    with pytest.raises(RuntimeError, match="CPU probe reached prewarm"):
        draft_worker.prewarm_sampling()


@pytest.mark.parametrize("is_prefill_only", [False, True])
def test_speculative_target_prefill_keeps_omni_hooks_and_samples_once(
    monkeypatch: pytest.MonkeyPatch, is_prefill_only: bool
) -> None:
    adapter, forward_batch, calls = make_speculative_target_probe(
        is_prefill_only=is_prefill_only
    )
    schedule_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[SimpleNamespace(rid="request", omni_data=SimpleNamespace())],
    )

    def init_new(
        batch, runner, *, capture_hidden_mode, return_hidden_states_before_norm
    ):
        assert batch is schedule_batch
        assert runner is adapter.model_runner
        assert capture_hidden_mode == CaptureHiddenMode.FULL
        assert not return_hidden_states_before_norm
        calls.append("init_prefill")
        return forward_batch

    monkeypatch.setattr(ForwardBatch, "init_new", init_new)

    result = adapter.forward_batch_generation(
        schedule_batch, capture_hidden_mode=CaptureHiddenMode.FULL
    )

    assert calls[:3] == ["init_prefill", "attention:None", "omni_prefill"]
    if is_prefill_only:
        assert calls[3:] == []
        assert result.next_token_ids.tolist() == [0]
    else:
        assert calls[3:] == ["sample"]
        assert result.next_token_ids.tolist() == [7]
    assert result.logits_output.hidden_states.shape == (3, 20)


def test_speculative_target_verification_preserves_native_batch_and_skips_sampling() -> (
    None
):
    adapter, forward_batch, calls = make_speculative_target_probe()
    forward_batch.forward_mode = ForwardMode.TARGET_VERIFY

    result = adapter.forward_batch_generation(
        batch=None,
        forward_batch=forward_batch,
        is_verify=True,
        skip_attn_backend_init=True,
    )

    assert calls == ["attention:True", "verify"]
    assert result.can_run_cuda_graph
    assert result.next_token_ids is None
    assert result.logits_output.hidden_states.shape == (3, 20)


@pytest.mark.parametrize("hidden_shape", [(3, 4), (2, 20), None])
def test_speculative_target_rejects_final_hidden_or_missing_auxiliary_rows(
    hidden_shape: tuple[int, int] | None,
) -> None:
    adapter, forward_batch, calls = make_speculative_target_probe()
    forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
    native_output = adapter.model_runner.forward(forward_batch=forward_batch)
    native_output.logits_output.hidden_states = (
        torch.zeros(hidden_shape) if hidden_shape is not None else None
    )

    with pytest.raises(AssertionError, match="DFlash target"):
        adapter.forward_batch_generation(None, forward_batch, is_verify=True)


def test_speculative_target_rejects_non_greedy_before_launch() -> None:
    adapter, forward_batch, calls = make_speculative_target_probe()
    forward_batch.forward_mode = ForwardMode.TARGET_VERIFY
    forward_batch.sampling_info.is_all_greedy = False

    with pytest.raises(ValueError, match="only greedy"):
        adapter.forward_batch_generation(None, forward_batch, is_verify=True)

    assert calls == []


@pytest.mark.parametrize("failing_phase", [None, "before", "custom", "forward"])
def test_prefill_sidecar_is_cleared_after_every_forward_exit(
    monkeypatch: pytest.MonkeyPatch, failing_phase: str | None
) -> None:
    adapter, forward_batch, calls = make_speculative_target_probe()
    forward_batch.replace_embeds = None
    request = SimpleNamespace(
        rid="middle-chunk",
        omni_data=SimpleNamespace(),
        inflight_middle_chunks=1,
        output_ids=[],
    )
    schedule_batch = SimpleNamespace(
        forward_mode=ForwardMode.EXTEND,
        reqs=[request],
        is_prefill_only=False,
        prefix_lens=[9],
        extend_lens=[3],
    )
    forward_batch.extend_prefix_lens_cpu = [9]
    forward_batch.extend_seq_lens_cpu = [3]
    forward_batch.mrope_positions = torch.arange(9, 12).repeat(3, 1)
    input_embeddings = torch.ones(3, 4)
    runner = adapter.thinker_runner
    runner.prepare_and_forward = MethodType(ModelRunner.prepare_and_forward, runner)

    def before_prefill(prepared_batch, batch, requests):
        assert batch is schedule_batch
        assert requests[0].data is request.omni_data
        attach_omni_prefill_inputs(
            prepared_batch, OmniPrefillInputs(input_embeds=input_embeddings)
        )
        calls.append("before")
        if failing_phase == "before":
            raise RuntimeError("before failed")
        else:
            pass

    def custom_prefill_forward(prepared_batch, batch, requests):
        calls.append("custom")
        if failing_phase == "custom":
            raise RuntimeError("custom failed")
        else:
            return None

    def standard_forward(prepared_batch):
        assert prepared_batch is forward_batch
        assert get_omni_prefill_inputs(prepared_batch).input_embeds is input_embeddings
        assert prepared_batch.extend_prefix_lens_cpu == [9]
        assert prepared_batch.extend_seq_lens_cpu == [3]
        assert prepared_batch.mrope_positions.tolist() == [[9, 10, 11]] * 3
        calls.append("forward")
        if failing_phase == "forward":
            raise RuntimeError("forward failed")
        else:
            return SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=torch.ones(3, 20)),
                next_token_ids=None,
                can_run_cuda_graph=False,
            )

    runner.before_prefill = before_prefill
    runner.custom_prefill_forward = custom_prefill_forward
    runner.cleanup_prefill = lambda *args: calls.append("cleanup")
    runner.sample_before_post_prefill = lambda *args: False
    runner.tp_worker.forward_batch_generation = standard_forward
    monkeypatch.setattr(ForwardBatch, "init_new", lambda *args, **kwargs: forward_batch)

    if failing_phase is None:
        result = adapter.forward_batch_generation(
            schedule_batch, capture_hidden_mode=CaptureHiddenMode.FULL
        )
        assert result.next_token_ids.tolist() == [7]
        assert calls == [
            "attention:None",
            "before",
            "custom",
            "forward",
            "cleanup",
            "sample",
        ]
    else:
        with pytest.raises(RuntimeError, match=f"{failing_phase} failed"):
            adapter.forward_batch_generation(
                schedule_batch, capture_hidden_mode=CaptureHiddenMode.FULL
            )
        assert calls[-1] == "cleanup"
        assert "sample" not in calls
    assert get_omni_prefill_inputs(forward_batch) is None
    assert request.output_ids == []
    assert request.inflight_middle_chunks == 1


def test_target_adapter_fails_closed_before_attaching_the_runner() -> None:
    adapter, forward_batch, calls = make_speculative_target_probe()
    adapter.thinker_runner = None

    with pytest.raises(RuntimeError, match="not attached"):
        adapter.forward_batch_generation(None, forward_batch, is_verify=True)

    assert calls == []
