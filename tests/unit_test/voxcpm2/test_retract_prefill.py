# SPDX-License-Identifier: Apache-2.0
"""Rebuild continuous audio inputs after SGLang releases a request's KV."""

from array import array
from types import SimpleNamespace

import pytest
import torch
from sglang.srt.managers.schedule_batch import Req
from sglang.srt.sampling.sampling_params import SamplingParams

from sglang_omni.model_runner.prefill_inputs import get_omni_prefill_inputs
from sglang_omni.models.voxcpm2.model_runner import VoxCPM2ModelRunner
from sglang_omni.models.voxcpm2.request_builders import (
    VoxCPM2PrefillInputs,
    VoxCPM2SGLangRequestData,
    build_stream_output,
)


class _Model:
    graph_feedback_buffer = None

    def build_input_embeds(self, text_token, audio_feat, text_mask, audio_mask):
        return text_token.float().unsqueeze(1).expand(-1, 2).clone()


def _request(generated=0, *, rid="test"):
    params = SamplingParams(max_new_tokens=20, temperature=0.0)
    params.normalize(None)
    req = Req(
        rid=rid,
        origin_input_text="",
        origin_input_ids=array("q", [3, 4, 5]),
        sampling_params=params,
        eos_token_ids=set(),
        vocab_size=100,
    )
    data = VoxCPM2SGLangRequestData(
        req=req,
        prefill=VoxCPM2PrefillInputs(
            text_token=torch.tensor([3, 4, 5], dtype=torch.int32),
            audio_feat=torch.zeros(3, 4, 8),
            text_mask=torch.tensor([1, 0, 1], dtype=torch.int32),
            audio_mask=torch.tensor([0, 1, 0], dtype=torch.int32),
        ),
    )
    data.decode_input_embeds = [torch.full((2,), 10.0 + i) for i in range(generated)]
    req.output_ids.extend([0] * generated)
    req._omni_data = data
    return SimpleNamespace(data=data)


def _retract_and_prepare(request):
    req = request.data.req
    # note (Xinhao Tan): real reset/init methods retain generated token IDs
    # even with radix disabled; only the GPU KV allocation is omitted here.
    req.reset_for_retract()
    req.init_next_round_input()
    fill_ids = req.full_untruncated_fill_ids
    req.extend_range = SimpleNamespace(start=0, end=len(fill_ids), length=len(fill_ids))
    req.is_retracted = False
    return SimpleNamespace(
        input_ids=torch.tensor(fill_ids), input_embeds=None, replace_embeds=None
    )


def _runner():
    runner = object.__new__(VoxCPM2ModelRunner)
    runner.model = _Model()
    return runner


@pytest.mark.parametrize("generated", [0, 1, 3])
def test_retraction_replays_prompt_and_all_generated_audio_inputs(generated):
    request = _request(generated)
    batch = _retract_and_prepare(request)
    assert request.data.req.input_embeds is None
    assert len(request.data.req.output_ids) == generated
    assert len(request.data.req.prefix_indices) == 0
    assert len(batch.input_ids) == 3 + generated
    _runner().before_prefill(batch, None, [request])
    sidecar = get_omni_prefill_inputs(batch)
    expected = torch.tensor([[3, 3], [4, 4], [5, 5]], dtype=torch.float32)
    if generated:
        expected = torch.cat([expected, torch.stack(request.data.decode_input_embeds)])
    torch.testing.assert_close(sidecar.input_embeds, expected, rtol=0, atol=0)
    assert sidecar.audio_mask.tolist() == [0, 1, 0] + [1] * generated


def test_retraction_rejects_missing_audio_history_before_forward():
    request = _request(3)
    batch = _retract_and_prepare(request)
    request.data.decode_input_embeds.pop()
    with pytest.raises(RuntimeError, match="history"):
        _runner().before_prefill(batch, None, [request])
    assert get_omni_prefill_inputs(batch) is None


def test_fresh_and_retracted_requests_keep_packed_masks_and_final_rows():
    retracted, fresh = _request(2), _request(0, rid="fresh")
    _retract_and_prepare(retracted)
    fresh.data.req.init_next_round_input()
    fresh.data.req.extend_range = SimpleNamespace(length=3)
    batch = SimpleNamespace(
        input_ids=torch.zeros(8, dtype=torch.long),
        input_embeds=None,
        replace_embeds=None,
    )
    runner = _runner()
    runner.before_prefill(batch, None, [retracted, fresh])
    inputs = get_omni_prefill_inputs(batch)
    assert inputs.audio_mask.tolist() == [0, 1, 0, 1, 1, 0, 1, 0]
    assert inputs.input_embeds[:, 0].tolist() == [3, 4, 5, 10, 11, 3, 4, 5]
    assert runner.prefill_rows([retracted, fresh]).tolist() == [4, 7]


def test_generated_feedback_is_saved_without_aliasing_sampler_outputs():
    request = _request()
    data = request.data
    data.state.max_len = 20
    data.state.min_len = 20
    embeddings = torch.tensor([[11.0, 12.0]], requires_grad=True)
    runner = _runner()
    runner.batch_cond = lambda _: torch.zeros(1, 4, 8)
    runner.batch_noise = lambda _: None
    runner.model.decode_patch = lambda *a, **kw: (torch.ones(1, 4, 8), embeddings)
    runner.model.stop_flags = lambda rows: [False]
    runner.advance([request], rows=torch.tensor([2]), is_prefill=True)
    assert len(data.decode_input_embeds) == 1
    assert not data.decode_input_embeds[0].requires_grad
    with torch.no_grad():
        embeddings.fill_(99)
    torch.testing.assert_close(data.decode_input_embeds[0], torch.tensor([11.0, 12.0]))


class _CumulativeModel(_Model):
    patch_size = 4
    feat_dim = 8

    def __init__(self, graph_feedback):
        self.graph_feedback_buffer = torch.zeros(4, 2) if graph_feedback else None
        self.last_hidden = None

    def parameters(self):
        yield torch.zeros(1)

    def write_feedback(self, embeddings):
        self.graph_feedback_buffer[: len(embeddings)].copy_(embeddings)

    def set_hidden_states(self, hidden):
        self.last_hidden = hidden

    def decode_patch(self, cond, *, noise, rows, **kwargs):
        hidden = self.last_hidden[rows]
        value = (hidden.mean(-1) + cond.mean((1, 2)) + noise.mean((1, 2))) / 4
        return (
            value[:, None, None].expand(-1, 4, 8).clone(),
            value[:, None].expand(-1, 2).clone(),
        )

    def stop_flags(self, rows):
        return [False] * len(rows)


def _generate_with_retractions(pattern, *, graph_feedback):
    request = _request()
    data = request.data
    data.state.stream = True
    data.state.min_len = 20
    data.state.max_len = 20
    data.noise_generator = torch.Generator().manual_seed(1234)
    runner = _runner()
    runner.model = _CumulativeModel(graph_feedback)
    hidden = None
    chunks = []
    for step, retract in enumerate([True, *pattern]):
        if retract:
            from sglang_omni.scheduling.omni_scheduler import (
                _compact_decode_input_history,
            )

            _compact_decode_input_history(data)
            batch = _retract_and_prepare(request)
            state_before = data.noise_generator.get_state().clone()
            patches_before = len(data.latent_patches)
            runner.before_prefill(batch, None, [request])
            assert torch.equal(data.noise_generator.get_state(), state_before)
            assert len(data.latent_patches) == patches_before
            inputs = get_omni_prefill_inputs(batch).input_embeds
            all_hidden = inputs.cumsum(0)
            hidden = all_hidden[-1:]
            result = SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=all_hidden)
            )
            runner.post_prefill(result, batch, None, [request])
        else:
            batch = SimpleNamespace(input_embeds=None)
            runner.before_decode(batch, None, [request])
            inputs = (
                runner.model.graph_feedback_buffer[:1]
                if graph_feedback
                else batch.input_embeds
            )
            hidden = hidden + inputs
            result = SimpleNamespace(
                logits_output=SimpleNamespace(hidden_states=hidden)
            )
            runner.post_decode(result, batch, None, [request])
        data.req.output_ids.append(0)
        chunks.extend(build_stream_output(data.req.rid, data, None))
        assert len(data.latent_patches) == step + 1
    return data, chunks


@pytest.mark.parametrize("graph_feedback", [False, True])
def test_repeated_retraction_preserves_noise_condition_and_stream_cursor(
    graph_feedback,
):
    baseline, expected_chunks = _generate_with_retractions(
        [False, False, False], graph_feedback=graph_feedback
    )
    resumed, actual_chunks = _generate_with_retractions(
        [True, False, True], graph_feedback=graph_feedback
    )
    assert len(resumed.decode_input_embeds) == 4
    assert torch.equal(
        baseline.noise_generator.get_state(), resumed.noise_generator.get_state()
    )
    torch.testing.assert_close(
        torch.stack(resumed.latent_patches), torch.stack(baseline.latent_patches)
    )
    assert [chunk.metadata["chunk_id"] for chunk in actual_chunks] == [0, 1, 2, 3]
    assert len(actual_chunks) == len(expected_chunks)
    for actual, expected in zip(actual_chunks, expected_chunks):
        torch.testing.assert_close(actual.data, expected.data)


def test_scheduler_requeues_voxcpm_history_before_reprefill(monkeypatch):
    from sglang.srt.disaggregation.utils import DisaggregationMode

    from sglang_omni.scheduling import omni_scheduler

    request = _request(2)
    req = request.data.req
    original_history = torch.stack(request.data.decode_input_embeds)
    released = []

    def release_without_gpu(**kwargs):
        for item in kwargs["reqs"]:
            released.append(item.rid)
            item.reset_for_retract()

    monkeypatch.setattr(omni_scheduler, "retract_all", release_without_gpu)
    scheduler = object.__new__(omni_scheduler.OmniScheduler)
    scheduler.disaggregation_mode = DisaggregationMode.NULL
    scheduler.enable_priority_scheduling = False
    scheduler.abort_on_priority_when_disabled = False
    scheduler.max_queued_requests = None
    scheduler.enable_hicache_storage = False
    scheduler.enable_hierarchical_cache = False
    scheduler.waiting_queue = []
    scheduler.chunked_req = None
    batch = SimpleNamespace(
        reqs=[req],
        is_empty=lambda: False,
        filter_batch=lambda: None,
        req_to_token_pool=object(),
        token_to_kv_pool_allocator=object(),
        tree_cache=object(),
        hisparse_coordinator=None,
        batch_is_full=True,
    )
    scheduler.running_batch = batch
    assert scheduler._retract_running_requests() == 1
    assert released == [req.rid]
    assert batch.reqs == []
    assert not batch.batch_is_full
    assert scheduler.waiting_queue == [req]
    torch.testing.assert_close(
        torch.stack(request.data.decode_input_embeds), original_history
    )
    req.init_next_round_input()
    req.extend_range = SimpleNamespace(length=len(req.full_untruncated_fill_ids))
    forward = SimpleNamespace(
        input_ids=torch.tensor(req.full_untruncated_fill_ids), replace_embeds=None
    )
    _runner().before_prefill(forward, None, [request])
    assert get_omni_prefill_inputs(forward).input_embeds[:, 0].tolist() == [
        3,
        4,
        5,
        10,
        11,
    ]
