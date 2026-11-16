# SPDX-License-Identifier: Apache-2.0
"""Correctness and routing tests for MOSS-TTS Delay sampling graphs."""

from __future__ import annotations

from collections import deque
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Literal

import pytest
import torch

from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts.model_runner import (
    MossDelayLaunchBuffer,
    MossTTSModelRunner,
)
from sglang_omni.models.moss_tts.sampler import (
    DelayGraphBatch,
    MossTTSDelayAudioGraphSampler,
    MossTTSDelayMixedGraphSampler,
)
from sglang_omni.models.moss_tts.sampling_cuda_graph import (
    MossTTSDelayMixedSamplingCudaGraphRunner,
    MossTTSDelaySamplingCudaGraphRunner,
)
from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel
from sglang_omni.proto.request import OmniRequest, StagePayload

INT64_MAX = torch.iinfo(torch.int64).max
MOSS_DELAY_N_VQ = 32
MOSS_DELAY_AUDIO_VOCAB = 1025


def make_config(*, n_vq: int = 2, audio_vocab: int = 5) -> SimpleNamespace:
    return SimpleNamespace(
        n_vq=n_vq,
        channels=n_vq + 1,
        pad_token_id=0,
        audio_start_token_id=10,
        audio_end_token_id=11,
        audio_assistant_gen_slot_token_id=12,
        audio_assistant_delay_slot_token_id=13,
        audio_pad_code=audio_vocab - 1,
        im_end_token_id=14,
        vocab_size_list=[20] + [audio_vocab] * n_vq,
    )


def sampling_data(
    *,
    profile: str,
    repetition_penalty: float = 1.0,
) -> SimpleNamespace:
    if profile == "default":
        values = (1.5, 1.0, 50, 1.7, 0.8, 25)
    elif profile == "greedy":
        values = (0.0, 0.7, 7, 0.0, 0.6, 5)
    elif profile == "custom":
        values = (1.0, 1.0, 50, 1.6, 0.8, 25)
    else:
        raise ValueError(profile)
    return SimpleNamespace(
        text_temperature=values[0],
        text_top_p=values[1],
        text_top_k=values[2],
        audio_temperature=values[3],
        audio_top_p=values[4],
        audio_top_k=values[5],
        audio_repetition_penalty=repetition_penalty,
    )


def test_sampling_graph_accepts_only_checkpoint_default_profile() -> None:
    assert MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        sampling_data(profile="default")
    )
    assert not MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        sampling_data(profile="greedy")
    )
    assert not MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        sampling_data(profile="custom")
    )
    assert not MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        sampling_data(profile="default", repetition_penalty=1.1)
    )


def test_model_runner_rejects_greedy_sampling_graph_batch() -> None:
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        is_sampling_cuda_graph_compatible=(
            MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible
        ),
        sampling_graph_available=lambda batch_size: batch_size == 1,
    )

    assert runner.can_use_sampling_cuda_graph(
        [sampling_data(profile="default")], is_audio=True
    )
    assert not runner.can_use_sampling_cuda_graph(
        [sampling_data(profile="greedy")], is_audio=True
    )
    assert not runner.can_use_sampling_cuda_graph(
        [sampling_data(profile="default"), sampling_data(profile="greedy")],
        is_audio=True,
    )


@pytest.mark.parametrize("async_decode_mode", ["all", "nonstream"])
def test_engine_builder_uses_backbone_capture_buckets(
    async_decode_mode: Literal["all", "nonstream"],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Factory scope must reach the runtime runner that admits streaming lookahead."""
    calls = []
    model = SimpleNamespace(
        config=make_config(),
        sampling_graph_available=lambda batch_size: True,
        is_sampling_cuda_graph_compatible=MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible,
        init_sampling_graphs=lambda batch_sizes, disable_padding: calls.append(
            (batch_sizes, disable_padding)
        ),
    )
    model_worker = SimpleNamespace(
        gpu_id=0,
        model_runner=SimpleNamespace(
            model=model,
            decode_cuda_graph_runner=SimpleNamespace(
                capture_bs=(1, 2, 4), disable_padding=True
            ),
        ),
    )
    builder = MossTtsEngineBuilder(async_decode_mode=async_decode_mode)
    builder.setup_model(
        model_worker=model_worker,
        checkpoint_dir="checkpoint",
        device="cuda:0",
        gpu_id=0,
        server_args=SimpleNamespace(),
    )
    builder.post_cuda_graph_setup(model, SimpleNamespace())

    assert calls == [([1, 2, 4], True)]
    monkeypatch.setattr(
        "sglang_omni.model_runner.base.current_platform.get_device",
        lambda gpu_id: torch.device("cpu"),
    )
    runtime_runner = builder.make_model_runner(model_worker, SimpleNamespace())
    assert runtime_runner.async_decode_mode == async_decode_mode
    requests = []
    for index in range(2):
        request_data = sampling_data(profile="default")
        request_data.is_audio = True
        request_data.output_rows = []
        request_data.audio_end_step = None
        request_data.audio_end_scanned_rows = 0
        request_data.generation_steps = 0
        request_data.sampling_steps = 0
        request_data.stage_payload = StagePayload(
            request_id=f"streaming-{index}",
            request=OmniRequest(inputs={}, params={"stream": True}),
            data=None,
        )
        requests.append(SimpleNamespace(omni_data=request_data))
    assert runtime_runner.lookahead_eligible(SimpleNamespace(reqs=requests)) is (
        async_decode_mode == "all"
    )


def test_model_runner_routes_supported_audio_batch_to_sampling_graph() -> None:
    calls = []

    class FakeModel:
        device = torch.device("cpu")
        hidden_size = 3
        text_control_token_ids = torch.tensor([12, 13], dtype=torch.long)

        @staticmethod
        def is_sampling_cuda_graph_compatible(data):
            return MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(data)

        @staticmethod
        def sampling_graph_available(batch_size):
            return batch_size == 1

        @staticmethod
        def compute_channel_logits(hidden_states, forward_batch, *, is_audio=False):
            del hidden_states, forward_batch
            assert is_audio is True
            return [
                torch.tensor([[3.0, 1.0]]),
                torch.zeros(1, 5),
                torch.zeros(1, 5),
            ]

        @staticmethod
        def sample_delay_graphed(
            control_logits,
            audio_logits,
            batch,
        ):
            calls.append((control_logits.clone(), audio_logits.clone(), batch))
            return SimpleNamespace(
                rows=torch.tensor([[12, 2, 4]], dtype=torch.long),
                next_delay_state=torch.tensor([[2, INT64_MAX, 1]], dtype=torch.long),
            )

        @staticmethod
        def prepare_multi_modal_inputs(rows):
            return rows.to(torch.float32)

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = FakeModel()
    runner.pending_rows = None
    runner.pending_embeds = None
    profile = sampling_data(profile="default")
    data = SimpleNamespace(
        is_audio=True,
        delay_state=torch.tensor([1, INT64_MAX, 1]),
        sampling_seed=123,
        generation_steps=1,
        text_temperature=profile.text_temperature,
        text_top_p=profile.text_top_p,
        text_top_k=profile.text_top_k,
        audio_temperature=profile.audio_temperature,
        audio_top_p=profile.audio_top_p,
        audio_top_k=profile.audio_top_k,
        audio_repetition_penalty=1.0,
    )
    request = SimpleNamespace(data=data)
    result = SimpleNamespace(
        logits_output=SimpleNamespace(
            customized_info=None,
            hidden_states=torch.zeros(1, 3),
        )
    )
    schedule_batch = SimpleNamespace()

    runner.collect_moss_step(
        result,
        SimpleNamespace(),
        schedule_batch,
        [request],
    )

    assert len(calls) == 1
    assert result.next_token_ids.tolist() == [12]
    assert runner.pending_rows.tolist() == [[12, 2, 4]]
    assert data.delay_state.tolist() == [2, INT64_MAX, 1]


def test_sampling_cuda_graph_setup_failure_uses_eager(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = SimpleNamespace(device=torch.device("cpu"))

    monkeypatch.setattr(torch.cuda, "device", lambda _device: nullcontext())
    monkeypatch.setattr(torch.cuda, "synchronize", lambda *args, **kwargs: None)

    def fail_static_input_allocation(self, max_bs):
        del self, max_bs
        raise torch.OutOfMemoryError("synthetic static-input OOM")

    monkeypatch.setattr(
        MossTTSDelaySamplingCudaGraphRunner,
        "make_static_inputs",
        fail_static_input_allocation,
    )

    runner = MossTTSDelaySamplingCudaGraphRunner.capture(
        model=model,
        capture_bs=(1,),
    )

    assert runner.graphs == {}
    assert runner.inputs is None
    assert not runner.can_replay(1)


class CudaGraphModel:
    def __init__(self, device: torch.device) -> None:
        self.config = make_config(
            n_vq=MOSS_DELAY_N_VQ,
            audio_vocab=MOSS_DELAY_AUDIO_VOCAB,
        )
        self.device = device
        self.delay_graph_sampler = MossTTSDelayAudioGraphSampler(self.config).to(device)

    def sample_delay_fixed_shape(
        self,
        control_logits: torch.Tensor,
        audio_logits: torch.Tensor,
        batch: DelayGraphBatch,
        *,
        audio_sample_output=None,
    ):
        return self.delay_graph_sampler(
            control_logits,
            audio_logits,
            batch,
            audio_sample_output=audio_sample_output,
        )


def request_data(
    delay_state: torch.Tensor,
    *,
    seed: int,
    generation_steps: int,
) -> SimpleNamespace:
    profile = sampling_data(profile="default")
    return SimpleNamespace(
        delay_state=delay_state,
        audio_length=int(delay_state[0]),
        delayed_length=int(delay_state[1]),
        is_audio=bool(int(delay_state[2])),
        sampling_seed=seed,
        generation_steps=generation_steps,
        text_temperature=profile.text_temperature,
        text_top_p=profile.text_top_p,
        text_top_k=profile.text_top_k,
        audio_temperature=profile.audio_temperature,
        audio_top_p=profile.audio_top_p,
        audio_top_k=profile.audio_top_k,
        audio_repetition_penalty=1.0,
        prompt_rows=None,
        output_rows=[],
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_shape_sampling_matches_current_compacted_eager() -> None:
    device = torch.device("cuda")
    config = make_config(
        n_vq=MOSS_DELAY_N_VQ,
        audio_vocab=MOSS_DELAY_AUDIO_VOCAB,
    )
    sampler = MossTTSDelayAudioGraphSampler(config).to(device)
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=config,
        text_control_token_ids=torch.tensor([12, 13], device=device),
    )
    initial_states = torch.tensor(
        [
            [1, INT64_MAX, 1],
            [17, INT64_MAX, 1],
            [MOSS_DELAY_N_VQ + 1, INT64_MAX, 1],
            [MOSS_DELAY_N_VQ + 1, 0, 1],
            [MOSS_DELAY_N_VQ + 1, MOSS_DELAY_N_VQ - 1, 1],
            [MOSS_DELAY_N_VQ + 1, MOSS_DELAY_N_VQ, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    seeds = torch.tensor(
        [101, 202, 303, 404, 505, 606], dtype=torch.long, device=device
    )
    steps = torch.tensor([1, 7, 8, 9, 10, 11], dtype=torch.long, device=device)
    datas = [
        request_data(
            initial_states[index].clone(),
            seed=int(seeds[index]),
            generation_steps=int(steps[index]),
        )
        for index in range(int(initial_states.shape[0]))
    ]
    generator = torch.Generator(device=device).manual_seed(20260722)
    batch_size = int(initial_states.shape[0])
    channel_logits = [
        torch.randn(batch_size, 2, device=device, generator=generator),
        *[
            torch.randn(
                batch_size,
                MOSS_DELAY_AUDIO_VOCAB,
                device=device,
                generator=generator,
            )
            for _ in range(MOSS_DELAY_N_VQ)
        ],
    ]

    eager_rows = runner.sample_rows(
        [logits.clone() for logits in channel_logits],
        datas,
        n_vq=MOSS_DELAY_N_VQ,
        is_audio=True,
    )
    eager_state = torch.stack([data.delay_state for data in datas])
    graph_batch = DelayGraphBatch(
        delay_state=initial_states,
        seeds=seeds,
        generation_steps=steps,
    )
    fixed = sampler(
        channel_logits[0],
        torch.stack(channel_logits[1:], dim=1),
        graph_batch,
    )
    torch.cuda.synchronize()

    assert torch.equal(eager_rows, fixed.rows)
    assert torch.equal(eager_state, fixed.next_delay_state)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampling_cuda_graph_replay_matches_fixed_shape_eager() -> None:
    device = torch.device("cuda")
    model = CudaGraphModel(device)
    runner = MossTTSDelaySamplingCudaGraphRunner.capture(
        model=model,
        capture_bs=(2,),
    )
    assert set(runner.graphs) == {2}
    assert runner.can_replay(1)
    assert runner.can_replay(2)

    generator = torch.Generator(device=device).manual_seed(20260722)
    for replay_index, batch_size in enumerate((2, 1, 2)):
        control_logits = torch.randn(
            batch_size,
            2,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        audio_logits = torch.randn(
            batch_size,
            MOSS_DELAY_N_VQ,
            MOSS_DELAY_AUDIO_VOCAB,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        batch = DelayGraphBatch(
            delay_state=torch.tensor(
                [[MOSS_DELAY_N_VQ + 1, INT64_MAX, 1]] * batch_size,
                device=device,
            ),
            seeds=torch.arange(batch_size, device=device, dtype=torch.long)
            + 1_234_567
            + replay_index * 100,
            generation_steps=torch.arange(batch_size, device=device, dtype=torch.long)
            + 7
            + replay_index,
        )

        expected = model.sample_delay_fixed_shape(
            control_logits,
            audio_logits,
            batch,
        )
        actual = runner.replay(
            control_logits,
            audio_logits,
            batch,
        )
        actual_rows = actual.rows.clone()
        actual_state = actual.next_delay_state.clone()
        torch.cuda.synchronize()

        assert torch.equal(expected.rows, actual_rows)
        assert torch.equal(expected.next_delay_state, actual_state)
        if batch_size == 1:
            static_inputs = runner.require_inputs()
            assert not torch.count_nonzero(static_inputs.control_logits[1:2])
            assert not torch.count_nonzero(static_inputs.audio_logits[1:2])
            assert not torch.count_nonzero(static_inputs.delay_state[1:2])
            assert not torch.count_nonzero(static_inputs.seeds[1:2])
            assert not torch.count_nonzero(static_inputs.generation_steps[1:2])
    runner.clear()


@pytest.mark.parametrize("use_mixed_graph", [False, True])
def test_lookahead_routes_delay_boundary_and_rejects_custom_sampling(
    use_mixed_graph: bool,
) -> None:
    config = make_config()
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.async_decode_mode = "all"
    runner.model = SimpleNamespace(
        config=config,
        sampling_graph_available=lambda batch_size: True,
        mixed_sampling_graph_available=lambda batch_size: use_mixed_graph,
        is_sampling_cuda_graph_compatible=MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible,
    )
    state = sampling_data(profile="default")
    state.is_audio = True
    state.output_rows = [torch.tensor([config.audio_assistant_delay_slot_token_id])]
    state.audio_end_step = None
    state.audio_end_scanned_rows = 0
    state.generation_steps = 1
    state.sampling_steps = 2
    batch = SimpleNamespace(reqs=[SimpleNamespace(omni_data=state)])
    assert runner.lookahead_eligible(batch)
    state.sampling_steps = 3
    assert runner.lookahead_eligible(batch) == use_mixed_graph
    state.audio_temperature = 0.0
    assert not runner.lookahead_eligible(batch)
    assert not runner.lookahead_eligible(SimpleNamespace(reqs=[]))


@pytest.mark.parametrize("async_decode_mode", ["all", "nonstream"])
@pytest.mark.parametrize(
    "streaming_requests", [(False, False), (True, False), (False, True), (True, True)]
)
def test_lookahead_keeps_streaming_and_mixed_batches_synchronous_when_requested(
    async_decode_mode: Literal["all", "nonstream"],
    streaming_requests: tuple[bool, bool],
) -> None:
    """Streaming batches must not inherit non-streaming lookahead eligibility."""
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.async_decode_mode = async_decode_mode
    runner.model = SimpleNamespace(
        config=make_config(),
        sampling_graph_available=lambda batch_size: True,
        is_sampling_cuda_graph_compatible=MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible,
    )
    requests = []
    for streaming in streaming_requests:
        request_data = sampling_data(profile="default")
        request_data.is_audio = True
        request_data.output_rows = []
        request_data.audio_end_step = None
        request_data.audio_end_scanned_rows = 0
        request_data.generation_steps = 0
        request_data.sampling_steps = 0
        request_data.stage_payload = SimpleNamespace(
            request=SimpleNamespace(params={"stream": streaming})
        )
        requests.append(SimpleNamespace(omni_data=request_data))
    assert runner.lookahead_eligible(SimpleNamespace(reqs=requests)) is (
        async_decode_mode == "all" or not any(streaming_requests)
    )


def test_lookahead_resolve_preserves_owned_rows_and_skips_dropped_requests() -> None:
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(config=make_config())
    requests = []
    for index in range(3):
        state = SimpleNamespace(
            req=SimpleNamespace(
                is_retracted=index == 1, finished=lambda i=index: i == 2
            ),
            delay_state=torch.tensor([99, 99, 1]),
            sampling_steps=12,
            pending_feedback_queue=deque(),
            output_rows=[],
            is_audio=True,
        )
        requests.append(SimpleNamespace(request_id=str(index), data=state))
    host_rows = torch.tensor([[12, 1, 2], [12, 2, 3], [12, 3, 1]])
    before = [torch.tensor([7, INT64_MAX, 1]) for _ in requests]
    launch = MossDelayLaunchBuffer(
        host_rows=host_rows,
        delay_states_before=before,
        sampling_steps_before=[8, 9, 10],
        feedback_embeds=torch.zeros(3, 4),
    )
    result = SimpleNamespace()
    runner.post_decode_resolve(launch, result, None, None, requests)
    host_rows.zero_()
    assert result.next_token_ids.tolist() == [12, 12, 12]
    assert requests[1].data.sampling_steps == 9
    assert torch.equal(requests[1].data.delay_state, before[1])
    outputs = {str(index): SimpleNamespace(data=12) for index in range(3)}
    runner.post_process_outputs(result, SimpleNamespace(requests=requests), outputs)
    assert requests[0].data.output_rows[0].tolist() == [12, 1, 2]
    assert not requests[1].data.output_rows
    assert not requests[2].data.output_rows
    assert not requests[0].data.pending_feedback_queue


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("ties", [False, True])
def test_mixed_sampling_graph_matches_eager_at_text_audio_boundaries(
    ties: bool,
) -> None:
    device = torch.device("cuda")
    config = make_config(n_vq=32, audio_vocab=1025)
    config.vocab_size_list[0] = 128
    sampler = MossTTSDelayMixedGraphSampler(config).to(device)
    model = SimpleNamespace(
        config=config,
        device=device,
        sample_mixed_fixed_shape=sampler,
    )
    graph = MossTTSDelayMixedSamplingCudaGraphRunner.capture(
        model=model, capture_bs=(8,), disable_padding=False
    )
    assert graph.can_replay(7)
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = model
    states = torch.tensor(
        [
            [0, INT64_MAX, 0],
            [1, INT64_MAX, 1],
            [40, INT64_MAX, 1],
            [40, 0, 1],
            [40, 31, 1],
            [40, 32, 1],
            [0, INT64_MAX, 0],
        ],
        dtype=torch.long,
        device=device,
    )
    seeds = torch.arange(7, device=device, dtype=torch.long) + 123
    steps = torch.tensor([0, 1, 40, 41, 72, 73, 74], device=device)
    generator = torch.Generator(device=device).manual_seed(42)
    for _ in range(3):
        text_logits = torch.randn(7, 128, device=device, generator=generator)
        audio_logits = torch.randn(7, 32, 1025, device=device, generator=generator)
        if ties:
            text_logits.round_()
            audio_logits.round_()
        requests = [
            request_data(
                states[index].clone(),
                seed=int(seeds[index]),
                generation_steps=int(steps[index]),
            )
            for index in range(7)
        ]
        expected = runner.sample_rows(
            [
                text_logits.clone(),
                *[audio_logits[:, index].clone() for index in range(32)],
            ],
            requests,
            n_vq=32,
            is_audio=False,
        )
        actual = graph.replay(
            text_logits,
            audio_logits,
            DelayGraphBatch(
                delay_state=states,
                seeds=seeds,
                generation_steps=steps,
            ),
        )
        assert torch.equal(actual.rows, expected)
        assert torch.equal(
            actual.next_delay_state,
            torch.stack([request.delay_state for request in requests]),
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampling_metadata_upload_owns_values_across_growth_and_reuse() -> None:
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.sampling_metadata_host_bufs = []
    runner.sampling_metadata_slot = 0
    runner.sampling_metadata_events = [None, None]
    copies = []
    for index, rows in enumerate([1, 3, 2, 8, 1, 8, 2, 4]):
        seeds = [1000 * index + row for row in range(rows)]
        positions = [100 * index + row for row in range(rows)]
        actual = runner.sampling_metadata(seeds, positions, torch.device("cuda"))
        copies.append((actual, seeds, positions))
    torch.cuda.synchronize()
    for (actual_seeds, actual_positions), seeds, positions in copies:
        assert actual_seeds.tolist() == seeds
        assert actual_positions.tolist() == positions
