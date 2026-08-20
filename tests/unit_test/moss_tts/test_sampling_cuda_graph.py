# SPDX-License-Identifier: Apache-2.0
"""Correctness and routing tests for MOSS-TTS Delay sampling graphs."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.moss_tts.engine_builder import MossTtsEngineBuilder
from sglang_omni.models.moss_tts.model_runner import MossTTSModelRunner
from sglang_omni.models.moss_tts.sampler import (
    DelayGraphBatch,
    MossTTSDelayAudioGraphSampler,
)
from sglang_omni.models.moss_tts.sampling_cuda_graph import (
    MossTTSDelaySamplingCudaGraphRunner,
)
from sglang_omni.models.moss_tts.sglang_model import MossTTSDelaySGLangModel

_INT64_MAX = torch.iinfo(torch.int64).max
_MOSS_DELAY_N_VQ = 32
_MOSS_DELAY_AUDIO_VOCAB = 1025


def _config(*, n_vq: int = 2, audio_vocab: int = 5) -> SimpleNamespace:
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


def _sampling_data(
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
        _sampling_data(profile="default")
    )
    assert not MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        _sampling_data(profile="greedy")
    )
    assert not MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        _sampling_data(profile="custom")
    )
    assert not MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible(
        _sampling_data(profile="default", repetition_penalty=1.1)
    )


def test_model_runner_rejects_greedy_sampling_graph_batch() -> None:
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        is_sampling_cuda_graph_compatible=(
            MossTTSDelaySGLangModel.is_sampling_cuda_graph_compatible
        ),
        sampling_graph_available=lambda batch_size: batch_size == 1,
    )

    assert runner._can_use_sampling_cuda_graph(
        [_sampling_data(profile="default")], is_audio=True
    )
    assert not runner._can_use_sampling_cuda_graph(
        [_sampling_data(profile="greedy")], is_audio=True
    )
    assert not runner._can_use_sampling_cuda_graph(
        [_sampling_data(profile="default"), _sampling_data(profile="greedy")],
        is_audio=True,
    )


def test_engine_builder_uses_backbone_capture_buckets() -> None:
    calls = []
    builder = MossTtsEngineBuilder()
    builder.setup_model(
        model_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                decode_cuda_graph_runner=SimpleNamespace(
                    capture_bs=(1, 2, 4), disable_padding=True
                )
            )
        ),
        checkpoint_dir="checkpoint",
        device="cuda:0",
        gpu_id=0,
        server_args=SimpleNamespace(),
    )
    model = SimpleNamespace(
        init_sampling_graphs=lambda batch_sizes, disable_padding: calls.append(
            (batch_sizes, disable_padding)
        )
    )

    builder.post_cuda_graph_setup(model, SimpleNamespace())

    assert calls == [([1, 2, 4], True)]


def test_engine_builder_allows_breakable_prefill_as_an_opt_in() -> None:
    builder = MossTtsEngineBuilder()

    assert builder.supports_breakable_prefill_cuda_graph is True
    assert "cuda_graph_backend_prefill" not in builder.generation_defaults(
        dtype="bfloat16"
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
                next_delay_state=torch.tensor([[2, _INT64_MAX, 1]], dtype=torch.long),
            )

        @staticmethod
        def _prepare_multi_modal_inputs(rows):
            return rows.to(torch.float32)

    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = FakeModel()
    runner._pending_rows = None
    runner._pending_embeds = None
    profile = _sampling_data(profile="default")
    data = SimpleNamespace(
        is_audio=True,
        delay_state=torch.tensor([1, _INT64_MAX, 1]),
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

    runner._collect_moss_step(
        result,
        SimpleNamespace(),
        schedule_batch,
        [request],
    )

    assert len(calls) == 1
    assert result.next_token_ids.tolist() == [12]
    assert runner._pending_rows.tolist() == [[12, 2, 4]]
    assert data.delay_state.tolist() == [2, _INT64_MAX, 1]


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
        "_make_static_inputs",
        fail_static_input_allocation,
    )

    runner = MossTTSDelaySamplingCudaGraphRunner.capture(
        model=model,
        capture_bs=(1,),
    )

    assert runner.graphs == {}
    assert runner._inputs is None
    assert not runner.can_replay(1)


class _CudaGraphModel:
    def __init__(self, device: torch.device) -> None:
        self.config = _config(
            n_vq=_MOSS_DELAY_N_VQ,
            audio_vocab=_MOSS_DELAY_AUDIO_VOCAB,
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


def _request_data(
    delay_state: torch.Tensor,
    *,
    seed: int,
    generation_steps: int,
) -> SimpleNamespace:
    profile = _sampling_data(profile="default")
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_fixed_shape_sampling_matches_current_compacted_eager() -> None:
    device = torch.device("cuda")
    config = _config(
        n_vq=_MOSS_DELAY_N_VQ,
        audio_vocab=_MOSS_DELAY_AUDIO_VOCAB,
    )
    sampler = MossTTSDelayAudioGraphSampler(config).to(device)
    runner = MossTTSModelRunner.__new__(MossTTSModelRunner)
    runner.model = SimpleNamespace(
        config=config,
        text_control_token_ids=torch.tensor([12, 13], device=device),
    )
    initial_states = torch.tensor(
        [
            [1, _INT64_MAX, 1],
            [17, _INT64_MAX, 1],
            [_MOSS_DELAY_N_VQ + 1, _INT64_MAX, 1],
            [_MOSS_DELAY_N_VQ + 1, 0, 1],
            [_MOSS_DELAY_N_VQ + 1, _MOSS_DELAY_N_VQ - 1, 1],
            [_MOSS_DELAY_N_VQ + 1, _MOSS_DELAY_N_VQ, 1],
        ],
        dtype=torch.long,
        device=device,
    )
    seeds = torch.tensor(
        [101, 202, 303, 404, 505, 606], dtype=torch.long, device=device
    )
    steps = torch.tensor([1, 7, 8, 9, 10, 11], dtype=torch.long, device=device)
    datas = [
        _request_data(
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
                _MOSS_DELAY_AUDIO_VOCAB,
                device=device,
                generator=generator,
            )
            for _ in range(_MOSS_DELAY_N_VQ)
        ],
    ]

    eager_rows = runner._sample_rows(
        [logits.clone() for logits in channel_logits],
        datas,
        n_vq=_MOSS_DELAY_N_VQ,
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_sampling_cuda_graph_replay_matches_fixed_shape_eager() -> None:
    device = torch.device("cuda")
    model = _CudaGraphModel(device)
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
            _MOSS_DELAY_N_VQ,
            _MOSS_DELAY_AUDIO_VOCAB,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        batch = DelayGraphBatch(
            delay_state=torch.tensor(
                [[_MOSS_DELAY_N_VQ + 1, _INT64_MAX, 1]] * batch_size,
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
            static_inputs = runner._require_inputs()
            assert not torch.count_nonzero(static_inputs.control_logits[1:2])
            assert not torch.count_nonzero(static_inputs.audio_logits[1:2])
            assert not torch.count_nonzero(static_inputs.delay_state[1:2])
            assert not torch.count_nonzero(static_inputs.seeds[1:2])
            assert not torch.count_nonzero(static_inputs.generation_steps[1:2])
    runner.clear()
