# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import queue
import sys
import threading
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

import sglang_omni.models.fun_cosyvoice3.streaming_vocoder as streaming_vocoder_module
from sglang_omni.client.client import Client
from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    FunCosyVoice3PipelineConfig,
)
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fun_cosyvoice3.test_flow_batch import _FakeFlow as _PackedFlow


class _FakeHiFT(torch.nn.Module):
    # cosyvoice3.yaml: upsample_rates [8, 5, 3], istft_params.hop_len 4.
    upsample_rates: ClassVar[list[int]] = [8, 5, 3]
    istft_params: ClassVar[dict[str, int]] = {"n_fft": 16, "hop_len": 4}

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def inference(self, *, speech_feat, finalize):
        self.calls.append((speech_feat, finalize))
        batch, _, frames = speech_feat.shape
        row = torch.arange(frames * 480, dtype=torch.float32).reshape(1, -1)
        return row.repeat(batch, 1), None


class _FakeEstimator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("batch adapter should be mocked in vocoder unit tests")


class _RunnableFakeFlow(_PackedFlow):
    def __init__(self):
        super().__init__(channels=80, max_frames=8192)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80)


def test_mlx_stream_scheduler_consumes_chunks_before_final_decode() -> None:
    class _FakeMlxVocoder:
        sample_rate = 24000

        async def decode_payload(self, payload):
            return payload

        async def decode_payloads(self, payloads):
            return payloads

        def decode_tokens(self, *, token, prompt_token, prompt_feat, embedding):
            del prompt_token, prompt_feat, embedding
            assert token.tolist() == [[11, 12]]
            return torch.ones(1, 16)

    scheduler = stages.FunCosyVoice3MlxStreamingVocoderScheduler(
        _FakeMlxVocoder(), max_batch_wait_ms=0
    )
    state = FunCosyVoice3State(
        stream=True,
        flow_prompt_speech_token=torch.tensor([[1, 2]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
        flow_embedding=torch.ones(1, 192),
    )
    payload = _payload(state)
    scheduler.stream_payloads["req"] = payload
    scheduler.on_streaming_new_request("req", payload)
    scheduler.on_stream_chunk(
        "req",
        StreamItem(
            chunk_id=0,
            data=torch.tensor([11, 12]),
            from_stage="tts_engine",
            metadata={"stream": True, "modality": "audio_codes"},
        ),
    )

    messages = scheduler.on_stream_done("req")

    assert [message.type for message in messages] == ["stream", "result"]


def test_mps_hift_adapter_moves_f0_to_cpu_before_float64() -> None:
    calls = []

    class _Predictor:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    hift = SimpleNamespace(f0_predictor=_Predictor())

    stages.MpsHiFTAdapter(hift, "mps")

    assert calls == [
        ((), {"device": "cpu"}),
        ((), {"dtype": torch.float64}),
    ]


def test_lightweight_loader_skips_llm_and_loads_flow_hift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    observed = {}

    class _Model:
        def __init__(self) -> None:
            self.loaded = None
            self.device = None
            self.evaluated = False

        def load_state_dict(self, state, strict=True):
            self.loaded = (state, strict)

        def to(self, device):
            self.device = device
            return self

        def eval(self):
            self.evaluated = True
            return self

    flow = _Model()
    flow.decoder = SimpleNamespace(estimator=torch.nn.Module())
    hift = _Model()

    def fake_load_hyperpyyaml(handle, overrides):
        observed.update(config=handle.name, overrides=overrides)
        return {"flow": flow, "hift": hift}

    monkeypatch.setitem(
        sys.modules,
        "hyperpyyaml",
        SimpleNamespace(load_hyperpyyaml=fake_load_hyperpyyaml),
    )
    for filename in ("cosyvoice3.yaml", "flow.pt", "hift.pt"):
        (tmp_path / filename).touch()

    def fake_torch_load(path, *, map_location, weights_only):
        assert map_location == "cpu"
        assert weights_only is True
        if str(path).endswith("flow.pt"):
            return {"flow_weight": torch.tensor(1)}
        return {
            "generator.hift_weight": torch.tensor(2),
            "unprefixed": torch.tensor(3),
        }

    monkeypatch.setattr(torch, "load", fake_torch_load)

    loaded_flow, loaded_hift = stages.load_cosyvoice3_flow_hift_lightweight(
        str(tmp_path),
        device="cpu",
    )

    assert isinstance(loaded_flow, stages.FunCosyVoice3Flow)
    assert loaded_flow.packed_estimator.dit is flow.decoder.estimator
    assert loaded_hift is hift
    assert observed["overrides"] == {
        "qwen_pretrain_path": str(tmp_path / "CosyVoice-BlankEN"),
        "llm": None,
        "hifigan": None,
    }
    assert flow.loaded == ({"flow_weight": torch.tensor(1)}, True)
    assert hift.loaded == (
        {
            "hift_weight": torch.tensor(2),
            "unprefixed": torch.tensor(3),
        },
        True,
    )
    assert flow.device == hift.device == "cpu"
    assert flow.evaluated is hift.evaluated is True


class _BatchCapableFakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.output_size = 80
        self.token_mel_ratio = 2
        self.input_embedding = torch.nn.Embedding(32, 80)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80)
        self.pre_lookahead_layer = lambda x, context=None: x
        self.pre_lookahead_len = 3
        self.decoder = SimpleNamespace(
            rand_noise=torch.zeros(1, 80, 1000),
            t_scheduler="cosine",
            inference_cfg_rate=0.7,
            estimator=_FakeEstimator(),
            forward_estimator=lambda *args, **kwargs: None,
        )


class _FakeFlow(torch.nn.Module):
    """CosyVoice-native Flow.inference(**kwargs) used by causal token2wav hops."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.decoder = SimpleNamespace(estimator=_FakeEstimator())

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        token_count = kwargs["token"].shape[1]
        return torch.ones(1, 80, token_count * 2), None


def _payload(
    state: FunCosyVoice3State, request_id: str = "req-vocoder"
) -> StagePayload:
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def test_cosyvoice3_vocoder_does_not_pad_or_rescale_short_sequences() -> None:
    flow = _FakeFlow()
    hift = _FakeHiFT()
    vocoder = stages.CosyVoice3Vocoder(flow, hift)

    # note (guozhihao-224): token 0 is a valid FSQ speech token, not padding.
    # Do not pad short sequences or apply speed inside HiFT.
    wav = vocoder.token2wav(
        token=torch.tensor([[0, 2]], dtype=torch.long),
        prompt_token=torch.tensor([[4]], dtype=torch.int32),
        prompt_feat=torch.zeros(1, 2, 80),
        embedding=torch.ones(1, 192),
    )

    flow_call = flow.calls[0]
    assert flow_call["token"].shape == (1, 2)
    assert flow_call["token"].tolist() == [[0, 2]]
    assert flow_call["token_len"].tolist() == [2]
    assert flow_call["prompt_token_len"].tolist() == [1]
    assert flow_call["prompt_feat_len"].tolist() == [2]
    assert flow_call["finalize"] is True
    assert flow_call["streaming"] is False
    assert hift.calls[0][0].shape[-1] == 4
    assert wav.device.type == "cpu"


def test_cosyvoice3_vocoder_raises_on_empty_token_sequence() -> None:
    vocoder = stages.CosyVoice3Vocoder(_FakeFlow(), _FakeHiFT())

    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        vocoder.token2wav(
            token=torch.zeros(1, 0, dtype=torch.long),
            prompt_token=torch.tensor([[4]], dtype=torch.int32),
            prompt_feat=torch.zeros(1, 2, 80),
            embedding=torch.ones(1, 192),
        )


def test_cosyvoice3_token2wav_chunk_slices_mel_and_hift_delta() -> None:
    flow = _FakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())
    token = torch.arange(28, dtype=torch.int32).unsqueeze(0)
    prompt_token = torch.zeros(1, 0, dtype=torch.int32)
    prompt_feat = torch.zeros(1, 0, 80)
    embedding = torch.ones(1, 192)

    delta, cached_mel, speech_offset = vocoder.token2wav_chunk(
        token=token,
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        embedding=embedding,
        token_offset=0,
        streaming=True,
        finalize=False,
        hift_mel=None,
        speech_offset=0,
    )

    assert flow.calls[0]["streaming"] is True
    assert flow.calls[0]["finalize"] is False
    assert cached_mel.shape[-1] == 56
    # speech_offset is waveform samples (mel_frames * HiFT stride=480).
    assert speech_offset == 56 * 480
    assert delta.shape[-1] == 56 * 480

    tail, cached_mel, speech_offset = vocoder.token2wav_chunk(
        token=token,
        prompt_token=prompt_token,
        prompt_feat=prompt_feat,
        embedding=embedding,
        token_offset=25,
        streaming=False,
        finalize=True,
        hift_mel=cached_mel,
        speech_offset=speech_offset,
    )

    assert flow.calls[1]["streaming"] is False
    assert flow.calls[1]["finalize"] is True
    # note (guozhihao-224): leftover hop slices from offset 25*2, concat onto
    # the 56-frame cache; HiFT emits the 6 new mel frames as 6*480 samples.
    assert cached_mel.shape[-1] == 62
    assert speech_offset == 62 * 480
    assert tail.shape[-1] == 6 * 480


def test_cosyvoice3_vocoder_prepare_and_store_audio_payload() -> None:
    vocoder = stages.CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    state = FunCosyVoice3State(
        text="hello",
        audio_codes=torch.tensor([[1, 2], [3, 4]]),
        flow_prompt_speech_token=torch.tensor([[5]], dtype=torch.int32),
        flow_embedding=torch.ones(1, 192),
    )
    payload = _payload(state)

    restored_state, codes = vocoder.prepare_item(payload)
    assert restored_state.text == "hello"
    assert torch.equal(codes, torch.tensor([1, 2, 3, 4]))

    stored = vocoder.store_result(
        payload, restored_state, torch.tensor([[0.1, 0.2]]), 24000
    )
    assert stored.data["audio_waveform_shape"] == [2]
    assert stored.data["audio_waveform_dtype"] == "float32"
    assert stored.data["sample_rate"] == 24000
    assert stored.data["modality"] == "audio"
    assert "audio_codes" not in stored.data


def test_cosyvoice3_vocoder_rejects_payload_without_audio_codes() -> None:
    vocoder = stages.CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    payload = _payload(FunCosyVoice3State(text="hello"))

    with pytest.raises(RuntimeError, match="requires audio_codes"):
        vocoder.prepare_item(payload)


def test_mlx_vocoder_audio_payload_survives_state_storage() -> None:
    state = FunCosyVoice3State(
        text="hello",
        audio_codes=torch.tensor([[1], [2]]),
        audio_samples=[9.0],
        prompt_tokens=3,
        completion_tokens=2,
    )
    waveform = np.array([[0.1, -0.2]], dtype=np.float32)

    mlx_vocoder = object.__new__(stages.CosyVoice3MlxVocoderAdapter)
    stored = mlx_vocoder.store_result(_payload(state), state, waveform, 24000)
    result = Client.default_result_builder(stored.request_id, stored.data)

    np.testing.assert_array_equal(result.audio_data, waveform.reshape(-1))
    assert result.sample_rate == 24000
    assert result.modality == "audio"
    assert result.usage.total_tokens == 5
    assert "audio_codes" not in stored.data
    assert "audio_samples" not in stored.data


def test_cosyvoice3_vocoder_rejects_missing_audio_output() -> None:
    vocoder = stages.CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    state = FunCosyVoice3State(text="hello")
    payload = _payload(state)

    with pytest.raises(RuntimeError, match="did not return audio"):
        vocoder.store_result(payload, state, None, 24000)


def test_cosyvoice3_vocoder_decode_batch_uses_state_conditioning(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())
    state = FunCosyVoice3State(
        speed=1.5,
        flow_prompt_speech_token=torch.tensor([[5]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.zeros(1, 1, 80),
        flow_embedding=torch.ones(1, 192),
    )

    results = asyncio.run(vocoder.decode_batch([(state, torch.tensor([1, 2]))]))

    assert len(results) == 1
    assert results[0][1] == 24000
    assert batch_calls[0][0].prompt_token.tolist() == [[5]]


def test_vocoder_autocast_uses_the_flow_device(monkeypatch) -> None:
    from contextlib import nullcontext

    observed = []
    monkeypatch.setattr(
        torch,
        "autocast",
        lambda *, device_type, dtype, enabled: observed.append(
            (device_type, dtype, enabled)
        )
        or nullcontext(),
    )
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(
        _BatchCapableFakeFlow(),
        _FakeHiFT(),
        autocast_dtype=torch.float16,
    )

    asyncio.run(vocoder.decode_batch([(_state(), torch.tensor([1, 2]))]))

    assert observed == [
        ("cpu", torch.float16, True),
        (stages.current_platform.device_type, None, False),
    ]


def _state(
    *,
    sample_rate: int = 24000,
    prompt_tokens: int = 1,
    prompt_feat_frames: int | None = None,
) -> FunCosyVoice3State:
    if prompt_feat_frames is None:
        prompt_feat_frames = prompt_tokens * 2
    return FunCosyVoice3State(
        sample_rate=sample_rate,
        flow_prompt_speech_token=torch.arange(prompt_tokens).reshape(1, -1),
        flow_prompt_speech_feat=torch.zeros(1, prompt_feat_frames, 80),
        flow_embedding=torch.ones(1, 192),
    )


def _codes(length: int, value: int = 1) -> torch.Tensor:
    return torch.full((length,), value, dtype=torch.long)


def _flow_requests(totals: list[int]) -> list[stages.PreparedFlowRequest]:
    flow_input = stages.FlowBatchInput(
        token=torch.empty((1, 0), dtype=torch.int32),
        prompt_token=torch.empty((1, 0), dtype=torch.int32),
        prompt_feat=torch.empty((1, 0, 80)),
        embedding=torch.empty((1, 192)),
    )
    return [
        stages.PreparedFlowRequest(
            index=index,
            sample_rate=24000,
            flow_input=flow_input,
            total_mel_frames=total,
        )
        for index, total in enumerate(totals)
    ]


def _install_fake_batch_adapter(monkeypatch, calls: list[list]) -> None:
    def fake_infer(flow, inputs):
        del flow
        calls.append(list(inputs))
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fake_infer)


def test_decode_batch_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(vocoder.decode_batch([(_state(), _codes(2))]))

    assert len(results) == 1
    assert [len(call) for call in batch_calls] == [1]
    assert len(hift.calls) == 1


def test_decode_payload_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())
    state = _state()
    state.audio_codes = _codes(2)

    result = asyncio.run(vocoder.decode_payload(_payload(state)))

    assert result.data["modality"] == "audio"
    assert [len(call) for call in batch_calls] == [1]


def test_decode_batch_merges_flow_preserving_hift_groups_and_order(
    monkeypatch,
) -> None:
    items = [
        (_state(sample_rate=16003, prompt_tokens=0), _codes(50, 3)),
        (_state(sample_rate=16001, prompt_tokens=0), _codes(24, 1)),
        (_state(sample_rate=16004, prompt_tokens=0), _codes(51, 4)),
        (_state(sample_rate=16002, prompt_tokens=0), _codes(25, 2)),
    ]

    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    flow_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, flow_calls)
    vocoder = stages.CosyVoice3Vocoder(
        flow,
        hift,
        flow_merge_max_gap_frames=4,
        flow_merge_pad_budget_percent=25,
    )
    results = asyncio.run(vocoder.decode_batch(items))
    hift_memberships = [
        tuple(int(value) for value in call[0][:, 0, 0].tolist()) for call in hift.calls
    ]

    assert [sample_rate for _, sample_rate in results] == [16003, 16001, 16004, 16002]
    assert [[item.token.shape[1] for item in call] for call in flow_calls] == [
        [24, 25],
        [50, 51],
    ]
    # Mel lengths 48/50 vs 100/102 exceed default HiFT waste=1.5, so HiFT
    # keeps the same cut Flow already made. Result order is still original.
    assert hift_memberships == [(1, 2), (3, 4)]


@pytest.mark.parametrize(
    (
        "totals",
        "flow_merge_max_gap_frames",
        "flow_merge_pad_budget_percent",
        "expected",
    ),
    [
        pytest.param(
            [10, 13, 30, 33],
            4,
            5,
            [[10], [13], [30, 33]],
            id="global-padding-cap",
        ),
        pytest.param(
            [10, 10, 10, 11, 13],
            3,
            10,
            [[10, 10, 10], [11, 13]],
            id="minimum-padded-work",
        ),
        pytest.param(
            [10, 10, 20, 40],
            40,
            30,
            [[10, 10, 20], [40]],
            id="maximum-merged-span",
        ),
        pytest.param(
            [450] + [500] * 15,
            384,
            25,
            [[450] + [500] * 15],
            id="b16-production-regime",
        ),
    ],
)
def test_flow_merge_partition_policy(
    totals: list[int],
    flow_merge_max_gap_frames: int,
    flow_merge_pad_budget_percent: float,
    expected: list[list[int]],
) -> None:
    groups = stages.adaptive_flow_requests_grouping(
        _flow_requests(totals),
        flow_merge_max_gap_frames=flow_merge_max_gap_frames,
        flow_merge_pad_budget_percent=flow_merge_pad_budget_percent,
    )

    assert [
        [request.total_mel_frames for request in group] for group in groups
    ] == expected


def test_decode_batch_runs_hift_once_over_padded_mels(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(
        vocoder.decode_batch(
            [(_state(), _codes(9)), (_state(), _codes(10)), (_state(), _codes(11))]
        )
    )

    # 9/10/11 tokens -> 18/20/22 mel frames. Right-zero-padded into one call.
    assert len(hift.calls) == 1
    speech_feat, finalize = hift.calls[0]
    assert finalize is True
    assert speech_feat.shape == (3, 80, 22)
    assert torch.count_nonzero(speech_feat[0, :, 18:]) == 0
    assert torch.count_nonzero(speech_feat[1, :, 20:]) == 0
    # Each request is sliced back to its own true length.
    assert [wav.shape[-1] for wav, _ in results] == [18 * 480, 20 * 480, 22 * 480]


def test_decode_batch_splits_hift_batch_when_padding_waste_is_large(
    monkeypatch,
) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    _install_fake_batch_adapter(monkeypatch, [])
    # max_waste=1.0 only accepts groups that need no padding at all.
    vocoder = stages.CosyVoice3Vocoder(flow, hift, hift_max_padding_waste=1.0)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))]))

    assert len(hift.calls) == 2


def test_decode_batch_long_singleton_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())

    asyncio.run(vocoder.decode_batch([(_state(prompt_tokens=0), _codes(2200, 1))]))

    assert [len(call) for call in batch_calls] == [1]
    assert batch_calls[0][0].token.shape[1] == 2200


def test_vocoder_rejects_non_pytorch_flow_estimator() -> None:
    flow = _BatchCapableFakeFlow()
    flow.decoder.estimator = object()

    with pytest.raises(RuntimeError, match="PyTorch module or a TensorRT wrapper"):
        stages.CosyVoice3Vocoder(flow, _FakeHiFT())


def test_vocoder_accepts_tensorrt_flow_estimator() -> None:
    class _FakeTRTEstimator:
        def acquire_estimator(self):
            return [None, None], None

        def execute(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("vocoder init must not run the estimator")

    flow = _BatchCapableFakeFlow()
    flow.decoder.estimator = _FakeTRTEstimator()
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())
    assert vocoder.flow is not None


def test_decode_batch_alignment_mismatch_fails() -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())

    with pytest.raises(ValueError, match="prompt feature length"):
        asyncio.run(
            vocoder.decode_batch(
                [
                    (_state(prompt_tokens=1, prompt_feat_frames=1), _codes(2)),
                    (_state(), _codes(3)),
                ]
            )
        )


def test_decode_batch_embedding_width_mismatch_fails() -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())
    invalid = _state()
    invalid.flow_embedding = torch.ones(1, 191)

    with pytest.raises(ValueError, match="embedding width"):
        asyncio.run(vocoder.decode_batch([(invalid, _codes(2)), (_state(), _codes(3))]))


def test_decode_batch_does_not_retry_after_batch_failure(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())

    def fail_batch(flow, inputs):
        del flow, inputs
        raise RuntimeError("batch estimator failed")

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fail_batch)

    with pytest.raises(RuntimeError, match="batch estimator failed"):
        asyncio.run(
            vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))])
        )


def _buffered_payload(
    request_id: str, code_length: int, value: int = 1
) -> StagePayload:
    state = _state(prompt_tokens=0)
    state.audio_codes = _codes(code_length, value)
    return _payload(state, request_id)


def _drain_buffered_results(
    scheduler: FunCosyVoice3StreamingVocoderScheduler,
) -> list:
    results = []
    while True:
        try:
            results.append(scheduler.outbox.get_nowait())
        except queue.Empty:
            return results


def _run_buffered_batch(
    scheduler: FunCosyVoice3StreamingVocoderScheduler,
    messages: list[IncomingMessage],
) -> None:
    scheduler.handle_new_request_batch(messages)
    while scheduler.has_ready_work():
        scheduler.run_ready_step()


def _audio_value(message) -> float:
    payload = message.data
    return float(np.frombuffer(payload.data["audio_waveform"], dtype=np.float32)[0])


def _buffered_scheduler(
    monkeypatch: pytest.MonkeyPatch,
    *,
    max_batch_size: int = 3,
    max_batch_wait_ms: int = 0,
) -> tuple[
    _BatchCapableFakeFlow,
    stages.CosyVoice3Vocoder,
    FunCosyVoice3StreamingVocoderScheduler,
]:
    flow = _BatchCapableFakeFlow()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(
        flow,
        _FakeHiFT(),
        flow_merge_max_gap_frames=0,
        flow_merge_pad_budget_percent=0,
    )
    scheduler = FunCosyVoice3StreamingVocoderScheduler(
        vocoder,
        max_batch_size=max_batch_size,
        max_batch_wait_ms=max_batch_wait_ms,
    )
    return flow, vocoder, scheduler


def test_buffered_vocoder_releases_first_flow_group_before_later_group(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch)
    first_group_started = threading.Event()
    release_first_group = threading.Event()
    call_count = 0

    def fake_inference(flow, inputs):
        nonlocal call_count
        del flow
        call_count += 1
        if call_count == 1:
            first_group_started.set()
            assert release_first_group.wait(timeout=5)
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fake_inference)
    messages = [
        IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
        IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
        IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
    ]
    scheduler.handle_new_request_batch(messages)
    worker = threading.Thread(target=scheduler.run_ready_step)
    worker.start()
    try:
        assert first_group_started.wait(timeout=5)
        with pytest.raises(queue.Empty):
            scheduler.outbox.get_nowait()
        release_first_group.set()
        worker.join(timeout=5)
    finally:
        release_first_group.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    first = _drain_buffered_results(scheduler)
    assert [(message.request_id, message.type) for message in first] == [
        ("a", "result"),
        ("b", "result"),
    ]

    scheduler.run_ready_step()
    later = _drain_buffered_results(scheduler)
    assert [(message.request_id, message.type) for message in later] == [
        ("c", "result")
    ]


def test_buffered_vocoder_rolls_pending_flow_groups_across_arrivals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, vocoder, scheduler = _buffered_scheduler(monkeypatch)
    prepare_calls: list[int] = []
    original_make_flow_input = vocoder.make_flow_input

    def record_make_flow_input(state, codes):
        prepare_calls.append(int(codes[0]))
        return original_make_flow_input(state, codes)

    monkeypatch.setattr(vocoder, "make_flow_input", record_make_flow_input)
    plan_calls: list[list[int]] = []
    original_grouping = streaming_vocoder_module.adaptive_flow_requests_grouping

    def spy_grouping(requests, **kwargs):
        plan_calls.append([request.index for request in requests])
        return original_grouping(requests, **kwargs)

    monkeypatch.setattr(
        streaming_vocoder_module, "adaptive_flow_requests_grouping", spy_grouping
    )
    flow_calls: list[list[int]] = []
    first_flow_started = threading.Event()
    release_first_flow = threading.Event()
    second_flow_started = threading.Event()
    release_second_flow = threading.Event()
    call_count = 0

    def block_between_flow_groups(flow, inputs):
        nonlocal call_count
        del flow
        call_count += 1
        members = [int(item.token[0, 0]) for item in inputs]
        flow_calls.append(members)
        if call_count == 1:
            first_flow_started.set()
            assert release_first_flow.wait(timeout=5)
        elif call_count == 2:
            second_flow_started.set()
            assert release_second_flow.wait(timeout=5)
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(
        stages.FunCosyVoice3Flow, "inference", block_between_flow_groups
    )
    initial = [
        IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
        IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
        IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
    ]
    for message in initial:
        scheduler.inbox.put(message)
    worker = threading.Thread(target=scheduler.start)
    worker.start()
    try:
        assert first_flow_started.wait(timeout=5)
        scheduler.inbox.put(
            IncomingMessage("d", "new_request", _buffered_payload("d", 3, 40))
        )
        scheduler.inbox.put(
            IncomingMessage("e", "new_request", _buffered_payload("e", 3, 50))
        )
        release_first_flow.set()

        assert second_flow_started.wait(timeout=5)
        assert flow_calls == [[10, 20], [30, 40, 50]]
        first_results = [
            scheduler.outbox.get(timeout=5),
            scheduler.outbox.get(timeout=5),
        ]
        assert [message.request_id for message in first_results] == ["a", "b"]
        assert all(message.type == "result" for message in first_results)
        with pytest.raises(queue.Empty):
            scheduler.outbox.get_nowait()

        release_second_flow.set()
        remaining_results = [scheduler.outbox.get(timeout=5) for _ in range(3)]
        assert [message.request_id for message in remaining_results] == [
            "c",
            "d",
            "e",
        ]
        assert all(message.type == "result" for message in remaining_results)
    finally:
        release_first_flow.set()
        release_second_flow.set()
        scheduler.stop()
        worker.join(timeout=5)

    assert not worker.is_alive()
    assert prepare_calls == [10, 20, 30, 40, 50]
    assert plan_calls == [[0, 1, 2], [2, 3, 4]]


def test_buffered_vocoder_reuses_plan_without_new_arrival(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch)
    plan_calls: list[list[int]] = []
    original_grouping = streaming_vocoder_module.adaptive_flow_requests_grouping

    def spy_grouping(requests, **kwargs):
        plan_calls.append([request.index for request in requests])
        return original_grouping(requests, **kwargs)

    monkeypatch.setattr(
        streaming_vocoder_module, "adaptive_flow_requests_grouping", spy_grouping
    )
    _run_buffered_batch(
        scheduler,
        [
            IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
            IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
            IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
        ],
    )

    assert plan_calls == [[0, 1, 2]]


def test_buffered_vocoder_executes_largest_groups_first_with_stable_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch, max_batch_size=5)
    flow_calls: list[list[int]] = []

    def fake_grouping(requests, **kwargs):
        del kwargs
        return [[requests[0]], requests[1:3], requests[3:5]]

    monkeypatch.setattr(
        streaming_vocoder_module, "adaptive_flow_requests_grouping", fake_grouping
    )

    def record_inference(flow, inputs):
        del flow
        flow_calls.append([int(item.token[0, 0]) for item in inputs])
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", record_inference)
    _run_buffered_batch(
        scheduler,
        [
            IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
            IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
            IncomingMessage("c", "new_request", _buffered_payload("c", 2, 30)),
            IncomingMessage("d", "new_request", _buffered_payload("d", 2, 40)),
            IncomingMessage("e", "new_request", _buffered_payload("e", 2, 50)),
        ],
    )

    assert flow_calls == [[20, 30], [40, 50], [10]]
    assert [message.request_id for message in _drain_buffered_results(scheduler)] == [
        "b",
        "c",
        "d",
        "e",
        "a",
    ]


def test_buffered_vocoder_skips_batch_wait_while_pending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch, max_batch_wait_ms=30)
    first = IncomingMessage("a", "new_request", _buffered_payload("a", 2))

    assert scheduler.new_request_batch_wait_s(first) == 0.03
    scheduler.handle_new_request_batch([first])
    assert scheduler.new_request_batch_wait_s(first) == 0.0


def test_buffered_vocoder_maps_results_after_adaptive_reordering(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, vocoder, scheduler = _buffered_scheduler(monkeypatch)
    monkeypatch.setattr(
        vocoder,
        "mel2wav_batch",
        lambda mels: [torch.tensor([[float(mel[0, 0, 0])]]) for mel in mels],
    )

    # The outer order is a, b, c; adaptive grouping executes b/c first because
    # they have equal shorter mel lengths, then a.
    messages = [
        IncomingMessage("a", "new_request", _buffered_payload("a", 3, 10)),
        IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
        IncomingMessage("c", "new_request", _buffered_payload("c", 2, 30)),
    ]
    _run_buffered_batch(scheduler, messages)

    results = _drain_buffered_results(scheduler)
    assert [message.request_id for message in results] == ["b", "c", "a"]
    assert {message.request_id: _audio_value(message) for message in results} == {
        "a": 10.0,
        "b": 20.0,
        "c": 30.0,
    }


def test_buffered_vocoder_emits_each_successful_request_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch)
    messages = [
        IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
        IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
        IncomingMessage("c", "new_request", _buffered_payload("c", 2, 30)),
    ]

    _run_buffered_batch(scheduler, messages)

    results = _drain_buffered_results(scheduler)
    assert [message.type for message in results] == ["result", "result", "result"]
    assert sorted(message.request_id for message in results) == ["a", "b", "c"]


def test_buffered_vocoder_later_group_failure_does_not_retroactively_fail(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch)
    call_count = 0

    def fail_second_group(flow, inputs):
        nonlocal call_count
        del flow
        call_count += 1
        if call_count == 2:
            raise RuntimeError("G1 failed")
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fail_second_group)
    _run_buffered_batch(
        scheduler,
        [
            IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
            IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
            IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
        ],
    )

    results = _drain_buffered_results(scheduler)
    by_request = {
        request_id: [message for message in results if message.request_id == request_id]
        for request_id in {message.request_id for message in results}
    }
    assert [message.type for message in by_request["a"]] == ["result"]
    assert [message.type for message in by_request["b"]] == ["result"]
    assert [message.type for message in by_request["c"]] == ["error"]
    assert "G1 failed" in str(by_request["c"][0].data)


def test_buffered_vocoder_waits_for_all_hift_subgroups_before_emitting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, vocoder, scheduler = _buffered_scheduler(monkeypatch)
    # Flow sees one group with mel lengths 4, 4, and 10. The padding budget
    # admits it as one adaptive group, while HiFT's 1.5x waste limit splits it
    # into [a, b] and [c].
    vocoder.flow_merge_max_gap_frames = 6
    vocoder.flow_merge_pad_budget_percent = 70
    first_hift_finished = threading.Event()
    second_hift_started = threading.Event()
    release_second_hift = threading.Event()
    hift_call_count = 0

    def fail_second_hift(mels):
        nonlocal hift_call_count
        hift_call_count += 1
        if hift_call_count == 1:
            first_hift_finished.set()
            return [torch.ones(1, 16) for _ in mels]
        second_hift_started.set()
        assert release_second_hift.wait(timeout=5)
        raise RuntimeError("H1 failed")

    monkeypatch.setattr(vocoder, "mel2wav_batch", fail_second_hift)
    messages = [
        IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
        IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
        IncomingMessage("c", "new_request", _buffered_payload("c", 5, 30)),
    ]
    scheduler.handle_new_request_batch(messages)
    worker = threading.Thread(target=scheduler.run_ready_step)
    worker.start()
    try:
        assert first_hift_finished.wait(timeout=5)
        assert second_hift_started.wait(timeout=5)
        with pytest.raises(queue.Empty):
            scheduler.outbox.get_nowait()
    finally:
        release_second_hift.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    results = _drain_buffered_results(scheduler)
    assert [(message.request_id, message.type) for message in results] == [
        ("a", "error"),
        ("b", "error"),
        ("c", "error"),
    ]
    assert all("H1 failed" in str(message.data) for message in results)


def test_buffered_vocoder_aborted_request_before_own_group_emits_nothing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch)
    first_group_started = threading.Event()
    release_first_group = threading.Event()
    call_count = 0

    def block_first_group(flow, inputs):
        nonlocal call_count
        del flow
        call_count += 1
        if call_count == 1:
            first_group_started.set()
            assert release_first_group.wait(timeout=5)
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", block_first_group)
    messages = [
        IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
        IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
        IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
    ]
    scheduler.handle_new_request_batch(messages)
    worker = threading.Thread(target=scheduler.run_ready_step)
    worker.start()
    try:
        assert first_group_started.wait(timeout=5)
        scheduler.abort("c")
    finally:
        release_first_group.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    results = _drain_buffered_results(scheduler)
    assert [message.request_id for message in results] == ["a", "b"]
    assert all(message.type == "result" for message in results)


def test_buffered_vocoder_active_group_abort_does_not_replan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch, max_batch_size=4)
    plan_calls: list[list[int]] = []
    original_grouping = streaming_vocoder_module.adaptive_flow_requests_grouping

    def spy_grouping(requests, **kwargs):
        plan_calls.append([request.index for request in requests])
        return original_grouping(requests, **kwargs)

    monkeypatch.setattr(
        streaming_vocoder_module, "adaptive_flow_requests_grouping", spy_grouping
    )
    first_group_started = threading.Event()
    release_first_group = threading.Event()
    flow_calls: list[list[int]] = []
    call_count = 0

    def block_first_group(flow, inputs):
        nonlocal call_count
        del flow
        call_count += 1
        flow_calls.append([int(item.token[0, 0]) for item in inputs])
        if call_count == 1:
            first_group_started.set()
            assert release_first_group.wait(timeout=5)
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", block_first_group)
    scheduler.handle_new_request_batch(
        [
            IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
            IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
            IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
            IncomingMessage("d", "new_request", _buffered_payload("d", 3, 40)),
        ]
    )
    worker = threading.Thread(target=scheduler.run_ready_step)
    worker.start()
    try:
        assert first_group_started.wait(timeout=5)
        with pytest.raises(queue.Empty):
            scheduler.outbox.get_nowait()
        scheduler.abort("b")
        release_first_group.set()
        worker.join(timeout=5)
    finally:
        release_first_group.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    scheduler.run_ready_step()
    results = _drain_buffered_results(scheduler)
    by_request = {
        request_id: [message for message in results if message.request_id == request_id]
        for request_id in {message.request_id for message in results}
    }
    assert [message.type for message in by_request["a"]] == ["result"]
    assert "b" not in by_request
    assert [message.type for message in by_request["c"]] == ["result"]
    assert [message.type for message in by_request["d"]] == ["result"]
    assert flow_calls == [[10, 20], [30, 40]]
    assert plan_calls == [[0, 1, 2, 3]]


def test_buffered_vocoder_replans_after_abort_invalidates_remaining_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, _, scheduler = _buffered_scheduler(monkeypatch)
    plan_calls: list[list[int]] = []
    original_grouping = streaming_vocoder_module.adaptive_flow_requests_grouping

    def spy_grouping(requests, **kwargs):
        plan_calls.append([request.index for request in requests])
        return original_grouping(requests, **kwargs)

    monkeypatch.setattr(
        streaming_vocoder_module, "adaptive_flow_requests_grouping", spy_grouping
    )
    first_group_started = threading.Event()
    release_first_group = threading.Event()
    call_count = 0

    def block_first_group(flow, inputs):
        nonlocal call_count
        del flow
        call_count += 1
        if call_count == 1:
            first_group_started.set()
            assert release_first_group.wait(timeout=5)
        return [
            torch.full(
                (1, 80, item.token.shape[1] * 2),
                float(item.token[0, 0]),
            )
            for item in inputs
        ]

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", block_first_group)
    scheduler.handle_new_request_batch(
        [
            IncomingMessage("a", "new_request", _buffered_payload("a", 2, 10)),
            IncomingMessage("b", "new_request", _buffered_payload("b", 2, 20)),
            IncomingMessage("c", "new_request", _buffered_payload("c", 3, 30)),
            IncomingMessage("d", "new_request", _buffered_payload("d", 3, 40)),
        ]
    )
    worker = threading.Thread(target=scheduler.run_ready_step)
    worker.start()
    try:
        assert first_group_started.wait(timeout=5)
        scheduler.abort("c")
        release_first_group.set()
        worker.join(timeout=5)
    finally:
        release_first_group.set()
        worker.join(timeout=5)

    assert not worker.is_alive()
    scheduler.run_ready_step()
    results = _drain_buffered_results(scheduler)
    assert [(message.request_id, message.type) for message in results] == [
        ("a", "result"),
        ("b", "result"),
        ("d", "result"),
    ]
    assert plan_calls == [[0, 1, 2], [3]]


def test_flow_scheduler_cost_uses_exact_frames() -> None:
    vocoder = stages.CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)

    assert vocoder.flow_scheduler_cost(_payload(state)) == 6


def test_flow_admission_defers_request_after_long_singleton(monkeypatch) -> None:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)
    monkeypatch.setattr(
        stages,
        "load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **kwargs: (
            _RunnableFakeFlow(),
            _FakeHiFT(),
        ),
    )
    # The default admission budget is sized for the real seed-tts-eval length
    # distribution, so pin it here: this test is about admission behaviour, not
    # about the default value.
    scheduler = stages.create_vocoder_executor(
        "model", device="cpu", flow_batch_admission_frames=2000
    )
    long_state = _state(prompt_tokens=0)
    long_state.audio_codes = _codes(2200)
    short_state = _state(prompt_tokens=0)
    short_state.audio_codes = _codes(2)
    first = IncomingMessage("long", "new_request", _payload(long_state))
    second = IncomingMessage("short", "new_request", _payload(short_state))
    scheduler.inbox.put(second)

    assert scheduler.max_batch_cost == 2000
    assert scheduler.collect_new_request_batch(first) == [first]
    assert scheduler.next_message() == second


def test_create_vocoder_executor_defaults_batch_for_real_lengths(monkeypatch) -> None:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)
    monkeypatch.setattr(
        stages,
        "load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **kwargs: (
            _RunnableFakeFlow(),
            _FakeHiFT(),
        ),
    )
    scheduler = stages.create_vocoder_executor("model", device="cpu")

    assert scheduler.max_batch_cost == stages.DEFAULT_FLOW_BATCH_ADMISSION_FRAMES
    assert (
        scheduler.max_batch_cost // 713 >= 8
    ), "default admission budget no longer holds a useful batch"
    assert scheduler.max_batch_size == 16
    assert scheduler.max_batch_wait_s == pytest.approx(0.03)
    assert scheduler.vocoder.flow_merge_max_gap_frames == 384
    assert scheduler.vocoder.flow_merge_pad_budget_percent == 25.0


def test_create_vocoder_executor_threads_batch_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_flow = _RunnableFakeFlow()
    fake_hift = _FakeHiFT()
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)

    def fake_load(checkpoint_dir, device, fp16, **kwargs):
        captured.update(
            {
                "checkpoint_dir": checkpoint_dir,
                "device": device,
                "fp16": fp16,
                "enable_flow_estimator_trt": kwargs.get(
                    "enable_flow_estimator_trt", False
                ),
            }
        )
        return fake_flow, fake_hift

    monkeypatch.setattr(stages, "load_cosyvoice3_flow_hift", fake_load)

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        dtype="float16",
        max_batch_size=6,
        max_batch_wait_ms=7,
        flow_batch_admission_frames=200,
        flow_merge_max_gap_frames=0,
        flow_merge_pad_budget_percent=0,
    )

    assert isinstance(scheduler, FunCosyVoice3StreamingVocoderScheduler)
    assert scheduler.max_batch_size == 6
    assert scheduler.max_batch_wait_s == pytest.approx(0.007)
    assert scheduler.max_batch_cost == 200
    assert callable(scheduler.request_cost_fn)
    assert scheduler.vocoder.flow_merge_max_gap_frames == 0
    assert scheduler.vocoder.flow_merge_pad_budget_percent == 0
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)
    assert scheduler.request_cost_fn(_payload(state)) == 6
    assert captured == {
        "checkpoint_dir": "/checkpoint",
        "device": "cpu",
        "fp16": True,
        "enable_flow_estimator_trt": False,
    }


def test_create_vocoder_executor_threads_trt_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)

    def fake_load(checkpoint_dir, device, fp16, **kwargs):
        captured.update(
            {
                "enable_flow_estimator_trt": kwargs.get("enable_flow_estimator_trt"),
            }
        )
        return _RunnableFakeFlow(), _FakeHiFT()

    monkeypatch.setattr(stages, "load_cosyvoice3_flow_hift", fake_load)

    stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=4,
        enable_flow_estimator_trt=True,
    )

    assert captured == {
        "enable_flow_estimator_trt": True,
    }


def _executor_compiles(monkeypatch, **kwargs) -> bool:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)
    monkeypatch.setattr(
        stages,
        "load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **_: (
            _RunnableFakeFlow(),
            _FakeHiFT(),
        ),
    )
    compiled: list[object] = []
    monkeypatch.setattr(
        stages,
        "compile_dit_backbone",
        lambda flow, autocast_dtype: compiled.append(flow),
    )
    stages.create_vocoder_executor("model", device="cpu", **kwargs)
    return bool(compiled)


def test_create_vocoder_executor_skips_dit_compile_by_default(monkeypatch) -> None:
    assert not _executor_compiles(monkeypatch)
    assert _executor_compiles(monkeypatch, enable_dit_torch_compile=True)


def test_create_vocoder_executor_trt_alone_skips_the_default_compile(
    monkeypatch,
) -> None:
    assert not _executor_compiles(monkeypatch, enable_flow_estimator_trt=True)


def test_create_vocoder_executor_rejects_trt_and_compile() -> None:
    with pytest.raises(ValueError, match="enable only one"):
        stages.create_vocoder_executor(
            "model",
            enable_dit_torch_compile=True,
            enable_flow_estimator_trt=True,
        )


def test_attach_flow_estimator_trt_requires_cuda(monkeypatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="requires NVIDIA CUDA"):
        stages.attach_flow_estimator_trt(object(), "/checkpoint", "cuda:0")


@pytest.mark.parametrize("device", ["cpu", "npu:0", "xpu:0"])
def test_attach_flow_estimator_trt_rejects_non_cuda_device(
    monkeypatch, device: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA vocoder device"):
        stages.attach_flow_estimator_trt(object(), "/checkpoint", device)


def test_attach_flow_estimator_trt_wraps_module_with_fallback(monkeypatch) -> None:
    from sglang_omni.models.fun_cosyvoice3.flow_estimator_trt import (
        FlowEstimatorTRTModule,
    )

    class _Fallback(torch.nn.Module):
        pass

    class _Decoder:
        def __init__(self) -> None:
            self.estimator = _Fallback()

    class _Flow:
        def __init__(self) -> None:
            self.decoder = _Decoder()

    flow = _Flow()
    fallback = flow.decoder.estimator
    captured: dict[str, object] = {}

    def fake_resolve(_checkpoint_dir: str) -> str:
        return "/tmp/fake.onnx"

    def fake_build(onnx_path, device, *, fallback=None, wrap_module=True, **kwargs):
        del kwargs
        captured["onnx_path"] = onnx_path
        captured["device"] = device
        captured["fallback"] = fallback
        captured["wrap_module"] = wrap_module

        class _FakeTRT:
            max_batch = 2

        return FlowEstimatorTRTModule(_FakeTRT(), fallback=fallback)

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    # note (PoTaTo-Mika) : attach_flow_estimator_trt also gates on current_platform.is_cuda(),
    # which stays False on CPU-only hosts (CI hides CUDA), not just on
    # torch.cuda.is_available().
    monkeypatch.setattr(stages.current_platform, "is_cuda", lambda: True)
    import sglang_omni.models.fun_cosyvoice3.flow_estimator_trt as trt_mod

    monkeypatch.setattr(trt_mod, "resolve_flow_estimator_onnx", fake_resolve)
    monkeypatch.setattr(trt_mod, "build_flow_estimator_trt", fake_build)

    stages.attach_flow_estimator_trt(flow, "/checkpoint", "cuda:0")

    assert captured["wrap_module"] is True
    assert captured["fallback"] is fallback
    assert isinstance(flow.decoder.estimator, FlowEstimatorTRTModule)
    assert flow.decoder.estimator._fallback is fallback


def test_preprocessing_executor_threads_max_concurrency() -> None:
    scheduler = stages.create_preprocessing_executor("model", max_concurrency=11)
    assert scheduler._max_concurrency == 11


def test_preprocessing_executor_rejects_non_positive_concurrency() -> None:
    with pytest.raises(ValueError, match="max_concurrency"):
        stages.create_preprocessing_executor("model", max_concurrency=0)


def test_onnx_intra_op_threads_reaches_both_encoders(monkeypatch) -> None:
    from sglang_omni.models.fun_cosyvoice3 import engine_builder, request_builders

    seen: dict[str, int] = {}

    def fake_tokenizer(model_path, device="cpu", intra_op_threads=1):
        seen["speech_tokenizer"] = intra_op_threads
        return object()

    def fake_encoder(model_path, device="cpu", intra_op_threads=1):
        seen["speaker_encoder"] = intra_op_threads
        return object()

    class _StubModel:
        def load_weights(self, weights) -> None:
            del weights

    monkeypatch.setattr(engine_builder, "SpeechTokenizerV3", fake_tokenizer)
    monkeypatch.setattr(engine_builder, "SpeakerEncoder", fake_encoder)
    monkeypatch.setattr(engine_builder, "CosyVoice3Tokenizer", lambda path: object())
    monkeypatch.setattr(engine_builder.torch, "load", lambda *a, **k: {})
    monkeypatch.setattr(
        request_builders, "set_cosyvoice3_preprocessing_context", lambda **kwargs: None
    )

    builder = engine_builder.FunCosyVoice3EngineBuilder(onnx_intra_op_threads=6)
    builder._checkpoint_root = "/tmp"
    builder.before_memory_pool(
        model_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=_StubModel(),
                model_config=SimpleNamespace(vocab_size=0),
            )
        ),
        checkpoint_dir="/tmp",
        device="cpu",
        gpu_id=0,
        server_args=object(),
    )

    assert seen == {"speech_tokenizer": 6, "speaker_encoder": 6}


def test_create_vocoder_executor_rejects_non_positive_admission_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )

    with pytest.raises(ValueError, match="flow_batch_admission_frames"):
        stages.create_vocoder_executor(
            "model",
            device="cpu",
            flow_batch_admission_frames=0,
        )


def test_pipeline_config_sets_flow_batch_admission_by_default() -> None:
    vocoder_stage = next(
        stage
        for stage in FunCosyVoice3PipelineConfig(model_path="model").stages
        if stage.name == "vocoder"
    )
    assert vocoder_stage.factory.model_dump(exclude_none=True) == {
        "dtype": "bfloat16",
        "flow_batch_admission_frames": 8000,
        "flow_merge_max_gap_frames": 384,
        "flow_merge_pad_budget_percent": 25.0,
        "flow_cuda_graph_capture_shapes": FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
        "max_batch_size": 16,
        "max_batch_wait_ms": 30,
        "enable_flow_cuda_graph": True,
        "enable_flow_estimator_trt": False,
        "token_hop_len": 25,
        "token_max_hop_len": 100,
        "disable_hop_growth": False,
    }


def test_vocoder_hift_defaults_to_float32(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(flow, _FakeHiFT())

    # bfloat16 gave HiFT no speedup, so the default keeps full precision.
    assert vocoder.hift_autocast_dtype is None
    with torch.autocast(
        device_type=stages.current_platform.device_type,
        dtype=vocoder.hift_autocast_dtype,
        enabled=vocoder.hift_autocast_dtype is not None,
    ):
        assert not torch.is_autocast_enabled()
