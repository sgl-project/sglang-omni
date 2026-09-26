# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import contextlib
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from sglang_omni.client.client import Client
from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import (
    FUN_COSYVOICE3_DEFAULT_FLOW_CUDA_GRAPH_CAPTURE_SHAPES,
    FunCosyVoice3PipelineConfig,
)
from sglang_omni.models.fun_cosyvoice3.packed_dit import PackedDiT
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage
from tests.unit_test.fun_cosyvoice3.test_flow_batch import FakeFlow as _PackedFlow


class FakeHiFT(torch.nn.Module):
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


class FakeEstimator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("batch adapter should be mocked in vocoder unit tests")


class RunnableFakeFlow(_PackedFlow):
    def __init__(self):
        super().__init__(channels=80, max_frames=8192)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80)


class GraphRunnableFakeFlow(RunnableFakeFlow):
    def __init__(self, events: list[str] | None = None) -> None:
        super().__init__()
        self.events = events
        self.attached_runner: stages.FlowCudaGraphRunner | None = None

    def attach_cuda_graph_runner(self, runner: stages.FlowCudaGraphRunner) -> None:
        self.attached_runner = runner
        if self.events is not None:
            self.events.append("attach")
        else:
            pass


class RecordingPackedDiT(PackedDiT):
    def __init__(self) -> None:
        self.is_ragged = True
        self.disable_calls = 0

    def compile(self, dtype: torch.dtype | None) -> bool:
        del dtype
        return True

    def disable_compile(self) -> None:
        self.disable_calls += 1


def packed_compile_scheduler(
    packed_estimator: RecordingPackedDiT,
    *,
    failure: str | None = None,
) -> tuple[
    FunCosyVoice3StreamingVocoderScheduler,
    list[list[stages.FlowBatchInput]],
    list[list[stages.FlowBatchInput]],
]:
    hop_batches: list[list[stages.FlowBatchInput]] = []
    leftover_batches: list[list[stages.FlowBatchInput]] = []
    flow = SimpleNamespace(
        output_size=80,
        token_mel_ratio=2,
        spk_embed_affine_layer=SimpleNamespace(in_features=192),
        packed_estimator=packed_estimator,
    )
    vocoder = SimpleNamespace(
        flow=flow,
        autocast_dtype=torch.bfloat16,
        stream_context=contextlib.nullcontext(),
    )

    def hop_batch(items: list[stages.FlowBatchInput]) -> list[torch.Tensor]:
        if failure == "hop":
            raise RuntimeError("causal materialization failed")
        else:
            pass
        hop_batches.append(list(items))
        return []

    def leftover_batch(items: list[stages.FlowBatchInput]) -> list[torch.Tensor]:
        leftover_batches.append(list(items))
        return []

    vocoder.hop_batch = hop_batch
    vocoder.leftover_batch = leftover_batch
    scheduler = FunCosyVoice3StreamingVocoderScheduler(vocoder)
    return scheduler, hop_batches, leftover_batches


def test_packed_dit_compile_warmup_materializes_serving_variants() -> None:
    packed_estimator = RecordingPackedDiT()
    scheduler, hop_batches, leftover_batches = packed_compile_scheduler(
        packed_estimator
    )

    scheduler.warmup_packed_dit_compile()

    assert len(hop_batches) == len(leftover_batches) == 1


def test_packed_dit_compile_warmup_failure_disables_compiled_path() -> None:
    packed_estimator = RecordingPackedDiT()
    scheduler, _, leftover_batches = packed_compile_scheduler(
        packed_estimator,
        failure="hop",
    )

    with pytest.raises(RuntimeError, match="causal materialization failed"):
        scheduler.warmup_packed_dit_compile()

    assert packed_estimator.disable_calls == 1
    assert leftover_batches == []


def test_mlx_stream_scheduler_consumes_chunks_before_final_decode() -> None:
    class FakeMlxVocoder:
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
        FakeMlxVocoder(), max_batch_wait_ms=0
    )
    state = FunCosyVoice3State(
        stream=True,
        flow_prompt_speech_token=torch.tensor([[1, 2]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
        flow_embedding=torch.ones(1, 192),
    )
    payload = make_payload(state)
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

    class Predictor:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    hift = SimpleNamespace(f0_predictor=Predictor())

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

    class Model:
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

    flow = Model()
    flow.decoder = SimpleNamespace(estimator=torch.nn.Module())
    hift = Model()

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


class BatchCapableFakeFlow(torch.nn.Module):
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
            estimator=FakeEstimator(),
            forward_estimator=lambda *args, **kwargs: None,
        )


class FakeFlow(torch.nn.Module):
    """CosyVoice-native Flow.inference(**kwargs) used by causal token2wav hops."""

    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []
        self.decoder = SimpleNamespace(estimator=FakeEstimator())

    def inference(self, **kwargs):
        self.calls.append(kwargs)
        token_count = kwargs["token"].shape[1]
        return torch.ones(1, 80, token_count * 2), None


def make_payload(state: FunCosyVoice3State) -> StagePayload:
    return StagePayload(
        request_id="req-vocoder",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def test_cosyvoice3_vocoder_does_not_pad_or_rescale_short_sequences() -> None:
    flow = FakeFlow()
    hift = FakeHiFT()
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
    vocoder = stages.CosyVoice3Vocoder(FakeFlow(), FakeHiFT())

    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        vocoder.token2wav(
            token=torch.zeros(1, 0, dtype=torch.long),
            prompt_token=torch.tensor([[4]], dtype=torch.int32),
            prompt_feat=torch.zeros(1, 2, 80),
            embedding=torch.ones(1, 192),
        )


def test_cosyvoice3_token2wav_chunk_slices_mel_and_hift_delta() -> None:
    flow = FakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())
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
    vocoder = stages.CosyVoice3Vocoder(BatchCapableFakeFlow(), FakeHiFT())
    state = FunCosyVoice3State(
        text="hello",
        audio_codes=torch.tensor([[1, 2], [3, 4]]),
        flow_prompt_speech_token=torch.tensor([[5]], dtype=torch.int32),
        flow_embedding=torch.ones(1, 192),
    )
    payload = make_payload(state)

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
    vocoder = stages.CosyVoice3Vocoder(BatchCapableFakeFlow(), FakeHiFT())
    payload = make_payload(FunCosyVoice3State(text="hello"))

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
    stored = mlx_vocoder.store_result(make_payload(state), state, waveform, 24000)
    result = Client.default_result_builder(stored.request_id, stored.data)

    np.testing.assert_array_equal(result.audio_data, waveform.reshape(-1))
    assert result.sample_rate == 24000
    assert result.modality == "audio"
    assert result.usage.total_tokens == 5
    assert "audio_codes" not in stored.data
    assert "audio_samples" not in stored.data


def test_cosyvoice3_vocoder_rejects_missing_audio_output() -> None:
    vocoder = stages.CosyVoice3Vocoder(BatchCapableFakeFlow(), FakeHiFT())
    state = FunCosyVoice3State(text="hello")
    payload = make_payload(state)

    with pytest.raises(RuntimeError, match="did not return audio"):
        vocoder.store_result(payload, state, None, 24000)


def test_cosyvoice3_vocoder_decode_batch_uses_state_conditioning(monkeypatch) -> None:
    flow = BatchCapableFakeFlow()
    batch_calls: list[list] = []
    install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())
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
    install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(
        BatchCapableFakeFlow(),
        FakeHiFT(),
        autocast_dtype=torch.float16,
    )

    asyncio.run(vocoder.decode_batch([(make_state(), torch.tensor([1, 2]))]))

    assert observed == [
        ("cpu", torch.float16, True),
        (stages.current_platform.device_type, None, False),
    ]


def make_state(
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


def make_codes(length: int, value: int = 1) -> torch.Tensor:
    return torch.full((length,), value, dtype=torch.long)


def flow_requests(totals: list[int]) -> list[stages.PreparedFlowRequest]:
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


def install_fake_batch_adapter(monkeypatch, calls: list[list]) -> None:
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
    flow = BatchCapableFakeFlow()
    hift = FakeHiFT()
    batch_calls: list[list] = []
    install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(vocoder.decode_batch([(make_state(), make_codes(2))]))

    assert len(results) == 1
    assert [len(call) for call in batch_calls] == [1]
    assert len(hift.calls) == 1


def test_decode_payload_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = BatchCapableFakeFlow()
    batch_calls: list[list] = []
    install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())
    state = make_state()
    state.audio_codes = make_codes(2)

    result = asyncio.run(vocoder.decode_payload(make_payload(state)))

    assert result.data["modality"] == "audio"
    assert [len(call) for call in batch_calls] == [1]


def test_decode_batch_merges_flow_preserving_hift_groups_and_order(
    monkeypatch,
) -> None:
    items = [
        (make_state(sample_rate=16003, prompt_tokens=0), make_codes(50, 3)),
        (make_state(sample_rate=16001, prompt_tokens=0), make_codes(24, 1)),
        (make_state(sample_rate=16004, prompt_tokens=0), make_codes(51, 4)),
        (make_state(sample_rate=16002, prompt_tokens=0), make_codes(25, 2)),
    ]

    flow = BatchCapableFakeFlow()
    hift = FakeHiFT()
    flow_calls: list[list] = []
    install_fake_batch_adapter(monkeypatch, flow_calls)
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
        flow_requests(totals),
        flow_merge_max_gap_frames=flow_merge_max_gap_frames,
        flow_merge_pad_budget_percent=flow_merge_pad_budget_percent,
    )

    assert [
        [request.total_mel_frames for request in group] for group in groups
    ] == expected


def test_decode_batch_runs_hift_once_over_padded_mels(monkeypatch) -> None:
    flow = BatchCapableFakeFlow()
    hift = FakeHiFT()
    install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(
        vocoder.decode_batch(
            [
                (make_state(), make_codes(9)),
                (make_state(), make_codes(10)),
                (make_state(), make_codes(11)),
            ]
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
    flow = BatchCapableFakeFlow()
    hift = FakeHiFT()
    install_fake_batch_adapter(monkeypatch, [])
    # max_waste=1.0 only accepts groups that need no padding at all.
    vocoder = stages.CosyVoice3Vocoder(flow, hift, hift_max_padding_waste=1.0)

    asyncio.run(
        vocoder.decode_batch(
            [(make_state(), make_codes(2)), (make_state(), make_codes(3))]
        )
    )

    assert len(hift.calls) == 2


def test_decode_batch_long_singleton_uses_batch_adapter(monkeypatch) -> None:
    flow = BatchCapableFakeFlow()
    batch_calls: list[list] = []
    install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())

    asyncio.run(
        vocoder.decode_batch([(make_state(prompt_tokens=0), make_codes(2200, 1))])
    )

    assert [len(call) for call in batch_calls] == [1]
    assert batch_calls[0][0].token.shape[1] == 2200


def test_vocoder_rejects_non_pytorch_flow_estimator() -> None:
    flow = BatchCapableFakeFlow()
    flow.decoder.estimator = object()

    with pytest.raises(RuntimeError, match="PyTorch module or a TensorRT wrapper"):
        stages.CosyVoice3Vocoder(flow, FakeHiFT())


def test_vocoder_accepts_tensorrt_flow_estimator() -> None:
    class FakeTRTEstimator:
        def acquire_estimator(self):
            return [None, None], None

        def execute(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("vocoder init must not run the estimator")

    flow = BatchCapableFakeFlow()
    flow.decoder.estimator = FakeTRTEstimator()
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())
    assert vocoder.flow is not None


def test_decode_batch_alignment_mismatch_fails() -> None:
    flow = BatchCapableFakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())

    with pytest.raises(ValueError, match="prompt feature length"):
        asyncio.run(
            vocoder.decode_batch(
                [
                    (make_state(prompt_tokens=1, prompt_feat_frames=1), make_codes(2)),
                    (make_state(), make_codes(3)),
                ]
            )
        )


def test_decode_batch_embedding_width_mismatch_fails() -> None:
    flow = BatchCapableFakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())
    invalid = make_state()
    invalid.flow_embedding = torch.ones(1, 191)

    with pytest.raises(ValueError, match="embedding width"):
        asyncio.run(
            vocoder.decode_batch(
                [(invalid, make_codes(2)), (make_state(), make_codes(3))]
            )
        )


def test_decode_batch_does_not_retry_after_batch_failure(monkeypatch) -> None:
    flow = BatchCapableFakeFlow()
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())

    def fail_batch(flow, inputs):
        del flow, inputs
        raise RuntimeError("batch estimator failed")

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fail_batch)

    with pytest.raises(RuntimeError, match="batch estimator failed"):
        asyncio.run(
            vocoder.decode_batch(
                [(make_state(), make_codes(2)), (make_state(), make_codes(3))]
            )
        )


def test_flow_scheduler_cost_uses_exact_frames() -> None:
    vocoder = stages.CosyVoice3Vocoder(BatchCapableFakeFlow(), FakeHiFT())
    state = make_state(prompt_tokens=1)
    state.audio_codes = make_codes(2)

    assert vocoder.flow_scheduler_cost(make_payload(state)) == 6


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
            RunnableFakeFlow(),
            FakeHiFT(),
        ),
    )
    # The default admission budget is sized for the real seed-tts-eval length
    # distribution, so pin it here: this test is about admission behaviour, not
    # about the default value.
    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        flow_batch_admission_frames=2000,
        enable_dit_torch_compile=False,
    )
    long_state = make_state(prompt_tokens=0)
    long_state.audio_codes = make_codes(2200)
    short_state = make_state(prompt_tokens=0)
    short_state.audio_codes = make_codes(2)
    first = IncomingMessage("long", "new_request", make_payload(long_state))
    second = IncomingMessage("short", "new_request", make_payload(short_state))
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
            RunnableFakeFlow(),
            FakeHiFT(),
        ),
    )
    scheduler = stages.create_vocoder_executor(
        "model", device="cpu", enable_dit_torch_compile=False
    )

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

    fake_flow = RunnableFakeFlow()
    fake_hift = FakeHiFT()
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
        enable_dit_torch_compile=False,
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
    state = make_state(prompt_tokens=1)
    state.audio_codes = make_codes(2)
    assert scheduler.request_cost_fn(make_payload(state)) == 6
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
        return RunnableFakeFlow(), FakeHiFT()

    monkeypatch.setattr(stages, "load_cosyvoice3_flow_hift", fake_load)

    stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=4,
        enable_dit_torch_compile=False,
        enable_flow_estimator_trt=True,
    )

    assert captured == {
        "enable_flow_estimator_trt": True,
    }


<<<<<<< HEAD
def executor_compiles(monkeypatch, **kwargs) -> bool:
=======
def create_scheduler_recording_native_compile(
    monkeypatch,
    **kwargs,
) -> tuple[list[torch.nn.Module], FunCosyVoice3StreamingVocoderScheduler]:
>>>>>>> upstream/main
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)
    monkeypatch.setattr(
        stages,
        "load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **_: (
            RunnableFakeFlow(),
            FakeHiFT(),
        ),
    )
    compiled: list[torch.nn.Module] = []

    def fake_compile(flow, autocast_dtype):
        assert autocast_dtype == torch.bfloat16
        compiled.append(flow)

    monkeypatch.setattr(stages, "compile_dit_backbone", fake_compile)
    scheduler = stages.create_vocoder_executor("model", device="cpu", **kwargs)
    return compiled, scheduler


@pytest.mark.parametrize("enable_dit_torch_compile", [False, True])
def test_create_vocoder_executor_compile_flag_controls_startup_materialization(
    monkeypatch,
    enable_dit_torch_compile: bool,
) -> None:
    packed_warmups: list[FunCosyVoice3StreamingVocoderScheduler] = []
    monkeypatch.setattr(
        FunCosyVoice3StreamingVocoderScheduler,
        "warmup_packed_dit_compile",
        lambda scheduler: packed_warmups.append(scheduler),
    )

    compiled, _scheduler = create_scheduler_recording_native_compile(
        monkeypatch,
        enable_dit_torch_compile=enable_dit_torch_compile,
    )
    assert len(compiled) == (1 if enable_dit_torch_compile else 0)
    assert len(packed_warmups) == (1 if enable_dit_torch_compile else 0)


FLOW_GRAPH_CAPTURE_SHAPES = ((2, 16),)


def prepare_vocoder_startup(
    monkeypatch: pytest.MonkeyPatch,
    startup_events: list[str],
    *,
    device_type: str,
    allow_native_compile: bool,
) -> GraphRunnableFakeFlow:
    fake_flow = GraphRunnableFakeFlow(startup_events)
    resolved_device = torch.device(device_type)
    monkeypatch.setattr(
        stages,
        "resolve_concrete_device",
        lambda device, gpu_id: resolved_device,
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(stages, "patch_chunk_mask", lambda: None)
    monkeypatch.setattr(
        stages,
        "load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, enable_flow_estimator_trt=False: (
            fake_flow,
            FakeHiFT(),
        ),
    )

    def record_native_compile(flow, autocast_dtype: torch.dtype | None) -> None:
        if allow_native_compile:
            assert flow is fake_flow
            assert autocast_dtype == torch.bfloat16
            startup_events.append("native_compile")
        else:
            raise AssertionError("native compile must stay disabled")

    monkeypatch.setattr(stages, "compile_dit_backbone", record_native_compile)
    monkeypatch.setattr(
        FunCosyVoice3StreamingVocoderScheduler,
        "warmup_now",
        lambda scheduler: startup_events.append("scheduler_warmup"),
    )
    monkeypatch.setattr(
        FunCosyVoice3StreamingVocoderScheduler,
        "warmup_packed_dit_compile",
        lambda scheduler: startup_events.append("packed_warmup"),
    )
    if device_type == "cuda":
        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

        class RecordingFlowCudaGraphRunner:
            def __init__(self, flow, *, device, autocast_dtype) -> None:
                assert flow is fake_flow
                assert device.type == "cuda"
                assert autocast_dtype == torch.bfloat16
                startup_events.append("runner_create")

            def capture(self, capture_shapes: tuple[tuple[int, int], ...]) -> None:
                assert capture_shapes == FLOW_GRAPH_CAPTURE_SHAPES
                startup_events.append("graph_capture")

        monkeypatch.setattr(stages, "FlowCudaGraphRunner", RecordingFlowCudaGraphRunner)
    else:
        pass
    return fake_flow


<<<<<<< HEAD
def test_create_vocoder_executor_skips_dit_compile_by_default(monkeypatch) -> None:
    assert not executor_compiles(monkeypatch)
    assert executor_compiles(monkeypatch, enable_dit_torch_compile=True)
=======
@pytest.mark.parametrize("enable_dit_torch_compile", [False, True])
def test_create_vocoder_executor_compiles_before_flow_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
    enable_dit_torch_compile: bool,
) -> None:
    startup_events: list[str] = []
    prepare_vocoder_startup(
        monkeypatch,
        startup_events,
        device_type="cuda",
        allow_native_compile=enable_dit_torch_compile,
    )

    _scheduler = stages.create_vocoder_executor(
        "model",
        device="cuda",
        enable_dit_torch_compile=enable_dit_torch_compile,
        enable_flow_cuda_graph=True,
        flow_cuda_graph_capture_shapes=FLOW_GRAPH_CAPTURE_SHAPES,
    )

    assert startup_events.count("graph_capture") == 1
    if enable_dit_torch_compile:
        assert startup_events.index("native_compile") < startup_events.index(
            "graph_capture"
        )
    else:
        assert "native_compile" not in startup_events
    assert ("packed_warmup" in startup_events) is enable_dit_torch_compile
>>>>>>> upstream/main


def test_create_vocoder_executor_trt_alone_skips_the_default_compile(
    monkeypatch,
) -> None:
<<<<<<< HEAD
    assert not executor_compiles(monkeypatch, enable_flow_estimator_trt=True)
=======
    compiled, _scheduler = create_scheduler_recording_native_compile(
        monkeypatch,
        enable_dit_torch_compile=False,
        enable_flow_estimator_trt=True,
    )
    assert compiled == []
>>>>>>> upstream/main


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

    class Fallback(torch.nn.Module):
        pass

    class Decoder:
        def __init__(self) -> None:
            self.estimator = Fallback()

    class Flow:
        def __init__(self) -> None:
            self.decoder = Decoder()

    flow = Flow()
    fallback = flow.decoder.estimator
    captured: dict[str, object] = {}

    def fake_resolve(checkpoint_dir: str) -> str:
        return "/tmp/fake.onnx"

    def fake_build(onnx_path, device, *, fallback=None, wrap_module=True, **kwargs):
        del kwargs
        captured["onnx_path"] = onnx_path
        captured["device"] = device
        captured["fallback"] = fallback
        captured["wrap_module"] = wrap_module

        class FakeTRT:
            max_batch = 2

        return FlowEstimatorTRTModule(FakeTRT(), fallback=fallback)

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
    assert flow.decoder.estimator.fallback is fallback


def test_preprocessing_executor_threads_max_concurrency() -> None:
    scheduler = stages.create_preprocessing_executor("model", max_concurrency=11)
    assert scheduler.max_concurrency == 11


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

    class StubModel:
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
    builder.checkpoint_root = "/tmp"
    builder.before_memory_pool(
        model_worker=SimpleNamespace(
            model_runner=SimpleNamespace(
                model=StubModel(),
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
            enable_dit_torch_compile=False,
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
        "enable_dit_torch_compile": True,
        "enable_flow_estimator_trt": False,
        "token_hop_len": 25,
        "token_max_hop_len": 100,
        "disable_hop_growth": False,
    }


def test_vocoder_hift_defaults_to_float32(monkeypatch) -> None:
    flow = BatchCapableFakeFlow()
    install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages.CosyVoice3Vocoder(flow, FakeHiFT())

    # bfloat16 gave HiFT no speedup, so the default keeps full precision.
    assert vocoder.hift_autocast_dtype is None
    with torch.autocast(
        device_type=stages.current_platform.device_type,
        dtype=vocoder.hift_autocast_dtype,
        enabled=vocoder.hift_autocast_dtype is not None,
    ):
        assert not torch.is_autocast_enabled()
