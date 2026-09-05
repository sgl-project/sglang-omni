# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


class _FakeHiFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.calls = []

    def inference(self, *, speech_feat, finalize):
        self.calls.append((speech_feat, finalize))
        return torch.arange(speech_feat.shape[-1]).reshape(1, -1).float(), None


class _FakeEstimator(torch.nn.Module):
    def forward(self, *args, **kwargs):
        del args, kwargs
        raise AssertionError("batch adapter should be mocked in vocoder unit tests")


class _BatchCapableFakeFlow(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.output_size = 80
        self.token_mel_ratio = 2
        self.input_embedding = torch.nn.Embedding(32, 80)
        self.spk_embed_affine_layer = torch.nn.Linear(192, 80)
        self.pre_lookahead_layer = torch.nn.Identity()
        self.decoder = SimpleNamespace(
            rand_noise=torch.zeros(1, 80, 1000),
            t_scheduler="cosine",
            inference_cfg_rate=0.7,
            estimator=_FakeEstimator(),
            forward_estimator=lambda *args, **kwargs: None,
        )


def _payload(state: FunCosyVoice3State) -> StagePayload:
    return StagePayload(
        request_id="req-vocoder",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def test_cosyvoice3_vocoder_prepare_and_store_audio_payload() -> None:
    vocoder = stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
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
    vocoder = stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    payload = _payload(FunCosyVoice3State(text="hello"))

    with pytest.raises(RuntimeError, match="requires audio_codes"):
        vocoder.prepare_item(payload)


def test_cosyvoice3_vocoder_rejects_missing_audio_output() -> None:
    vocoder = stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT())
    state = FunCosyVoice3State(text="hello")
    payload = _payload(state)

    with pytest.raises(RuntimeError, match="did not return audio"):
        vocoder.store_result(payload, state, None, 24000)


def test_cosyvoice3_vocoder_decode_batch_uses_state_conditioning(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
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


def _run_decode_with_coalescing(
    monkeypatch,
    items,
    *,
    span_frames: int,
    max_added_padding_pct: float,
    bucket_frames: int = 50,
):
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(
        flow,
        hift,
        flow_batch_bucket_frames=bucket_frames,
        flow_batch_coalesce_span_frames=span_frames,
        flow_batch_coalesce_max_added_padding_pct=max_added_padding_pct,
    )
    results = asyncio.run(vocoder.decode_batch(items))
    return results, batch_calls, hift


def test_decode_batch_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    results = asyncio.run(vocoder.decode_batch([(_state(), _codes(2))]))

    assert len(results) == 1
    assert [len(call) for call in batch_calls] == [1]
    assert len(hift.calls) == 1


def test_decode_payload_size_one_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    state = _state()
    state.audio_codes = _codes(2)

    result = asyncio.run(vocoder.decode_payload(_payload(state)))

    assert result.data["modality"] == "audio"
    assert [len(call) for call in batch_calls] == [1]


def test_decode_batch_singleton_buckets_use_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(26))]))

    assert [len(call) for call in batch_calls] == [1, 1]
    assert len(hift.calls) == 2


def test_decode_batch_same_bucket_batches_flow_once(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))]))

    assert len(batch_calls) == 1
    assert len(batch_calls[0]) == 2
    assert len(hift.calls) == 2


def test_decode_batch_long_singleton_uses_batch_adapter(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    asyncio.run(vocoder.decode_batch([(_state(prompt_tokens=0), _codes(2200, 1))]))

    assert [len(call) for call in batch_calls] == [1]
    assert batch_calls[0][0].token.shape[1] == 2200


def test_decode_batch_different_buckets_do_not_share_padding(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    items = [
        (_state(), _codes(9)),
        (_state(), _codes(10)),
        (_state(), _codes(25)),
        (_state(), _codes(26)),
    ]

    asyncio.run(vocoder.decode_batch(items))

    assert [len(call) for call in batch_calls] == [2, 2]
    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [9, 10],
        [25, 26],
    ]


def test_flow_coalescing_disabled_preserves_current_bucket_groups(monkeypatch) -> None:
    items = [
        (_state(sample_rate=16001, prompt_tokens=0), _codes(27)),
        (_state(sample_rate=16002, prompt_tokens=0), _codes(24)),
        (_state(sample_rate=16003, prompt_tokens=0), _codes(25)),
    ]

    results, batch_calls, _ = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=0,
        max_added_padding_pct=0,
    )

    assert [sample_rate for _, sample_rate in results] == [16001, 16002, 16003]
    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [27],
        [24, 25],
    ]


def test_flow_coalescing_merges_and_restores_result_order(monkeypatch) -> None:
    items = [
        (_state(sample_rate=16001, prompt_tokens=0), _codes(27)),
        (_state(sample_rate=16002, prompt_tokens=0), _codes(25)),
    ]

    results, batch_calls, hift = _run_decode_with_coalescing(
        monkeypatch,
        items,
        span_frames=64,
        max_added_padding_pct=5,
    )

    assert [sample_rate for _, sample_rate in results] == [16001, 16002]
    assert [[item.token.shape[1] for item in call] for call in batch_calls] == [
        [25, 27]
    ]
    assert len(hift.calls) == 2


def _make_flow_buckets(
    total_mel_frames: list[int], *, bucket_frames: int = 50
) -> dict[int, list[stages._PreparedFlowRequest]]:
    buckets: dict[int, list[stages._PreparedFlowRequest]] = {}
    for index, total in enumerate(total_mel_frames):
        bucket_key = (total + bucket_frames - 1) // bucket_frames
        buckets.setdefault(bucket_key, []).append(
            stages._PreparedFlowRequest(
                index=index,
                sample_rate=24000,
                flow_input=stages.FlowBatchInput(
                    token=torch.empty((1, 0), dtype=torch.int32),
                    prompt_token=torch.empty((1, 0), dtype=torch.int32),
                    prompt_feat=torch.empty((1, 0, 80)),
                    embedding=torch.empty((1, 192)),
                ),
                total_mel_frames=total,
            )
        )
    return buckets


@pytest.mark.parametrize(
    ("total_mel_frames", "bucket_frames", "span_frames", "padding_pct", "expected"),
    [
        pytest.param(
            [48, 50, 54],
            50,
            1,
            5,
            [[48, 50], [54]],
            id="atomic-baseline-buckets",
        ),
        pytest.param(
            [104, 50, 54],
            50,
            64,
            30,
            [[50, 54], [104]],
            id="sort-before-coarsening",
        ),
        pytest.param(
            [50, 94],
            50,
            40,
            100,
            [[50], [94]],
            id="span-limit",
        ),
        pytest.param(
            [50, 54],
            50,
            64,
            0,
            [[50], [54]],
            id="positive-span-zero-padding",
        ),
        pytest.param(
            [10, 13, 30, 33],
            10,
            4,
            5,
            [[10], [13], [30, 33]],
            id="whole-outer-padding-cap",
        ),
        pytest.param(
            [50, 54, 104],
            50,
            64,
            100,
            [[50, 54, 104]],
            id="fewest-solves",
        ),
        pytest.param(
            [50, 54, 104],
            50,
            64,
            30,
            [[50, 54], [104]],
            id="least-padded-work",
        ),
        pytest.param(
            [10, 10, 20, 40],
            10,
            40,
            30,
            [[10, 10, 20], [40]],
            id="smallest-maximum-merge-span",
        ),
        pytest.param(
            [10, 20, 30],
            10,
            20,
            40,
            [[10], [20, 30]],
            id="bucket-range-signature",
        ),
    ],
)
def test_flow_coalescing_policy_directly(
    total_mel_frames: list[int],
    bucket_frames: int,
    span_frames: int,
    padding_pct: float,
    expected: list[list[int]],
) -> None:
    groups = stages._group_flow_requests(
        _make_flow_buckets(total_mel_frames, bucket_frames=bucket_frames),
        coalesce_span_frames=span_frames,
        coalesce_max_added_padding_pct=padding_pct,
    )

    assert [
        [request.total_mel_frames for request in group] for group in groups
    ] == expected


def test_decode_batch_preserves_input_order_across_buckets(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    batch_calls: list[list] = []
    _install_fake_batch_adapter(monkeypatch, batch_calls)
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    items = [
        (_state(sample_rate=16001), _codes(9, 1)),
        (_state(sample_rate=16002), _codes(25, 2)),
        (_state(sample_rate=16003), _codes(10, 3)),
        (_state(sample_rate=16004), _codes(26, 4)),
    ]

    results = asyncio.run(vocoder.decode_batch(items))

    assert [sample_rate for _, sample_rate in results] == [16001, 16002, 16003, 16004]
    assert [len(call) for call in batch_calls] == [2, 2]


def test_vocoder_rejects_non_pytorch_flow_estimator() -> None:
    flow = _BatchCapableFakeFlow()
    flow.decoder.estimator = object()

    with pytest.raises(RuntimeError, match="PyTorch Flow estimator"):
        stages._CosyVoice3Vocoder(flow, _FakeHiFT())


def test_decode_batch_alignment_mismatch_fails() -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

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
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    invalid = _state()
    invalid.flow_embedding = torch.ones(1, 191)

    with pytest.raises(ValueError, match="embedding width"):
        asyncio.run(vocoder.decode_batch([(invalid, _codes(2)), (_state(), _codes(3))]))


def test_decode_batch_does_not_retry_after_batch_failure(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    def fail_batch(flow, inputs):
        del flow, inputs
        raise RuntimeError("batch estimator failed")

    monkeypatch.setattr(stages.FunCosyVoice3Flow, "inference", fail_batch)

    with pytest.raises(RuntimeError, match="batch estimator failed"):
        asyncio.run(
            vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))])
        )


def test_vocoder_rejects_non_positive_flow_bucket_size() -> None:
    with pytest.raises(ValueError, match="flow_batch_bucket_frames"):
        stages._CosyVoice3Vocoder(
            _BatchCapableFakeFlow(), _FakeHiFT(), flow_batch_bucket_frames=0
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        (
            {"flow_batch_coalesce_span_frames": -1},
            "flow_batch_coalesce_span_frames",
        ),
        (
            {"flow_batch_coalesce_max_added_padding_pct": -1},
            "flow_batch_coalesce_max_added_padding_pct",
        ),
        (
            {"flow_batch_coalesce_max_added_padding_pct": float("nan")},
            "flow_batch_coalesce_max_added_padding_pct",
        ),
        (
            {"flow_batch_coalesce_max_added_padding_pct": float("inf")},
            "flow_batch_coalesce_max_added_padding_pct",
        ),
        (
            {
                "flow_batch_coalesce_span_frames": 0,
                "flow_batch_coalesce_max_added_padding_pct": 1,
            },
            "flow_batch_coalesce_max_added_padding_pct",
        ),
    ],
)
def test_vocoder_rejects_invalid_flow_coalescing_configuration(
    kwargs, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        stages._CosyVoice3Vocoder(_BatchCapableFakeFlow(), _FakeHiFT(), **kwargs)


def test_flow_scheduler_cost_rounds_to_bucket() -> None:
    vocoder = stages._CosyVoice3Vocoder(
        _BatchCapableFakeFlow(), _FakeHiFT(), flow_batch_bucket_frames=50
    )
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)

    assert vocoder._flow_scheduler_cost(_payload(state)) == 50


def test_flow_admission_defers_request_after_long_singleton(monkeypatch) -> None:
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16: (_BatchCapableFakeFlow(), _FakeHiFT()),
    )
    scheduler = stages.create_vocoder_executor("model", device="cpu")
    long_state = _state(prompt_tokens=0)
    long_state.audio_codes = _codes(2200)
    short_state = _state(prompt_tokens=0)
    short_state.audio_codes = _codes(2)
    first = IncomingMessage("long", "new_request", _payload(long_state))
    second = IncomingMessage("short", "new_request", _payload(short_state))
    scheduler.inbox.put(second)

    assert scheduler._max_batch_cost == stages._DEFAULT_FLOW_BATCH_ADMISSION_FRAMES
    assert scheduler._collect_batch(first) == [first]
    assert scheduler._next_message() == second


def test_create_vocoder_executor_threads_batch_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_flow = _BatchCapableFakeFlow()
    fake_hift = _FakeHiFT()
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")

    def fake_load(checkpoint_dir, device, fp16):
        captured.update(
            {
                "checkpoint_dir": checkpoint_dir,
                "device": device,
                "fp16": fp16,
            }
        )
        return fake_flow, fake_hift

    monkeypatch.setattr(stages, "_load_cosyvoice3_flow_hift", fake_load)

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        dtype="float16",
        max_batch_size=6,
        max_batch_wait_ms=7,
        flow_batch_bucket_frames=100,
        flow_batch_admission_frames=200,
        flow_batch_coalesce_span_frames=40,
        flow_batch_coalesce_max_added_padding_pct=3,
    )

    assert isinstance(scheduler, stages.SimpleScheduler)
    assert scheduler._max_batch_size == 6
    assert scheduler._max_batch_wait_s == pytest.approx(0.007)
    assert scheduler._max_batch_cost == 200
    assert callable(scheduler._request_cost_fn)
    vocoder = scheduler._request_cost_fn.__self__
    assert vocoder._flow_batch_coalesce_span_frames == 40
    assert vocoder._flow_batch_coalesce_max_added_padding_pct == 3
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)
    assert scheduler._request_cost_fn(_payload(state)) == 100
    assert captured == {
        "checkpoint_dir": "/checkpoint",
        "device": "cpu",
        "fp16": True,
    }


def test_create_vocoder_executor_rejects_large_batch_for_coalescing() -> None:
    with pytest.raises(ValueError, match=r"max_batch_size.*8"):
        stages.create_vocoder_executor(
            "model-that-must-not-load",
            max_batch_size=9,
            flow_batch_coalesce_span_frames=64,
            flow_batch_coalesce_max_added_padding_pct=5,
        )


def test_create_vocoder_executor_allows_large_batch_when_coalescing_disabled(
    monkeypatch,
) -> None:
    fake_flow = _BatchCapableFakeFlow()
    fake_hift = _FakeHiFT()
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16: (fake_flow, fake_hift),
    )

    scheduler = stages.create_vocoder_executor(
        "model",
        device="cpu",
        max_batch_size=16,
        flow_batch_coalesce_span_frames=0,
        flow_batch_coalesce_max_added_padding_pct=0,
    )

    assert scheduler._max_batch_size == 16


def test_create_vocoder_executor_rejects_non_positive_admission_budget(
    monkeypatch,
) -> None:
    monkeypatch.setattr(stages, "resolve_device_spec", lambda device, gpu_id: "cpu")

    with pytest.raises(ValueError, match="flow_batch_admission_frames"):
        stages.create_vocoder_executor(
            "model",
            device="cpu",
            flow_batch_admission_frames=0,
        )


def test_pipeline_config_sets_flow_batch_bucket_by_default() -> None:
    vocoder_stage = next(
        stage
        for stage in FunCosyVoice3PipelineConfig(model_path="model").stages
        if stage.name == "vocoder"
    )

    assert vocoder_stage.factory.model_dump(exclude_none=True) == {
        "dtype": "bfloat16",
        "flow_batch_bucket_frames": 50,
        "flow_batch_admission_frames": 2000,
        "flow_batch_coalesce_span_frames": 0,
        "flow_batch_coalesce_max_added_padding_pct": 0.0,
        "enable_dit_torch_compile": False,
    }
