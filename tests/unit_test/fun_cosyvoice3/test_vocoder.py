# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import numpy as np
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


class _FakeMlxCore:
    gpu = object()
    int32 = np.int32
    float32 = np.float32

    def __init__(self) -> None:
        self.thread_local_streams = []
        self.entered_streams = []
        self.evaluated = []

    def new_thread_local_stream(self, device):
        stream = object()
        self.thread_local_streams.append((device, stream))
        return stream

    def stream(self, stream):
        self.entered_streams.append(stream)
        return nullcontext()

    @staticmethod
    def array(value, *, dtype):
        return np.asarray(value, dtype=dtype)

    def eval(self, value):
        self.evaluated.append(value)


class _FakeMlxVocoder:
    sample_rate = 24000
    token_mel_ratio = 2

    def __init__(self) -> None:
        self.calls = []
        self.output = np.asarray([0.1, 0.2, 0.3], dtype=np.float32)

    def decode_mx(self, **kwargs):
        self.calls.append(kwargs)
        return self.output


def test_mps_hift_adapter_moves_f0_to_cpu_before_float64() -> None:
    calls = []

    class _Predictor:
        def to(self, *args, **kwargs):
            calls.append((args, kwargs))
            return self

    hift = SimpleNamespace(f0_predictor=_Predictor())

    stages._MpsHiFTAdapter(hift, "mps")

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

    loaded_flow, loaded_hift = stages._load_cosyvoice3_flow_hift_lightweight(
        str(tmp_path),
        device="cpu",
    )

    assert isinstance(loaded_flow, stages.FunCosyVoice3Flow)
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


def test_mlx_artifact_resolution_is_revision_aware_and_gated(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.hardware_backend.mlx import remote_code_gate

    observed = {}
    monkeypatch.setattr(
        remote_code_gate,
        "resolve_model_directory",
        lambda model_path, revision: observed.update(
            model_path=model_path,
            revision=revision,
        )
        or Path("/resolved/mlx-cosyvoice3"),
    )
    monkeypatch.setattr(
        remote_code_gate,
        "ensure_remote_code_allowed",
        lambda model_dir, trust_remote_code: observed.update(
            gated_model_dir=model_dir,
            trust_remote_code=trust_remote_code,
        ),
    )

    resolved = stages._resolve_cosyvoice3_mlx_artifact(
        "mlx-org/cosyvoice3",
        revision="revision-a",
    )

    assert resolved == "/resolved/mlx-cosyvoice3"
    assert observed == {
        "model_path": "mlx-org/cosyvoice3",
        "revision": "revision-a",
        "gated_model_dir": Path("/resolved/mlx-cosyvoice3"),
        "trust_remote_code": False,
    }


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


def test_mlx_vocoder_adapter_decodes_state_on_thread_local_stream(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mx = _FakeMlxCore()
    backend = _FakeMlxVocoder()
    monkeypatch.setattr(stages, "_get_mlx_core", lambda: mx)
    vocoder = stages._CosyVoice3MlxVocoderAdapter(backend)
    state = _state(prompt_tokens=2)

    results = asyncio.run(vocoder.decode_batch([(state, _codes(3))]))

    assert len(results) == 1
    wav, sample_rate = results[0]
    assert sample_rate == 24000
    assert np.array_equal(wav, np.asarray([0.1, 0.2, 0.3], dtype=np.float32))
    assert len(mx.thread_local_streams) == 1
    assert mx.entered_streams == [mx.thread_local_streams[0][1]]
    assert mx.evaluated[0] is backend.output
    call = backend.calls[0]
    assert call["token"].shape == (1, 3)
    assert call["token"].dtype == np.int32
    assert call["prompt_token"].shape == (1, 2)
    assert call["prompt_token"].dtype == np.int32
    assert call["prompt_feat"].shape == (1, 4, 80)
    assert call["prompt_feat"].dtype == np.float32
    assert call["embedding"].shape == (1, 192)
    assert call["embedding"].dtype == np.float32


def test_mlx_vocoder_adapter_rejects_non_singleton_batch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(stages, "_get_mlx_core", _FakeMlxCore)
    vocoder = stages._CosyVoice3MlxVocoderAdapter(_FakeMlxVocoder())

    with pytest.raises(RuntimeError, match="exactly one request"):
        asyncio.run(
            vocoder.decode_batch(
                [
                    (_state(), _codes(2)),
                    (_state(), _codes(3)),
                ]
            )
        )


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
    vocoder = stages._CosyVoice3Vocoder(
        _BatchCapableFakeFlow(),
        _FakeHiFT(),
        compute_dtype=torch.float16,
    )

    asyncio.run(vocoder.decode_batch([(_state(), torch.tensor([1, 2]))]))

    assert observed == [("cpu", torch.float16, True)]


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
    assert scheduler._max_batch_size == 8
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
    )

    assert isinstance(scheduler, stages.SimpleScheduler)
    assert scheduler._max_batch_size == 6
    assert scheduler._max_batch_wait_s == pytest.approx(0.007)
    assert scheduler._max_batch_cost == 200
    assert callable(scheduler._request_cost_fn)
    state = _state(prompt_tokens=1)
    state.audio_codes = _codes(2)
    assert scheduler._request_cost_fn(_payload(state)) == 100
    assert captured == {
        "checkpoint_dir": "/checkpoint",
        "device": "cpu",
        "fp16": True,
    }


def test_create_vocoder_executor_selects_native_mlx_and_forwards_revision(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from sglang.srt.utils import tensor_bridge

    observed = {}
    backend = _FakeMlxVocoder()
    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    monkeypatch.setattr(stages.current_platform, "is_mps", lambda: True)
    monkeypatch.setattr(stages, "_get_mlx_core", _FakeMlxCore)
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_mlx_vocoder",
        lambda model_path, revision, expected_dtype: observed.update(
            model_path=model_path,
            revision=revision,
            expected_dtype=expected_dtype,
        )
        or backend,
    )
    monkeypatch.setattr(
        stages,
        "resolve_checkpoint",
        lambda model_path: pytest.fail(
            f"MLX vocoder must not resolve the official checkpoint: {model_path}"
        ),
    )

    scheduler = stages.create_vocoder_executor(
        "official-model",
        mlx_model_path="mlx-org/cosyvoice3",
        mlx_model_revision="revision-a",
    )

    assert observed == {
        "model_path": "mlx-org/cosyvoice3",
        "revision": "revision-a",
        "expected_dtype": None,
    }
    assert scheduler._max_batch_size == 1
    assert scheduler._max_batch_cost is None
    assert scheduler._request_cost_fn is None


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({}, "mlx_model_path"),
        (
            {"mlx_model_path": "mlx-org/cosyvoice3", "max_batch_size": 2},
            "max_batch_size=1",
        ),
        (
            {
                "mlx_model_path": "mlx-org/cosyvoice3",
                "enable_dit_torch_compile": True,
            },
            "unavailable",
        ),
        (
            {
                "mlx_model_path": "mlx-org/cosyvoice3",
                "device": "cpu",
            },
            "requires an MPS device",
        ),
    ],
)
def test_create_native_mlx_vocoder_rejects_incompatible_configuration(
    monkeypatch: pytest.MonkeyPatch,
    kwargs: dict[str, object],
    message: str,
) -> None:
    from sglang.srt.utils import tensor_bridge

    monkeypatch.setattr(tensor_bridge, "use_mlx", lambda: True)
    monkeypatch.setattr(stages.current_platform, "is_mps", lambda: True)

    with pytest.raises(ValueError, match=message):
        stages.create_vocoder_executor("official-model", **kwargs)


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
        "flow_batch_bucket_frames": 50,
        "flow_batch_admission_frames": 2000,
        "enable_dit_torch_compile": False,
    }
