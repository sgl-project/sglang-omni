# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import sys
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest
import torch

from sglang_omni.client.client import Client
from sglang_omni.models.fun_cosyvoice3 import stages
from sglang_omni.models.fun_cosyvoice3.config import FunCosyVoice3PipelineConfig
from sglang_omni.models.fun_cosyvoice3.payload_types import FunCosyVoice3State
from sglang_omni.models.fun_cosyvoice3.streaming_vocoder import (
    FunCosyVoice3StreamingVocoderScheduler,
)
from sglang_omni.pipeline.stage.stream_queue import StreamItem
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.messages import IncomingMessage


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

    scheduler = stages._FunCosyVoice3MlxStreamingVocoderScheduler(
        _FakeMlxVocoder(), max_batch_wait_ms=0
    )
    state = FunCosyVoice3State(
        stream=True,
        flow_prompt_speech_token=torch.tensor([[1, 2]], dtype=torch.int32),
        flow_prompt_speech_feat=torch.ones(1, 2, 80),
        flow_embedding=torch.ones(1, 192),
    )
    payload = _payload(state)
    scheduler._stream_payloads["req"] = payload
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


def _payload(state: FunCosyVoice3State) -> StagePayload:
    return StagePayload(
        request_id="req-vocoder",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )


def test_cosyvoice3_vocoder_does_not_pad_or_rescale_short_sequences() -> None:
    flow = _FakeFlow()
    hift = _FakeHiFT()
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

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
    vocoder = stages._CosyVoice3Vocoder(_FakeFlow(), _FakeHiFT())

    with pytest.raises(RuntimeError, match="no usable speech tokens"):
        vocoder.token2wav(
            token=torch.zeros(1, 0, dtype=torch.long),
            prompt_token=torch.tensor([[4]], dtype=torch.int32),
            prompt_feat=torch.zeros(1, 2, 80),
            embedding=torch.ones(1, 192),
        )


def test_cosyvoice3_token2wav_chunk_slices_mel_and_hift_delta() -> None:
    flow = _FakeFlow()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
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


def test_mlx_vocoder_audio_payload_survives_state_storage() -> None:
    state = FunCosyVoice3State(
        text="hello",
        audio_codes=torch.tensor([[1], [2]]),
        audio_samples=[9.0],
        prompt_tokens=3,
        completion_tokens=2,
    )
    waveform = np.array([[0.1, -0.2]], dtype=np.float32)

    mlx_vocoder = object.__new__(stages._CosyVoice3MlxVocoderAdapter)
    stored = mlx_vocoder.store_result(_payload(state), state, waveform, 24000)
    result = Client._default_result_builder(stored.request_id, stored.data)

    np.testing.assert_array_equal(result.audio_data, waveform.reshape(-1))
    assert result.sample_rate == 24000
    assert result.modality == "audio"
    assert result.usage.total_tokens == 5
    assert "audio_codes" not in stored.data
    assert "audio_samples" not in stored.data


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

    assert observed == [("cpu", torch.float16, True), ("mps", None, False)]


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
    # HiFT runs once over the padded batch rather than once per request.
    assert len(hift.calls) == 1
    assert hift.calls[0][0].shape[0] == 2


def test_decode_batch_runs_hift_once_over_padded_mels(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    hift = _FakeHiFT()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages._CosyVoice3Vocoder(flow, hift)

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
    vocoder = stages._CosyVoice3Vocoder(flow, hift, hift_max_padding_waste=1.0)

    asyncio.run(vocoder.decode_batch([(_state(), _codes(2)), (_state(), _codes(3))]))

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

    with pytest.raises(RuntimeError, match="PyTorch module or a TensorRT wrapper"):
        stages._CosyVoice3Vocoder(flow, _FakeHiFT())


def test_vocoder_accepts_tensorrt_flow_estimator() -> None:
    class _FakeTRTEstimator:
        def acquire_estimator(self):
            return [None, None], None

        def execute(self, *args, **kwargs):
            del args, kwargs
            raise AssertionError("vocoder init must not run the estimator")

    flow = _BatchCapableFakeFlow()
    flow.decoder.estimator = _FakeTRTEstimator()
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())
    assert vocoder._flow is not None


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
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **kwargs: (
            _BatchCapableFakeFlow(),
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

    assert scheduler._max_batch_cost == 2000
    assert scheduler._collect_new_request_batch(first) == [first]
    assert scheduler._next_message() == second


def test_create_vocoder_executor_defaults_batch_for_real_lengths(monkeypatch) -> None:
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **kwargs: (
            _BatchCapableFakeFlow(),
            _FakeHiFT(),
        ),
    )
    scheduler = stages.create_vocoder_executor("model", device="cpu")

    assert scheduler._max_batch_cost == stages._DEFAULT_FLOW_BATCH_ADMISSION_FRAMES
    assert (
        scheduler._max_batch_cost // 713 >= 8
    ), "default admission budget no longer holds a useful batch"
    assert scheduler._max_batch_size == 16
    assert scheduler._max_batch_wait_s == pytest.approx(0.03)


def test_create_vocoder_executor_threads_batch_configuration(monkeypatch) -> None:
    captured: dict[str, object] = {}

    fake_flow = _BatchCapableFakeFlow()
    fake_hift = _FakeHiFT()
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")

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

    assert isinstance(scheduler, FunCosyVoice3StreamingVocoderScheduler)
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
        "enable_flow_estimator_trt": False,
    }


def test_create_vocoder_executor_threads_trt_flag(monkeypatch) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        stages, "resolve_concrete_device", lambda device, gpu_id: torch.device("cpu")
    )
    monkeypatch.setattr(stages, "resolve_checkpoint", lambda model_path: "/checkpoint")

    def fake_load(checkpoint_dir, device, fp16, **kwargs):
        captured.update(
            {
                "enable_flow_estimator_trt": kwargs.get("enable_flow_estimator_trt"),
            }
        )
        return _BatchCapableFakeFlow(), _FakeHiFT()

    monkeypatch.setattr(stages, "_load_cosyvoice3_flow_hift", fake_load)

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
    monkeypatch.setattr(
        stages,
        "_load_cosyvoice3_flow_hift",
        lambda checkpoint_dir, device, fp16, **_: (
            _BatchCapableFakeFlow(),
            _FakeHiFT(),
        ),
    )
    compiled: list[object] = []
    monkeypatch.setattr(
        stages,
        "_compile_dit_backbone",
        lambda flow, compute_dtype: compiled.append(flow),
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
        stages._attach_flow_estimator_trt(object(), "/checkpoint", "cuda:0")


@pytest.mark.parametrize("device", ["cpu", "npu:0", "xpu:0"])
def test_attach_flow_estimator_trt_rejects_non_cuda_device(
    monkeypatch, device: str
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    with pytest.raises(RuntimeError, match="CUDA vocoder device"):
        stages._attach_flow_estimator_trt(object(), "/checkpoint", device)


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
    # note (PoTaTo-Mika) : _attach_flow_estimator_trt also gates on current_platform.is_cuda(),
    # which stays False on CPU-only hosts (CI hides CUDA), not just on
    # torch.cuda.is_available().
    monkeypatch.setattr(stages.current_platform, "is_cuda", lambda: True)
    import sglang_omni.models.fun_cosyvoice3.flow_estimator_trt as trt_mod

    monkeypatch.setattr(trt_mod, "resolve_flow_estimator_onnx", fake_resolve)
    monkeypatch.setattr(trt_mod, "build_flow_estimator_trt", fake_build)

    stages._attach_flow_estimator_trt(flow, "/checkpoint", "cuda:0")

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
    builder.setup_model(
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


def test_pipeline_config_sets_flow_batch_bucket_by_default() -> None:
    vocoder_stage = next(
        stage
        for stage in FunCosyVoice3PipelineConfig(model_path="model").stages
        if stage.name == "vocoder"
    )

    assert vocoder_stage.factory.model_dump(exclude_none=True) == {
        "flow_batch_bucket_frames": 50,
        "flow_batch_admission_frames": 8000,
        "max_batch_wait_ms": 30,
        "enable_flow_estimator_trt": False,
        "token_hop_len": 25,
        "token_max_hop_len": 100,
        "disable_hop_growth": False,
    }


def test_vocoder_hift_defaults_to_float32(monkeypatch) -> None:
    flow = _BatchCapableFakeFlow()
    _install_fake_batch_adapter(monkeypatch, [])
    vocoder = stages._CosyVoice3Vocoder(flow, _FakeHiFT())

    # bfloat16 gave HiFT no speedup, so the default keeps full precision.
    assert vocoder._hift_compute_dtype is None
    with vocoder._hift_autocast():
        assert not torch.is_autocast_enabled()
