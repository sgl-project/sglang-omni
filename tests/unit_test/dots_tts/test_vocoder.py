# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import asyncio
import copy
import importlib.util
import threading
from types import SimpleNamespace

import pytest
import torch

from sglang_omni.config.runtime import resolve_stage_typed_kwargs
from sglang_omni.models.dots_tts import alias_free
from sglang_omni.models.dots_tts.alias_free import FusedAliasFree
from sglang_omni.models.dots_tts.codec import DotsAudioCodec
from sglang_omni.models.dots_tts.compat import import_dots_tts
from sglang_omni.models.dots_tts.config import DotsTTSPipelineConfig
from sglang_omni.models.dots_tts.payload_types import DotsTTSState
from sglang_omni.models.dots_tts.vocoder import (
    DotsTTSBatchVocoder,
    DotsTTSStreamingVocoder,
)
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.message import IncomingMessage

try:
    from sglang_omni.models.dots_tts import stages
except ImportError:
    stages = None


class FakeInference:
    def __init__(self, hop_size: int) -> None:
        self.hop_size = hop_size
        self.inputs: list[torch.Tensor] = []
        self.input_data_ptrs: list[int] = []

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        self.input_data_ptrs.append(latents.data_ptr())
        self.inputs.append(latents.clone())
        rows = []
        for row in latents:
            samples = row[:, 0].repeat_interleave(self.hop_size)
            rows.append(samples.unsqueeze(0))
        return torch.stack(rows)


def make_codec(*, latent_dim: int = 3, hop_size: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        device=torch.device("cpu"),
        latent_dim=latent_dim,
        hop_size=hop_size,
        sample_rate=48000,
        patch_size=4,
        lock=threading.RLock(),
        inference=FakeInference(hop_size),
    )


def make_latents(frames: int, value: float, *, latent_dim: int = 3) -> torch.Tensor:
    return torch.full((1, frames, latent_dim), value)


def decode(
    vocoder: DotsTTSBatchVocoder, latents: list[torch.Tensor]
) -> list[tuple[torch.Tensor, int]]:
    items = [(DotsTTSState(), item) for item in latents]
    return asyncio.run(vocoder.decode_batch(items))


def payload(
    request_id: str, frames: int, value: float, *, stream: bool = False
) -> StagePayload:
    state = DotsTTSState(generated_latents=make_latents(frames, value))
    return StagePayload(
        request_id=request_id,
        request=OmniRequest(inputs="hello", params={"stream": stream}),
        data=state.to_dict(),
    )


def test_equal_length_inputs_use_one_audiovae_forward() -> None:
    codec = make_codec()
    vocoder = DotsTTSBatchVocoder(codec)
    outputs = decode(
        vocoder,
        [make_latents(16, 1), make_latents(16, 2), make_latents(16, 3)],
    )

    assert vocoder.logged_batch
    assert len(codec.inference.inputs) == 1
    assert codec.inference.inputs[0].shape == (3, 16, 3)
    assert [waveform.shape for waveform, _ in outputs] == [(1, 1, 32)] * 3


def test_mixed_length_bucket_pads_and_crops_each_output() -> None:
    codec = make_codec()
    outputs = decode(
        DotsTTSBatchVocoder(codec),
        [make_latents(17, 1), make_latents(31, 2)],
    )

    [padded] = codec.inference.inputs
    assert padded.shape == (2, 31, 3)
    assert torch.count_nonzero(padded[0, 17:]) == 0
    assert [waveform.shape[-1] for waveform, _ in outputs] == [34, 62]
    assert torch.all(outputs[0][0] == 1)
    assert torch.all(outputs[1][0] == 2)


def test_multiple_buckets_restore_original_request_order() -> None:
    codec = make_codec()
    outputs = decode(
        DotsTTSBatchVocoder(codec),
        [
            make_latents(33, 1),
            make_latents(16, 2),
            make_latents(40, 3),
            make_latents(8, 4),
        ],
    )

    assert [batch.shape[0] for batch in codec.inference.inputs] == [2, 2]
    assert [waveform[0, 0, 0].item() for waveform, _ in outputs] == [1, 2, 3, 4]


def test_single_input_preserves_batch_and_waveform_shapes() -> None:
    codec = make_codec()
    vocoder = DotsTTSBatchVocoder(codec)
    latents = make_latents(12, 5)
    [output] = decode(vocoder, [latents])

    assert not vocoder.logged_batch
    assert codec.inference.input_data_ptrs == [latents.data_ptr()]
    assert codec.inference.inputs[0].shape == (1, 12, 3)
    assert output[0].shape == (1, 1, 24)
    assert output[1] == 48000


@pytest.mark.parametrize(
    ("latents", "message"),
    [
        (torch.zeros(4, 3), "shape"),
        (torch.zeros(2, 4, 3), "shape"),
        (torch.zeros(1, 0, 3), "at least one frame"),
        (torch.zeros(1, 4, 2), "latent_dim"),
    ],
)
def test_invalid_latents_are_rejected(latents: torch.Tensor, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        decode(DotsTTSBatchVocoder(make_codec()), [latents])


def test_streaming_vocoder_enables_payload_and_chunk_batching() -> None:
    codec = make_codec()
    scheduler = DotsTTSStreamingVocoder(
        codec,
        optimize=False,
    )

    assert scheduler.batch_fn is not None
    assert scheduler.max_batch_size == 4
    assert scheduler.stream_chunk_batch_max == 4
    assert scheduler.max_batch_wait_s == 0.002
    assert scheduler.can_batch_stream_chunks
    results = asyncio.run(
        scheduler.batch_fn([payload("a", 16, 1), payload("b", 16, 2)])
    )
    assert len(codec.inference.inputs) == 1
    assert [result.request_id for result in results] == ["a", "b"]
    assert scheduler.is_streaming_payload(payload("stream", 16, 1, stream=True))


def test_non_streaming_batch_isolates_invalid_payload() -> None:
    codec = make_codec()
    scheduler = DotsTTSStreamingVocoder(codec, optimize=False)
    invalid = payload("bad", 16, 2)
    invalid.data["generated_latents"] = torch.zeros(2, 4, 3)

    scheduler.handle_new_request_batch(
        [
            IncomingMessage("good-a", "new_request", payload("good-a", 16, 1)),
            IncomingMessage("bad", "new_request", invalid),
            IncomingMessage("good-b", "new_request", payload("good-b", 16, 3)),
        ]
    )

    outputs = [scheduler.outbox.get_nowait() for _ in range(3)]
    by_request = {output.request_id: output for output in outputs}
    assert by_request["bad"].type == "error"
    assert isinstance(by_request["bad"].data, ValueError)
    assert by_request["good-a"].type == "result"
    assert by_request["good-b"].type == "result"
    assert [item.shape for item in codec.inference.inputs] == [(2, 16, 3)]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_batch_size": 0}, "max_batch_size"),
        ({"max_batch_wait_ms": -1}, "max_batch_wait_ms"),
        ({"stream_slots": 0}, "stream_slots"),
    ],
)
def test_invalid_batch_config_is_rejected(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DotsTTSStreamingVocoder(make_codec(), optimize=False, **kwargs)


class TestVocoderFactorySignature:
    """The vocoder factory declares every kwarg it accepts.

    With a ``**kwargs`` catch-all, a mistyped ``factory.*`` key -- or a
    correctly spelled one the factory never reads -- would be swallowed
    silently; without it, the typed-kwargs check refuses it."""

    def test_an_unknown_factory_key_is_refused(self) -> None:
        if stages is None:
            pytest.skip("Requires the SGLang runtime")
        else:
            pass
        from sglang_omni.config.runtime import apply_typed_stage_kwargs

        with pytest.raises(ValueError, match="stream_slotz"):
            apply_typed_stage_kwargs(
                stages.create_vocoder_executor,
                {},
                {"stream_slotz": 8},
                stage_name="vocoder",
            )

    def test_declared_kwargs_still_pass(self) -> None:
        stages = pytest.importorskip("sglang_omni.models.dots_tts.stages")
        from sglang_omni.config.runtime import apply_typed_stage_kwargs

        out = apply_typed_stage_kwargs(
            stages.create_vocoder_executor,
            {},
            {"stream_slots": 8},
            stage_name="vocoder",
        )
        assert out == {"stream_slots": 8}


class TestAliasFreeFusion:
    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_native_fallback_and_checkpoint_keys(self, dtype: torch.dtype) -> None:
        activation = torch.nn.Module()
        activation.up_ratio = activation.down_ratio = 2
        activation.upsample = torch.nn.ConvTranspose1d(3, 3, 1)
        activation.downsample = torch.nn.Conv1d(3, 3, 1)
        activation.act = torch.nn.Identity()
        activation.act.alpha = torch.nn.Parameter(torch.zeros(3))
        activation.act.beta = torch.nn.Parameter(torch.zeros(3))
        activation.act.alpha_logscale = True
        activation.act.no_div_by_zero = 1e-9
        activation = activation.to(dtype=dtype).eval()
        candidate = FusedAliasFree(activation)
        inputs = torch.randn(2, 3, 11, dtype=dtype)
        expected = activation.downsample(activation.act(activation.upsample(inputs)))
        actual = candidate(inputs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        assert set(candidate.state_dict()) == set(activation.state_dict())
        actual.sum().backward()
        assert candidate.upsample.weight.grad is not None

    @pytest.mark.parametrize("first_enabled", [False, True])
    def test_shared_codec_mode_is_fixed(
        self, monkeypatch: pytest.MonkeyPatch, first_enabled: bool
    ) -> None:
        codec = DotsAudioCodec.__new__(DotsAudioCodec)
        codec.lock = threading.RLock()
        codec.alias_free_fusion_enabled = None
        codec.vocoder = SimpleNamespace(decoder=torch.nn.Identity())
        installed: list[torch.nn.Module] = []
        monkeypatch.setattr(
            "sglang_omni.models.dots_tts.codec.install_alias_free_fusion",
            lambda decoder: installed.append(decoder),
        )
        codec.configure_alias_free_fusion(first_enabled)
        codec.configure_alias_free_fusion(first_enabled)
        assert len(installed) == int(first_enabled)
        with pytest.raises(RuntimeError, match="different.*enable_alias_free_fusion"):
            codec.configure_alias_free_fusion(not first_enabled)
        assert codec.alias_free_fusion_enabled == first_enabled

    @pytest.mark.parametrize(
        ("optimize", "enabled"), [(False, True), (True, False), (True, True)]
    )
    def test_factory_configures_codec_before_pool(
        self, monkeypatch: pytest.MonkeyPatch, optimize: bool, enabled: bool
    ) -> None:
        if stages is None:
            pytest.skip("Requires the SGLang runtime")
        else:
            pass

        events: list[str | bool] = []

        class Codec:
            def configure_alias_free_fusion(self, flag: bool) -> None:
                events.append(flag)

        class Vocoder:
            def __init__(self, codec: Codec, **kwargs) -> None:
                self.merge_steps = 4
                self.stream_slots = 16
                self.stream_chunk_batch_max = 4
                events.append("constructor")

            def ensure_slot_pool(self) -> None:
                events.append("pool")

        monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
        monkeypatch.setattr(
            stages, "load_dots_audio_codec", lambda *args, **kwargs: Codec()
        )
        monkeypatch.setattr(stages, "DotsTTSStreamingVocoder", Vocoder)
        stages.create_vocoder_executor(
            "model", device="cpu", optimize=optimize, enable_alias_free_fusion=enabled
        )
        assert events == [optimize and enabled, "constructor", "pool"]

    def test_fusion_config_is_typed_and_disabled_by_default(self) -> None:
        config = DotsTTSPipelineConfig(model_path="model")
        stage = config.stage_named("vocoder")
        assert resolve_stage_typed_kwargs(stage)["enable_alias_free_fusion"] is False
        stage.factory.enable_alias_free_fusion = True
        assert resolve_stage_typed_kwargs(stage)["enable_alias_free_fusion"] is True


@pytest.fixture
def cuda_alias_free_activation(request: pytest.FixtureRequest) -> torch.nn.Module:
    if (
        not torch.cuda.is_available()
        or alias_free.triton is None
        or torch.version.hip is not None
    ):
        pytest.skip("Requires CUDA and Triton")
    else:
        pass
    if importlib.util.find_spec("dots_tts") is None:
        pytest.skip("Requires dots.tts")
    else:
        pass
    import_dots_tts()
    from dots_tts.modules.vocoder.alias_free_act import Activation1d, SnakeBeta

    activation = (
        Activation1d(
            SnakeBeta(7, alpha_logscale=True), causal=True, fixed_filter=request.param
        )
        .cuda()
        .eval()
    )
    with torch.no_grad():
        activation.act.alpha.uniform_(-2, 2)
        activation.act.beta.uniform_(-2, 2)
        activation.upsample.filter.add_(
            torch.randn_like(activation.upsample.filter) * 0.01
        )
        activation.downsample.lowpass.filter.add_(
            torch.randn_like(activation.downsample.lowpass.filter) * 0.01
        )
    return activation


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_alias_free_activation", [False, True], indirect=True)
@pytest.mark.parametrize("frames", [1, 2, 3, 11, 17, 257])
@pytest.mark.parametrize(
    "input_kind", ["random", "impulse_left", "impulse_right", "noncontiguous"]
)
@torch.inference_mode()
def test_alias_free_cuda_boundaries(
    cuda_alias_free_activation: torch.nn.Module, frames: int, input_kind: str
) -> None:
    activation = cuda_alias_free_activation
    inputs = torch.randn(2, 7, frames, device="cuda")
    if input_kind == "noncontiguous":
        inputs = torch.randn(2, 7, frames * 2, device="cuda")[..., ::2]
    elif input_kind in ("impulse_left", "impulse_right"):
        inputs.zero_()
        inputs[..., 0 if input_kind == "impulse_left" else frames - 1] = 1
    else:
        pass
    expected = activation(inputs)
    candidate = FusedAliasFree(activation)
    torch.testing.assert_close(candidate(inputs), expected, rtol=1e-4, atol=1e-4)
    torch.testing.assert_close(activation(inputs), expected, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_alias_free_activation", [False, True], indirect=True)
@torch.inference_mode()
def test_alias_free_graph_input_replay(
    cuda_alias_free_activation: torch.nn.Module,
) -> None:
    activation = cuda_alias_free_activation
    candidate = FusedAliasFree(activation)
    inputs = torch.randn(2, 7, 17, device="cuda")
    first_inputs = inputs.clone()
    second_inputs = torch.randn_like(inputs)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            candidate(inputs)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = candidate(inputs)
    for replay_inputs in (first_inputs, second_inputs, first_inputs):
        inputs.copy_(replay_inputs)
        graph.replay()
        torch.testing.assert_close(output, activation(inputs), rtol=1e-4, atol=1e-4)


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_alias_free_activation", [False, True], indirect=True)
def test_alias_free_install_is_atomic(
    cuda_alias_free_activation: torch.nn.Module,
) -> None:
    activation = cuda_alias_free_activation
    unsupported = copy.deepcopy(activation)
    unsupported.upsample.pad = 1
    decoder = torch.nn.Sequential(activation, unsupported)
    assert alias_free.install_alias_free_fusion(decoder) == 0
    assert decoder[0] is activation
    assert decoder[1] is unsupported
    decoder = torch.nn.Sequential(activation)
    checkpoint_keys = set(decoder.state_dict())
    assert alias_free.install_alias_free_fusion(decoder) == 1
    assert set(decoder.state_dict()) == checkpoint_keys
    assert decoder[0].upsample.filter is activation.upsample.filter
    assert decoder[0].downsample.lowpass.filter is activation.downsample.lowpass.filter


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_alias_free_activation", [False, True], indirect=True)
@pytest.mark.parametrize("update_kind", ["load_state_dict", "train_then_eval"])
@torch.inference_mode()
def test_alias_free_parameter_updates(
    cuda_alias_free_activation: torch.nn.Module, update_kind: str
) -> None:
    activation = cuda_alias_free_activation
    decoder = torch.nn.Sequential(FusedAliasFree(activation))
    inputs = torch.randn(2, 7, 17, device="cuda")
    if update_kind == "load_state_dict":
        checkpoint = {
            name: value.clone() for name, value in decoder.state_dict().items()
        }
        checkpoint["0.act.alpha"].add_(0.5)
        checkpoint["0.act.beta"].add_(0.5)
        decoder.load_state_dict(checkpoint)
    else:
        decoder.train()
        activation.act.alpha.add_(0.5)
        activation.act.beta.add_(0.5)
        decoder.eval()
    torch.testing.assert_close(
        decoder(inputs), activation(inputs), rtol=1e-4, atol=1e-4
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_alias_free_activation", [False, True], indirect=True)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_alias_free_autocast_falls_back(
    cuda_alias_free_activation: torch.nn.Module, dtype: torch.dtype
) -> None:
    activation = cuda_alias_free_activation
    candidate = FusedAliasFree(activation)
    inputs = torch.randn(2, 7, 17, device="cuda")
    with torch.autocast("cuda", dtype=dtype):
        expected = activation(inputs)
        actual = candidate(inputs)
    assert actual.dtype == expected.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.parametrize("cuda_alias_free_activation", [False, True], indirect=True)
@pytest.mark.parametrize("frames", [1, 3, 17, 257])
@pytest.mark.parametrize("bias_stride", [1, 2])
@torch.inference_mode()
def test_alias_free_folded_bias_matches_bias_then_activation(
    cuda_alias_free_activation: torch.nn.Module, frames: int, bias_stride: int
) -> None:
    activation = cuda_alias_free_activation
    candidate = FusedAliasFree(activation)
    inputs = torch.randn(2, 7, frames, device="cuda")
    bias = torch.randn(7 * bias_stride, device="cuda")[::bias_stride]
    expected = candidate(inputs + bias.view(1, -1, 1))
    torch.testing.assert_close(candidate(inputs, bias=bias), expected, rtol=0, atol=0)

    residual = torch.randn_like(inputs)
    torch.testing.assert_close(
        alias_free.residual_bias_add(inputs, bias, residual),
        (inputs + bias.view(1, -1, 1)) + residual,
        rtol=0,
        atol=0,
    )
