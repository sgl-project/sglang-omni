# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from safetensors.torch import save_file
from transformers import MimiConfig, MimiModel
from transformers.models.mimi.modeling_mimi import (
    MimiConv1d,
    MimiEncoderOutput,
    MimiEuclideanCodebook,
)

from sglang_omni.models.qwen3_tts import request_builders as qwen3_request_builders
from sglang_omni.models.qwen3_tts import stages as qwen3_stages
from sglang_omni.models.qwen3_tts.reference_encoder_cuda_graph import (
    Qwen3TTSReferenceEncoderCudaGraphRunner,
    exact_length_encode,
    move_conv_padding_to_host,
    smallest_bucket,
)

HOP = 16


def small_mimi_config() -> MimiConfig:
    """Ratios 4 and 2 with the stride 2 downsample conv: 16 samples per frame."""
    return MimiConfig(
        hidden_size=16,
        num_filters=4,
        num_residual_layers=1,
        upsampling_ratios=[4, 2],
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=8,
        intermediate_size=32,
        vector_quantization_hidden_dimension=16,
        codebook_dim=16,
        codebook_size=8,
        num_quantizers=4,
        num_semantic_quantizers=1,
        sliding_window=16,
        upsample_groups=16,
    )


def test_move_conv_padding_to_host_keeps_the_conv_arithmetic() -> None:
    torch.manual_seed(3)
    config = small_mimi_config()
    conv = MimiConv1d(config, 4, 4, kernel_size=7, stride=2)
    values = torch.randn(1, 4, 21)
    before = conv(values)

    assert move_conv_padding_to_host(conv) == 1
    assert conv.stride.device.type == "cpu"
    assert conv.kernel_size.device.type == "cpu"
    assert conv.padding_total.device.type == "cpu"
    assert conv.padding_left.device.type == "cpu"
    assert torch.equal(conv(values), before)


def test_speech_tokenizer_loader_moves_the_encoder_padding_to_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = small_mimi_config()

    class FakeQwen3TTSTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            tokenizer = cls()
            tokenizer.model = SimpleNamespace(encoder=MimiModel(config).encoder)
            return tokenizer

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSTokenizer = FakeQwen3TTSTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts_module)
    monkeypatch.setattr(
        qwen3_stages, "apply_qwen_tts_transformers_compatibility_patches", lambda: None
    )
    monkeypatch.setattr(qwen3_stages, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(qwen3_stages, "_SPEECH_TOKENIZERS", {})

    tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
        "/ckpt",
        device="cpu",
        dtype="float32",
        attn_implementation=None,
        float32_encoder=False,
    )

    convs = [m for m in tokenizer.model.encoder.modules() if isinstance(m, MimiConv1d)]
    assert len(convs) == 1 + 2 * (2 + 1) + 1
    assert all(conv.padding_total.device.type == "cpu" for conv in convs)


def fake_speech_tokenizer(encoder: MimiModel, quantizers: int) -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            encoder=encoder,
            encode_downsample_rate=HOP,
            encoder_valid_num_quantizers=quantizers,
        ),
        _normalize_audio_inputs=lambda waveforms, sr: [
            np.asarray(waveform, dtype=np.float32) for waveform in waveforms
        ],
    )


def exact_length_codes(model: MimiModel, waveform: np.ndarray) -> torch.Tensor:
    return (
        model.encode(
            torch.from_numpy(waveform).view(1, 1, -1),
            num_quantizers=4,
            return_dict=True,
        )
        .audio_codes[0]
        .transpose(0, 1)
    )


def test_reference_codes_match_the_exact_length_encode() -> None:
    torch.manual_seed(7)
    model = MimiModel(small_mimi_config())
    for module in model.modules():
        if isinstance(module, MimiEuclideanCodebook):
            module.embed_sum.normal_()
    model.eval()
    unhooked_model = copy.deepcopy(model)
    batcher = qwen3_request_builders.Qwen3TTSRefCodeBatcher(
        fake_speech_tokenizer(model, quantizers=4), max_batch_wait_ms=0
    )
    waveforms = [torch.randn(frames * HOP + 7).numpy() for frames in (2, 5, 9)]

    try:
        codes = [batcher.encode(waveform, 24000) for waveform in waveforms]
    finally:
        batcher.close()

    with torch.inference_mode():
        want = [exact_length_codes(unhooked_model, waveform) for waveform in waveforms]
        codes_after_batcher = [
            exact_length_codes(model, waveform) for waveform in waveforms
        ]
    assert [tuple(clip_codes.shape) for clip_codes in codes] == [
        (3, 4),
        (6, 4),
        (10, 4),
    ]
    assert all(torch.equal(got, wanted) for got, wanted in zip(codes, want))
    assert all(
        torch.equal(got, wanted) for got, wanted in zip(codes_after_batcher, want)
    )


def test_reference_code_batcher_refuses_an_empty_reference() -> None:
    batcher = qwen3_request_builders.Qwen3TTSRefCodeBatcher(
        fake_speech_tokenizer(MimiModel(small_mimi_config()).eval(), quantizers=4),
        max_batch_wait_ms=0,
    )
    try:
        with pytest.raises(ValueError, match="no samples"):
            batcher.encode(np.zeros(0, dtype=np.float32), 24000)
    finally:
        batcher.close()


@pytest.mark.parametrize("float32_encoder", [False, True])
def test_speech_tokenizer_loader_gives_the_encoder_float32_checkpoint_weights_when_asked(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, float32_encoder: bool
) -> None:
    torch.manual_seed(5)
    config = small_mimi_config()
    checkpoint_encoder = MimiModel(config)
    checkpoint = {
        name: tensor.contiguous()
        for name, tensor in checkpoint_encoder.state_dict().items()
    }
    (tmp_path / "speech_tokenizer").mkdir()
    save_file(
        {f"encoder.{name}": tensor for name, tensor in checkpoint.items()},
        str(tmp_path / "speech_tokenizer" / "model.safetensors"),
    )

    class FakeDecoder(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = torch.nn.Linear(4, 4)
            self.register_buffer("inv_freq", torch.rand(4), persistent=False)

    class FakeQwen3TTSTokenizer:
        @classmethod
        def from_pretrained(cls, path, **kwargs):
            decoder = FakeDecoder()
            decoder.proj.to(kwargs["dtype"])
            tokenizer = cls()
            tokenizer.model = SimpleNamespace(
                encoder=copy.deepcopy(checkpoint_encoder).to(kwargs["dtype"]),
                decoder=decoder,
            )
            return tokenizer

    qwen_tts_module = types.ModuleType("qwen_tts")
    qwen_tts_module.Qwen3TTSTokenizer = FakeQwen3TTSTokenizer
    monkeypatch.setitem(sys.modules, "qwen_tts", qwen_tts_module)
    monkeypatch.setattr(
        qwen3_stages, "apply_qwen_tts_transformers_compatibility_patches", lambda: None
    )
    monkeypatch.setattr(qwen3_stages, "_resolve_checkpoint", lambda path: path)
    monkeypatch.setattr(qwen3_stages, "_SPEECH_TOKENIZERS", {})

    tokenizer = qwen3_stages.load_qwen3_tts_tokenizer(
        str(tmp_path),
        device="cpu",
        dtype="bfloat16",
        attn_implementation=None,
        float32_encoder=float32_encoder,
    )

    encoder_dtype = torch.float32 if float32_encoder else torch.bfloat16
    loaded = tokenizer.model.encoder.state_dict()
    assert loaded.keys() == checkpoint.keys()
    for name, tensor in checkpoint.items():
        assert loaded[name].dtype == encoder_dtype
        assert torch.equal(loaded[name], tensor.to(encoder_dtype))
    codebooks = [
        module
        for module in tokenizer.model.encoder.modules()
        if isinstance(module, MimiEuclideanCodebook)
    ]
    assert codebooks
    for codebook in codebooks:
        table = codebook._embed  # noqa: leading-underscore  # production name
        if float32_encoder:
            assert torch.equal(
                table,
                codebook.embed_sum
                / codebook.cluster_usage.clamp(min=codebook.epsilon)[:, None],
            )
        else:
            assert table is None
    assert tokenizer.model.decoder.proj.weight.dtype == torch.bfloat16
    assert tokenizer.model.decoder.inv_freq.dtype == torch.float32


def test_smallest_bucket_picks_the_first_bucket_that_fits() -> None:
    assert smallest_bucket(40, (32, 48, 64)) == 48
    assert smallest_bucket(48, (32, 48, 64)) == 48
    assert smallest_bucket(1, (32, 48, 64)) == 32
    assert smallest_bucket(65, (32, 48, 64)) is None
    assert smallest_bucket(3, ()) is None


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reference_encoder_graph_replays_match_eager_and_miss_above_the_largest_bucket() -> (
    None
):
    torch.manual_seed(11)
    device = torch.device("cuda", torch.cuda.current_device())
    config = small_mimi_config()
    model = MimiModel(config)
    for module in model.modules():
        if isinstance(module, MimiEuclideanCodebook):
            module.embed_sum.normal_()
    model = model.to(device).eval()
    move_conv_padding_to_host(model)
    hop = HOP
    quantizers = 2
    stream = torch.cuda.Stream(device)
    runner = Qwen3TTSReferenceEncoderCudaGraphRunner(
        model, hop=hop, num_quantizers=quantizers, bucket_frames=(4, 8), stream=stream
    )
    runner.capture()
    assert runner.stats()["enabled"]
    assert runner.stats()["captured"] == [4, 8]

    short = torch.randn(3 * hop + 5, device=device)
    long = torch.randn(7 * hop, device=device)
    long_other = torch.randn(6 * hop + 9, device=device)
    with torch.inference_mode(), torch.cuda.stream(stream):
        first = runner.encode(short)
        second = runner.encode(long)
        third = runner.encode(long_other)
        too_long = runner.encode(torch.randn(9 * hop, device=device))
    torch.cuda.synchronize(device)

    assert too_long is None
    assert runner.stats()["replays"] == 3
    assert runner.stats()["misses"] == 1
    assert first.shape == (4, quantizers)
    assert second.shape == (7, quantizers)
    assert third.shape == (7, quantizers)
    eager = []
    with torch.inference_mode():
        for waveform, bucket in ((short, 4), (long, 8), (long_other, 8)):
            values = torch.zeros((1, 1, bucket * hop), device=device)
            values[0, 0, : waveform.numel()].copy_(waveform)
            clip_samples = torch.tensor(waveform.numel(), device=device)
            with exact_length_encode(model, clip_samples):
                codes = model.encode(
                    values, num_quantizers=quantizers, return_dict=True
                ).audio_codes
            frames = -(-waveform.numel() // hop)
            eager.append(codes[0, :, :frames].transpose(0, 1))
    assert not torch.equal(eager[1], eager[2])
    assert torch.equal(first, eager[0])
    assert torch.equal(second, eager[1])
    assert torch.equal(third, eager[2])


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_reference_code_batcher_replays_captured_buckets_and_encodes_the_rest(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    torch.manual_seed(5)
    device = torch.device("cuda", torch.cuda.current_device())
    config = small_mimi_config()
    model = MimiModel(config)
    for module in model.modules():
        if isinstance(module, MimiEuclideanCodebook):
            module.embed_sum.normal_()
    model = model.to(device).eval()
    move_conv_padding_to_host(model)
    batcher = qwen3_request_builders.Qwen3TTSRefCodeBatcher(
        fake_speech_tokenizer(model, quantizers=2),
        max_batch_wait_ms=0,
        graph_bucket_frames=(4,),
    )
    beyond_waveforms = [torch.rand(frames * HOP).numpy() for frames in (5, 7)]
    encode = model.encode
    encoded_samples = []

    def recording_encode(
        values: torch.Tensor, *, num_quantizers: int, return_dict: bool
    ) -> MimiEncoderOutput:
        encoded_samples.append(int(values.shape[-1]))
        return encode(values, num_quantizers=num_quantizers, return_dict=return_dict)

    monkeypatch.setattr(model, "encode", recording_encode)
    try:
        assert batcher.graph_runner is not None
        inside = batcher.encode(torch.rand(3 * HOP + 1).numpy(), 24000)
        beyond = [batcher.encode(waveform, 24000) for waveform in beyond_waveforms]
    finally:
        batcher.close()
    expected = []
    with torch.inference_mode():
        for waveform in beyond_waveforms:
            values = torch.zeros((1, 1, 8 * HOP), device=device)
            values[0, 0, : waveform.size].copy_(torch.from_numpy(waveform))
            clip_samples = torch.tensor(waveform.size, device=device)
            with exact_length_encode(model, clip_samples):
                codes = encode(values, num_quantizers=2, return_dict=True).audio_codes
            expected.append(codes[0, :, : waveform.size // HOP].transpose(0, 1))

    assert all(int(codes.unique().numel()) > 1 for codes in expected)
    assert inside.shape == (4, 2) and inside.is_cuda
    assert encoded_samples == [8 * HOP, 8 * HOP]
    assert [tuple(codes.shape) for codes in beyond] == [(5, 2), (7, 2)]
    assert all(torch.equal(codes, want) for codes, want in zip(beyond, expected))
    assert batcher.graph_runner.stats() == {
        "enabled": True,
        "disable_reason": None,
        "bucket_frames": [4],
        "captured": [4],
        "replays": 1,
        "misses": 2,
    }
