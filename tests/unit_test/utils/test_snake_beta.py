# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from importlib import import_module
from importlib.util import find_spec
from unittest.mock import Mock

import pytest
import torch

from sglang_omni.models.qwen3_tts.compat import (
    apply_qwen_tts_transformers_compatibility_patches,
)
from sglang_omni.platforms import current_platform
from sglang_omni.utils import snake_beta
from sglang_omni.utils.channels_last_conv import channels_last_weight

if find_spec("cutlass") is not None and find_spec("qwen_tts") is not None:
    from sglang_omni.models.qwen3_tts.cutedsl.vocoder import (
        ChannelsLastUpsample,
        CuTeDSLCausalConv1d,
        CuTeDSLResidualUnit,
        fuse_cutedsl_convolutions,
    )
    from sglang_omni.models.qwen3_tts.incremental_codec import (
        Qwen3TTSIncrementalCodecState,
        incremental_causal_conv1d,
    )
else:
    CuTeDSLResidualUnit = None

if find_spec("cutlass") is not None:
    from sglang_omni.models.qwen3_tts.cutedsl.conv1d import (
        compile_kernel,
        conv1d,
        pointwise_conv,
    )
    from sglang_omni.models.qwen3_tts.cutedsl.snake_beta import fuse_cutedsl_vocoder
else:
    fuse_cutedsl_vocoder = None


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or CuTeDSLResidualUnit is None,
    reason="requires CUDA, CUTLASS DSL, and qwen-tts",
)
@pytest.mark.parametrize("windowed", [False, True])
@torch.inference_mode()
def test_cutedsl_decoder_replacement_and_graph_replay(windowed: bool) -> None:
    """Replacing decoder blocks preserves waveform alignment in full/window decode and graph replay."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    else:
        pass
    torch.manual_seed(42)
    apply_qwen_tts_transformers_compatibility_patches()
    tokenizer = import_module(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
    )
    configuration = tokenizer.Qwen3TTSTokenizerV2DecoderConfig(
        hidden_size=64,
        latent_dim=64,
        codebook_dim=64,
        codebook_size=32,
        num_quantizers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=16,
        intermediate_size=128,
        num_hidden_layers=1,
        max_position_embeddings=64,
        decoder_dim=512,
        attn_implementation="sdpa",
    )
    original = (
        tokenizer.Qwen3TTSTokenizerV2Decoder(configuration).cuda().bfloat16().eval()
    )
    for name, parameter in original.named_parameters():
        if name.endswith("embedding_sum"):
            parameter.normal_()
        else:
            pass
    fused = copy.deepcopy(original)
    assert fuse_cutedsl_vocoder(fused) > 0
    assert fuse_cutedsl_convolutions(fused) > 0
    assert all(
        isinstance(block.block[1], ChannelsLastUpsample)
        for block in fused.decoder[1:-2]
    )
    if windowed:
        reference_decode = original.chunked_decode
        fused_decode = fused.chunked_decode
        decode_arguments = {"chunk_size": 3, "left_context_size": 2}
    else:
        reference_decode = original
        fused_decode = fused
        decode_arguments = {}
    codes = torch.randint(32, (2, 2, 7), device="cuda")
    # note (Hao Wang): Random decoder weights produce quiet audio, so scale tolerance to waveform RMS.
    expected = reference_decode(codes, **decode_arguments)
    absolute_tolerance = 0.04 * expected.float().square().mean().sqrt().item()
    torch.testing.assert_close(
        fused_decode(codes, **decode_arguments),
        expected,
        atol=absolute_tolerance,
        rtol=0.02,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        waveform = fused_decode(codes, **decode_arguments)
    for _ in range(2):
        codes.random_(0, 32)
        graph.replay()
        expected = reference_decode(codes, **decode_arguments)
        absolute_tolerance = 0.04 * expected.float().square().mean().sqrt().item()
        torch.testing.assert_close(
            waveform, expected, atol=absolute_tolerance, rtol=0.02
        )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or fuse_cutedsl_vocoder is None,
    reason="requires CUDA and CUTLASS DSL",
)
@pytest.mark.parametrize(
    "channels_last,length", [(False, 33), (True, 33), (False, 65536), (True, 65536)]
)
@torch.inference_mode()
def test_cutedsl_snake_preserves_checkpoint_and_graph_replay(
    channels_last: bool, length: int
) -> None:
    """Prepared coefficients preserve checkpoint keys, eager rounding, and compiled graph replay."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    torch.manual_seed(42)
    original = StubSnakeBeta(64).to(device="cuda", dtype=torch.bfloat16).eval()
    decoder = torch.nn.Sequential(original)
    checkpoint = {name: value.clone() for name, value in decoder.state_dict().items()}
    assert fuse_cutedsl_vocoder(decoder) == 1
    assert fuse_cutedsl_vocoder(decoder) == 0
    assert decoder.state_dict().keys() == checkpoint.keys()
    assert all(
        torch.equal(value, checkpoint[name])
        for name, value in decoder.state_dict().items()
    )
    x = torch.randn(1, 64, length, device="cuda", dtype=torch.bfloat16)
    if channels_last:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    compiled = torch.compile(decoder, fullgraph=True)
    torch.testing.assert_close(compiled(x), original(x), atol=0, rtol=0)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = decoder(x)
    x.normal_()
    graph.replay()
    assert actual.stride() == x.stride()
    torch.testing.assert_close(actual, original(x), atol=0, rtol=0)
    resized = torch.randn(2, 64, length + 8, device="cuda", dtype=torch.bfloat16)
    if channels_last:
        resized = resized.transpose(1, 2).contiguous().transpose(1, 2)
    torch.testing.assert_close(decoder(resized), original(resized), atol=0, rtol=0)


class StubSnakeBeta(torch.nn.Module):
    """Stand-in with the qwen-tts SnakeBeta attribute layout."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.alpha = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.beta = torch.nn.Parameter(torch.randn(channels) * 0.1)
        self.no_div_by_zero = 1e-9

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        alpha = torch.exp(self.alpha.unsqueeze(0).unsqueeze(-1))
        beta = torch.exp(self.beta.unsqueeze(0).unsqueeze(-1))
        return hidden_states + (1.0 / (beta + self.no_div_by_zero)) * torch.pow(
            torch.sin(hidden_states * alpha), 2
        )


StubSnakeBeta.__name__ = "SnakeBeta"


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or fuse_cutedsl_vocoder is None,
    reason="requires CUDA and CUTLASS DSL",
)
@torch.inference_mode()
def test_cutedsl_snake_large_offsets() -> None:
    """Buffers above 2**31 elements must not overflow launch dimensions or element offsets."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    else:
        pass
    if torch.cuda.mem_get_info()[0] < 12 * 1024**3:
        pytest.skip("Large-offset regression needs 12 GiB free device memory")
    else:
        pass
    torch.manual_seed(42)
    original = StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    decoder = torch.nn.Sequential(original)
    assert fuse_cutedsl_vocoder(decoder) == 1
    hidden_states = torch.ones(32, 96, 699056, device="cuda", dtype=torch.bfloat16)
    actual = decoder(hidden_states)
    expected = original(hidden_states[:, :, :8])
    torch.testing.assert_close(actual[:, :, :8], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :, -8:], expected, atol=0, rtol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or CuTeDSLResidualUnit is None,
    reason="requires CUDA, CUTLASS DSL, and qwen-tts",
)
@pytest.mark.parametrize("channels,dilation", [(96, 3), (192, 9), (768, 1)])
@torch.inference_mode()
def test_cutedsl_residual_graph_and_incremental_cache(
    channels: int, dilation: int
) -> None:
    """Fusion preserves checkpoint keys, causal history, and graph replay across audio lengths."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    else:
        pass
    torch.manual_seed(42)
    apply_qwen_tts_transformers_compatibility_patches()
    tokenizer = import_module(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
    )
    original = (
        tokenizer.Qwen3TTSTokenizerV2DecoderDecoderResidualUnit(channels, dilation)
        .cuda()
        .bfloat16()
        .eval()
    )
    assert fuse_cutedsl_vocoder(original) == 2
    fused = CuTeDSLResidualUnit(original).eval()
    assert fused.state_dict().keys() == original.state_dict().keys()
    assert all(
        torch.equal(value, original.state_dict()[name])
        for name, value in fused.state_dict().items()
    )
    for batch_size, length in ((1, 65), (2, 257)):
        hidden_states = torch.randn(
            batch_size, channels, length, device="cuda", dtype=torch.bfloat16
        )
        torch.testing.assert_close(
            fused(hidden_states), original(hidden_states), atol=0.03125, rtol=0.02
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = fused(hidden_states)
        hidden_states.normal_()
        graph.replay()
        torch.testing.assert_close(output, fused(hidden_states), atol=0, rtol=0)
        original_state = Qwen3TTSIncrementalCodecState()
        fused_state = Qwen3TTSIncrementalCodecState()
        for chunk in hidden_states.split(32, dim=-1):
            expected = incremental_causal_conv1d(
                original.conv1, chunk, original_state, "conv"
            )
            actual = incremental_causal_conv1d(fused.conv1, chunk, fused_state, "conv")
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
        torch.testing.assert_close(
            fused_state.conv_histories["conv"],
            original_state.conv_histories["conv"],
            atol=0,
            rtol=0,
        )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or fuse_cutedsl_vocoder is None,
    reason="requires CUDA and CUTLASS DSL",
)
@torch.inference_mode()
def test_cutedsl_input_convolution_causal_boundaries() -> None:
    """Channel expansion and partial temporal tiles preserve left-zero causal padding."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    else:
        pass
    torch.manual_seed(42)
    weight = (
        torch.randn(1536, 7, 1024, device="cuda", dtype=torch.bfloat16)
        / (7 * 1024) ** 0.5
    )
    bias = torch.randn(1536, device="cuda", dtype=torch.bfloat16)
    for batch_size, length in ((1, 1), (2, 129)):
        hidden_states = torch.randn(
            batch_size, length, 1024, device="cuda", dtype=torch.bfloat16
        )
        expected = torch.nn.functional.conv1d(
            torch.nn.functional.pad(hidden_states.transpose(1, 2).contiguous(), (6, 0)),
            weight.transpose(1, 2).contiguous(),
            bias,
        ).transpose(1, 2)
        torch.testing.assert_close(
            conv1d(hidden_states, weight, bias, 1), expected, atol=0.03125, rtol=0.02
        )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or CuTeDSLResidualUnit is None,
    reason="requires CUDA, CUTLASS DSL, and qwen-tts",
)
@torch.inference_mode()
def test_cutedsl_output_convolution_compiled_fallback() -> None:
    """The one-channel vocoder fallback passes both bias and length to the convolution helper."""
    torch.manual_seed(42)
    apply_qwen_tts_transformers_compatibility_patches()
    tokenizer = import_module(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2"
    )
    original = (
        tokenizer.Qwen3TTSTokenizerV2CausalConvNet(96, 1, 7).cuda().bfloat16().eval()
    )
    wrapped = CuTeDSLCausalConv1d(original)
    hidden_states = torch.randn(1, 96, 65, device="cuda", dtype=torch.bfloat16)
    torch.testing.assert_close(
        wrapped(hidden_states), original(hidden_states), atol=0.03125, rtol=0.02
    )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or fuse_cutedsl_vocoder is None,
    reason="requires CUDA and CUTLASS DSL",
)
@torch.inference_mode()
def test_cutedsl_convolution_dynamic_sizes() -> None:
    """Changing batch and length reuses full/partial-tile binaries with correct boundaries."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    else:
        pass
    torch.manual_seed(42)
    weight = (
        torch.randn(96, 7, 96, device="cuda", dtype=torch.bfloat16) / (7 * 96) ** 0.5
    )
    compile_kernel.cache_clear()
    for batch_size, length in ((1, 1), (2, 65), (3, 257), (1, 33), (2, 128), (1, 384)):
        hidden_states = torch.randn(
            batch_size, length, 96, device="cuda", dtype=weight.dtype
        )
        expected = torch.nn.functional.conv1d(
            torch.nn.functional.pad(
                hidden_states.transpose(1, 2).contiguous(), (18, 0)
            ),
            weight.transpose(1, 2).contiguous(),
            dilation=3,
        ).transpose(1, 2)
        torch.testing.assert_close(
            conv1d(hidden_states, weight, None, 3), expected, atol=0.03125, rtol=0.02
        )
    assert compile_kernel.cache_info().misses == 2


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or CuTeDSLResidualUnit is None,
    reason="requires CUDA, CUTLASS DSL, and qwen-tts",
)
@pytest.mark.parametrize("channels", [96, 768])
@torch.inference_mode()
def test_cutedsl_incremental_pointwise_preserves_outputs_and_replay(
    channels: int,
) -> None:
    """The streaming pointwise convolution leaves bias and residual to its consumer, including graph replay."""
    if torch.cuda.get_device_capability() not in ((9, 0), (10, 0), (10, 3)):
        pytest.skip("requires Hopper or datacenter Blackwell")
    else:
        pass
    torch.manual_seed(42)
    module = torch.nn.Conv1d(channels, channels, 1).cuda().bfloat16().eval()
    weight = channels_last_weight(module).transpose(1, 2)
    for length in (1, 32, 7):
        hidden_states = torch.randn(
            2, length, channels, device="cuda", dtype=torch.bfloat16
        )
        expected = torch.nn.functional.conv1d(
            hidden_states.transpose(1, 2), module.weight
        ).transpose(1, 2)
        actual = pointwise_conv(hidden_states, weight, None, None)
        torch.testing.assert_close(actual, expected, atol=0.03125, rtol=0.02)

    compiled = torch.compile(pointwise_conv, fullgraph=True)
    compiled(hidden_states, weight, None, None)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = compiled(hidden_states, weight, None, None)
    hidden_states.normal_()
    graph.replay()
    expected = pointwise_conv(hidden_states, weight, None, None)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)


def test_fuse_vocoder_decoder_keeps_originals_on_prewarm_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    first = StubSnakeBeta(4)
    second = StubSnakeBeta(4)
    decoder = torch.nn.Sequential(first, torch.nn.Sequential(second))

    monkeypatch.setattr(snake_beta, "HAS_TRITON", True)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)

    def fail_prewarm(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("prewarm failed")

    monkeypatch.setattr(snake_beta, "prewarm_replacements", fail_prewarm)

    assert snake_beta.fuse_vocoder_decoder(decoder) == 0
    assert decoder[0] is first
    assert decoder[1][0] is second


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused SnakeBeta parity needs CUDA"
)
@pytest.mark.parametrize(
    ("batch", "channels", "frames"),
    [
        (1, 96, 33),
        (2, 192, 96),
        (1, 384, 192),
        (1, 768, 257),
        (1, 96, 65536),
        (1, 96, 122880),
        (4, 1536, 1),
        (8, 384, 64),
        (16, 96, 37845),
        (32, 96, 64),
        (2, 4, 33),
    ],
)
@pytest.mark.parametrize("channels_last", [False, True])
def test_fused_snake_beta_cuda_parity_uses_kernel(
    monkeypatch: pytest.MonkeyPatch,
    batch: int,
    channels: int,
    frames: int,
    channels_last: bool,
) -> None:
    # note (db-ol): on the accelerator runner a missing Triton must fail
    # loudly, a skip here would hide the kernel from CI again.
    assert snake_beta.HAS_TRITON, "Triton is required on accelerator CI"

    torch.manual_seed(0)
    device = torch.device("cuda")
    original = StubSnakeBeta(channels).to(device=device, dtype=torch.bfloat16)
    if channels_last:
        x = torch.randn(
            (batch, frames, channels), device=device, dtype=torch.bfloat16
        ).transpose(1, 2)
    else:
        x = torch.randn(
            (batch, channels, frames),
            device=device,
            dtype=torch.bfloat16,
        )
    expected = original(x)
    launches: list[tuple[int, int, int]] = []
    original_launch = snake_beta.launch

    def record_launch(
        hidden_states: torch.Tensor,
        alpha: torch.Tensor,
        beta: torch.Tensor,
        eps: float,
    ) -> torch.Tensor:
        launches.append(tuple(hidden_states.shape))
        return original_launch(hidden_states, alpha, beta, eps)

    monkeypatch.setattr(snake_beta, "launch", record_launch)

    actual = snake_beta.fused_snake_beta(
        x, original.alpha, original.beta, original.no_div_by_zero
    )

    assert actual is not None
    assert launches == [(batch, channels, frames)]
    assert torch.equal(actual, expected)
    assert actual.stride() == x.stride()


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_shared_snake_preserves_parameters_and_cpu_fallback(dtype: torch.dtype) -> None:
    original = StubSnakeBeta(96).to(dtype=dtype).eval()
    nested = StubSnakeBeta(96).to(dtype=dtype).eval()
    decoder = torch.nn.Sequential(original, torch.nn.Sequential(nested))
    x = torch.randn(2, 96, 17).to(dtype=dtype)
    expected = decoder(x)
    state = {name: value.clone() for name, value in decoder.state_dict().items()}

    assert snake_beta.fuse_vocoder_decoder(decoder) == 2
    assert snake_beta.fuse_vocoder_decoder(decoder) == 0
    assert isinstance(decoder[1][0], snake_beta.FusedSnakeBeta)
    assert decoder[0].alpha is original.alpha
    assert decoder[0].beta is original.beta
    assert decoder[1][0].alpha is nested.alpha
    assert not decoder[0].training
    assert decoder.state_dict().keys() == state.keys()
    assert all(
        torch.equal(value, state[name]) for name, value in decoder.state_dict().items()
    )
    assert torch.equal(decoder(x), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_uses_the_module_epsilon() -> None:
    original = StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    x = torch.ones(1, 96, 257, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        original.alpha.zero_()
        original.beta.zero_()
        original.no_div_by_zero = 0.5
        expected = original(x)
        original.no_div_by_zero = 1e-9
        default_epsilon = original(x)
        original.no_div_by_zero = 0.5
        assert not torch.equal(expected, default_epsilon)
        fused = snake_beta.fused_snake_beta(
            x, original.alpha, original.beta, original.no_div_by_zero
        )
        assert fused is not None
        assert torch.equal(fused, expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("kind", ["dtype", "layout", "empty", "length"])
def test_shared_snake_cuda_falls_back_outside_envelope(kind: str) -> None:
    batch, channels, frames = 2, 96, 17
    if kind == "empty":
        frames = 0
    elif kind == "length":
        batch, channels, frames = 1, 1, snake_beta.MAX_T + 1
    dtype = torch.float32 if kind == "dtype" else torch.bfloat16
    original = StubSnakeBeta(channels).to(device="cuda", dtype=dtype).eval()
    x = torch.randn(batch, channels, frames, device="cuda", dtype=dtype)
    if kind == "layout":
        x = x[..., ::2]
    with torch.inference_mode():
        assert (
            snake_beta.fused_snake_beta(
                x, original.alpha, original.beta, original.no_div_by_zero
            )
            is None
        )
        assert torch.equal(snake_beta.FusedSnakeBeta(original)(x), original(x))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("channels_last", [False, True])
def test_shared_snake_bf16_encodings_and_denormals(channels_last: bool) -> None:
    original = StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    encodings = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16)
    values = encodings.view(torch.bfloat16)
    values = values[torch.isfinite(values)]
    x = values.repeat(96).reshape(1, 96, -1)
    if channels_last:
        x = x.transpose(1, 2).contiguous().transpose(1, 2)
    else:
        pass
    with torch.inference_mode():
        original.alpha.copy_(
            torch.tensor([-90.0, -80.0, 0.0], device="cuda").repeat(32)
        )
        original.beta.copy_(torch.tensor([-90.0, 0.0, 80.0], device="cuda").repeat(32))
        actual = snake_beta.fused_snake_beta(
            x, original.alpha, original.beta, original.no_div_by_zero
        )
        assert actual is not None
        assert torch.equal(actual, original(x))


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_prewarm_covers_new_shapes_and_capture(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:0")
    snake_beta.prewarm(device)
    compile_kernel = Mock(side_effect=AssertionError("runtime Triton compilation"))
    monkeypatch.setattr(snake_beta.snake_beta_kernel, "compile", compile_kernel)
    monkeypatch.setattr(
        snake_beta.snake_beta_channels_last_kernel, "compile", compile_kernel
    )
    with torch.inference_mode():
        for index, frames in enumerate(
            (1, 16, 33, 64, 65, 96, 128, 129, 192, 256, 257, 1024, 122880)
        ):
            channels = (96, 192, 384, 768, 1536)[index % 5]
            original = (
                StubSnakeBeta(channels).to(device=device, dtype=torch.bfloat16).eval()
            )
            original.no_div_by_zero = (1e-9, 1e-3)[index % 2]
            batch = 16 if frames <= 1024 else 1
            x = torch.randn(
                batch, channels, frames, device=device, dtype=torch.bfloat16
            )
            if index % 2:
                x = x.transpose(1, 2).contiguous().transpose(1, 2)
            else:
                pass
            expected = original(x)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                actual = snake_beta.fused_snake_beta(
                    x, original.alpha, original.beta, original.no_div_by_zero
                )
            graph.replay()
            assert actual is not None
            assert torch.equal(actual, expected)
    compile_kernel.assert_not_called()


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available(), reason="fused SnakeBeta compile needs CUDA"
)
def test_fused_snake_beta_survives_a_fullgraph_compile() -> None:
    """A fullgraph compile of a fused decoder raised Unsupported on the launch."""
    torch.manual_seed(0)
    device = torch.device("cuda")
    original = StubSnakeBeta(96).to(device=device, dtype=torch.bfloat16)
    decoder = torch.nn.Sequential(original)
    x = torch.randn((2, 96, 320), device=device, dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(x)

        assert snake_beta.fuse_vocoder_decoder(decoder) == 1
        compiled = torch.compile(decoder, dynamic=False, fullgraph=True)

        assert torch.equal(compiled(x), expected)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_shared_snake_graph_reads_current_inputs_and_parameters() -> None:
    original = StubSnakeBeta(96).to(device="cuda", dtype=torch.bfloat16).eval()
    decoder = torch.nn.Sequential(original)
    x = torch.zeros(1, 96, 33, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        assert snake_beta.fuse_vocoder_decoder(decoder) == 1
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            actual = decoder(x)
        for value in (0.1, -3.0):
            x.fill_(value)
            original.alpha.fill_(value)
            original.beta.fill_(-value)
            graph.replay()
            assert torch.equal(actual, original(x))


@pytest.mark.accelerator
@pytest.mark.parametrize("channels_last", [False, True])
def test_fused_snake_beta_matches_eager_on_this_host(channels_last: bool) -> None:
    """The parity tests above allocate CUDA tensors, so this is what exercises the
    launch and its rounding options on a non-CUDA Triton device."""
    device_type = current_platform.device_type
    if device_type not in snake_beta.TRITON_DEVICE_TYPES:
        pytest.skip(f"{device_type} has no Triton backend")
    else:
        pass
    assert snake_beta.HAS_TRITON, "Triton is required on accelerator CI"
    device_module = torch.get_device_module(device_type)
    device = torch.device(device_type, device_module.current_device())
    snake_beta.prewarm(device)

    torch.manual_seed(0)
    original = StubSnakeBeta(96).to(device=device, dtype=torch.bfloat16)
    for frames in (2, 33, 1024):
        if channels_last:
            x = torch.randn(
                (1, frames, 96), device=device, dtype=torch.bfloat16
            ).transpose(1, 2)
        else:
            x = torch.randn((1, 96, frames), device=device, dtype=torch.bfloat16)
        expected = original(x)
        actual = snake_beta.fused_snake_beta(
            x, original.alpha, original.beta, original.no_div_by_zero
        )
        assert actual is not None, frames
        assert torch.equal(actual, expected), frames
        assert actual.stride() == x.stride(), frames
