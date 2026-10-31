# SPDX-License-Identifier: Apache-2.0
"""CPU equivalence checks against a small production Fish codec."""

import subprocess
import sys
import textwrap
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from sglang_omni.models.fishaudio_drama3.channels_last_decoder import (
    ChannelsLastDecoder,
)
from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
from sglang_omni.models.fishaudio_drama3.stateful_waveform import (
    StatefulWaveformDecoder,
)
from sglang_omni.models.fishaudio_s2_pro.fish_speech.models.dac.modded_dac import (
    DAC,
    ModelArgs,
    WindowLimitedTransformer,
)
from tests.unit_test.fixtures.fishaudio_codec import tiny_fishaudio_codec


@pytest.fixture
def codec() -> DAC:
    return tiny_fishaudio_codec()


@pytest.mark.parametrize(
    "folded_codebooks,gemm_upsample", [(False, True), (True, False), (True, True)]
)
@torch.no_grad()
def test_optimized_quantizer_preserves_incremental_audio(
    codec: DAC, folded_codebooks: bool, gemm_upsample: bool
) -> None:
    decoder = StatefulDACDecoder(
        codec,
        capacity=1,
        gemm_upsample=gemm_upsample,
        folded_codebooks=folded_codebooks,
    )
    codes = torch.randint(0, 16, (1, decoder.num_codebooks, 17))
    expected = codec.from_indices(codes)
    state = decoder.create_state()
    outputs: list[torch.Tensor] = []
    position = 0
    for frame_count in (1, 2, 3, 11):
        outputs.append(
            decoder.decode_delta(codes[..., position : position + frame_count], state)
        )
        position += frame_count
    decoder.release_state(state)
    torch.testing.assert_close(
        torch.cat(outputs, dim=-1), expected, atol=1e-4, rtol=1e-4
    )


@pytest.mark.accelerator
@pytest.mark.parametrize("compile_attention", [False, True])
@torch.no_grad()
def test_fused_history_preserves_interleaved_streams(
    codec: DAC, compile_attention: bool
) -> None:
    codec = codec.cuda().half()
    codec.fold_weight_norm()
    native = StatefulDACDecoder(codec, dtype=torch.float16, capacity=2)
    fused = StatefulDACDecoder(
        codec,
        dtype=torch.float16,
        capacity=2,
        fuse_snake_history=True,
        compile_attention=compile_attention,
    )
    native_states = [native.create_state() for _ in range(2)]
    fused_states = [fused.create_state() for _ in range(2)]
    generator = torch.Generator(device="cuda").manual_seed(73)
    codes = torch.randint(
        0, 16, (2, fused.num_codebooks, 43), device="cuda", generator=generator
    )
    position = 0
    for frame_count in (1, 2, 7, 10, 23):
        for stream_index in range(2):
            delta = codes[
                stream_index : stream_index + 1, :, position : position + frame_count
            ]
            expected = native.decode_delta(delta, native_states[stream_index])
            actual = fused.decode_delta(delta, fused_states[stream_index])
            torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
        position += frame_count
    for native_state, fused_state in zip(native_states, fused_states, strict=True):
        native.release_state(native_state)
        fused.release_state(fused_state)
    delta = codes[:1, :, :7]
    native_state = native.create_state()
    fused_state = fused.create_state()
    expected = native.decode_delta(delta, native_state, is_final=True)
    actual = fused.decode_delta(delta, fused_state, is_final=True)
    torch.testing.assert_close(actual, expected, atol=0.02, rtol=0.02)
    native.release_state(native_state)
    fused.release_state(fused_state)


@torch.no_grad()
def test_partial_failure_retains_lease_until_owner_cleanup(codec: DAC) -> None:
    decoder = StatefulDACDecoder(codec, capacity=1)
    state = decoder.create_state()
    codes = torch.zeros((1, decoder.num_codebooks, 10), dtype=torch.long)
    original = decoder.decode_slot

    def execute_then_fail(
        codes: torch.Tensor, slot_index: torch.Tensor, frame_count: int
    ) -> torch.Tensor:
        original(codes, slot_index, frame_count)
        raise RuntimeError("Failure after arena mutation")

    with patch.object(decoder, "decode_slot", side_effect=execute_then_fail):
        with pytest.raises(RuntimeError, match="after arena mutation"):
            decoder.decode_delta(codes, state)
    assert state.is_failed
    assert state.slot_index in decoder.arena.active_slots
    assert all(
        torch.count_nonzero(tensor[state.slot_index]) == 0
        for tensor in decoder.arena.persistent_tensors()
    )
    with pytest.raises(ValueError, match="failed stream"):
        decoder.decode_delta(codes, state)
    with pytest.raises(ValueError, match="capacity"):
        decoder.create_state()
    decoder.release_state(state)
    replacement = decoder.create_state()
    torch.testing.assert_close(
        decoder.decode_delta(codes, replacement),
        codec.from_indices(codes.clone()),
        atol=1e-4,
        rtol=1e-4,
    )
    with pytest.raises(ValueError, match="stale"):
        decoder.release_state(state)


@torch.no_grad()
def test_channels_last_incremental_history_and_unbiased_carry(codec: DAC) -> None:
    pytest.importorskip("triton")
    backend = StatefulDACDecoder(codec, capacity=1)
    decoder = StatefulWaveformDecoder(
        ChannelsLastDecoder(codec, dtype=torch.float32), backend.arena
    )
    state = backend.create_state()
    slot_index = torch.tensor([state.slot_index])
    for module in (codec.decoder,):
        for child in module.modules():
            if type(child).__name__ in ("CausalConvNet", "CausalTransConvNet"):
                inputs = torch.randn(1, child.conv.in_channels, 17)
                expected = child(inputs)
                history_path = "isolated_convolution"
                if type(child).__name__ == "CausalConvNet":
                    channels = child.conv.in_channels
                    history_length = (
                        child.conv.kernel_size[0] - 1
                    ) * child.conv.dilation[0]
                else:
                    channels = child.conv.out_channels
                    history_length = child.conv.kernel_size[0] - child.conv.stride[0]
                backend.arena.histories[history_path] = torch.zeros(
                    2, channels, history_length
                )
                backend.arena.histories[history_path].zero_()
                outputs = []
                position = 0
                for frames in (1, 2, 3, 11):
                    outputs.append(
                        decoder.decode_module(
                            child,
                            inputs[..., position : position + frames]
                            .transpose(1, 2)
                            .contiguous(),
                            slot_index,
                            history_path,
                        ).transpose(1, 2)
                    )
                    position += frames
                torch.testing.assert_close(
                    torch.cat(outputs, dim=-1), expected, atol=1e-4, rtol=1e-4
                )
            else:
                pass


@pytest.fixture
def forbid_cuda_initialization(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail_initialization() -> None:
        raise AssertionError("CPU codec must not initialize CUDA")

    monkeypatch.setattr(torch.cuda, "_lazy_init", fail_initialization)


@pytest.mark.usefixtures("forbid_cuda_initialization")
@torch.no_grad()
def test_native_whole_prefix_causality(codec: DAC) -> None:
    codes = torch.randint(0, 16, (1, 10, 21))
    whole = codec.from_indices(codes.clone())
    assert whole.shape == (1, 1, 21 * 16)
    for frame_count in (1, 2, 4, 7, 8, 9, 17, 20):
        prefix = codec.from_indices(codes[..., :frame_count].clone())
        torch.testing.assert_close(
            prefix, whole[..., : frame_count * 16], atol=1e-4, rtol=1e-4
        )


@pytest.mark.usefixtures("forbid_cuda_initialization")
@pytest.mark.parametrize("chunk_sizes", [(1,), (2,), (4,), (3, 1, 7, 2, 4)])
@torch.no_grad()
def test_incremental_matches_whole(codec: DAC, chunk_sizes: tuple[int, ...]) -> None:
    codes = torch.randint(0, 16, (1, 10, 37))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    chunks = []
    position = 0
    chunk_index = 0
    while position < codes.shape[-1]:
        end = min(
            position + chunk_sizes[chunk_index % len(chunk_sizes)], codes.shape[-1]
        )
        chunks.append(decoder.decode_delta(codes[..., position:end], state))
        position = end
        chunk_index += 1
        torch.testing.assert_close(
            torch.cat(chunks, dim=-1), expected[..., : end * 16], atol=1e-4, rtol=1e-4
        )
    tail = decoder.decode_delta(codes[..., :0], state, is_final=True)
    assert tail.shape == (1, 1, 0)
    assert state.processed_frames == 37
    assert state.is_final


@torch.no_grad()
def test_long_stream_bounded_history(codec: DAC) -> None:
    codes = torch.randint(0, 16, (1, 10, 500))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    chunks = []
    tensors = decoder.arena.persistent_tensors()
    addresses = [tensor.data_ptr() for tensor in tensors]
    retained_elements = sum(tensor.numel() for tensor in tensors)
    for position in range(500):
        chunks.append(decoder.decode_delta(codes[..., position : position + 1], state))
        assert [
            tensor.data_ptr() for tensor in decoder.arena.persistent_tensors()
        ] == addresses
        assert (
            sum(tensor.numel() for tensor in decoder.arena.persistent_tensors())
            == retained_elements
        )
    torch.testing.assert_close(
        torch.cat(chunks, dim=-1), expected, atol=1e-4, rtol=1e-4
    )
    assert state.processed_frames == 500
    assert decoder.arena.positions[state.slot_index] == 500


@torch.no_grad()
def test_interleaved_requests_and_batch_one_contract(codec: DAC) -> None:
    codes = torch.randint(0, 16, (2, 10, 19))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=3)
    first = decoder.create_state()
    second = decoder.create_state()
    first_chunks = []
    second_chunks = []
    for position in range(19):
        first_chunks.append(
            decoder.decode_delta(codes[:1, :, position : position + 1], first)
        )
        second_chunks.append(
            decoder.decode_delta(codes[1:, :, position : position + 1], second)
        )
    torch.testing.assert_close(
        torch.cat(first_chunks, dim=-1), expected[:1], atol=1e-4, rtol=1e-4
    )
    torch.testing.assert_close(
        torch.cat(second_chunks, dim=-1), expected[1:], atol=1e-4, rtol=1e-4
    )
    batch_state = decoder.create_state()
    with pytest.raises(ValueError, match="batch size one"):
        decoder.decode_delta(codes, batch_state)
    assert batch_state.processed_frames == 0


@torch.no_grad()
def test_state_validation_and_fresh_stream(codec: DAC) -> None:
    decoder = StatefulDACDecoder(codec, capacity=3)
    codes = torch.randint(0, 16, (1, 10, 3))
    original_codes = codes.clone()
    checkpoint_keys = tuple(codec.state_dict())
    state = decoder.create_state()
    expected = decoder.decode_delta(codes, state, is_final=True)
    with pytest.raises(ValueError, match="finalized"):
        decoder.decode_delta(codes, state)
    fresh = decoder.create_state()
    torch.testing.assert_close(
        decoder.decode_delta(codes, fresh), expected, atol=1e-4, rtol=1e-4
    )
    assert torch.equal(codes, original_codes)
    assert tuple(codec.state_dict()) == checkpoint_keys
    assert codec.quantizer.post_module.layers[0].attention.kv_cache is None
    with pytest.raises(ValueError, match="another decoder"):
        StatefulDACDecoder(codec, capacity=3).decode_delta(codes, fresh)
    with pytest.raises(ValueError, match="batch size"):
        decoder.decode_delta(codes.expand(2, -1, -1), fresh)


@pytest.mark.parametrize(
    "failure", ["noncausal", "lookahead", "unbounded", "training", "dtype"]
)
def test_reject_unsupported_codec(codec: DAC, failure: str) -> None:
    if failure == "noncausal":
        codec.quantizer.post_module.causal = False
    elif failure == "lookahead":
        codec.quantizer.post_module.look_ahead_conv = torch.nn.Conv1d(16, 16, 3)
    elif failure == "unbounded":
        codec.quantizer.post_module.window_size = None
    elif failure == "training":
        codec.train()
    else:
        codec.double()
    with pytest.raises(ValueError):
        StatefulDACDecoder(codec, capacity=3)


@torch.no_grad()
def test_only_new_frames_reach_projections(codec: DAC) -> None:
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    codes = torch.randint(0, 16, (1, 10, 17))
    observed_lengths: list[int] = []

    def record_projection(
        module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]
    ) -> None:
        observed_lengths.append(inputs[0].shape[1])

    hook = codec.quantizer.post_module.layers[
        0
    ].attention.wqkv.register_forward_pre_hook(record_projection)
    for start, end in ((0, 1), (1, 5), (5, 7), (7, 17)):
        decoder.decode_delta(codes[..., start:end], state)
    decoder.decode_delta(codes[..., :0], state, is_final=True)
    hook.remove()
    assert observed_lengths == [1, 4, 2, 10]
    assert sum(observed_lengths) == state.processed_frames == 17


@torch.no_grad()
def test_absolute_position_capacity_rejected_before_state_change(codec: DAC) -> None:
    codec.quantizer.post_module.freqs_cis = codec.quantizer.post_module.freqs_cis[:4]
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    codes = torch.randint(0, 16, (1, 10, 5))
    decoder.decode_delta(codes[..., :3], state)
    with pytest.raises(ValueError, match="position capacity"):
        decoder.decode_delta(codes[..., 3:], state)
    assert state.processed_frames == 3
    assert decoder.arena.positions[state.slot_index] == 3
    decoder.decode_delta(codes[..., 3:4], state, is_final=True)
    assert state.processed_frames == 4


@torch.no_grad()
def test_multiple_layers_and_projection_dimensions(codec: DAC) -> None:
    codec.quantizer.post_module = (
        WindowLimitedTransformer(
            ModelArgs(
                n_layer=2,
                n_head=4,
                dim=32,
                head_dim=8,
                intermediate_size=64,
                dropout_rate=0,
                attn_dropout_rate=0,
            ),
            input_dim=16,
            window_size=8,
            causal=True,
        )
        .eval()
        .float()
    )
    codes = torch.randint(0, 16, (1, 10, 31))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    chunks = [
        decoder.decode_delta(codes[..., :5], state),
        decoder.decode_delta(codes[..., 5:17], state),
        decoder.decode_delta(codes[..., 17:], state, is_final=True),
    ]
    torch.testing.assert_close(
        torch.cat(chunks, dim=-1), expected, atol=1e-4, rtol=1e-4
    )


@torch.no_grad()
def test_window_one_and_empty_first_chunk(codec: DAC) -> None:
    codec.quantizer.post_module.window_size = 1
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    codes = torch.randint(0, 16, (1, 10, 9))
    assert decoder.decode_delta(codes[..., :0], state).shape == (1, 1, 0)
    assert state.processed_frames == 0
    expected = codec.from_indices(codes.clone())
    chunks = [
        decoder.decode_delta(codes[..., :2], state),
        decoder.decode_delta(codes[..., 2:], state, is_final=True),
    ]
    torch.testing.assert_close(
        torch.cat(chunks, dim=-1), expected, atol=1e-4, rtol=1e-4
    )


@torch.no_grad()
def test_transposed_convolution_bias_is_not_accumulated(codec: DAC) -> None:
    for module in codec.modules():
        if isinstance(module, torch.nn.ConvTranspose1d) and module.bias is not None:
            module.bias.uniform_(-0.3, 0.3)
        else:
            pass
    codes = torch.randint(0, 16, (1, 10, 13))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    chunks = [
        decoder.decode_delta(codes[..., position : position + 1], state)
        for position in range(13)
    ]
    torch.testing.assert_close(
        torch.cat(chunks, dim=-1), expected, atol=1e-4, rtol=1e-4
    )


@pytest.mark.usefixtures("forbid_cuda_initialization")
def test_unused_encoder_dtype_does_not_reject_decode(codec: DAC) -> None:
    codec.encoder.double()
    decoder = StatefulDACDecoder(codec, capacity=3)
    codes = torch.randint(0, 16, (1, 10, 2))
    waveform = decoder.decode_delta(codes, decoder.create_state(), is_final=True)
    assert waveform.dtype == torch.float32


@torch.no_grad()
def test_unused_quantizer_weights_do_not_reject_decode(codec: DAC) -> None:
    codec.quantizer.pre_module = torch.nn.Linear(16, 16).double()
    codec.quantizer.semantic_predictor_module = torch.nn.Linear(16, 16).double()
    codec.quantizer.downsample.double()
    for vector_quantizer in (
        *codec.quantizer.semantic_quantizer.quantizers,
        *codec.quantizer.quantizer.quantizers,
    ):
        vector_quantizer.in_proj.double()
    codes = torch.randint(0, 16, (1, 10, 9))
    expected = codec.from_indices(codes.clone())
    decoder = StatefulDACDecoder(codec, capacity=3)
    waveform = decoder.decode_delta(codes, decoder.create_state(), is_final=True)
    torch.testing.assert_close(waveform, expected, atol=1e-4, rtol=1e-4)
    assert waveform.dtype == torch.float32


@torch.no_grad()
def test_negative_code_lookup_discards_failed_lease(codec: DAC) -> None:
    decoder = StatefulDACDecoder(codec, capacity=3)
    state = decoder.create_state()
    with pytest.raises(IndexError):
        decoder.decode_delta(torch.full((1, 10, 1), -1, dtype=torch.long), state)
    assert state.batch_size is None
    assert state.processed_frames == 0
    with pytest.raises(ValueError, match="failed stream"):
        decoder.decode_delta(torch.zeros((1, 10, 1), dtype=torch.long), state)
    failed_slot = state.slot_index
    decoder.release_state(state)
    state = decoder.create_state(failed_slot)
    assert all(
        torch.count_nonzero(tensor[state.slot_index]) == 0
        for tensor in decoder.arena.persistent_tensors()
    )
    codes = torch.randint(0, 16, (1, 10, 3))
    expected = codec.from_indices(codes.clone())
    actual = decoder.decode_delta(codes, state, is_final=True)
    torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-4)
    assert state.batch_size == 1


def test_stateful_codec_import_and_cpu_decode_without_triton() -> None:
    """Optional stateful imports must not prevent CPU decoding without Triton."""
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            textwrap.dedent(
                """
            import builtins
            import torch
            from sglang_omni.platforms import current_platform
            original_import = builtins.__import__
            def without_triton(name, *args, **kwargs):
                if name == "triton" or name.startswith("triton."):
                    raise ModuleNotFoundError(name, name=name)
                else:
                    return original_import(name, *args, **kwargs)
            builtins.__import__ = without_triton
            import torch
            assert not torch.cuda.is_initialized()
            from sglang_omni.models.fishaudio_drama3 import stateful_waveform
            from sglang_omni.models.fishaudio_drama3.stateful_decoder import StatefulDACDecoder
            from tests.unit_test.fixtures.fishaudio_codec import tiny_fishaudio_codec
            assert stateful_waveform.triton is None
            codec = tiny_fishaudio_codec()
            decoder = StatefulDACDecoder(codec, capacity=1)
            state = decoder.create_state()
            codes = torch.zeros((1, decoder.num_codebooks, 3), dtype=torch.long)
            audio = decoder.decode_delta(codes, state, is_final=True)
            assert audio.shape[-1] == 3 * codec.frame_length
            assert torch.isfinite(audio).all()
            decoder.release_state(state)
            assert not torch.cuda.is_initialized()
        """
            ),
        ],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stdout + result.stderr
