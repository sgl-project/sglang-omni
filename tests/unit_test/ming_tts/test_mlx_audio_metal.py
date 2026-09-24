# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

if not torch.backends.mps.is_available():
    pytest.skip("Requires an accessible Apple Metal device", allow_module_level=True)

mx = pytest.importorskip("mlx.core")

from sglang_omni.models.ming_tts.mlx.audio_vae import (  # noqa: E402
    AudioVAE,
    Decoder,
    Encoder,
    ISTFT,
    StreamingLinearUpsample,
)


def backbone_config(*, window: int | None = None) -> dict[str, Any]:
    return dict(
        vocab_size=8, hidden_size=16, intermediate_size=24, num_hidden_layers=2,
        num_attention_heads=2, num_key_value_heads=1, max_position_embeddings=256,
        rms_norm_eps=1e-6, rope_theta=10000.0, use_sliding_window=window is not None,
        sliding_window=window, max_window_layers=0, _attn_implementation="sdpa",
    )


def copy_weights(native: Any, reference: torch.nn.Module) -> None:
    native.load_weights([
        (key, mx.array(value.detach().float().numpy()))
        for key, value in reference.state_dict().items()
    ], strict=True)
    native.eval()


def assert_close(actual: Any, expected: torch.Tensor) -> None:
    np.testing.assert_allclose(
        np.array(actual.astype(mx.float32)), expected.detach().float().numpy(),
        atol=3e-5, rtol=3e-4,
    )


@pytest.mark.parametrize("lengths", [(2,), (2, 3), (1, 2, 1, 3)])
def test_streaming_interpolation_matches_torch(lengths: tuple[int, ...]) -> None:
    from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import (
        StreamingLinearUpsample as TorchUpsample,
    )

    native = StreamingLinearUpsample(4)
    reference = TorchUpsample(4)
    values = torch.linspace(-1, 1, sum(lengths) * 3).reshape(1, -1, 3)
    native_state = torch_state = None
    offset = 0
    outputs = []
    for index, length in enumerate(lengths):
        chunk = values[:, offset:offset + length]
        terminal = index == len(lengths) - 1
        actual, native_state = native(mx.array(chunk.numpy()), native_state, last_chunk=terminal)
        expected, torch_state = reference(chunk, torch_state, is_last=terminal)
        if expected is None:
            assert actual is None
        else:
            assert_close(actual, expected)
            outputs.append(actual)
        offset += length
    assert native_state is None
    full = torch.nn.functional.interpolate(values.transpose(1, 2), scale_factor=4, mode="linear", align_corners=False).transpose(1, 2)
    assert_close(mx.concatenate(outputs, axis=1), full)


@pytest.mark.parametrize("lengths", [(12,), (4, 4, 4), (5, 3, 4)])
@pytest.mark.parametrize("n_fft,hop_length", [(32, 8), (3528, 882)])
def test_istft_overlap_and_flush(
    lengths: tuple[int, ...], n_fft: int, hop_length: int
) -> None:
    from sglang_omni.models.ming_omni.talker.audio_vae.istft import ISTFT as TorchISTFT

    native = ISTFT(n_fft, hop_length)
    reference = TorchISTFT(n_fft, hop_length, n_fft)
    torch.manual_seed(12)
    bins = n_fft // 2 + 1
    spectrum = torch.complex(torch.randn(1, bins, 12), torch.randn(1, bins, 12))
    state = None
    audio_buffer = window_buffer = None
    outputs = []
    offset = 0
    for index, length in enumerate(lengths):
        chunk = spectrum[:, :, offset:offset + length]
        terminal = index == len(lengths) - 1
        actual, state = native(mx.array(chunk.transpose(1, 2).numpy()), overlap=state, streaming=True, last_chunk=terminal)
        expected, audio_buffer, window_buffer = reference(
            chunk, audio_buffer, window_buffer, streaming=True, last_chunk=terminal,
        )
        assert_close(actual, expected)
        outputs.append(actual)
        offset += length
    full, _, _ = reference(spectrum)
    assert_close(mx.concatenate(outputs, axis=1), full)
    assert state is None


@pytest.mark.parametrize("window", [None, 5])
@pytest.mark.parametrize("lengths", [(7,), (1, 2, 1, 3), (2, 2, 3)])
def test_decoder_full_and_streaming_torch_parity(
    window: int | None, lengths: tuple[int, ...]
) -> None:
    from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import Decoder as TorchDecoder

    torch.manual_seed(42)
    config = dict(backbone=backbone_config(window=window), output_dim=8, latent_dim=4)
    reference = TorchDecoder(config["backbone"], output_dim=8, latent_dim=4, patch_size=4).eval()
    native = Decoder(config, 4)
    copy_weights(native, reference)
    latent = torch.randn(1, sum(lengths), 4)
    with torch.no_grad():
        expected, _, _ = reference.low_level_reconstruct(latent, stream_state=(None, None, None), last_chunk=True)
    actual, _ = native(mx.array(latent.numpy()))
    assert_close(actual, expected[:, 0])
    state = None
    outputs = []
    offset = 0
    for index, length in enumerate(lengths):
        chunk = mx.array(latent[:, offset:offset + length].numpy())
        waveform, state = native(chunk, state=state, streaming=True, last_chunk=index == len(lengths) - 1)
        mx.eval(waveform)
        outputs.append(waveform)
        if state is not None and window is not None:
            assert all(entry.keys is None or entry.keys.shape[2] <= window for entry in state.cache)
        offset += length
    assert state is None
    assert_close(mx.concatenate(outputs, axis=1), expected[:, 0])


@pytest.mark.parametrize("window", [None, 5])
def test_reference_encoder_and_fixed_posterior_noise(window: int | None) -> None:
    from sglang_omni.models.ming_omni.talker.audio_vae.vae_modules import Encoder as TorchEncoder

    torch.manual_seed(7)
    config = dict(backbone=dict(backbone_config(window=window), num_hidden_layers=4), input_dim=8, hop_size=8, latent_dim=4)
    reference = TorchEncoder(config["backbone"], input_dim=8, hop_size=8, latent_dim=4, patch_size=4).eval()
    native = Encoder(config, 4)
    copy_weights(native, reference)
    waveform = torch.randn(1, 91)
    with torch.no_grad():
        expected, _ = reference(waveform)
    actual = native(mx.array(waveform.numpy()))
    assert_close(actual, expected)
    vae = AudioVAE(dict(sample_rate=44100, enc_kwargs=config, patch_size=4), component="encoder")
    vae.encoder = native
    mean, scale = expected.chunk(2, dim=-1)
    noise = torch.randn_like(mean)
    assert_close(
        vae.encode_latent(mx.array(waveform.numpy()), noise=mx.array(noise.numpy())),
        mean + (torch.nn.functional.softplus(scale) + 1e-4) * noise,
    )


def test_decoder_slot_cleanup_after_terminal_and_error(monkeypatch: pytest.MonkeyPatch) -> None:
    from sglang_omni.models.ming_tts.mlx.audio_io import MingMlxAudioDecoder

    config = dict(sample_rate=44100, patch_size=4, dec_kwargs=dict(
        backbone=backbone_config(window=5), output_dim=8, latent_dim=4,
    ))
    vae = AudioVAE(config, component="decoder")
    decoder = MingMlxAudioDecoder(vae)
    decoder.prepare_streaming()
    patch = torch.zeros(2, 4)
    first = decoder.run_streaming(slot_ids=(0,), patch_groups=((patch,),), terminal_flags=(False,))
    assert first[0].numel() == 0
    assert 0 in decoder._states
    terminal = decoder.run_streaming(slot_ids=(0,), patch_groups=((patch,),), terminal_flags=(True,))
    assert terminal[0].numel() == 4 * 4 * 8
    assert decoder._states == {}
    decoder.run_streaming(slot_ids=(0,), patch_groups=((patch,),), terminal_flags=(False,))
    decoder.reset_stream_rows((0,))
    assert decoder._states == {}

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("synthetic decoder failure")

    monkeypatch.setattr(type(vae.decoder), "__call__", fail)
    with pytest.raises(RuntimeError, match="synthetic"):
        decoder.run_streaming(slot_ids=(0,), patch_groups=((patch,),), terminal_flags=(False,))
    assert decoder._states == {}
    decoder.close()
    assert not decoder.streaming_ready
