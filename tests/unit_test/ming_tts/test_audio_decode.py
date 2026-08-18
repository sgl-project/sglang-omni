# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

from sglang_omni.models.ming_omni.talker.audio_vae.configuration_audio_vae import (
    AudioVAEconfig,
)
from sglang_omni.models.ming_omni.talker.audio_vae.modeling_audio_vae import AudioVAE
from sglang_omni.models.ming_tts.audio_decode import (
    MingAudioDecoder,
    MingAudioDecoderState,
    decode_ming_tts_audio_payload,
)
from sglang_omni.models.ming_tts.config import (
    MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
    MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
)
from sglang_omni.models.ming_tts.payload_types import MingTTSState
from sglang_omni.proto import OmniRequest, StagePayload


class _FakeDecoder:
    sample_rate = 44100
    device = torch.device("cpu")
    dtype = torch.float32

    def __init__(self) -> None:
        self.calls = 0

    def decode_nonstreaming(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        assert latents.shape == (0, 2, 3)
        self.calls += 1
        return torch.empty((0,), dtype=torch.float32)


class _RecordingDecoder:
    sample_rate = 44100

    def __init__(self, waveform: torch.Tensor) -> None:
        self.waveform = waveform
        self.calls: list[torch.Tensor] = []

    def decode_nonstreaming(self, latents: torch.Tensor) -> torch.Tensor:
        self.calls.append(latents.clone())
        return self.waveform.clone()


class _FailingAudioVAE(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.empty(()))

    @property
    def decoder(self):
        raise AssertionError("empty latents should not call the AudioVAE decoder")


def test_ming_audio_decoder_skips_audio_vae_for_empty_latents() -> None:
    decoder = MingAudioDecoder(_FailingAudioVAE(), sample_rate=44100)

    waveform = decoder.decode_nonstreaming(torch.empty((0, 2, 3), dtype=torch.float32))

    assert waveform.shape == (0,)
    assert waveform.dtype == torch.float32


def _make_tiny_audio_decoder() -> MingAudioDecoder:
    backbone = {
        "_attn_implementation": "sdpa",
        "attention_dropout": 0.0,
        "hidden_act": "silu",
        "hidden_size": 8,
        "initializer_range": 0.02,
        "intermediate_size": 16,
        "max_position_embeddings": 256,
        "max_window_layers": 0,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "rms_norm_eps": 1e-6,
        "rope_theta": 10_000.0,
        "sliding_window": 64,
        "use_cache": False,
        "use_sliding_window": True,
        "vocab_size": 1,
    }
    config = AudioVAEconfig(
        sample_rate=44100,
        enc_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 4},
            "input_dim": 4,
            "hop_size": 4,
            "latent_dim": 4,
        },
        dec_kwargs={
            "backbone": {**backbone, "num_hidden_layers": 1},
            "output_dim": 4,
            "latent_dim": 4,
        },
        patch_size=4,
    )
    return MingAudioDecoder(AudioVAE(config).eval(), sample_rate=44100)


def _assert_incremental_matches_full_sequence(
    chunk_patches: tuple[int, ...],
) -> list[torch.Tensor]:
    assert chunk_patches and all(count > 0 for count in chunk_patches)
    num_patches = sum(chunk_patches)

    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(0)
        decoder = _make_tiny_audio_decoder()
        latents = torch.randn(num_patches, 4, 4)

        full = decoder.decode_nonstreaming(latents)
        state = MingAudioDecoderState()
        incremental_parts = []
        patch_start = 0
        for chunk_index, patch_count in enumerate(chunk_patches):
            patch_end = patch_start + patch_count
            incremental_parts.append(
                decoder.decode_streaming_step(
                    latents[patch_start:patch_end].flatten(0, 1),
                    state=state,
                    is_last=chunk_index == len(chunk_patches) - 1,
                )
            )
            patch_start = patch_end

    incremental = torch.cat(incremental_parts)
    assert full.numel() > 0
    assert incremental.shape == full.shape
    torch.testing.assert_close(incremental, full, rtol=1e-4, atol=1e-6)
    return incremental_parts


def test_ming_audio_decoder_incremental_matches_full_sequence_on_cpu() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence((1, 2, 2))

    assert incremental_parts[0].numel() == 0
    assert incremental_parts[-1].numel() > 0


def test_ming_audio_decoder_single_terminal_patch_matches_full_on_cpu() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence((1,))

    assert incremental_parts[0].numel() > 0


def test_ming_audio_decoder_matches_full_after_window_saturation_on_cpu() -> None:
    # Eleven patches upsample to 176 decoder frames. With sliding_window=64,
    # the fifth non-terminal call runs after the cache has crossed the window.
    incremental_parts = _assert_incremental_matches_full_sequence((1, 2, 2, 2, 2, 2))

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def test_ming_audio_decoder_matches_full_at_shipped_cadence_on_cpu() -> None:
    initial = MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES
    steady = MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES
    incremental_parts = _assert_incremental_matches_full_sequence(
        (initial, steady, steady, steady)
    )

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def test_ming_audio_decoder_matches_full_when_initial_exceeds_steady() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence((4, 2, 2))

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


def test_ming_audio_decoder_short_terminal_group_matches_full_on_cpu() -> None:
    incremental_parts = _assert_incremental_matches_full_sequence(
        (
            MING_TTS_DEFAULT_INITIAL_CHUNK_PATCHES,
            MING_TTS_DEFAULT_STEADY_CHUNK_PATCHES,
            2,
        )
    )

    assert incremental_parts[0].numel() == 0
    assert all(part.numel() > 0 for part in incremental_parts[1:])


@pytest.mark.parametrize("keep_latents", [False, True])
def test_ming_tts_nonstreaming_payload_decodes_full_sequence_once(
    keep_latents: bool,
) -> None:
    latents = torch.arange(12, dtype=torch.float32).reshape(2, 2, 3)
    state = MingTTSState(
        text="hello",
        prompt_tokens=3,
        completion_tokens=2,
        generated_latents=latents,
    )
    payload = StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    waveform = torch.tensor([0.25, -0.5, 0.75, -1.0], dtype=torch.float32)
    decoder = _RecordingDecoder(waveform)

    result = decode_ming_tts_audio_payload(
        payload,
        decoder,
        keep_latents=keep_latents,
    )

    assert len(decoder.calls) == 1
    torch.testing.assert_close(decoder.calls[0], latents)
    restored = MingTTSState.from_dict(result.data)
    if keep_latents:
        torch.testing.assert_close(restored.generated_latents, latents)
    else:
        assert restored.generated_latents is None
    assert restored.sample_rate == 44100
    assert restored.duration_s == pytest.approx(waveform.numel() / 44100)
    assert result.data["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "total_tokens": 5,
    }
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    np.testing.assert_array_equal(audio, waveform.numpy())


def test_ming_tts_audio_decode_accepts_empty_generated_latents() -> None:
    state = MingTTSState(
        text="hello",
        prompt_tokens=3,
        completion_tokens=0,
        generated_latents=torch.empty((0, 2, 3), dtype=torch.float32),
    )
    payload = StagePayload(
        request_id="req-ming-tts",
        request=OmniRequest(inputs="hello"),
        data=state.to_dict(),
    )
    decoder = _FakeDecoder()

    result = decode_ming_tts_audio_payload(payload, decoder)

    assert decoder.calls == 1
    assert result.data["sample_rate"] == 44100
    assert result.data["duration_s"] == 0.0
    assert result.data["audio_waveform_shape"] == [0]
    audio = np.frombuffer(result.data["audio_waveform"], dtype=np.float32)
    assert audio.tolist() == []
