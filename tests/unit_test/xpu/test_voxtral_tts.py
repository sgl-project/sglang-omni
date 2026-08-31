# SPDX-License-Identifier: Apache-2.0
"""Intel XPU policy tests for Voxtral TTS."""

from __future__ import annotations

import pytest
import torch

from sglang_omni import platforms
from sglang_omni.models.voxtral_tts import audio_tokenizer
from sglang_omni.models.voxtral_tts.pipeline import engine_builder, stages


def _mock_xpu_platform(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: True)
    monkeypatch.setattr(platforms.current_platform, "device_type", "xpu", raising=False)


def test_voxtral_tts_xpu_defaults_to_eager_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)

    defaults = engine_builder.VoxtralTtsEngineBuilder().generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["disable_cuda_graph"] is True
    assert defaults["enable_torch_compile"] is False


def test_voxtral_tts_non_xpu_generation_defaults_are_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(platforms.current_platform, "is_xpu", lambda: False)

    defaults = engine_builder.VoxtralTtsEngineBuilder().generation_defaults(
        dtype="bfloat16"
    )

    assert defaults["disable_cuda_graph"] is False
    assert defaults["enable_torch_compile"] is True


def test_voxtral_tts_vocoder_follows_xpu_placement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _mock_xpu_platform(monkeypatch)
    seen_devices: list[str] = []

    monkeypatch.setattr(stages, "_resolve_checkpoint", lambda model_path: model_path)
    monkeypatch.setattr(
        stages,
        "_load_audio_tokenizer",
        lambda checkpoint_dir, state_dict, device: (
            seen_devices.append(device) or object()
        ),
    )

    stages.create_vocoder_executor("model", gpu_id=2)
    stages.create_vocoder_executor("model", device="cpu", gpu_id=2)
    stages.create_vocoder_executor("model", device="cuda:0", gpu_id=2)

    assert seen_devices == ["xpu:2", "cpu", "cuda:2"]


def test_voxtral_native_attention_accepts_noncontiguous_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = audio_tokenizer.AudioTokenizerArgs(
        dim=8,
        hidden_dim=16,
        head_dim=4,
        n_heads=2,
        n_kv_heads=2,
        qk_norm=False,
        layer_scale=False,
    )
    attention = audio_tokenizer.Attention(args, layer_id=0)

    def fake_native_attention(xq, xk, xv):
        del xk, xv
        output = torch.zeros(
            xq.shape[0], xq.shape[1], xq.shape[3], xq.shape[2]
        ).transpose(-1, -2)
        assert not output.is_contiguous()
        return output

    monkeypatch.setattr(audio_tokenizer, "HAS_FLASH_ATTN", False)
    monkeypatch.setattr(attention, "_native_attention", fake_native_attention)

    output = attention(torch.zeros(3, args.dim))

    assert output.shape == (3, args.dim)
