# SPDX-License-Identifier: Apache-2.0
"""NPU platform adaptation tests for Higgs TTS."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest
import torch

import sglang_omni.platforms as platforms_mod
import sglang_omni.utils.device as device_mod
from sglang_omni.models.higgs_tts import sampler as higgs_sampler
from sglang_omni.models.higgs_tts import stages as higgs_stages
from sglang_omni.platforms import current_platform


class _FakePlatform:
    def __init__(self, device_type: str, *, npu: bool = False) -> None:
        self.device_type = device_type
        self._npu = npu

    def is_npu(self) -> bool:
        return self._npu

    def enable_code2wav_graph(self) -> bool:
        return not self._npu


def test_sampler_renorm_falls_back_to_torch_on_npu(monkeypatch) -> None:
    monkeypatch.setattr(
        higgs_sampler,
        "current_platform",
        _FakePlatform("npu", npu=True),
    )

    top_k, top_p = higgs_sampler._resolve_renorm_kernels()

    assert top_k is higgs_sampler._top_k_renorm_prob_torch
    assert top_p is higgs_sampler._top_p_renorm_prob_torch


def test_torch_top_k_renorm_keeps_top_k_and_renormalizes() -> None:
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.25, 0.25, 0.25, 0.25]])

    out = higgs_sampler._top_k_renorm_prob_torch(probs, torch.tensor([2, 2]))

    assert torch.allclose(out.sum(dim=-1), torch.ones(2))
    # Row 0: top-2 are {0.3, 0.4}; 0.1/0.2 are filtered out.
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert torch.allclose(out[0, 2:].sum(), torch.ones(()))
    # Row 1: ties at the k-th boundary are kept, so no mass is dropped.
    assert torch.allclose(out[1], torch.full((4,), 0.25))


def test_torch_top_p_renorm_keeps_top_token_and_renormalizes() -> None:
    probs = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.6, 0.4, 0.0, 0.0]])

    out = higgs_sampler._top_p_renorm_prob_torch(probs, torch.tensor([0.6, 0.4]))

    assert torch.allclose(out.sum(dim=-1), torch.ones(2))
    # Row 0 (p=0.6): keep {0.3, 0.4}, filter 0.1/0.2.
    assert out[0, 0] == 0.0 and out[0, 1] == 0.0
    assert out[0, 2] > 0.0 and out[0, 3] > 0.0
    # Row 1 (p=0.4): the highest-probability token is force-kept, the rest cut.
    assert out[1, 0] > 0.0 and out[1, 1] == 0.0
    assert torch.allclose(out[1, 0], torch.ones(()))


def test_batched_sampler_works_with_torch_renorm_fallback(monkeypatch) -> None:
    """The NPU torch renorm path must produce valid in-support samples and keep
    the greedy short-circuit (top_k == 1 → argmax) intact."""
    monkeypatch.setattr(
        higgs_sampler, "_top_k_renorm", higgs_sampler._top_k_renorm_prob_torch
    )
    monkeypatch.setattr(
        higgs_sampler, "_top_p_renorm", higgs_sampler._top_p_renorm_prob_torch
    )

    B, N, V = 3, 8, 64
    torch.manual_seed(0)
    logits = torch.randn(B, N, V)
    top_k_buf = torch.tensor([1, 8, 32])

    codes = higgs_sampler._sample_independent_batched(
        logits,
        temperature=torch.full((B,), 1.0),
        top_p=torch.tensor([0.0, 0.5, 0.9]),
        top_k_buf=top_k_buf,
    )

    assert codes.shape == (B, N)
    assert codes.dtype == torch.long
    assert bool((codes >= 0).all().item()) and bool((codes < V).all().item())
    # Row 0 is greedy (top_k == 1): it must equal argmax over the raw logits.
    assert torch.equal(codes[0], logits[0].argmax(dim=-1))


def test_stage_devices_resolve_from_platform_type() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="unused")

    for stage in config.stages:
        if "device" in stage.factory_args:
            assert stage.factory_args["device"] == current_platform.device_type


def test_vocoder_decode_graph_domain_follows_platform_capability() -> None:
    from sglang_omni.models.higgs_tts.config import HiggsTtsPipelineConfig

    config = HiggsTtsPipelineConfig(model_path="unused")
    vocoder = next(stage for stage in config.stages if stage.name == "vocoder")

    counts = vocoder.factory_args["decode_cuda_graph_frame_counts"]
    if current_platform.enable_code2wav_graph():
        assert counts == tuple(range(1, 151))
    else:
        assert counts == ()


def test_capabilities_torch_compile_reflects_platform() -> None:
    from sglang_omni.models import higgs_tts as higgs_pkg

    assert higgs_pkg.CAPABILITIES.supports_torch_compile is (
        not current_platform.is_npu()
    )


class _FakeTokenizer:
    @staticmethod
    def from_file(path):  # noqa: ANN001
        return object()


def _fake_encoder_codec() -> SimpleNamespace:
    return SimpleNamespace(
        SAMPLE_RATE=24000,
        model=SimpleNamespace(acoustic_encoder=torch.nn.Linear(4, 4)),
        encode_reference=lambda *a, **k: torch.zeros(1),
    )


def _install_audio_encoder_fakes(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    codec: SimpleNamespace,
) -> None:
    monkeypatch.setattr(platforms_mod, "current_platform", platform)
    monkeypatch.setattr(
        device_mod, "resolve_device_spec", lambda device, index=None: device
    )
    monkeypatch.setattr(
        higgs_stages, "resolve_checkpoint", lambda model_path: "ckpt_dir"
    )
    monkeypatch.setattr(higgs_stages, "Tokenizer", _FakeTokenizer)
    monkeypatch.setattr(
        higgs_stages,
        "PreTrainedTokenizerFast",
        lambda tokenizer_object=None: object(),
    )
    monkeypatch.setattr(
        higgs_stages, "HiggsTokenizerAdapter", lambda tokenizer: object()
    )
    monkeypatch.setattr(
        higgs_stages, "get_or_load_codec", lambda ckpt, device, dtype: codec
    )
    monkeypatch.setattr(
        higgs_stages, "ReferenceEncodeService", lambda *a, **k: object()
    )
    monkeypatch.setattr(higgs_stages, "get_speaker_artifact_cache", lambda: object())
    monkeypatch.setattr(higgs_stages, "SimpleScheduler", lambda *a, **k: "scheduler")


def test_audio_encoder_keeps_acoustic_encoder_eager_on_npu(monkeypatch) -> None:
    codec = _fake_encoder_codec()
    original = codec.model.acoustic_encoder
    _install_audio_encoder_fakes(monkeypatch, _FakePlatform("npu", npu=True), codec)

    result = higgs_stages.create_audio_encoder_executor("model", device="npu:0")

    assert result == "scheduler"
    assert codec.model.acoustic_encoder is original


def _fake_vocoder_codec() -> SimpleNamespace:
    return SimpleNamespace(
        model=SimpleNamespace(
            config=SimpleNamespace(num_quantizers=8),
            decode=lambda codes: SimpleNamespace(audio_values=torch.zeros(1)),
        ),
        decode=lambda *a, **k: torch.zeros(16),
        decode_batch=lambda *a, **k: [torch.zeros(16)],
        capture_decode_cuda_graphs=lambda frame_counts: None,
    )


def _install_vocoder_fakes(
    monkeypatch: pytest.MonkeyPatch,
    platform: _FakePlatform,
    codec: SimpleNamespace,
) -> None:
    monkeypatch.setattr(platforms_mod, "current_platform", platform)
    monkeypatch.setattr(
        device_mod, "resolve_device_spec", lambda device, index=None: device
    )
    monkeypatch.setattr(
        higgs_stages, "resolve_checkpoint", lambda model_path: "ckpt_dir"
    )
    monkeypatch.setattr(
        higgs_stages, "get_or_load_codec", lambda ckpt, device, dtype: codec
    )
    monkeypatch.setattr(
        higgs_stages,
        "HiggsStreamingVocoderScheduler",
        lambda *a, **k: "scheduler",
    )


def test_vocoder_compile_decode_falls_back_to_eager_on_npu(
    monkeypatch, caplog
) -> None:
    codec = _fake_vocoder_codec()
    original = codec.model.decode
    _install_vocoder_fakes(monkeypatch, _FakePlatform("npu", npu=True), codec)

    with caplog.at_level(
        logging.WARNING, logger="sglang_omni.models.higgs_tts.stages"
    ):
        result = higgs_stages.create_vocoder_executor(
            "model", device="npu:0", compile_decode=True
        )

    assert result == "scheduler"
    assert codec.model.decode is original
    assert "no torch.compile backend" in caplog.text


def test_vocoder_decode_cuda_graphs_skipped_on_npu(monkeypatch) -> None:
    codec = _fake_vocoder_codec()
    captured: list[tuple] = []
    codec.capture_decode_cuda_graphs = lambda frame_counts: captured.append(
        frame_counts
    )
    _install_vocoder_fakes(monkeypatch, _FakePlatform("npu", npu=True), codec)

    higgs_stages.create_vocoder_executor(
        "model", device="npu:0", decode_cuda_graph_frame_counts=(1, 2, 3)
    )

    assert captured == []
