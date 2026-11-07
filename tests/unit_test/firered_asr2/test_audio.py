# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from numpy.typing import NDArray

import sglang_omni.models.firered_asr2.engine_builder as engine_builder
from sglang_omni.config.runtime import resolve_stage_factory_args
from sglang_omni.models.firered_asr2.audio import (
    FireRedAdapter,
    FireRedFeatureExtractor,
    num_audio_tokens,
)
from sglang_omni.models.firered_asr2.config import FireRedASR2PipelineConfig
from sglang_omni.models.firered_asr2.conformer import (
    ConformerEncoder,
    RelPosMultiHeadAttention,
)
from sglang_omni.models.firered_asr2.request_builders import build_prompt_ids
from sglang_omni.models.registry import PIPELINE_CONFIG_REGISTRY


@pytest.mark.parametrize("frames", [5, 7, 8, 9, 15, 16, 17, 100, 301])
def test_audio_length_matches_encoder_and_adapter(frames: int) -> None:
    torch.manual_seed(1)
    encoder = ConformerEncoder(80, 1, 2, 16, pe_maxlen=512).eval()
    adapter = FireRedAdapter(16, 24, downsample_rate=2).eval()
    with torch.no_grad():
        output, lengths, _ = encoder(torch.randn(1, frames, 80), torch.tensor([frames]))
        projected, lengths = adapter(output, lengths)
    assert int(lengths[0]) == num_audio_tokens(frames)
    assert projected.shape == (1, num_audio_tokens(frames), 24)


def test_variable_length_batch_preserves_valid_encoder_outputs() -> None:
    torch.manual_seed(2)
    encoder = ConformerEncoder(80, 2, 2, 16, pe_maxlen=512).eval()
    adapter = FireRedAdapter(16, 24, downsample_rate=2).eval()
    first, second = torch.randn(21, 80), torch.randn(71, 80)
    with torch.no_grad():
        alone, lengths, _ = encoder(first.unsqueeze(0), torch.tensor([21]))
        alone, _ = adapter(alone, lengths)
        batch = torch.nn.utils.rnn.pad_sequence([first, second], batch_first=True)
        together, lengths, _ = encoder(batch, torch.tensor([21, 71]))
        together, lengths = adapter(together, lengths)
    assert lengths.tolist() == [num_audio_tokens(21), num_audio_tokens(71)]
    torch.testing.assert_close(
        alone[0], together[0, : alone.shape[1]], atol=2e-6, rtol=2e-5
    )


def test_frontend_matches_kaldi_and_cmvn() -> None:
    knf = pytest.importorskip("kaldi_native_fbank")
    waveform = np.sin(np.arange(16000, dtype=np.float32) * 0.03) * 0.1
    means, inv_std = np.arange(80) * 0.01, np.ones(80) * 0.5
    actual = FireRedFeatureExtractor(means, inv_std)(waveform)
    options = knf.FbankOptions()
    options.frame_opts.dither = 0.0
    options.mel_opts.num_bins = 80
    options.mel_opts.debug_mel = False
    reference = knf.OnlineFbank(options)
    reference.accept_waveform(16000, (waveform * 32768).tolist())
    expected = np.stack(
        [reference.get_frame(i) for i in range(reference.num_frames_ready)]
    )
    np.testing.assert_allclose(
        actual.numpy(), ((expected - means) * inv_std).astype(np.float32)
    )


@pytest.mark.parametrize(
    "waveform", [np.zeros(1), np.array([np.nan]), np.zeros((2, 100))]
)
def test_frontend_rejects_invalid_audio(waveform: NDArray[np.float32]) -> None:
    pytest.importorskip("kaldi_native_fbank")
    with pytest.raises(ValueError):
        FireRedFeatureExtractor(np.zeros(80), np.ones(80))(waveform)


def test_frontend_rejects_a_checkpoint_with_incompatible_sample_rate() -> None:
    with pytest.raises(ValueError, match="16 kHz"):
        FireRedFeatureExtractor(np.zeros(80), np.ones(80), sample_rate=8000)


def test_prompt_places_exact_audio_span() -> None:
    class Tokenizer:
        unk_token_id = None

        def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
            return [1, 2] if text.endswith("user\n") else [3, 4, 5]

        def convert_tokens_to_ids(self, token: str) -> int:
            return 99

    ids, start, speech = build_prompt_ids(Tokenizer(), 7)
    assert start == 2 and speech == 99
    assert ids == [1, 2] + [99] * 7 + [3, 4, 5]


def test_pipeline_config_resolves_factory_arguments() -> None:
    assert (
        PIPELINE_CONFIG_REGISTRY.get_config("FireRedASR2ForConditionalGeneration")
        is FireRedASR2PipelineConfig
    )
    pipeline = FireRedASR2PipelineConfig(model_path="firered-checkpoint")
    stage = pipeline.stage_named("asr")
    stage.engine.max_running_requests = 4
    arguments = resolve_stage_factory_args(stage, pipeline)
    assert arguments["max_running_requests"] == 4
    assert arguments["server_args_overrides"]["max_running_requests"] == 4
    assert arguments["max_new_tokens"] == 256
    assert arguments["mm_embedding_cache_size_bytes"] == 256 * 1024 * 1024
    assert arguments["enable_encoder_cuda_graph"] is True


def test_attention_respects_loader_device_and_dtype_context() -> None:
    original = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.bfloat16)
        with torch.device("meta"):
            attention = RelPosMultiHeadAttention(2, 16)
        assert all(p.device.type == "meta" for p in attention.parameters())
        assert all(p.dtype == torch.bfloat16 for p in attention.parameters())
    finally:
        torch.set_default_dtype(original)


def test_prefill_chunk_smaller_than_a_cache_page_fails_before_loading(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        engine_builder.AsrEngineBuilder,
        "validate_before_infrastructure",
        lambda self, server_args: None,
    )
    monkeypatch.setattr(
        engine_builder, "resolved_view", lambda server_args: server_args
    )
    builder = engine_builder.FireRedASR2EngineBuilder(
        max_new_tokens=256,
        max_running_requests=8,
        mm_embedding_cache_size_bytes=0,
        enable_encoder_cuda_graph=True,
    )
    with pytest.raises(ValueError, match="KV cache page"):
        builder.validate_before_infrastructure(
            SimpleNamespace(chunked_prefill_size=16, page_size=128)
        )
    builder.validate_before_infrastructure(
        SimpleNamespace(chunked_prefill_size=128, page_size=128)
    )
