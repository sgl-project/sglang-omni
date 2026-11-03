# SPDX-License-Identifier: Apache-2.0
"""Intel XPU adaptation tests for Higgs TTS.

The platform case needs no device. The accelerator cases run Higgs stages and
the sampler on an Intel XPU with a seeded random codec, and skip elsewhere.
"""

from __future__ import annotations

import copy
import math
from types import SimpleNamespace

import pytest
import torch
from torch._dynamo.utils import counters
from transformers import HiggsAudioV2TokenizerConfig, HiggsAudioV2TokenizerModel

from sglang_omni.models.higgs_tts import audio_codec
from sglang_omni.models.higgs_tts import config as higgs_config
from sglang_omni.models.higgs_tts import stages as higgs_stages
from sglang_omni.models.higgs_tts.sampler import (
    K_MAX,
    HiggsBatchedSamplerState,
    batched_step,
    sample_independent_batched,
)
from sglang_omni.platforms.xpu import XPUOmniPlatform
from tests.unit_test.fixtures.accelerator import require_device_streams

# The bf16 decode error measured on Arc Pro B60 peaked at 1.4% of the clip's peak.
DECODE_ERROR_PEAK_FRACTION = 0.05
# Compiled and eager bf16 reference codes agreed on at least 98.2% of 12 clips.
MIN_COMPILED_CODE_AGREEMENT = 0.95


@pytest.fixture
def xpu_device() -> torch.device:
    device = require_device_streams()
    if device.type != "xpu":
        pytest.skip(f"needs an Intel XPU, found {device.type}")
    else:
        pass
    return device


def random_codec_model(seed: int) -> HiggsAudioV2TokenizerModel:
    """The bundled codec architecture with seeded weights and codebooks.

    Transformers fills every codebook with one constant vector, so without
    reseeding them decode would return the same audio for any codes.
    """
    config = HiggsAudioV2TokenizerConfig.from_json_file(
        audio_codec._BUNDLED_CODEC_CONFIG_PATH  # noqa: leading-underscore  # production name
    )
    generator = torch.Generator().manual_seed(seed)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = HiggsAudioV2TokenizerModel(config).eval().requires_grad_(False)
    for quantizer in model.quantizer.quantizers:
        quantizer.codebook.embed.copy_(
            torch.randn(quantizer.codebook.embed.shape, generator=generator)
        )
    return model


def serve_random_codec(
    monkeypatch: pytest.MonkeyPatch, model: HiggsAudioV2TokenizerModel
) -> list[audio_codec.HiggsAudioCodec]:
    """Hand the stage factories the random codec on the device and dtype they ask for."""
    built: list[audio_codec.HiggsAudioCodec] = []

    def load_codec(path: str, device: str, dtype: str) -> audio_codec.HiggsAudioCodec:
        built.append(
            audio_codec.HiggsAudioCodec(
                copy.deepcopy(model).to(device=device, dtype=getattr(torch, dtype)),
                device=torch.device(device),
            )
        )
        return built[-1]

    monkeypatch.setattr(
        higgs_stages, "resolve_checkpoint", lambda model_path: model_path
    )
    monkeypatch.setattr(higgs_stages, "get_or_load_codec", load_codec)
    return built


def test_xpu_keeps_the_default_vocoder_decode_eager(monkeypatch) -> None:
    """XPU declines codec decode graphs, so its vocoder captures no shapes at startup."""
    monkeypatch.setattr(higgs_config, "current_platform", XPUOmniPlatform())
    config = higgs_config.HiggsTtsPipelineConfig(model_path="unused")

    vocoder_kwargs = config.stage_factory_kwargs("vocoder")

    assert vocoder_kwargs["decode_cuda_graph_frame_counts"] == ()


@pytest.mark.accelerator
def test_vocoder_decodes_single_and_batched_windows_like_cpu(
    monkeypatch, xpu_device
) -> None:
    """The vocoder XPU serves decodes eagerly, and its windows match a CPU fp32 decode."""
    model = random_codec_model(seed=0)
    serve_random_codec(monkeypatch, model)
    vocoder_kwargs = higgs_config.HiggsTtsPipelineConfig(
        model_path="unused"
    ).stage_factory_kwargs("vocoder")

    vocoder = higgs_stages.create_vocoder_executor(
        "model", device=xpu_device.type, gpu_id=xpu_device.index, **vocoder_kwargs
    )

    assert vocoder.codec.decode_cuda_graphs == {}
    cpu_codec = audio_codec.HiggsAudioCodec(model, device=torch.device("cpu"))
    generator = torch.Generator().manual_seed(1)
    windows = [
        torch.randint(
            0,
            model.config.codebook_size,
            (frame_count, model.config.num_quantizers),
            generator=generator,
        )
        for frame_count in (7, 7, 150)
    ]
    expected = [cpu_codec.decode(window) for window in windows]
    assert not torch.allclose(expected[0], expected[1])
    for decoded in (
        [vocoder.codec.decode(window) for window in windows],
        vocoder.codec.decode_batch(windows),
    ):
        for actual, reference, window in zip(decoded, expected, windows, strict=True):
            assert actual.shape == (window.shape[0] * vocoder.samples_per_frame,)
            assert torch.isfinite(actual).all()
            error = (actual.float() - reference).abs().max()
            assert error <= DECODE_ERROR_PEAK_FRACTION * reference.abs().max()


@pytest.mark.accelerator
def test_audio_encoder_compiles_reference_encoding(monkeypatch, xpu_device) -> None:
    """The stage compiles the acoustic encoder on XPU, and its codes track eager."""
    model = random_codec_model(seed=0)
    built = serve_random_codec(monkeypatch, model)
    monkeypatch.setattr(
        higgs_stages, "Tokenizer", SimpleNamespace(from_file=lambda path: object())
    )
    monkeypatch.setattr(
        higgs_stages, "PreTrainedTokenizerFast", lambda tokenizer_object: object()
    )
    monkeypatch.setattr(
        higgs_stages, "HiggsTokenizerAdapter", lambda tokenizer: object()
    )

    torch.compiler.reset()
    compiled_frames = counters["frames"]["ok"]

    higgs_stages.create_audio_encoder_executor(
        "model", device=xpu_device.type, gpu_id=xpu_device.index
    )

    # Dynamo compiles on first call, so only a compiled warmup encode adds a frame.
    assert counters["frames"]["ok"] > compiled_frames
    (codec,) = built
    eager_codec = audio_codec.HiggsAudioCodec(
        copy.deepcopy(model).to(device=xpu_device, dtype=codec.dtype),
        device=xpu_device,
    )
    generator = torch.Generator().manual_seed(1)
    for seconds in (2, 3):
        clip = 0.1 * torch.randn(codec.SAMPLE_RATE * seconds, generator=generator)
        compiled_codes = codec.encode_reference(clip)
        eager_codes = eager_codec.encode_reference(clip)
        assert compiled_codes.shape == eager_codes.shape
        agreement = (compiled_codes == eager_codes).float().mean().item()
        assert agreement >= MIN_COMPILED_CODE_AGREEMENT


@pytest.mark.accelerator
def test_seeded_draws_depend_only_on_the_row_seed_and_step(xpu_device) -> None:
    """Off CPU the seeded sampler hashes with Triton, which XPU runs too."""
    generator = torch.Generator(device=xpu_device).manual_seed(0)
    logits = torch.randn(3, 8, 64, generator=generator, device=xpu_device)

    def draw(seeds: list[int], step: int, rows: slice = slice(None)) -> torch.Tensor:
        row_logits = logits[rows]
        return sample_independent_batched(
            row_logits,
            temperature=torch.ones(row_logits.shape[0], device=xpu_device),
            top_p=None,
            seeds_B=torch.tensor(seeds, device=xpu_device),
            step_B=torch.full(
                (row_logits.shape[0],), step, dtype=torch.long, device=xpu_device
            ),
        )

    codes = draw([55, 77, 99], step=3)

    assert torch.equal(codes, draw([55, 77, 99], step=3))
    assert not torch.equal(codes, draw([56, 78, 100], step=3))
    assert not torch.equal(codes, draw([55, 77, 99], step=4))
    assert torch.equal(draw([55], step=3, rows=slice(0, 1))[0], codes[0])


@pytest.mark.accelerator
def test_seeded_draws_keep_the_sampling_distribution(xpu_device) -> None:
    batch = 20_000
    probs = torch.tensor([0.9, 0.1], device=xpu_device)
    logits = probs.log().view(1, 1, 2).expand(batch, 1, 2).contiguous()

    sampled = sample_independent_batched(
        logits,
        temperature=torch.ones(batch, device=xpu_device),
        top_p=None,
        seeds_B=torch.arange(1, batch + 1, device=xpu_device),
        step_B=torch.zeros(batch, dtype=torch.long, device=xpu_device),
    )

    token0_rate = (sampled[:, 0] == 0).float().mean().item()
    expected_rate = probs[0].item()
    # One draw per row, so a correct sampler stays within 4 standard errors.
    standard_error = math.sqrt(expected_rate * (1 - expected_rate) / batch)
    assert abs(token0_rate - expected_rate) <= 4 * standard_error


@pytest.mark.accelerator
def test_batched_step_keeps_each_rows_top_k(xpu_device) -> None:
    """XPU renormalizes with the torch fallback; rows must not share one top_k."""
    num_codebooks, vocab_size = 8, 1026
    pool = HiggsBatchedSamplerState(2, num_codebooks, device=xpu_device)
    pool.delay_count.fill_(num_codebooks)
    allowed_codes = (
        torch.tensor([2, 3, 5, 7, 11, 13, 17, 19, 23, 29], device=xpu_device),
        torch.tensor([101, 103, 107, 109, 113], device=xpu_device),
    )
    logits = torch.full((2, num_codebooks, vocab_size), -1e9, device=xpu_device)
    for row, codes in enumerate(allowed_codes):
        logits[row, :, codes] = 5.0

    sampled = batched_step(
        logits,
        pool,
        torch.arange(2, device=xpu_device),
        temperature=torch.ones(2, device=xpu_device),
        top_k_buf=torch.tensor([K_MAX, 5], dtype=torch.long, device=xpu_device),
    )

    assert sampled.shape == (2, num_codebooks)
    for row, codes in enumerate(allowed_codes):
        assert torch.isin(sampled[row], codes).all(), sampled[row].tolist()
