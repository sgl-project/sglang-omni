# SPDX-License-Identifier: Apache-2.0
"""Native frontend and encoder numerical contracts."""

import librosa
import pytest
import torch
from torch import nn

from sglang_omni.models.nemotron_diarization.model import SortformerModules
from sglang_omni.models.nemotron_transcribe.fp8_linear import (
    FP8_DTYPE,
    FP8_MAX,
    Fp8Linear,
    dequantize_fp8_linears,
    fp8_gemm_supported,
)
from sglang_omni.models.nemotron_transcribe.hf_config import num_audio_tokens
from sglang_omni.models.nemotron_transcribe.mel_features import (
    LogMelFrontend,
    normalize_per_feature,
)
from sglang_omni.models.nemotron_transcribe.parallel_expert import ParallelExpertEncoder
from sglang_omni.models.nemotron_transcribe.perception import (
    AudioPerception,
    MelPreprocessor,
    load_perception_weights,
)
from sglang_omni.models.nemotron_transcribe.sortformer_diarizer import (
    SortformerDiarizer,
)
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    FeatureStacking,
    RotaryEmbedding,
    SpeechTransformerEncoder,
)


def test_frontend_filterbank_matches_slaney_reference() -> None:
    frontend = LogMelFrontend()
    reference = librosa.filters.mel(
        sr=16000, n_fft=512, n_mels=128, htk=False, norm="slaney"
    )
    torch.testing.assert_close(
        frontend.fb.squeeze(0), torch.from_numpy(reference), rtol=1e-4, atol=1e-6
    )


def test_frontend_lengths_agree_with_audio_placeholders() -> None:
    frontend = LogMelFrontend()
    signals = torch.randn(2, 16000)
    signals[1, 8000:] = 0
    mel, lengths = frontend(signals, torch.tensor([16000, 8000]))
    assert lengths.tolist() == [100, 50]
    assert torch.isfinite(mel).all()
    assert torch.count_nonzero(mel[1, :, 50:]) == 0
    stacking = FeatureStacking(8, 128, 16)
    encoded, encoded_lengths = stacking(mel, lengths)
    assert encoded_lengths.tolist() == [num_audio_tokens(16000), num_audio_tokens(8000)]
    assert encoded.shape[1] >= int(encoded_lengths.max())


def test_normalization_ignores_padding_and_preserves_finite_silence() -> None:
    features = torch.tensor([[[1.0, 3.0, 5.0, 1000.0]], [[0.0, 0.0, 1000.0, 1000.0]]])
    normalized = normalize_per_feature(features, torch.tensor([3, 2]))
    torch.testing.assert_close(normalized[0, 0, :3].mean(), torch.tensor(0.0))
    torch.testing.assert_close(
        normalized[0, 0, :3].std(), torch.tensor(1.0), rtol=1e-4, atol=1e-4
    )
    assert normalized[0, 0, 3] == 0
    assert torch.count_nonzero(normalized[1]) == 0
    assert torch.isfinite(normalized).all()


def test_frontend_stays_fp32_when_built_under_bfloat16() -> None:
    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        perception = AudioPerception(
            MelPreprocessor(LogMelFrontend(features=4)),
            small_parallel_encoder(),
            output_width=16,
        )
    finally:
        torch.set_default_dtype(previous_dtype)

    featurizer = perception.preprocessor.featurizer
    assert featurizer.fb.dtype == torch.bfloat16
    checkpoint = {
        name: tensor.float() for name, tensor in perception.state_dict().items()
    }
    checkpoint["preprocessor.featurizer.fb"].view(-1)[0] = 1.0001
    load_perception_weights(perception, checkpoint, torch.bfloat16)
    mel, _ = perception.preprocessor(torch.zeros(1, 16000), torch.tensor([16000]))

    assert mel.dtype == torch.float32
    torch.testing.assert_close(featurizer.fb, checkpoint["preprocessor.featurizer.fb"])
    assert perception.encoder.asr_encoder.pre_encode.proj.weight.dtype == torch.bfloat16
    assert perception.proj.weight.dtype == torch.bfloat16


def test_fp8_encoder_linears_keep_the_checkpoint_fp8_weights() -> None:
    torch.manual_seed(2)
    perception = small_perception()
    checkpoint, names = fp8_checkpoint(perception)
    load_perception_weights(perception, checkpoint, torch.bfloat16)

    assert len(names) == 4
    for name in names:
        linear = perception.get_submodule(name)
        assert isinstance(linear, Fp8Linear)
        assert torch.equal(
            linear.weight.view(torch.uint8),
            checkpoint[f"{name}.weight"].view(torch.uint8),
        )
        assert linear.weight_scale.item() == checkpoint[f"{name}.weight_scale"].item()
        assert linear.input_scale.item() == pytest.approx(0.01)
    assert isinstance(perception.encoder.asr_encoder.pre_encode.proj, nn.Linear)
    assert perception.proj.weight.dtype == torch.bfloat16


def test_dequantized_fp8_linears_fold_the_weight_scale() -> None:
    torch.manual_seed(3)
    perception = small_perception()
    checkpoint, names = fp8_checkpoint(perception)
    load_perception_weights(perception, checkpoint, torch.bfloat16)

    assert dequantize_fp8_linears(perception) == 4
    for name in names:
        linear = perception.get_submodule(name)
        expected = (
            checkpoint[f"{name}.weight"].float() * checkpoint[f"{name}.weight_scale"]
        ).to(torch.bfloat16)
        assert type(linear) is nn.Linear
        assert linear.weight.dtype == torch.bfloat16
        torch.testing.assert_close(linear.weight, expected, rtol=0, atol=0)


def test_fp8_perception_weight_without_scale_is_rejected() -> None:
    perception = small_perception()
    checkpoint = dict(perception.state_dict())
    checkpoint["proj.weight"] = checkpoint["proj.weight"].to(FP8_DTYPE)
    with pytest.raises(ValueError, match="without a weight_scale"):
        load_perception_weights(perception, checkpoint, torch.bfloat16)


def test_fp8_linear_without_an_input_scale_is_rejected() -> None:
    perception = small_perception()
    checkpoint, names = fp8_checkpoint(perception)
    del checkpoint[f"{names[0]}.input_scale"]
    with pytest.raises(ValueError, match="one input_scale"):
        load_perception_weights(perception, checkpoint, torch.bfloat16)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not fp8_gemm_supported(torch.device("cuda")),
    reason="FP8 GEMMs need a CUDA GPU with compute capability 8.9 or newer",
)
@torch.no_grad()
def test_fp8_linear_matches_a_dequantized_reference() -> None:
    torch.manual_seed(5)
    dense = nn.Linear(1280, 3840, device="cuda", dtype=torch.bfloat16)
    fp8 = Fp8Linear.replacing(dense)
    hidden = torch.randn(2, 37, 1280, device="cuda", dtype=torch.bfloat16)
    weight_scale = dense.weight.float().abs().max() / FP8_MAX
    input_scale = hidden.float().abs().max() / FP8_MAX
    fp8.weight.copy_((dense.weight.float() / weight_scale).to(FP8_DTYPE))
    fp8.weight_scale.copy_(weight_scale)
    fp8.input_scale.copy_(input_scale)
    fp8.bias.copy_(dense.bias)

    actual = fp8(hidden)
    quantized = (hidden / input_scale).clamp(-FP8_MAX, FP8_MAX).to(FP8_DTYPE)
    reference = torch.nn.functional.linear(
        quantized.float() * input_scale,
        fp8.weight.float() * weight_scale,
        dense.bias.float(),
    )

    assert actual.shape == (2, 37, 3840)
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual.float(), reference, rtol=1e-2, atol=1e-2)


def fp8_checkpoint(
    perception: AudioPerception,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """ModelOpt's FP8 export: per-tensor FP8 weights and scales on the ASR layers."""
    checkpoint = dict(perception.state_dict())
    layers = perception.encoder.asr_encoder.layers
    names = [
        f"encoder.asr_encoder.layers.{name}"
        for name, module in layers.named_modules()
        if isinstance(module, nn.Linear)
    ]
    for name in names:
        weight = checkpoint[f"{name}.weight"].float()
        scale = weight.abs().max() / FP8_MAX
        checkpoint[f"{name}.weight"] = (weight / scale).to(FP8_DTYPE)
        checkpoint[f"{name}.weight_scale"] = scale
        checkpoint[f"{name}.input_scale"] = torch.tensor(0.01)
    return checkpoint, names


def small_perception() -> AudioPerception:
    return AudioPerception(
        MelPreprocessor(LogMelFrontend(features=4)),
        small_parallel_encoder(),
        output_width=16,
    )


def small_parallel_encoder() -> ParallelExpertEncoder:
    asr = SpeechTransformerEncoder(
        feat_in=4,
        d_model=16,
        n_heads=2,
        n_layers=1,
        subsampling_factor=2,
        drop_rate=0.0,
    )
    diarization_encoder = SpeechTransformerEncoder(
        feat_in=4, d_model=8, n_heads=2, n_layers=1, subsampling_factor=2, drop_rate=0.0
    )
    speaker_modules = SortformerModules(
        fc_d_model=8, tf_d_model=4, speakers=2, upsample=2
    )
    diarizer = SortformerDiarizer(
        diarization_encoder, speaker_modules, output_subsampling_factor=2
    )
    return ParallelExpertEncoder(
        asr,
        diarizer,
        asr_normalize="per_feature",
        diar_normalize=None,
        speaker_activity_threshold=0.5,
        speaker_kernel_scale=1.0,
        online_frames=4,
        left_context_frames=1,
        right_context_frames=1,
        cache_limit=4,
        recent_limit=2,
        cache_update_frames=4,
    )


def complex_rotary_reference(
    tokens: torch.Tensor, positions: torch.Tensor
) -> torch.Tensor:
    frequencies = 10000.0 ** (-torch.arange(0, 4, 2).float() / 4)
    phases = positions[:, None] * frequencies
    rotated = torch.complex(tokens[..., :2], tokens[..., 2:4]) * torch.polar(
        torch.ones_like(phases), phases
    )
    return torch.cat([rotated.real, rotated.imag, tokens[..., 4:]], dim=-1)


def test_rotary_matches_complex_reference_after_extension() -> None:
    torch.manual_seed(0)
    rotary = RotaryEmbedding(8, 0.5, 10000.0, 2)
    queries = torch.randn(2, 2, 5, 8)
    keys = torch.randn(2, 2, 5, 8)
    actual_queries, actual_keys = rotary.apply_rotary(queries, keys)
    torch.testing.assert_close(
        actual_queries, complex_rotary_reference(queries, torch.arange(5))
    )
    torch.testing.assert_close(
        actual_keys, complex_rotary_reference(keys, torch.arange(5))
    )


def test_transformer_batch_padding_matches_individual_inference() -> None:
    torch.manual_seed(1)
    encoder = SpeechTransformerEncoder(
        feat_in=4,
        d_model=16,
        n_heads=2,
        n_layers=2,
        subsampling_factor=2,
        drop_rate=0.0,
        qk_norm=True,
    ).eval()
    signals = torch.randn(2, 4, 10)
    signals[1, :, 5:] = 0
    with torch.no_grad():
        batched, lengths = encoder(signals, torch.tensor([10, 5]))
        individual, individual_lengths = encoder(signals[1:2, :, :5], torch.tensor([5]))
    assert lengths.tolist() == [5, 3]
    assert individual_lengths.tolist() == [3]
    torch.testing.assert_close(batched[1, :, :3], individual[0], rtol=1e-5, atol=1e-5)


@torch.no_grad()
def test_windowed_speaker_fusion_handles_multiple_chunks_and_padding() -> None:
    torch.manual_seed(4)
    encoder = small_parallel_encoder().eval()
    mel = torch.randn(2, 4, 25)
    mel[1, :, 17:] = 0
    batched, lengths = encoder(mel, torch.tensor([25, 17]))
    individual, individual_lengths = encoder(mel[1:2, :, :17], torch.tensor([17]))
    assert lengths.tolist() == [13, 9]
    assert individual_lengths.tolist() == [9]
    assert torch.isfinite(batched).all()
    torch.testing.assert_close(batched[1, :, :9], individual[0], rtol=1e-4, atol=1e-4)
