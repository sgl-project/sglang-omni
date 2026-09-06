# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from transformers import WhisperConfig  # noqa: E402
from transformers.models.whisper.modeling_whisper import (  # noqa: E402
    WhisperEncoder as HFWhisperEncoder,
)

from sglang_omni.models.whisper_asr.mlx.config import ModelConfig  # noqa: E402
from sglang_omni.models.whisper_asr.mlx.model import WhisperMlxModel  # noqa: E402

D_MODEL = 64
LAYERS = 2
HEADS = 4
FFN = 128
MELS = 8
POSITIONS = 20


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=D_MODEL,
        encoder_layers=LAYERS,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=FFN,
        num_mel_bins=MELS,
        max_source_positions=POSITIONS,
        decoder_layers=1,
        decoder_attention_heads=HEADS,
        decoder_ffn_dim=FFN,
        max_target_positions=16,
        vocab_size=64,
        pad_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=1,
    )


def _hf_encoder() -> HFWhisperEncoder:
    torch.manual_seed(0)
    config = WhisperConfig(
        d_model=D_MODEL,
        encoder_layers=LAYERS,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=FFN,
        decoder_layers=1,
        decoder_attention_heads=HEADS,
        decoder_ffn_dim=FFN,
        num_mel_bins=MELS,
        max_source_positions=POSITIONS,
        max_target_positions=16,
        vocab_size=64,
        activation_function="gelu",
        scale_embedding=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )
    encoder = HFWhisperEncoder(config).eval()
    # Whisper initialises embed_positions sinusoidally, which is symmetric
    # enough that an off-by-one slice can still pass. Randomise it so the
    # position lookup is actually exercised.
    with torch.no_grad():
        encoder.embed_positions.weight.copy_(torch.randn(POSITIONS, D_MODEL) * 0.05)
    return encoder


def _loaded_model(encoder: HFWhisperEncoder) -> WhisperMlxModel:
    model = WhisperMlxModel(_tiny_config())
    raw = {
        f"model.encoder.{name}": mx.array(tensor.detach().numpy())
        for name, tensor in encoder.state_dict().items()
    }
    model.load_weights(list(model.sanitize(raw).items()), strict=True)
    model.eval()
    return model


def test_encoder_matches_hf_reference() -> None:
    """The MLX encoder must reproduce the Hugging Face reference elementwise."""
    encoder = _hf_encoder()
    model = _loaded_model(encoder)

    features = torch.randn(2, MELS, POSITIONS * 2)
    with torch.no_grad():
        expected = encoder(features).last_hidden_state.numpy()
    got = np.array(model.encode(mx.array(features.numpy())))

    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 2e-4


def test_encoder_halves_the_frame_count() -> None:
    """conv2 has stride 2, so 30 s of mel frames become max_source_positions."""
    model = WhisperMlxModel(_tiny_config())
    out = model.encode(mx.zeros((1, MELS, POSITIONS * 2)))
    assert out.shape == (1, POSITIONS, D_MODEL)


def test_key_projection_has_no_bias() -> None:
    """Whisper checkpoints ship k_proj without a bias; the module must match."""
    model = WhisperMlxModel(_tiny_config())
    attn = model.encoder.layers[0].self_attn
    assert "bias" not in attn.k_proj
    assert "bias" in attn.q_proj
    assert "bias" in attn.v_proj


def test_sanitize_transposes_conv1d_weights() -> None:
    """PyTorch stores (out, in, kernel); MLX wants (out, kernel, in)."""
    model = WhisperMlxModel(_tiny_config())
    torch_weight = mx.zeros((D_MODEL, MELS, 3))

    out = model.sanitize({"model.encoder.conv1.weight": torch_weight})

    assert out["model.encoder.conv1.weight"].shape == (D_MODEL, 3, MELS)


def test_sanitize_is_idempotent() -> None:
    """Re-running sanitize on converted weights must not transpose them back."""
    model = WhisperMlxModel(_tiny_config())
    key = "model.encoder.conv1.weight"

    once = model.sanitize({key: mx.zeros((D_MODEL, MELS, 3))})
    twice = model.sanitize(once)

    assert twice[key].shape == once[key].shape == (D_MODEL, 3, MELS)


def test_sanitize_drops_decoder_weights() -> None:
    """The decoder is not implemented yet, so its tensors must not be loaded."""
    model = WhisperMlxModel(_tiny_config())

    out = model.sanitize(
        {
            "model.encoder.layer_norm.weight": mx.zeros((D_MODEL,)),
            "model.decoder.layers.0.encoder_attn.q_proj.weight": mx.zeros((4, 4)),
            "proj_out.weight": mx.zeros((4, 4)),
        }
    )

    assert list(out) == ["model.encoder.layer_norm.weight"]


def test_sanitize_rejects_unmappable_conv_weight() -> None:
    """A kernel that fits neither layout is a checkpoint mismatch, not a no-op."""
    model = WhisperMlxModel(_tiny_config())

    with pytest.raises(ValueError, match="cannot map"):
        model.sanitize({"model.encoder.conv1.weight": mx.zeros((D_MODEL, 5, 7))})
