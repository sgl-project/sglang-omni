# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")

from transformers import WhisperConfig  # noqa: E402
from transformers.models.whisper.modeling_whisper import (  # noqa: E402
    WhisperForConditionalGeneration as HFWhisperForConditionalGeneration,
)

from sglang_omni.models.whisper_asr.mlx.config import ModelConfig  # noqa: E402
from sglang_omni.models.whisper_asr.mlx.model import WhisperMlxModel  # noqa: E402

D_MODEL = 64
LAYERS = 2
DECODER_LAYERS = 2
HEADS = 4
FFN = 128
MELS = 8
POSITIONS = 20
TARGET_POSITIONS = 16
VOCAB = 64


def _tiny_config() -> ModelConfig:
    return ModelConfig(
        d_model=D_MODEL,
        encoder_layers=LAYERS,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=FFN,
        num_mel_bins=MELS,
        max_source_positions=POSITIONS,
        decoder_layers=DECODER_LAYERS,
        decoder_attention_heads=HEADS,
        decoder_ffn_dim=FFN,
        max_target_positions=TARGET_POSITIONS,
        vocab_size=VOCAB,
        pad_token_id=0,
        eos_token_id=2,
        decoder_start_token_id=1,
    )


def _hf_config() -> WhisperConfig:
    return WhisperConfig(
        d_model=D_MODEL,
        encoder_layers=LAYERS,
        encoder_attention_heads=HEADS,
        encoder_ffn_dim=FFN,
        decoder_layers=DECODER_LAYERS,
        decoder_attention_heads=HEADS,
        decoder_ffn_dim=FFN,
        num_mel_bins=MELS,
        max_source_positions=POSITIONS,
        max_target_positions=TARGET_POSITIONS,
        vocab_size=VOCAB,
        activation_function="gelu",
        scale_embedding=False,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        decoder_start_token_id=1,
    )


def _hf_seq2seq() -> HFWhisperForConditionalGeneration:
    torch.manual_seed(0)
    model = HFWhisperForConditionalGeneration(_hf_config()).eval()
    # Whisper initialises the encoder's embed_positions sinusoidally, which is
    # symmetric enough that an off-by-one slice can still pass. Randomise it so
    # the position lookup is actually exercised.
    with torch.no_grad():
        model.model.encoder.embed_positions.weight.copy_(
            torch.randn(POSITIONS, D_MODEL) * 0.05
        )
    return model


def _loaded_seq2seq(hf: HFWhisperForConditionalGeneration) -> WhisperMlxModel:
    model = WhisperMlxModel(_tiny_config())
    raw = {
        name: mx.array(tensor.detach().numpy())
        for name, tensor in hf.state_dict().items()
    }
    model.load_weights(list(model.sanitize(raw).items()), strict=True)
    model.eval()
    return model


def test_encoder_matches_hf_reference() -> None:
    """The MLX encoder must reproduce the Hugging Face reference elementwise."""
    hf = _hf_seq2seq()
    model = _loaded_seq2seq(hf)

    features = torch.randn(2, MELS, POSITIONS * 2)
    with torch.no_grad():
        expected = hf.model.encoder(features).last_hidden_state.numpy()
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


def test_sanitize_drops_tied_output_projection() -> None:
    """proj_out is tied to the decoder embedding, so it must not be loaded."""
    model = WhisperMlxModel(_tiny_config())

    out = model.sanitize(
        {
            "model.encoder.layer_norm.weight": mx.zeros((D_MODEL,)),
            "model.decoder.layers.0.encoder_attn.q_proj.weight": mx.zeros((4, 4)),
            "proj_out.weight": mx.zeros((4, 4)),
        }
    )

    assert "proj_out.weight" not in out
    assert "model.decoder.layers.0.encoder_attn.q_proj.weight" in out


def test_sanitize_rejects_unmappable_conv_weight() -> None:
    """A kernel that fits neither layout is a checkpoint mismatch, not a no-op."""
    model = WhisperMlxModel(_tiny_config())

    with pytest.raises(ValueError, match="cannot map"):
        model.sanitize({"model.encoder.conv1.weight": mx.zeros((D_MODEL, 5, 7))})


def test_decoder_matches_hf_reference() -> None:
    """Encoder, decoder and the tied output projection must match end to end."""
    hf = _hf_seq2seq()
    model = _loaded_seq2seq(hf)

    features = torch.randn(1, MELS, POSITIONS * 2)
    tokens = torch.tensor([[1, 5, 9, 13]])

    with torch.no_grad():
        expected = hf(input_features=features, decoder_input_ids=tokens).logits.numpy()
    got = np.array(
        model.decode(mx.array(tokens.numpy()), model.encode(mx.array(features.numpy())))
    )

    assert got.shape == expected.shape
    assert np.abs(got - expected).max() < 2e-4


def test_incremental_decode_matches_full_sequence() -> None:
    """Stepping one token at a time through the cache must equal one big pass.

    This is what proves the two cache lifetimes are wired correctly: the
    self-attention cache has to grow while the cross-attention cache stays put.
    """
    hf = _hf_seq2seq()
    model = _loaded_seq2seq(hf)

    encoded = model.encode(mx.array(torch.randn(1, MELS, POSITIONS * 2).numpy()))
    tokens = [1, 5, 9, 13]

    full = np.array(model.decode(mx.array([tokens]), encoded))

    cache = model.make_cache()
    stepped = [
        np.array(model.decode(mx.array([[token]]), encoded, cache=cache))[:, 0]
        for token in tokens
    ]

    assert np.abs(np.stack(stepped, axis=1) - full).max() < 2e-4


def test_cross_attention_cache_is_fixed_while_self_attention_grows() -> None:
    """Cross-attention keys come from the encoder, so they must not accumulate."""
    hf = _hf_seq2seq()
    model = _loaded_seq2seq(hf)
    encoded = model.encode(mx.array(torch.randn(1, MELS, POSITIONS * 2).numpy()))

    cache = model.make_cache()
    model.decode(mx.array([[1]]), encoded, cache=cache)
    cross_after_one = cache[0][1][0]
    assert cache[0][0].offset == 1
    assert cross_after_one.shape[2] == POSITIONS

    model.decode(mx.array([[5]]), encoded, cache=cache)
    assert cache[0][0].offset == 2
    assert cache[0][1][0].shape == cross_after_one.shape
    assert mx.array_equal(cache[0][1][0], cross_after_one)
