# SPDX-License-Identifier: Apache-2.0
"""Parity of the MLX Parakeet port against the Transformers implementation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
transformers = pytest.importorskip("transformers")
parakeet = pytest.importorskip("transformers.models.parakeet")

from sglang_omni.models.parakeet.mlx.model import (  # noqa: E402
    ParakeetMlxConfig,
    ParakeetModel,
    subsampled_lengths,
    torch_layout_to_mlx,
)

TINY_ENCODER = {
    "hidden_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 2,
    "intermediate_size": 64,
    "num_mel_bins": 16,
    "subsampling_conv_channels": 8,
    "attention_bias": False,
    "convolution_bias": False,
}


@pytest.fixture(autouse=True)
def mlx_on_cpu():
    # note: MLX's GPU fp32 matmul rounds more coarsely than PyTorch; CPU keeps
    # the comparison tight enough to catch layout or math mistakes.
    previous = mx.default_device()
    mx.set_default_device(mx.cpu)
    yield
    mx.set_default_device(previous)


def randomize_batch_norm(model: torch.nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm1d):
            module.running_mean.uniform_(-0.5, 0.5)
            module.running_var.uniform_(0.5, 1.5)
        else:
            pass


def tiny_torch_model(architecture: str) -> torch.nn.Module:
    torch.manual_seed(0)
    if architecture == "ParakeetForCTC":
        config = parakeet.ParakeetCTCConfig(
            vocab_size=12, pad_token_id=11, encoder_config=TINY_ENCODER
        )
    elif architecture == "ParakeetForRNNT":
        config = parakeet.ParakeetRNNTConfig(
            vocab_size=12,
            blank_token_id=11,
            pad_token_id=2,
            decoder_hidden_size=16,
            encoder_config=TINY_ENCODER,
        )
    else:
        config = parakeet.ParakeetTDTConfig(
            vocab_size=12,
            blank_token_id=11,
            pad_token_id=2,
            decoder_hidden_size=16,
            encoder_config=TINY_ENCODER,
        )
    model = getattr(transformers, architecture)(config)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(0.0, 0.3)
        randomize_batch_norm(model)
    if architecture != "ParakeetForCTC":
        model.generation_config.decoder_start_token_id = config.blank_token_id
        model.generation_config.suppress_tokens = (
            list(
                range(
                    config.vocab_size,
                    config.vocab_size + len(getattr(config, "durations", None) or ()),
                )
            )
            or None
        )
    else:
        pass
    return model.eval()


def to_mlx(torch_model: torch.nn.Module) -> ParakeetModel:
    config = torch_model.config.to_dict()
    config["architectures"] = [type(torch_model).__name__]
    model = ParakeetModel(ParakeetMlxConfig.from_dict(config))
    weights = {
        name: mx.array(tensor.detach().numpy())
        for name, tensor in torch_model.state_dict().items()
    }
    model.load_weights(list(torch_layout_to_mlx(weights).items()), strict=True)
    model.eval()
    return model


def features(batch_lengths: list[int]) -> tuple[torch.Tensor, torch.Tensor]:
    rng = np.random.default_rng(1)
    frames = max(batch_lengths)
    values = rng.standard_normal((len(batch_lengths), frames, 16)).astype(np.float32)
    mask = np.arange(frames)[None, :] < np.array(batch_lengths)[:, None]
    values = values * mask[:, :, None]
    return torch.from_numpy(values), torch.from_numpy(mask)


@pytest.mark.parametrize(
    "architecture", ["ParakeetForCTC", "ParakeetForRNNT", "ParakeetForTDT"]
)
def test_padded_batch_encoder_matches_transformers(architecture: str) -> None:
    torch_model = tiny_torch_model(architecture)
    model = to_mlx(torch_model)
    values, mask = features([120, 77])

    with torch.inference_mode():
        expected = torch_model.encoder(
            input_features=values, attention_mask=mask
        ).last_hidden_state.numpy()
    hidden, lengths = model.encoder(
        mx.array(values.numpy()), mx.array(mask.sum(-1).numpy().astype(np.int32))
    )
    hidden, lengths = np.array(hidden), np.array(lengths)

    assert lengths.tolist() == [15, 10]
    for row, length in enumerate(lengths):
        np.testing.assert_allclose(
            hidden[row, :length], expected[row, :length], rtol=1e-4, atol=1e-4
        )


def test_unpadded_encoder_skips_masks_with_the_same_result() -> None:
    torch_model = tiny_torch_model("ParakeetForTDT")
    model = to_mlx(torch_model)
    values, mask = features([96])

    masked, _ = model.encoder(mx.array(values.numpy()), mx.array([96]))
    unmasked, lengths = model.encoder(mx.array(values.numpy()), None)

    assert np.array(lengths).tolist() == [12]
    np.testing.assert_allclose(np.array(masked), np.array(unmasked), atol=1e-5)


def strip(ids: list[int], drop: set[int]) -> list[int]:
    return [token for token in ids if token not in drop]


@pytest.mark.parametrize(
    "architecture", ["ParakeetForCTC", "ParakeetForRNNT", "ParakeetForTDT"]
)
def test_greedy_decode_matches_transformers_generate(architecture: str) -> None:
    torch_model = tiny_torch_model(architecture)
    model = to_mlx(torch_model)
    values, mask = features([120, 77])
    config = torch_model.config

    decoded = model.greedy_decode(
        mx.array(values.numpy()), mx.array(mask.sum(-1).numpy().astype(np.int32))
    )

    if architecture == "ParakeetForCTC":
        with torch.inference_mode():
            sequences = torch_model.generate(input_features=values, attention_mask=mask)
        lengths = subsampled_lengths(mx.array([120, 77]), model.config.encoder)
        for row, length in enumerate(np.array(lengths)):
            assert decoded[row] == sequences[row, :length].tolist()
    else:
        # note: compare against one-utterance generate calls; batched
        # Transformers transducer decoding reads past a short row's last frame.
        drop = {config.blank_token_id, config.pad_token_id}
        expected = []
        for row, length in enumerate([120, 77]):
            with torch.inference_mode():
                output = torch_model.generate(
                    input_features=values[row : row + 1, :length],
                    attention_mask=mask[row : row + 1, :length],
                )
            expected.append(strip(output.sequences[0].tolist(), drop))
        assert decoded == expected
        if architecture == "ParakeetForTDT":
            assert all(decoded), "the TDT weights should emit tokens on both rows"
        else:
            pass


def test_weight_conversion_transposes_convolutions_and_drops_counters() -> None:
    converted = torch_layout_to_mlx(
        {
            "conv2d.weight": mx.zeros((8, 1, 3, 5)),
            "conv1d.weight": mx.zeros((6, 4, 9)),
            "linear.weight": mx.zeros((3, 2)),
            "norm.num_batches_tracked": mx.array(7),
        }
    )

    assert converted["conv2d.weight"].shape == (8, 3, 5, 1)
    assert converted["conv1d.weight"].shape == (6, 9, 4)
    assert converted["linear.weight"].shape == (3, 2)
    assert "norm.num_batches_tracked" not in converted


def test_config_rejects_non_parakeet_and_incomplete_tdt_checkpoints() -> None:
    with pytest.raises(ValueError, match="Hugging Face format"):
        ParakeetMlxConfig.from_dict(
            {"architectures": ["WhisperForConditionalGeneration"]}
        )
    with pytest.raises(ValueError, match="durations"):
        ParakeetMlxConfig.from_dict(
            {
                "architectures": ["ParakeetForTDT"],
                "encoder_config": {},
                "vocab_size": 8,
                "durations": [],
            }
        )
