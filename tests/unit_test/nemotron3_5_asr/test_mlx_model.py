# SPDX-License-Identifier: Apache-2.0
"""Numerical parity with the independent Torch implementation on tiny models."""

import numpy as np
import pytest
import torch

mx = pytest.importorskip("mlx.core")
if not mx.metal.is_available():
    pytest.skip("MLX Metal is required", allow_module_level=True)

from sglang_omni.models.nemotron3_5_asr.hf_compat import (
    Nemotron3_5AsrConfig,
    Nemotron3_5AsrForRNNT,
)
from sglang_omni.models.nemotron3_5_asr.mlx.model import Model, sanitize_weights


def models():
    torch.manual_seed(7)
    c = Nemotron3_5AsrConfig(
        encoder_config=dict(
            hidden_size=16,
            intermediate_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            num_mel_bins=8,
            subsampling_conv_channels=4,
            attention_bias=False,
            convolution_bias=False,
            scale_input=False,
            sliding_window=9,
        ),
        decoder_hidden_size=8,
        num_decoder_layers=2,
        vocab_size=10,
        blank_token_id=9,
        decoder_start_token_id=9,
        num_prompts=4,
        default_prompt_id=0,
        prompt_intermediate_size=24,
    )
    reference = Nemotron3_5AsrForRNNT(c).eval()
    native = Model(c)
    native.load_weights(
        list(
            sanitize_weights(
                {k: mx.array(v.numpy()) for k, v in reference.state_dict().items()}
            ).items()
        )
    )
    native.eval()
    return reference, native


@pytest.mark.parametrize("lookahead", [0, 3, 6, 13])
@pytest.mark.parametrize("lengths", [[65], [65, 41]])
def test_encoder_parity(lookahead, lengths):
    reference, native = models()
    torch.manual_seed(8)
    features = torch.randn(len(lengths), max(lengths), 8)
    mask = torch.arange(features.shape[1])[None, :] < torch.tensor(lengths)[:, None]
    features *= mask[..., None]
    prompts = torch.arange(len(lengths))
    with torch.inference_mode():
        expected = reference.get_audio_features(
            input_features=features,
            attention_mask=mask,
            prompt_ids=prompts,
            num_lookahead_tokens=lookahead,
        )
    actual, out_lengths = native.encode(
        mx.array(features.numpy()),
        mx.array(lengths),
        mx.array(prompts.numpy()),
        lookahead,
    )
    for i, length in enumerate(out_lengths.tolist()):
        np.testing.assert_allclose(
            np.array(actual[i, :length]),
            expected.pooler_output[i, :length].numpy(),
            rtol=2e-4,
            atol=2e-5,
        )


def test_predictor_and_joint_parity():
    reference, native = models()
    state = None
    tokens = [9, 2, 7, 1]
    for index, token in enumerate(tokens):
        with torch.inference_mode():
            expected = reference.decoder(torch.tensor([tokens[: index + 1]]))[:, -1]
        actual, state = native.decoder(token, state)
        np.testing.assert_allclose(
            np.array(actual), expected.numpy(), rtol=2e-4, atol=2e-5
        )
        encoded = np.ones((1, 8), dtype=np.float32)
        with torch.inference_mode():
            expected_logits = reference.joint(expected, torch.from_numpy(encoded))
        np.testing.assert_allclose(
            np.array(native.joint(mx.array(encoded), actual)),
            expected_logits.numpy(),
            rtol=2e-4,
            atol=2e-5,
        )


@pytest.mark.parametrize("max_new_tokens", [None, 1, 12])
def test_greedy_tokens_and_request_state_isolation(max_new_tokens):
    reference, native = models()
    features = torch.randn(1, 25, 8)
    mask = torch.ones(1, 25, dtype=torch.long)
    with torch.inference_mode():
        expected = (
            reference.generate(
                input_features=features,
                attention_mask=mask,
                prompt_ids=torch.tensor([0]),
                num_lookahead_tokens=3,
                max_new_tokens=max_new_tokens,
            )
            .sequences[0]
            .tolist()
        )
    encoded, lengths = native.encode(
        mx.array(features.numpy()), mx.array([25]), mx.array([0]), 3
    )
    actual = native.decode(encoded[0], int(lengths[0]), max_new_tokens=max_new_tokens)
    assert actual == expected
    assert (
        native.decode(encoded[0], int(lengths[0]), max_new_tokens=max_new_tokens)
        == actual
    )
