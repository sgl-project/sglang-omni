# SPDX-License-Identifier: Apache-2.0
"""Encoder replay correctness across requests, shapes and compilation."""

import pytest
import torch

from sglang_omni.models.nemotron_transcribe.encoder_cuda_graph import EncoderCudaGraphs
from sglang_omni.models.nemotron_transcribe.speech_transformer import (
    SpeechTransformerEncoder,
)

pytestmark = [
    pytest.mark.accelerator,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
]


def masked_projection(
    hidden_states: torch.Tensor, valid_key_mask: torch.Tensor | None
) -> torch.Tensor:
    output = hidden_states.sin() * 2.0
    if valid_key_mask is not None:
        output = output * valid_key_mask.unsqueeze(-1)
    else:
        pass
    return output


@torch.no_grad()
def test_replay_and_shape_limit_preserve_independent_outputs() -> None:
    replay = EncoderCudaGraphs(masked_projection)
    retained_outputs = []
    for frame_count in [4, 7, 4, 11, 13, 7, 17, 4]:
        hidden_states = torch.randn(2, frame_count, 8, device="cuda")
        valid_mask = torch.rand(2, frame_count, device="cuda") > 0.3
        output = replay(hidden_states, valid_mask)
        reference = masked_projection(hidden_states, valid_mask)
        torch.testing.assert_close(output, reference)
        retained_outputs.append((output, reference))
    for output, reference in retained_outputs:
        torch.testing.assert_close(output, reference)


@torch.no_grad()
def test_compiled_encoder_replays_changed_content_and_batches() -> None:
    torch.manual_seed(3)
    encoder = (
        SpeechTransformerEncoder(
            feat_in=4,
            d_model=16,
            n_heads=2,
            n_layers=1,
            subsampling_factor=2,
            drop_rate=0.0,
        )
        .cuda()
        .eval()
    )
    samples = [
        (
            torch.randn(batch_size, 4, 12, device="cuda"),
            torch.full((batch_size,), 12, device="cuda"),
        )
        for batch_size in [1, 3, 2, 3]
    ]
    references = [encoder(signal, lengths)[0].clone() for signal, lengths in samples]
    encoder.compile_layers(mode="default", cuda_graphs=True, dynamic=True)
    for (signal, lengths), reference in zip(samples + samples, references + references):
        output, output_lengths = encoder(signal, lengths)
        torch.testing.assert_close(output, reference, rtol=1e-4, atol=1e-4)
        assert output_lengths.tolist() == [6] * signal.shape[0]
