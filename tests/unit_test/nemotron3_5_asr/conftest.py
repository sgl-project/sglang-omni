# SPDX-License-Identifier: Apache-2.0
"""Small real encoder shared by state-pool and graph tests."""

from collections.abc import Iterator

import pytest
import torch

from sglang_omni.vendor.nemotron3_5_asr.configuration_nemotron3_5_asr import (
    Nemotron3_5AsrConfig,
)
from sglang_omni.vendor.nemotron3_5_asr.modeling_nemotron3_5_asr import (
    Nemotron3_5AsrForRNNT,
)


@pytest.fixture
def model() -> Iterator[Nemotron3_5AsrForRNNT]:
    config = Nemotron3_5AsrConfig(
        vocab_size=16,
        decoder_hidden_size=8,
        num_decoder_layers=1,
        blank_token_id=15,
        num_prompts=4,
        prompt_intermediate_size=8,
        default_prompt_id=1,
        encoder_config={
            "hidden_size": 8,
            "num_hidden_layers": 2,
            "num_attention_heads": 2,
            "intermediate_size": 16,
            "subsampling_factor": 8,
            "subsampling_conv_channels": 2,
            "num_mel_bins": 4,
            "subsampling_conv_kernel_size": 3,
            "subsampling_conv_stride": 2,
            "conv_kernel_size": 9,
            "sliding_window": 9,
        },
    )
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(7)
        yield Nemotron3_5AsrForRNNT(config).eval()
