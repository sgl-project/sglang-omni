# SPDX-License-Identifier: Apache-2.0
import pytest
import torch

from sglang_omni.models.personaplex.components.mimi import MimiCodec


@pytest.fixture
def random_codec() -> MimiCodec:
    """A full-size Mimi with small random weights and every codebook entry in use."""
    torch.manual_seed(0)
    codec = MimiCodec().eval()
    with torch.no_grad():
        for parameter in codec.parameters():
            parameter.normal_(std=0.05)
        for module in codec.modules():
            if hasattr(module, "embedding_sum"):
                module.embedding_sum.normal_()
                module.cluster_usage.fill_(1.0)
    return codec
