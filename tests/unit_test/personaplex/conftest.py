# SPDX-License-Identifier: Apache-2.0
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from sglang_omni.models.personaplex.components.mimi import MimiCodec


def create_random_codec() -> MimiCodec:
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


@pytest.fixture
def random_codec() -> MimiCodec:
    return create_random_codec()


@pytest.fixture(scope="session")
def mimi_checkpoint(
    tmp_path_factory: pytest.TempPathFactory, request: pytest.FixtureRequest
) -> Path:
    """A full-size checkpoint using the released Mimi tensor names."""
    codec = create_random_codec()
    checkpoint_weights: dict[str, torch.Tensor] = {}
    for parameter_name, tensor in codec.state_dict().items():
        checkpoint_name = (
            parameter_name.replace(".conv.", ".conv.conv.")
            .replace(".convtr.", ".convtr.convtr.")
            .replace("_transformer.layers.", "_transformer.transformer.layers.")
            .replace(".codebook.", "._codebook.")
        )
        checkpoint_weights[checkpoint_name] = tensor
    checkpoint_weights["quantizer.rvq_rest.vq.layers.7._codebook.embedding_sum"] = (
        torch.zeros(1)
    )
    checkpoint_path = tmp_path_factory.mktemp("mimi") / "tokenizer-test.safetensors"
    save_file(checkpoint_weights, str(checkpoint_path))
    request.addfinalizer(checkpoint_path.unlink)
    return checkpoint_path
