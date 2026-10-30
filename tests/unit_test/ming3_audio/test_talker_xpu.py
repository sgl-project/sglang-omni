# SPDX-License-Identifier: Apache-2.0
"""Acoustic graph capture on Intel XPU."""

import pytest
import torch
from transformers import Qwen2Config

from sglang_omni.models.ming3_audio.talker import Ming3Talker, TalkerConfig

pytestmark = pytest.mark.accelerator

ACOUSTIC_MODULE = {
    "in_channels": 4,
    "hidden_size": 32,
    "depth": 1,
    "num_heads": 2,
    "attn_backend": "torch",
}


@pytest.fixture
def talker() -> Ming3Talker:
    if not torch.xpu.is_available():
        pytest.skip("Acoustic XPU graph tests require an Intel XPU")
    else:
        pass
    torch.manual_seed(42)
    acoustic_config = TalkerConfig(
        aggregator=ACOUSTIC_MODULE,
        flowmodel=ACOUSTIC_MODULE,
        patch_size=4,
        history_patch_size=8,
        steps=2,
        cfg_strength=2.0,
    )
    language_config = Qwen2Config(
        vocab_size=128,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        num_key_value_heads=1,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = Ming3Talker(acoustic_config, language_config, enable_cuda_graph=False)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.normal_(std=0.05)
        model.stop_head.weight.zero_()
        model.stop_head.bias.copy_(torch.tensor([10.0, -10.0]))
    return model.eval().to(device="xpu", dtype=torch.bfloat16)


def test_xpu_graph_matches_eager_across_requests(talker: Ming3Talker) -> None:
    """Capturing the acoustic step used to raise on XPU because it asked for a CUDA stream."""
    with torch.inference_mode():
        for prompt_tokens, reference_frames, seed in [(3, 0, 9), (7, 5, 17), (4, 2, 9)]:
            embeddings = torch.randn(
                1, prompt_tokens, 32, device="xpu", dtype=torch.bfloat16
            )
            reference_latents = torch.randn(
                1, reference_frames, 4, device="xpu", dtype=torch.bfloat16
            )
            talker.enable_cuda_graph = False
            expected = talker.generate_latents(
                embeddings,
                reference_latents,
                6,
                torch.Generator(device="xpu").manual_seed(seed),
            )
            talker.enable_cuda_graph = True
            actual = talker.generate_latents(
                embeddings,
                reference_latents,
                6,
                torch.Generator(device="xpu").manual_seed(seed),
            )
            assert actual.shape == (1, 24, 4)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
