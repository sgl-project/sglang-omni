# SPDX-License-Identifier: Apache-2.0
"""Speech generation parity with the eager acoustic decoder."""

import pytest
import torch
from transformers import Qwen2Config

from sglang_omni.models.ming3_audio.talker import Ming3Talker, TalkerConfig

pytestmark = pytest.mark.accelerator


@pytest.fixture
def talker() -> Ming3Talker:
    if not torch.cuda.is_available():
        pytest.skip("Acoustic CUDA graph tests require CUDA")
    else:
        pass
    torch.manual_seed(42)
    acoustic_config = TalkerConfig(
        aggregator={
            "in_channels": 4,
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 2,
            "attn_backend": "torch",
        },
        flowmodel={
            "in_channels": 4,
            "hidden_size": 32,
            "depth": 1,
            "num_heads": 2,
            "attn_backend": "torch",
        },
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
    return model.eval().to(device="cuda", dtype=torch.bfloat16)


def test_cuda_graph_matches_eager_across_requests(talker: Ming3Talker) -> None:
    with torch.inference_mode():
        for prompt_tokens, reference_frames, seed in [(3, 0, 9), (7, 5, 17), (4, 2, 9)]:
            embeddings = torch.randn(
                1, prompt_tokens, 32, device="cuda", dtype=torch.bfloat16
            )
            reference_latents = torch.randn(
                1, reference_frames, 4, device="cuda", dtype=torch.bfloat16
            )
            talker.enable_cuda_graph = False
            expected = talker.generate_latents(
                embeddings,
                reference_latents,
                6,
                torch.Generator(device="cuda").manual_seed(seed),
            )
            talker.enable_cuda_graph = True
            actual = talker.generate_latents(
                embeddings,
                reference_latents,
                6,
                torch.Generator(device="cuda").manual_seed(seed),
            )
            assert actual.shape == (1, 24, 4)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_speech_stop_preserves_minimum_context(
    talker: Ming3Talker, enable_cuda_graph: bool
) -> None:
    talker.enable_cuda_graph = enable_cuda_graph
    with torch.inference_mode():
        talker.stop_head.bias.copy_(torch.tensor([-10.0, 10.0], device="cuda"))
        actual = talker.generate_latents(
            torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, 4, device="cuda", dtype=torch.bfloat16),
            12,
            torch.Generator(device="cuda").manual_seed(9),
        )
    assert actual.shape == (1, 20, 4)
