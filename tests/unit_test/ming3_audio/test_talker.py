# SPDX-License-Identifier: Apache-2.0
"""Speech generation parity with the eager acoustic decoder."""

from dataclasses import dataclass

import pytest
import torch
from transformers import DynamicCache, Qwen2Config

from sglang_omni.models.ming3_audio.talker import (
    GenerationStopReason,
    Ming3Talker,
    TalkerConfig,
)


@dataclass(frozen=True, kw_only=True)
class FakeLanguageOutput:
    last_hidden_state: torch.Tensor


class FakeLanguageModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config: Qwen2Config = Qwen2Config(
            vocab_size=32,
            hidden_size=4,
            intermediate_size=8,
            num_hidden_layers=1,
            num_attention_heads=1,
            num_key_value_heads=1,
        )

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        past_key_values: DynamicCache,
        use_cache: bool,
    ) -> FakeLanguageOutput:
        return FakeLanguageOutput(last_hidden_state=inputs_embeds[:, -1:])


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


@pytest.mark.accelerator
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
            expected = talker.generate_latents_with_status(
                embeddings,
                reference_latents,
                6,
                torch.Generator(device="cuda").manual_seed(seed),
            )
            talker.enable_cuda_graph = True
            actual = talker.generate_latents_with_status(
                embeddings,
                reference_latents,
                6,
                torch.Generator(device="cuda").manual_seed(seed),
            )
            assert actual.latents.shape == (1, 24, 4)
            assert actual.actual_decode_steps == expected.actual_decode_steps == 6
            assert (
                actual.stop_reason == expected.stop_reason == "max_decode_steps_reached"
            )
            torch.testing.assert_close(actual.latents, expected.latents, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.parametrize("enable_cuda_graph", [False, True])
def test_speech_stop_preserves_minimum_context(
    talker: Ming3Talker, enable_cuda_graph: bool
) -> None:
    talker.enable_cuda_graph = enable_cuda_graph
    with torch.inference_mode():
        talker.stop_head.bias.copy_(torch.tensor([-10.0, 10.0], device="cuda"))
        actual = talker.generate_latents_with_status(
            torch.randn(1, 3, 32, device="cuda", dtype=torch.bfloat16),
            torch.empty(1, 0, 4, device="cuda", dtype=torch.bfloat16),
            12,
            torch.Generator(device="cuda").manual_seed(9),
        )
    assert actual.latents.shape == (1, 20, 4)
    assert actual.actual_decode_steps == 5
    assert actual.stop_reason == "natural_stop"


@pytest.mark.parametrize(
    (
        "stop_probabilities",
        "max_decode_steps",
        "expected_decode_steps",
        "expected_stop_reason",
    ),
    [
        ((0.0, 0.0, 0.0, 0.0, 0.9, 0.0), 6, 5, "natural_stop"),
        ((0.0, 0.0, 0.0, 0.0), 4, 4, "max_decode_steps_reached"),
        ((0.0, 0.0, 0.0, 0.0, 0.9), 5, 5, "natural_stop"),
        ((0.9, 0.9, 0.9, 0.9, 0.9), 8, 5, "natural_stop"),
    ],
    ids=[
        "early-natural-stop",
        "cap-exhausted",
        "natural-stop-on-final-step",
        "minimum-step-gate",
    ],
)
def test_generation_status_comes_from_the_production_stop_loop(
    monkeypatch: pytest.MonkeyPatch,
    stop_probabilities: tuple[float, ...],
    max_decode_steps: int,
    expected_decode_steps: int,
    expected_stop_reason: GenerationStopReason,
) -> None:
    talker = Ming3Talker.__new__(Ming3Talker)
    torch.nn.Module.__init__(talker)
    talker.config = TalkerConfig(
        aggregator={
            "in_channels": 2,
            "hidden_size": 4,
            "depth": 1,
            "num_heads": 1,
            "attn_backend": "torch",
        },
        flowmodel={
            "in_channels": 2,
            "hidden_size": 4,
            "depth": 1,
            "num_heads": 1,
            "attn_backend": "torch",
        },
        patch_size=1,
        history_patch_size=2,
        steps=1,
        cfg_strength=2.0,
    )
    talker.model = FakeLanguageModel()
    talker.enable_cuda_graph = False
    talker.acoustic_graph = None
    decoded_latents: list[torch.Tensor] = []

    def decode_acoustic(
        hidden: torch.Tensor,
        history: torch.Tensor,
        noise: torch.Tensor,
        timesteps: torch.Tensor,
        diffusion_parameters: torch.Tensor,
        diffusion_noise: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        latent = torch.full(
            (1, 1, 2),
            float(len(decoded_latents) + 1),
            device=hidden.device,
            dtype=hidden.dtype,
        )
        decoded_latents.append(latent)
        stop_probability = torch.tensor(
            stop_probabilities[len(decoded_latents) - 1],
            device=hidden.device,
            dtype=hidden.dtype,
        )
        return latent, hidden, stop_probability

    monkeypatch.setattr(talker, "decode_acoustic", decode_acoustic)
    embeddings = torch.ones((1, 2, 4), dtype=torch.float32)
    reference_latents = torch.empty((1, 0, 2), dtype=torch.float32)
    generator = torch.Generator(device="cpu").manual_seed(7)

    outcome = talker.generate_latents_with_status(
        embeddings, reference_latents, max_decode_steps, generator
    )

    expected_latents = torch.tensor(
        [[[float(step), float(step)] for step in range(1, expected_decode_steps + 1)]]
    )
    assert outcome.actual_decode_steps == expected_decode_steps
    assert outcome.stop_reason == expected_stop_reason
    assert len(decoded_latents) == expected_decode_steps
    torch.testing.assert_close(outcome.latents, expected_latents, rtol=0, atol=0)

    decoded_latents.clear()
    legacy_tensor = talker.generate_latents(
        embeddings,
        reference_latents,
        max_decode_steps,
        torch.Generator(device="cpu").manual_seed(7),
    )
    assert isinstance(legacy_tensor, torch.Tensor)
    assert len(decoded_latents) == expected_decode_steps
    torch.testing.assert_close(legacy_tensor, expected_latents, rtol=0, atol=0)
