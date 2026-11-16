# SPDX-License-Identifier: Apache-2.0
"""CPU coverage for fragment stop handling and cap compatibility."""

import pytest
import torch

from sglang_omni.models.ming3_audio.talker import (
    GenerationOutcome,
    GenerationStopReason,
    MaxDecodeStepsReachedError,
    Ming3SpeechGenerator,
    ReferenceConditioning,
)
from tests.unit_test.fixtures.ming3_audio import make_ming3_speech_generator


class FakeTokenizer:
    def encode(self, text: str) -> list[int]:
        return [1]

    def convert_tokens_to_ids(self, token: str) -> int:
        return 2


class FakeEmbeddingModel:
    def embed_tokens(self, token_ids: torch.Tensor) -> torch.Tensor:
        return torch.zeros((1, token_ids.shape[1], 2), dtype=torch.float32)


class FakeTalker:
    def __init__(self, generation_outcomes: list[GenerationOutcome]) -> None:
        self.enable_cuda_graph: bool = False
        self.model: FakeEmbeddingModel = FakeEmbeddingModel()
        self.generation_outcomes: list[GenerationOutcome] = generation_outcomes
        self.generation_call_count: int = 0

    def generate_latents_with_status(
        self,
        embeddings: torch.Tensor,
        reference_latents: torch.Tensor,
        max_decode_steps: int,
        generator: torch.Generator,
    ) -> GenerationOutcome:
        outcome = self.generation_outcomes[self.generation_call_count]
        self.generation_call_count += 1
        return outcome


class FakeVAE:
    def __init__(self) -> None:
        self.decoded_latents: list[torch.Tensor] = []

    def decode(
        self,
        latents: torch.Tensor,
        *,
        use_cache: bool,
        last_chunk: bool,
    ) -> tuple[torch.Tensor, None, None]:
        self.decoded_latents.append(latents)
        return torch.tensor([9.0]), None, None


def install_fake_generation_path(
    speech_generator: Ming3SpeechGenerator,
    generation_outcomes: list[GenerationOutcome],
) -> tuple[FakeTalker, FakeVAE]:
    fake_talker = FakeTalker(generation_outcomes)
    fake_vae = FakeVAE()
    speech_generator.talker = fake_talker
    speech_generator.tokenizer = FakeTokenizer()
    speech_generator.vae = fake_vae
    speech_generator.device = torch.device("cpu")
    return fake_talker, fake_vae


@pytest.mark.parametrize("failure_fragment_index", [1, 2])
def test_segmented_synthesis_fails_before_decoding_a_capped_fragment(
    monkeypatch: pytest.MonkeyPatch,
    failure_fragment_index: int,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=True,
        max_decode_steps=4,
    )
    generation_outcomes: list[GenerationOutcome] = []
    for fragment_index in range(1, 4):
        if fragment_index == failure_fragment_index:
            stop_reason: GenerationStopReason = "max_decode_steps_reached"
            actual_decode_steps = 4
        else:
            stop_reason = "natural_stop"
            actual_decode_steps = 3
        generation_outcomes.append(
            GenerationOutcome(
                latents=torch.ones((1, 1, 2), dtype=torch.float32),
                actual_decode_steps=actual_decode_steps,
                stop_reason=stop_reason,
            )
        )
    fake_talker, fake_vae = install_fake_generation_path(
        speech_generator, generation_outcomes
    )
    conditioning = ReferenceConditioning(
        reference_latents=torch.empty((1, 0, 2), dtype=torch.float32),
        reference_embeddings=torch.empty((1, 0, 2), dtype=torch.float32),
        speaker_embedding=None,
    )
    prepare_call_count = 0

    def prepare_reference(
        reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        nonlocal prepare_call_count
        prepare_call_count += 1
        return conditioning

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)

    with pytest.raises(
        MaxDecodeStepsReachedError,
        match=(
            rf"fragment {failure_fragment_index}/3.*4 actual decode steps.*"
            "stop_reason=max_decode_steps_reached"
        ),
    ):
        speech_generator.synthesize("甲乙。丙丁。戊己。", None, "", 123)

    assert prepare_call_count == 1
    assert fake_talker.generation_call_count == failure_fragment_index
    assert len(fake_vae.decoded_latents) == failure_fragment_index - 1


def test_segmentation_off_warns_but_returns_capped_one_shot_audio(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=False,
        max_decode_steps=1,
        max_semantic_length=64,
        max_fragment_characters=128,
        max_fragments=1,
    )
    fake_talker, fake_vae = install_fake_generation_path(
        speech_generator,
        [
            GenerationOutcome(
                latents=torch.ones((1, 1, 2), dtype=torch.float32),
                actual_decode_steps=1,
                stop_reason="max_decode_steps_reached",
            )
        ],
    )
    conditioning = ReferenceConditioning(
        reference_latents=torch.empty((1, 0, 2), dtype=torch.float32),
        reference_embeddings=torch.empty((1, 0, 2), dtype=torch.float32),
        speaker_embedding=None,
    )

    def prepare_reference(
        reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        return conditioning

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)

    raw_markdown = "# **Raw Markdown**"
    with caplog.at_level("INFO"):
        actual = speech_generator.synthesize(raw_markdown, None, "", 321)

    torch.testing.assert_close(actual, torch.tensor([9.0]), rtol=0, atol=0)
    assert fake_talker.generation_call_count == 1
    assert len(fake_vae.decoded_latents) == 1
    assert "raw one-shot fragment 1/1" in caplog.text
    assert f"characters={len(raw_markdown)}" in caplog.text
    assert "actual_decode_steps=1" in caplog.text
    assert "after 1 actual decode steps" in caplog.text
    assert "max_decode_steps_reached" in caplog.text
    assert "cuda_graph_enabled=False" in caplog.text
    assert raw_markdown not in caplog.text
