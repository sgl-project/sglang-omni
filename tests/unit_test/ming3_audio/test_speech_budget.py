# SPDX-License-Identifier: Apache-2.0
"""CPU regression tests for Ming 3's conservative fragment decode budget."""

import pytest
import torch

from sglang_omni.models.ming3_audio.talker import ReferenceConditioning
from sglang_omni.models.ming3_audio.text_processing import TextTooLargeError
from tests.unit_test.fixtures.ming3_audio import make_ming3_speech_generator


@pytest.fixture
def reference_conditioning() -> ReferenceConditioning:
    return ReferenceConditioning(
        reference_latents=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
        reference_embeddings=torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
        speaker_embedding=torch.tensor([[5.0, 6.0]], dtype=torch.float32),
    )


def test_synthesize_rejects_total_decode_work_before_reference_preparation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=True,
        max_total_decode_steps=2,
    )
    prepare_call_count = 0

    def prepare_reference(
        reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        nonlocal prepare_call_count
        prepare_call_count += 1
        raise AssertionError("reference preparation must not start")

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)

    with pytest.raises(TextTooLargeError, match="decode steps"):
        speech_generator.synthesize("甲乙。丙丁。戊己。", None, "", 123)

    assert prepare_call_count == 0


@pytest.mark.parametrize(
    (
        "max_decode_steps",
        "max_total_decode_steps",
        "fragment_count",
        "should_admit",
    ),
    [
        (512, 16384, 32, True),
        (512, 16384, 33, False),
        (256, 16384, 64, True),
        (256, 16384, 65, False),
        (7, 20, 2, True),
        (7, 20, 3, False),
    ],
)
def test_decode_budget_uses_conservative_per_fragment_caps(
    max_decode_steps: int,
    max_total_decode_steps: int,
    fragment_count: int,
    should_admit: bool,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=True,
        max_decode_steps=max_decode_steps,
        max_fragments=max(fragment_count, 3),
        max_total_decode_steps=max_total_decode_steps,
    )
    expected_admitted_fragment_count = max_total_decode_steps // max_decode_steps

    if should_admit:
        assert (
            speech_generator.get_max_admitted_fragment_count()
            == expected_admitted_fragment_count
        )
        speech_generator.validate_decode_budget(fragment_count)
    else:
        with pytest.raises(TextTooLargeError, match="at most"):
            speech_generator.validate_decode_budget(fragment_count)


@pytest.mark.parametrize(
    ("max_decode_steps", "max_total_decode_steps", "error_message"),
    [
        (0, 1, "max_decode_steps"),
        (1, 0, "max_total_decode_steps"),
        (2, 1, "at least one fragment"),
    ],
)
def test_decode_budget_rejects_non_admissible_configuration(
    max_decode_steps: int,
    max_total_decode_steps: int,
    error_message: str,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=True,
        max_decode_steps=max_decode_steps,
        max_total_decode_steps=max_total_decode_steps,
    )

    with pytest.raises(ValueError, match=error_message):
        speech_generator.get_max_admitted_fragment_count()


@pytest.mark.parametrize(
    ("fragment_count", "should_admit"),
    [(32, True), (33, False)],
)
def test_synthesize_enforces_32_fragment_admission_boundary(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
    fragment_count: int,
    should_admit: bool,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=True,
        max_decode_steps=512,
        max_semantic_length=1,
        max_fragment_characters=2,
        max_fragments=128,
        max_input_characters=256,
        max_total_decode_steps=16384,
    )
    fragment_calls: list[str] = []
    prepare_call_count = 0
    text = "甲。" * fragment_count

    def prepare_reference(
        reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        nonlocal prepare_call_count
        prepare_call_count += 1
        return reference_conditioning

    def synthesize_fragment(
        text: str,
        reference_text: str,
        conditioning: ReferenceConditioning,
        seed: int,
        fragment_index: int,
        fragment_count: int,
    ) -> torch.Tensor:
        fragment_calls.append(text)
        return torch.tensor([1.0], dtype=torch.float32)

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    if should_admit:
        actual = speech_generator.synthesize(text, None, "", 123)
        assert len(fragment_calls) == fragment_count
        assert actual.shape == (fragment_count,)
        assert prepare_call_count == 1
    else:
        with pytest.raises(TextTooLargeError, match="at most 32 fragments"):
            speech_generator.synthesize(text, None, "", 123)
        assert fragment_calls == []
        assert prepare_call_count == 0


def test_synthesize_processes_fragments_at_exact_budget_ceiling(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=True,
        max_decode_steps=4,
        max_semantic_length=1,
        max_fragment_characters=2,
        max_fragments=2,
        max_total_decode_steps=8,
    )
    fragment_calls: list[str] = []

    def prepare_reference(
        reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        return reference_conditioning

    def synthesize_fragment(
        text: str,
        reference_text: str,
        conditioning: ReferenceConditioning,
        seed: int,
        fragment_index: int,
        fragment_count: int,
    ) -> torch.Tensor:
        fragment_calls.append(text)
        return torch.tensor([7.0], dtype=torch.float32)

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    actual = speech_generator.synthesize("甲。乙。", None, "", 123)

    assert fragment_calls == ["甲。", "乙。"]
    torch.testing.assert_close(
        actual, torch.tensor([7.0, 7.0], dtype=torch.float32), rtol=0, atol=0
    )
