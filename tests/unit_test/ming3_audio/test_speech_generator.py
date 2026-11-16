# SPDX-License-Identifier: Apache-2.0
"""CPU-only orchestration tests for Ming 3 speech synthesis."""

import asyncio

import pytest
import torch

from sglang_omni.models.ming3_audio.talker import ReferenceConditioning
from tests.unit_test.fixtures.ming3_audio import make_ming3_speech_generator


@pytest.fixture
def reference_conditioning() -> ReferenceConditioning:
    return ReferenceConditioning(
        reference_latents=torch.tensor([[[1.0, 2.0]]], dtype=torch.float32),
        reference_embeddings=torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
        speaker_embedding=torch.tensor([[5.0, 6.0]], dtype=torch.float32),
    )


def test_synthesize_segments_and_concatenates_without_gap(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
) -> None:
    speech_generator = make_ming3_speech_generator(enable_text_segmentation=True)
    expected_fragments = ("甲乙。", "丙丁。", "戊己。")
    fragment_waveforms = {
        "甲乙。": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "丙丁。": torch.tensor([10.0, 11.0, 12.0], dtype=torch.float32),
        "戊己。": torch.tensor([20.0, 21.0], dtype=torch.float32),
    }
    prepare_calls: list[tuple[bytes | None, int]] = []
    fragment_calls: list[tuple[str, str, ReferenceConditioning, int]] = []

    def prepare_reference(
        reference_audio: bytes | None, seed: int
    ) -> ReferenceConditioning:
        prepare_calls.append((reference_audio, seed))
        return reference_conditioning

    def synthesize_fragment(
        text: str,
        reference_text: str,
        conditioning: ReferenceConditioning,
        seed: int,
        fragment_index: int,
        fragment_count: int,
    ) -> torch.Tensor:
        fragment_calls.append((text, reference_text, conditioning, seed))
        return fragment_waveforms[text]

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    actual = speech_generator.synthesize(
        "甲乙。丙丁。戊己。", b"reference audio", "参考文本", 123
    )

    assert prepare_calls == [(b"reference audio", 123)]
    assert [call[0] for call in fragment_calls] == list(expected_fragments)
    assert [call[1] for call in fragment_calls] == ["参考文本"] * 3
    assert all(call[2] is reference_conditioning for call in fragment_calls)
    assert [call[3] for call in fragment_calls] == [123] * 3
    assert actual.ndim == 1
    assert actual.dtype == torch.float32
    assert actual.device.type == "cpu"
    torch.testing.assert_close(
        actual,
        torch.tensor([1.0, 2.0, 10.0, 11.0, 12.0, 20.0, 21.0], dtype=torch.float32),
        rtol=0,
        atol=0,
    )


def test_synthesize_stops_on_fragment_failure_without_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
) -> None:
    speech_generator = make_ming3_speech_generator(enable_text_segmentation=True)
    prepare_call_count = 0
    fragment_calls: list[tuple[str, ReferenceConditioning, int]] = []

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
        assert reference_text == "参考文本"
        fragment_calls.append((text, conditioning, seed))
        if text == "丙丁。":
            raise ValueError("injected fragment failure")
        else:
            return torch.tensor([1.0], dtype=torch.float32)

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    with pytest.raises(RuntimeError, match="fragment 2/3"):
        speech_generator.synthesize(
            "甲乙。丙丁。戊己。", b"reference audio", "参考文本", 123
        )

    assert prepare_call_count == 1
    assert [call[0] for call in fragment_calls] == ["甲乙。", "丙丁。"]
    assert all(call[1] is reference_conditioning for call in fragment_calls)
    assert [call[2] for call in fragment_calls] == [123, 123]


def test_synthesize_without_segmentation_passes_raw_markdown_as_one_fragment(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
) -> None:
    speech_generator = make_ming3_speech_generator(
        enable_text_segmentation=False,
        max_decode_steps=32768,
        max_semantic_length=1,
        max_fragment_characters=1,
        max_fragments=1,
        max_input_characters=8,
        max_total_decode_steps=1,
    )
    raw_markdown = (
        "# **Raw Markdown**\n- [link](https://example.com/a_b?q=1)\n" + "a" * 8193
    )
    fragment_calls: list[tuple[str, str, ReferenceConditioning, int]] = []

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
        fragment_calls.append((text, reference_text, conditioning, seed))
        return torch.tensor([7.0], dtype=torch.float32)

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    actual = speech_generator.synthesize(raw_markdown, None, "", 321)

    assert len(fragment_calls) == 1
    assert fragment_calls[0][0] == raw_markdown
    assert fragment_calls[0][1] == ""
    assert fragment_calls[0][2] is reference_conditioning
    assert fragment_calls[0][3] == 321
    torch.testing.assert_close(
        actual, torch.tensor([7.0], dtype=torch.float32), rtol=0, atol=0
    )


@pytest.mark.parametrize(
    ("waveform", "error_message"),
    [
        (torch.empty((0,), dtype=torch.float32), "empty or non-flat waveform"),
        (torch.tensor([float("nan")]), "non-finite audio"),
    ],
)
def test_segmentation_off_keeps_waveform_errors_contextualized(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
    waveform: torch.Tensor,
    error_message: str,
) -> None:
    speech_generator = make_ming3_speech_generator(enable_text_segmentation=False)

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
        return waveform

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    with pytest.raises(RuntimeError, match=f"{error_message}.*fragment 1/1"):
        speech_generator.synthesize("raw text", None, "", 123)


@pytest.mark.parametrize(
    "interruption_type",
    [asyncio.CancelledError, KeyboardInterrupt],
    ids=["async-cancellation", "keyboard-interrupt"],
)
def test_synthesize_propagates_cancellation_without_partial_result(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
    interruption_type: type[BaseException],
) -> None:
    speech_generator = make_ming3_speech_generator(enable_text_segmentation=True)
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
        if text == "丙丁。":
            raise interruption_type()
        else:
            return torch.tensor([1.0], dtype=torch.float32)

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    with pytest.raises(interruption_type):
        speech_generator.synthesize("甲乙。丙丁。戊己。", None, "", 123)

    assert fragment_calls == ["甲乙。", "丙丁。"]


@pytest.mark.parametrize(
    ("invalid_waveform", "error_message"),
    [
        (torch.empty(0, dtype=torch.float32), "empty or non-flat waveform"),
        (torch.tensor([0.0, float("nan")]), "non-finite audio"),
    ],
)
def test_synthesize_rejects_invalid_fragment_waveforms(
    monkeypatch: pytest.MonkeyPatch,
    reference_conditioning: ReferenceConditioning,
    invalid_waveform: torch.Tensor,
    error_message: str,
) -> None:
    speech_generator = make_ming3_speech_generator(enable_text_segmentation=True)
    prepare_call_count = 0
    fragment_call_count = 0

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
        nonlocal fragment_call_count
        fragment_call_count += 1
        return invalid_waveform

    monkeypatch.setattr(speech_generator, "prepare_reference", prepare_reference)
    monkeypatch.setattr(speech_generator, "synthesize_fragment", synthesize_fragment)

    with pytest.raises(RuntimeError, match=error_message):
        speech_generator.synthesize("甲乙。", None, "", 123)

    assert prepare_call_count == 1
    assert fragment_call_count == 1
