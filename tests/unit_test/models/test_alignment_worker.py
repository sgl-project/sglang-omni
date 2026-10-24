# SPDX-License-Identifier: Apache-2.0
"""The forced-alignment worker's request contract; it runs where the worker's `qwen-asr` package is installed."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

processor_module = pytest.importorskip("qwen_asr.inference.qwen3_forced_aligner")

worker = pytest.importorskip("examples.streaming_cascade.alignment_worker")


@pytest.mark.parametrize(
    "text, expected_ends",
    [
        ("Version 2.0.1.", [7, 11, 11, 11, 13]),
        ("It takes 3-5 days.", [2, 8, 10, 11, 12, 17]),
        ("Wait 3.5-5 hours.", [4, 8, 8, 8, 9, 10, 16]),
    ],
)
def test_worker_maps_numbers_separated_by_punctuation(
    text: str, expected_ends: list[int], monkeypatch: pytest.MonkeyPatch
) -> None:
    processor = processor_module.Qwen3ForceAlignProcessor()

    def align(*, audio, text: str, language: str):
        return [
            [
                SimpleNamespace(text=word, end_time=0.5)
                for word in processor.tokenize_space_lang(text)
            ]
        ]

    monkeypatch.setattr(
        worker.Qwen3ForcedAligner,
        "from_pretrained",
        lambda *args, **kwargs: SimpleNamespace(align=align),
    )
    with TestClient(worker.create_app("test-model", "cpu", "English")) as client:
        response = client.post(
            "/align",
            json={
                "text": text,
                "audio": "AAA=",
                "sample_rate_hz": 24000,
                "audio_format": "pcm16",
                "language": "English",
            },
        )
    assert response.status_code == 200, response.text
    ends = [boundary["text_end_chars"] for boundary in response.json()["boundaries"]]
    assert ends == expected_ends


@pytest.mark.parametrize("language", ["Chinese", None])
def test_a_request_names_its_language_or_leaves_it_to_the_worker(language):
    body = {"text": "1", "audio": "", "sample_rate_hz": 24000, "audio_format": "pcm16"}
    if language is None:
        pass
    else:
        body["language"] = language
    assert worker.AlignmentRequest.model_validate(body).language == language
