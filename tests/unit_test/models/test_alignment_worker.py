# SPDX-License-Identifier: Apache-2.0
"""The forced-alignment worker's request contract; it runs where the worker's `qwen-asr` package is installed."""

import base64
from collections.abc import Iterator
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.models.streaming_cascade.alignment import (
    Alignment,
    AlignmentConfig,
    SentenceAudio,
)
from sglang_omni.models.streaming_cascade.dialog import heard_prefix

processor_module = pytest.importorskip("qwen_asr.inference.qwen3_forced_aligner")

worker = pytest.importorskip("examples.streaming_cascade.alignment_worker")


@dataclass
class TimestampAligner:
    aligner_processor: processor_module.Qwen3ForceAlignProcessor = field(
        default_factory=processor_module.Qwen3ForceAlignProcessor
    )
    token_texts: list[str] | None = None

    def align(
        self, *, audio: tuple[np.ndarray, int], text: str, language: str
    ) -> list[list[SimpleNamespace]]:
        token_texts = (
            self.aligner_processor.encode_timestamp(text, language)[0]
            if self.token_texts is None
            else self.token_texts
        )
        return [
            [
                SimpleNamespace(text=token, end_time=(index + 1) / 10)
                for index, token in enumerate(token_texts)
            ]
        ]


@pytest.fixture
def alignment_client(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[TestClient, TimestampAligner]]:
    aligner = TimestampAligner()
    monkeypatch.setattr(
        worker.Qwen3ForcedAligner, "from_pretrained", lambda *args, **kwargs: aligner
    )
    with TestClient(worker.create_app("test-model", "cpu", "English")) as client:
        yield client, aligner


def request_alignment(
    client: TestClient, text: str, language: str = "English"
) -> httpx.Response:
    return client.post(
        "/align",
        json={
            "text": text,
            "audio": base64.b64encode(bytes(48000)).decode(),
            "sample_rate_hz": 24000,
            "audio_format": "pcm16",
            "language": language,
        },
    )


@pytest.mark.parametrize(
    "text, expected_ends",
    [
        ("Version 2.0.1.", [7, 11, 13]),
        ("It takes 3-5 days.", [2, 8, 10, 11, 12, 17]),
        ("Wait 3.5-5 hours.", [4, 8, 9, 10, 16]),
    ],
)
def test_worker_maps_numbers_separated_by_punctuation(
    text: str,
    expected_ends: list[int],
    alignment_client: tuple[TestClient, TimestampAligner],
) -> None:
    client, _ = alignment_client
    response = request_alignment(client, text)
    assert response.status_code == 200, response.text
    ends = [boundary["text_end_chars"] for boundary in response.json()["boundaries"]]
    assert ends == expected_ends


@pytest.mark.parametrize(
    "text, language, expected_prefixes",
    [
        ("Don’t stop.", "English", ["Don’t", "Don’t stop"]),
        ("Don't stop.", "English", ["Don't", "Don't stop"]),
        ("foo/bar", "English", ["foo/bar"]),
        ("U.S. troops", "English", ["U.S", "U.S. troops"]),
        ("e\u0301lan", "English", ["e\u0301lan"]),
        ("cafe\u0301 noir", "English", ["cafe\u0301", "cafe\u0301 noir"]),
        (
            "ＡＢＣのﾃｽﾄです。",
            "Japanese",
            ["ＡＢＣ", "ＡＢＣの", "ＡＢＣのﾃｽﾄ", "ＡＢＣのﾃｽﾄです"],
        ),
        (
            "ｶﾞｯﾂﾎﾟｰｽﾞです。",
            "Japanese",
            ["ｶﾞｯﾂ", "ｶﾞｯﾂﾎﾟｰｽﾞ", "ｶﾞｯﾂﾎﾟｰｽﾞです"],
        ),
        ("e\u0301です。", "Japanese", ["e\u0301", "e\u0301です"]),
        ("안녕하세요.", "Korean", ["안녕", "안녕하세요"]),
        (
            "中文A-B测试。",
            "Chinese",
            ["中", "中文", "中文A-B", "中文A-B测", "中文A-B测试"],
        ),
        (
            "😀 long-term go go!",
            "English",
            ["😀 long-term", "😀 long-term go", "😀 long-term go go"],
        ),
    ],
)
def test_worker_preserves_original_offsets_after_token_normalization(
    text: str,
    language: str,
    expected_prefixes: list[str],
    alignment_client: tuple[TestClient, TimestampAligner],
) -> None:
    """Valid punctuation removal previously rejected the entire sentence."""
    client, _ = alignment_client
    response = request_alignment(client, text, language)
    assert response.status_code == 200, response.text
    boundaries = response.json()["boundaries"]
    assert [
        text[0 : boundary["text_end_chars"]] for boundary in boundaries
    ] == expected_prefixes
    assert [boundary["audio_end_samples"] for boundary in boundaries] == [
        (index + 1) * 2400 for index in range(len(expected_prefixes))
    ]


@pytest.mark.parametrize(
    "text, token_texts",
    [
        ("go go stop", ["go", "stop"]),
        ("go stop", ["go", "go", "stop"]),
        ("go stop", ["go", "other"]),
        ("go stop", ["go"]),
        ("go stop", [""]),
    ],
)
def test_worker_rejects_missing_repeated_or_changed_words(
    text: str,
    token_texts: list[str],
    alignment_client: tuple[TestClient, TimestampAligner],
) -> None:
    """Substring search previously skipped an omitted word or accepted a missing tail."""
    client, aligner = alignment_client
    aligner.token_texts = token_texts
    response = request_alignment(client, text)
    assert response.status_code == 422


@pytest.mark.parametrize(
    "text, language, complete_ms, expected_prefix",
    [
        ("21 apples.", "English", 200, "21"),
        ("2024 ends.", "English", 300, "2024"),
        ("3.14 matters.", "English", 400, "3.14"),
        ("21个。", "Chinese", 300, "21"),
        ("㍿です。", "Japanese", 200, "㍿"),
    ],
)
def test_worker_commits_expanded_source_units_only_after_the_last_word(
    text: str,
    language: str,
    complete_ms: int,
    expected_prefix: str,
    alignment_client: tuple[TestClient, TimestampAligner],
) -> None:
    """Expanded numbers and compatibility characters commit only at their last token."""
    client, _ = alignment_client
    response = request_alignment(client, text, language)
    assert response.status_code == 200, response.text
    progress = [
        (boundary["text_end_chars"], boundary["audio_end_samples"] / 24)
        for boundary in response.json()["boundaries"]
    ]
    before_end, _ = heard_prefix(progress, complete_ms - 1, text)
    after_end, _ = heard_prefix(progress, complete_ms, text)
    assert text[:before_end] == ""
    assert text[:after_end] == expected_prefix


@pytest.mark.asyncio
async def test_normalized_worker_boundaries_retain_reply_offsets(
    alignment_client: tuple[TestClient, TimestampAligner],
) -> None:
    """Worker offsets must survive sentence offsets and drive the original history cut."""
    client, _ = alignment_client
    earlier_text = "Earlier. "
    sentence_text = "😀 long-term plan costs 21 dollars."
    node = Alignment(AlignmentConfig(method="forced", alignment_url="http://alignment"))
    await node.client.aclose()
    node.client = httpx.AsyncClient(transport=httpx.ASGITransport(app=client.app))
    try:
        progress = await node.align(
            SentenceAudio(
                turn=1,
                text=sentence_text,
                start=len(earlier_text),
                start_ms=1000,
                samples=np.zeros(24000, dtype=np.float32),
                language="English",
            )
        )
    finally:
        await node.close()
    full_text = earlier_text + sentence_text
    before_chars, _ = heard_prefix(progress, 1499, full_text)
    after_chars, _ = heard_prefix(progress, 1500, full_text)
    assert full_text[:before_chars] == earlier_text + "😀 long-term plan costs"
    assert full_text[:after_chars] == earlier_text + "😀 long-term plan costs 21"


@pytest.mark.parametrize("language", ["Chinese", None])
def test_a_request_names_its_language_or_leaves_it_to_the_worker(language):
    body = {"text": "1", "audio": "", "sample_rate_hz": 24000, "audio_format": "pcm16"}
    if language is None:
        pass
    else:
        body["language"] = language
    assert worker.AlignmentRequest.model_validate(body).language == language
