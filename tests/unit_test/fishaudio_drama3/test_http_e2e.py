# SPDX-License-Identifier: Apache-2.0
"""Drama 3 over HTTP on a GPU with a synthetic checkpoint and the public codec; the
model makes noise, so this checks serving, chunking and windows, not audio quality."""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import wave
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np
import pytest

from sglang_omni.client.audio import encode_wav
from sglang_omni.models.fishaudio_drama3.config import DEFAULT_CODEC_PATH
from sglang_omni.models.fishaudio_drama3.stages import resolve_codec_directory
from sglang_omni.utils.connection import find_available_port
from tests.unit_test.fishaudio_drama3.http_server import MODEL_NAME
from tests.unit_test.fishaudio_drama3.mtp_fixtures import (
    CHECKPOINT_CONFIG,
    DECODER_CONFIG,
    TEXT_CONFIG,
    write_synthetic_checkpoint,
)
from tests.unit_test.fishaudio_drama3.test_prompt import (
    CODEBOOK_SIZE,
    NUM_CODEBOOKS,
    build_toy_tokenizer,
)
from tests.utils import server_log_file, start_server_from_cmd, stop_server

pytestmark = pytest.mark.accelerator

REPO_ROOT = Path(__file__).resolve().parents[3]
# The public S2-Pro codec decodes every frame to exactly this many samples.
SAMPLES_PER_FRAME = 2048
SAMPLE_RATE = 44100
# Windows of 10, 10, 12, 32, 64 and 64 frames, then a 22-frame tail.
STREAMED_FRAME_COUNT = 214
STREAMED_CHUNK_FRAMES = [10, 10, 12, 32, 64, 64, 22]
MAX_RUNNING_REQUESTS = 16
STARTUP_TIMEOUT_S = 900
REQUEST_TIMEOUT_S = 300.0
REFERENCE_WAV = encode_wav(
    (0.2 * np.sin(np.arange(SAMPLE_RATE) * 2 * np.pi * 220 / SAMPLE_RATE)).astype(
        np.float32
    ),
    SAMPLE_RATE,
)


def write_checkpoint(checkpoint_dir: Path) -> None:
    """The synthetic test model resized to the toy tokenizer's ten codebooks of 4096
    codes, with the tokenizer beside it."""
    tokenizer = build_toy_tokenizer()
    semantic_begin_id = tokenizer.token_to_id("<|semantic:0|>")
    write_synthetic_checkpoint(
        checkpoint_dir,
        {
            **CHECKPOINT_CONFIG,
            "text_config": {
                **TEXT_CONFIG,
                "vocab_size": tokenizer.get_vocab_size(),
                "max_position_embeddings": 16384,
            },
            "audio_decoder_config": {**DECODER_CONFIG, "vocab_size": CODEBOOK_SIZE},
            "semantic_start_token_id": semantic_begin_id,
            "semantic_end_token_id": semantic_begin_id + CODEBOOK_SIZE - 1,
            "eos_token_id": tokenizer.token_to_id("<|im_end|>"),
        },
    )
    tokenizer.save(str(checkpoint_dir / "tokenizer.json"))


@pytest.fixture(scope="module")
def base_url(tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    checkpoint_dir = tmp_path_factory.mktemp("drama3-checkpoint")
    write_checkpoint(checkpoint_dir)
    # The server runs offline, so fetch the pinned codec into the Hub cache first.
    resolve_codec_directory(
        DEFAULT_CODEC_PATH, download_timeout_seconds=STARTUP_TIMEOUT_S
    )
    port = find_available_port()
    server = start_server_from_cmd(
        [
            sys.executable,
            "-c",
            "from tests.unit_test.fishaudio_drama3.http_server import main; main()",
            "--checkpoint",
            str(checkpoint_dir),
            "--port",
            str(port),
        ],
        server_log_file(tmp_path_factory, "drama3_http_logs"),
        port,
        timeout=STARTUP_TIMEOUT_S,
        env={
            "PYTHONPATH": os.pathsep.join(
                [str(REPO_ROOT), os.environ.get("PYTHONPATH", "")]
            ),
            "HF_HUB_OFFLINE": "1",
        },
        strip_proxy=True,
    )
    yield f"http://127.0.0.1:{port}"
    stop_server(server)


def speech(base_url: str, request_fields: dict[str, object]) -> httpx.Response:
    return httpx.post(
        f"{base_url}/v1/audio/speech",
        json={"model": MODEL_NAME, "input": "hello", **request_fields},
        timeout=REQUEST_TIMEOUT_S,
    )


def wav_sample_count(response: httpx.Response) -> int:
    assert response.status_code == 200, response.text
    with wave.open(io.BytesIO(response.content)) as wav_file:
        assert wav_file.getframerate() == SAMPLE_RATE
        return wav_file.getnframes()


@pytest.mark.parametrize(
    "references",
    [
        pytest.param(None, id="no-reference"),
        pytest.param(
            [
                {
                    "text": "a short take",
                    "data": base64.b64encode(REFERENCE_WAV).decode("ascii"),
                    "media_type": "audio/wav",
                }
            ],
            id="reference-audio",
        ),
        pytest.param(
            [{"text": "a short take", "vq_codes": [[7] * 12] * NUM_CODEBOOKS}],
            id="reference-codes",
        ),
    ],
)
def test_speech_holds_exactly_the_frames_asked_for(
    base_url: str, references: list[dict[str, object]] | None
) -> None:
    response = speech(base_url, {"max_new_tokens": 30, "references": references})

    assert wav_sample_count(response) == 30 * SAMPLES_PER_FRAME


def streamed_events(
    base_url: str, request_fields: dict[str, object]
) -> list[dict[str, object]]:
    events = []
    with httpx.stream(
        "POST",
        f"{base_url}/v1/audio/speech",
        json={
            "model": MODEL_NAME,
            "input": "hello",
            "stream": True,
            "stream_format": "sse",
            "response_format": "pcm",
            **request_fields,
        },
        timeout=REQUEST_TIMEOUT_S,
    ) as response:
        assert response.status_code == 200, response.read()
        assert response.headers["X-Sample-Rate"] == str(SAMPLE_RATE)
        for line in response.iter_lines():
            if line.startswith("data: "):
                events.append(json.loads(line.removeprefix("data: ")))
            else:
                pass
    return events


def test_streamed_speech_arrives_in_the_fixed_windows(base_url: str) -> None:
    events = streamed_events(base_url, {"max_new_tokens": STREAMED_FRAME_COUNT})

    chunk_sample_counts = [
        len(base64.b64decode(str(event["audio"]))) // 2
        for event in events
        if event["type"] == "speech.audio.delta"
    ]
    assert chunk_sample_counts == [
        frame_count * SAMPLES_PER_FRAME for frame_count in STREAMED_CHUNK_FRAMES
    ]
    done = events[-1]
    assert done["type"] == "speech.audio.done"
    usage = done["usage"]
    assert isinstance(usage, dict)
    assert usage["output_tokens"] == STREAMED_FRAME_COUNT
    assert usage["input_tokens"] > 0


def full_round(base_url: str) -> list[int]:
    """Sample counts of as many concurrent requests as the engine runs at once."""
    with ThreadPoolExecutor(max_workers=MAX_RUNNING_REQUESTS) as pool:
        responses = list(
            pool.map(
                lambda index: speech(
                    base_url, {"input": f"line {index}", "max_new_tokens": 20}
                ),
                range(MAX_RUNNING_REQUESTS),
            )
        )
    return [wav_sample_count(response) for response in responses]


def test_concurrent_requests_all_complete(base_url: str) -> None:
    assert full_round(base_url) == [20 * SAMPLES_PER_FRAME] * MAX_RUNNING_REQUESTS


REFUSED_REQUESTS = [
    pytest.param({"instructions": "whisper"}, id="instructions"),
    pytest.param({"stage_sampling": {}}, id="stage-sampling"),
    pytest.param(
        {"references": [{"text": "take", "speaker": "a", "vq_codes": [[0]] * 10}]},
        id="reference-speaker",
    ),
    pytest.param({"repetition_penalty": 1.2}, id="repetition-penalty"),
    pytest.param({"input": "<|speaker:1|> hello"}, id="speaker-one"),
]


@pytest.mark.parametrize("request_fields", REFUSED_REQUESTS)
def test_requests_the_model_cannot_honor_are_400_without_streaming(
    base_url: str, request_fields: dict[str, object]
) -> None:
    response = speech(base_url, request_fields)

    assert response.status_code == 400, response.text


@pytest.mark.parametrize("request_fields", REFUSED_REQUESTS)
def test_requests_the_model_cannot_honor_are_400_before_the_first_chunk(
    base_url: str, request_fields: dict[str, object]
) -> None:
    response = speech(
        base_url,
        {
            "stream": True,
            "stream_format": "sse",
            "response_format": "pcm",
            **request_fields,
        },
    )

    assert response.status_code == 400, response.text


def test_a_full_round_completes_after_streams_are_dropped(base_url: str) -> None:
    """Shows the server serves on after aborts; it does not show that pools drained."""
    for _ in range(3):
        with httpx.stream(
            "POST",
            f"{base_url}/v1/audio/speech",
            json={
                "model": MODEL_NAME,
                "input": "hello",
                "stream": True,
                "stream_format": "sse",
                "response_format": "pcm",
                "max_new_tokens": 2048,
            },
            timeout=REQUEST_TIMEOUT_S,
        ) as response:
            assert response.status_code == 200
            next(line for line in response.iter_lines() if line.startswith("data: "))

    assert full_round(base_url) == [20 * SAMPLES_PER_FRAME] * MAX_RUNNING_REQUESTS
    assert httpx.get(f"{base_url}/health", timeout=REQUEST_TIMEOUT_S).status_code == 200
