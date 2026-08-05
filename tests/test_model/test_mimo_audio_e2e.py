# SPDX-License-Identifier: Apache-2.0
"""Opt-in A100 MiMo-Audio real-file transcription validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import requests

from tests.test_model.omni_router_utils import launch_managed_router

MIMO_MODEL_ENV = "SGLANG_OMNI_TEST_MIMO_MODEL"
MIMO_AUDIO_ENV = "SGLANG_OMNI_TEST_MIMO_AUDIO"
MIMO_EXPECT_ENV = "SGLANG_OMNI_TEST_MIMO_EXPECT"
DEFAULT_AUDIO = Path("/workspace/references/MiMo-Audio/examples/北京.mp3")


def _required_path(
    env_name: str,
    default: Path | None = None,
    *,
    file_only: bool = False,
) -> Path:
    raw = os.environ.get(env_name)
    path = Path(raw) if raw else default
    valid = path is not None and (path.is_file() if file_only else path.exists())
    if not valid:
        kind = "file" if file_only else "path"
        pytest.skip(f"set {env_name} to an existing local {kind}")
    return path


@pytest.mark.benchmark
def test_mimo_audio_real_file_to_meaningful_text(
    tmp_path_factory: pytest.TempPathFactory,
) -> None:
    """Exercise real audio through `/v1/audio/transcriptions` on one GPU."""

    if not os.environ.get(MIMO_MODEL_ENV):
        pytest.skip(f"set {MIMO_MODEL_ENV} after approving the MiMo checkpoint")
    model_path = _required_path(MIMO_MODEL_ENV)
    audio_path = _required_path(MIMO_AUDIO_ENV, DEFAULT_AUDIO, file_only=True)

    with (
        launch_managed_router(
            tmp_path_factory=tmp_path_factory,
            model_path=str(model_path),
            model_name="mimo-audio",
            worker_extra_args="",
            num_workers=1,
            num_gpus_per_worker=1,
            wait_timeout=900,
            log_prefix="mimo_audio_e2e_logs",
        ) as router,
        audio_path.open("rb") as audio,
    ):
        response = requests.post(
            f"http://127.0.0.1:{router.port}/v1/audio/transcriptions",
            data={
                "model": "mimo-audio",
                "language": "zh",
                "temperature": "0",
                "max_new_tokens": "128",
            },
            files={"file": (audio_path.name, audio, "audio/mpeg")},
            timeout=300,
        )

    assert response.status_code == 200, response.text
    text = response.json()["text"].strip()
    assert len(text) >= 2
    assert "<|empty|>" not in text
    assert "<|sostm|>" not in text
    expected = os.environ.get(MIMO_EXPECT_ENV)
    if expected:
        assert expected.casefold() in text.casefold()
