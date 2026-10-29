# SPDX-License-Identifier: Apache-2.0
"""AURA Omni on Intel XPU: speech, video and both together reach AURA, and its answer comes back as text and speech.

Starts examples/aura_omni/serve.py with AURA_OMNI_AV_CONFIG (default: the B60 config),
or reuses a running server named by AURA_OMNI_AV_API_BASE (and AURA_OMNI_AV_SERVER_PID
for the device checks). Greedy decoding throughout; assertions name a keyword, never a sentence.

    pytest tests/test_model/test_aura_omni_xpu.py -s -x
"""

from __future__ import annotations

import base64
import io
import json
import os
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import psutil
import pytest
import requests
import soundfile
import torch

from tests.utils import (
    disable_proxy,
    server_log_file,
    start_server_from_cmd,
    stop_server,
)
from tests.utils.synthetic_video import Scene, Shape, write_scene_video

pytestmark = pytest.mark.skipif(
    not torch.xpu.is_available(), reason="AURA Omni XPU tests need Intel XPUs"
)

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "tests" / "data"
QUESTIONS_DIR = DATA_DIR / "aura_omni"
DEFAULT_CONFIG = REPO_ROOT / "examples" / "configs" / "aura_omni_xpu_b60.yaml"
SERVER_PORT = 18961
STARTUP_TIMEOUT_S = 1800
REQUEST_TIMEOUT_S = 300
# AURA's weights alone are 16.5 GiB, more than any speech stage holds.
AURA_MIN_VRAM_BYTES = 15 * 1024**3
SPEECH_MIN_VRAM_BYTES = 1024**3
MIN_REPLY_SPEECH_S = 0.5


def shape(kind: str, rgb: tuple[int, int, int], center_x_pixels: int = 320) -> Shape:
    return Shape(
        kind=kind,
        rgb=rgb,
        center_x_pixels=center_x_pixels,
        center_y_pixels=240,
        half_size_pixels=90,
    )


RED = (220, 0, 0)
BLUE = (0, 0, 230)
GREEN = (0, 170, 0)


@dataclass
class ChatReply:
    content: str
    prompt_tokens: int
    completion_tokens: int
    audio_wav: bytes | None
    latency_s: float


@dataclass
class AuraServer:
    api_base: str
    pid: int | None


@pytest.fixture(scope="module")
def server(tmp_path_factory: pytest.TempPathFactory) -> Iterator[AuraServer]:
    api_base = os.environ.get("AURA_OMNI_AV_API_BASE")
    if api_base:
        server_pid = os.environ.get("AURA_OMNI_AV_SERVER_PID")
        yield AuraServer(api_base, int(server_pid) if server_pid else None)
        return
    else:
        pass
    config = os.environ.get("AURA_OMNI_AV_CONFIG", str(DEFAULT_CONFIG))
    cmd = [
        sys.executable,
        str(REPO_ROOT / "examples" / "aura_omni" / "serve.py"),
        "--config",
        config,
        "--port",
        str(SERVER_PORT),
    ]
    proc = start_server_from_cmd(
        cmd, server_log_file(tmp_path_factory), SERVER_PORT, timeout=STARTUP_TIMEOUT_S
    )
    yield AuraServer(f"http://localhost:{SERVER_PORT}", proc.pid)
    stop_server(proc)


@pytest.fixture(scope="module")
def clips(tmp_path_factory: pytest.TempPathFactory) -> dict[str, str]:
    """Synthetic clips whose content the assertions know exactly."""
    clip_dir = tmp_path_factory.mktemp("aura_clips")
    scenes = {
        "blue_ball": [Scene(duration_s=6.0, shapes=(shape("circle", BLUE),))],
        "green_square": [Scene(duration_s=6.0, shapes=(shape("square", GREEN),))],
        "red_then_blue": [
            Scene(duration_s=3.0, shapes=(shape("square", RED),)),
            Scene(duration_s=3.0, shapes=(shape("circle", BLUE),)),
        ],
        "blue_then_red": [
            Scene(duration_s=3.0, shapes=(shape("circle", BLUE),)),
            Scene(duration_s=3.0, shapes=(shape("square", RED),)),
        ],
        "one_frame": [Scene(duration_s=0.1, shapes=(shape("circle", BLUE),))],
    }
    paths = {
        name: str(write_scene_video(clip_dir / f"{name}.mp4", scene_list))
        for name, scene_list in scenes.items()
    }
    corrupt = clip_dir / "corrupt.mp4"
    corrupt.write_bytes(b"\x00\x00\x00\x18ftypmp42" + b"\x07" * 4096)
    paths["corrupt"] = str(corrupt)
    return paths


def video_part(path: str) -> dict[str, object]:
    return {"type": "video_url", "video_url": {"url": f"file://{path}"}}


def audio_part(path: Path) -> dict[str, object]:
    return {"type": "audio_url", "audio_url": {"url": f"file://{path}"}}


def text_part(text: str) -> dict[str, object]:
    return {"type": "text", "text": text}


def chat_body(
    content: list[dict[str, object]], *, modalities: list[str], stream: bool
) -> dict[str, object]:
    return {
        "model": "aura",
        "messages": [{"role": "user", "content": content}],
        "modalities": modalities,
        "audio": {"voice": "ryan", "format": "wav"},
        "temperature": 0.0,
        "max_tokens": 96,
        "stream": stream,
    }


def chat(
    server: AuraServer,
    content: list[dict[str, object]],
    *,
    modalities: list[str] | None = None,
) -> ChatReply:
    started = time.monotonic()
    with disable_proxy():
        response = requests.post(
            f"{server.api_base}/v1/chat/completions",
            json=chat_body(content, modalities=modalities or ["text"], stream=False),
            timeout=REQUEST_TIMEOUT_S,
        )
    assert response.status_code == 200, response.text
    body = response.json()
    message = body["choices"][0]["message"]
    audio = message.get("audio")
    return ChatReply(
        content=message.get("content") or "",
        prompt_tokens=body["usage"]["prompt_tokens"],
        completion_tokens=body["usage"]["completion_tokens"],
        audio_wav=base64.b64decode(audio["data"]) if audio else None,
        latency_s=time.monotonic() - started,
    )


def status_of(server: AuraServer, content: list[dict[str, object]]) -> int:
    with disable_proxy():
        return requests.post(
            f"{server.api_base}/v1/chat/completions",
            json=chat_body(content, modalities=["text"], stream=False),
            timeout=REQUEST_TIMEOUT_S,
        ).status_code


def first_color(answer: str) -> str:
    positions = {
        color: answer.lower().find(color)
        for color in ("red", "blue")
        if color in answer.lower()
    }
    assert positions, answer
    return min(positions, key=positions.get)


def test_audio_only_question_is_answered_from_speech(server: AuraServer) -> None:
    """The question exists only as speech, so a right answer means Qwen3-ASR's transcript reached AURA."""
    reply = chat(server, [audio_part(QUESTIONS_DIR / "capital_of_france.wav")])

    assert "paris" in reply.content.lower(), reply
    assert reply.prompt_tokens < 200, reply


def test_video_only_question_is_answered_from_frames(
    server: AuraServer, clips: dict[str, str]
) -> None:
    reply = chat(
        server,
        [video_part(clips["blue_ball"]), text_part("What color is the ball?")],
    )

    assert "blue" in reply.content.lower(), reply
    assert reply.prompt_tokens > 500, reply


def test_spoken_question_about_the_clip(
    server: AuraServer, clips: dict[str, str]
) -> None:
    reply = chat(
        server,
        [video_part(clips["blue_ball"]), audio_part(QUESTIONS_DIR / "ball_color.wav")],
    )

    assert "blue" in reply.content.lower(), reply


def test_answer_needs_both_modalities(
    server: AuraServer, clips: dict[str, str]
) -> None:
    """The secret word is only spoken and the shape's color is only shown, and dropping either source loses its half."""
    question = QUESTIONS_DIR / "secret_word_and_shape.wav"
    both = chat(server, [video_part(clips["green_square"]), audio_part(question)])
    speech_only = chat(server, [audio_part(question)])
    video_only = chat(
        server,
        [
            video_part(clips["green_square"]),
            text_part(
                "What is the secret word, and what color is the shape in the video?"
            ),
        ],
    )

    assert "banana" in both.content.lower() and "green" in both.content.lower(), both
    assert "green" not in speech_only.content.lower(), speech_only
    assert "banana" not in video_only.content.lower(), video_only


@pytest.mark.parametrize(
    "clip_name,expected_first", [("red_then_blue", "red"), ("blue_then_red", "blue")]
)
def test_temporal_order_follows_the_clip(
    server: AuraServer, clips: dict[str, str], clip_name: str, expected_first: str
) -> None:
    reply = chat(
        server,
        [video_part(clips[clip_name]), audio_part(QUESTIONS_DIR / "first_color.wav")],
    )

    assert first_color(reply.content) == expected_first, reply


def test_video_wins_when_speech_contradicts_it(
    server: AuraServer, clips: dict[str, str]
) -> None:
    """The speaker says the ball is red while the clip shows a blue one."""
    reply = chat(
        server,
        [
            video_part(clips["blue_ball"]),
            audio_part(QUESTIONS_DIR / "red_ball_claim.wav"),
        ],
    )

    assert "blue" in reply.content.lower(), reply


def test_real_clip_spoken_question(server: AuraServer) -> None:
    reply = chat(
        server,
        [
            video_part(str(DATA_DIR / "draw.mp4")),
            audio_part(DATA_DIR / "query_to_draw.wav"),
        ],
    )

    assert "guitar" in reply.content.lower(), reply


def test_spoken_reply_is_real_speech(server: AuraServer, clips: dict[str, str]) -> None:
    reply = chat(
        server,
        [video_part(clips["blue_ball"]), audio_part(QUESTIONS_DIR / "ball_color.wav")],
        modalities=["text", "audio"],
    )

    assert reply.audio_wav is not None, reply
    waveform, sample_rate = soundfile.read(io.BytesIO(reply.audio_wav), dtype="float32")
    assert len(waveform) / sample_rate >= MIN_REPLY_SPEECH_S
    assert float(np.sqrt(np.mean(waveform**2))) > 0.01
    assert "blue" in reply.content.lower(), reply


def test_streaming_delivers_text_then_speech(
    server: AuraServer, clips: dict[str, str]
) -> None:
    text_deltas: list[str] = []
    audio_chunks = 0
    with disable_proxy():
        response = requests.post(
            f"{server.api_base}/v1/chat/completions",
            json=chat_body(
                [
                    video_part(clips["blue_ball"]),
                    audio_part(QUESTIONS_DIR / "ball_color.wav"),
                ],
                modalities=["text", "audio"],
                stream=True,
            ),
            timeout=REQUEST_TIMEOUT_S,
            stream=True,
        )
        assert response.status_code == 200, response.text
        for line in response.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: ") or line == "data: [DONE]":
                continue
            else:
                pass
            delta = json.loads(line[len("data: ") :])["choices"][0]["delta"]
            if delta.get("content"):
                text_deltas.append(delta["content"])
            elif delta.get("audio", {}).get("data"):
                audio_chunks += 1
            else:
                pass

    assert "blue" in "".join(text_deltas).lower(), text_deltas
    assert audio_chunks >= 1


@pytest.mark.parametrize("clip_name", ["corrupt", "one_frame"])
def test_unreadable_video_is_a_bad_request(
    server: AuraServer, clips: dict[str, str], clip_name: str
) -> None:
    status = status_of(
        server, [video_part(clips[clip_name]), text_part("What do you see?")]
    )

    assert status == 400


def test_stages_hold_their_weights_on_xpu(server: AuraServer) -> None:
    """Every model stage keeps its weights in XPU memory, and none of them is allowed to fall back to CPU kernels."""
    if server.pid is None:
        pytest.skip(
            "set AURA_OMNI_AV_SERVER_PID to check a server this test did not start"
        )
    else:
        pass
    vram_by_card: dict[str, int] = {}
    for process in psutil.Process(server.pid).children(recursive=True):
        assert "PYTORCH_ENABLE_XPU_FALLBACK" not in process.environ(), process
        for fdinfo in Path(f"/proc/{process.pid}/fdinfo").iterdir():
            fields = dict(
                line.split(":", 1)
                for line in fdinfo.read_text().splitlines()
                if ":" in line
            )
            if "drm-total-vram0" in fields and "drm-pdev" in fields:
                card = fields["drm-pdev"].strip()
                vram_kib = int(fields["drm-total-vram0"].split()[0])
                vram_by_card[card] = max(vram_by_card.get(card, 0), vram_kib * 1024)
            else:
                pass

    held = sorted(vram_by_card.values(), reverse=True)
    assert len(held) >= 2, vram_by_card
    assert held[0] >= AURA_MIN_VRAM_BYTES, vram_by_card
    assert held[1] >= SPEECH_MIN_VRAM_BYTES, vram_by_card
