# SPDX-License-Identifier: Apache-2.0
"""Send the audio-only, video-only and joint audio-video AURA Omni demos to a running server.

    python examples/aura_omni/demo_requests.py --api-base http://127.0.0.1:8000 --out-dir /tmp/aura_demo

Each demo prints its request, the reply text, token usage and latency, and writes the
spoken reply (when requested) next to a JSON record of the whole exchange.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

import requests
import soundfile

from tests.utils.synthetic_video import Scene, Shape, write_scene_video

REPO_ROOT = Path(__file__).resolve().parents[2]
QUESTIONS_DIR = REPO_ROOT / "tests" / "data" / "aura_omni"
REQUEST_TIMEOUT_S = 300


def media_part(kind: str, path: Path) -> dict[str, object]:
    return {"type": f"{kind}_url", f"{kind}_url": {"url": f"file://{path}"}}


def run_demo(
    api_base: str,
    out_dir: Path,
    name: str,
    content: list[dict[str, object]],
    modalities: list[str],
) -> None:
    body = {
        "model": "aura",
        "messages": [{"role": "user", "content": content}],
        "modalities": modalities,
        "audio": {"voice": "ryan", "format": "wav"},
        "temperature": 0.0,
        "max_tokens": 96,
    }
    started = time.monotonic()
    response = requests.post(
        f"{api_base}/v1/chat/completions", json=body, timeout=REQUEST_TIMEOUT_S
    )
    latency_s = time.monotonic() - started
    record: dict[str, object] = {
        "demo": name,
        "request": body,
        "status": response.status_code,
        "latency_s": round(latency_s, 3),
    }
    if response.status_code == 200:
        reply = response.json()
        message = reply["choices"][0]["message"]
        record["content"] = message.get("content")
        record["usage"] = reply.get("usage")
        audio = message.get("audio")
        if audio:
            wav_path = out_dir / f"{name}.wav"
            wav_path.write_bytes(base64.b64decode(audio["data"]))
            waveform, sample_rate = soundfile.read(io.BytesIO(wav_path.read_bytes()))
            record["reply_audio"] = {
                "path": str(wav_path),
                "duration_s": round(len(waveform) / sample_rate, 3),
                "sample_rate": sample_rate,
            }
        else:
            pass
    else:
        record["error"] = response.text
    (out_dir / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps({key: value for key, value in record.items() if key != "request"}))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--api-base", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    blue_ball = write_scene_video(
        args.out_dir / "blue_ball.mp4",
        [
            Scene(
                duration_s=6.0,
                shapes=(
                    Shape(
                        kind="circle",
                        rgb=(0, 0, 230),
                        center_x_pixels=320,
                        center_y_pixels=240,
                        half_size_pixels=90,
                    ),
                ),
            )
        ],
    )
    run_demo(
        args.api_base,
        args.out_dir,
        "audio_only",
        [media_part("audio", QUESTIONS_DIR / "capital_of_france.wav")],
        ["text", "audio"],
    )
    run_demo(
        args.api_base,
        args.out_dir,
        "video_only",
        [
            media_part("video", blue_ball),
            {"type": "text", "text": "What color is the ball?"},
        ],
        ["text"],
    )
    run_demo(
        args.api_base,
        args.out_dir,
        "audio_video",
        [
            media_part("video", blue_ball),
            media_part("audio", QUESTIONS_DIR / "secret_word_and_shape.wav"),
        ],
        ["text", "audio"],
    )
    run_demo(
        args.api_base,
        args.out_dir,
        "audio_video_real_clip",
        [
            media_part("video", REPO_ROOT / "tests" / "data" / "draw.mp4"),
            media_part("audio", REPO_ROOT / "tests" / "data" / "query_to_draw.wav"),
        ],
        ["text", "audio"],
    )


if __name__ == "__main__":
    main()
