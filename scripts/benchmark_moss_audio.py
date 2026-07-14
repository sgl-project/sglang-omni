#!/usr/bin/env python3
"""Benchmark an OpenAI-compatible MOSS transcription endpoint on local audio."""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import statistics
import subprocess
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

import requests


DEFAULT_PROMPT = (
    "请将音频转写为文本，每一段需以起始时间戳和说话人编号"
    "（[S01]、[S02]、[S03]…）开头，正文为对应的语音内容，"
    "并在段末标注结束时间戳，以清晰标明该段语音范围。"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default="OpenMOSS-Team/MOSS-Transcribe-Diarize")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=11500,
        help=(
            "Per-request output budget. The model repository default (5120) "
            "truncates roughly 17-minute conversational recordings."
        ),
    )
    parser.add_argument("--timeout", type=float, default=3600)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--omit-prompt",
        action="store_true",
        help="Exercise the server-side default instead of sending the official prompt.",
    )
    parser.add_argument(
        "--reuse-identical-audio",
        action="store_true",
        help=(
            "Send the exact same waveform to every concurrent request. By "
            "default, concurrent runs use lossless volume variants so the "
            "multimodal cache cannot inflate throughput."
        ),
    )
    return parser.parse_args()


def wav_duration(path: Path) -> float:
    with wave.open(str(path), "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def build_audio_variants(
    args: argparse.Namespace, work_dir: Path
) -> list[Path]:
    if args.concurrency == 1 or args.reuse_identical_audio:
        return [args.audio] * args.concurrency

    variants: list[Path] = []
    for request_id in range(args.concurrency):
        output = work_dir / f"input-{request_id}.wav"
        # Change only PCM amplitude. The recording remains lossless and its
        # duration is unchanged, but every request receives distinct content.
        factor = 0.82 + (args.concurrency * 7 + request_id + 1) / 10000
        subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-loglevel",
                "error",
                "-y",
                "-i",
                str(args.audio),
                "-filter:a",
                f"volume={factor:.4f}",
                "-c:a",
                "pcm_s16le",
                str(output),
            ],
            check=True,
        )
        variants.append(output)
    return variants


def transcribe(
    args: argparse.Namespace, request_id: int, audio_path: Path
) -> dict[str, Any]:
    form = {
        "model": args.model,
        "response_format": "verbose_json",
        "max_new_tokens": str(args.max_new_tokens),
    }
    if not args.omit_prompt:
        form["prompt"] = DEFAULT_PROMPT
    started = time.perf_counter()
    try:
        with audio_path.open("rb") as audio_file:
            response = requests.post(
                f"{args.base_url.rstrip('/')}/v1/audio/transcriptions",
                data=form,
                files={"file": (audio_path.name, audio_file, "audio/wav")},
                timeout=args.timeout,
            )
        latency = time.perf_counter() - started
        try:
            payload: Any = response.json()
        except ValueError:
            payload = {"body": response.text}
        return {
            "request_id": request_id,
            "ok": response.ok,
            "status": response.status_code,
            "latency_s": latency,
            "response": payload,
        }
    except requests.RequestException as exc:
        return {
            "request_id": request_id,
            "ok": False,
            "status": None,
            "latency_s": time.perf_counter() - started,
            "error": str(exc),
        }


def write_markdown(args: argparse.Namespace, summary: dict[str, Any], results: list[dict[str, Any]]) -> None:
    lines = [
        f"# {args.audio.stem} — MOSS / SGLang-Omni XPU",
        "",
        f"- 音频时长：{summary['audio_duration_s']:.2f} 秒",
        f"- 并发数：{summary['concurrency']}",
        f"- 最大输出 token：{summary['max_new_tokens']}",
        f"- 并发音频：{'content-distinct 无损变体' if summary['content_distinct_audio'] else '同一波形'}",
        f"- 总耗时：{summary['wall_clock_s']:.2f} 秒",
        f"- 成功/失败：{summary['successful_requests']}/{summary['failed_requests']}",
        f"- 聚合处理速度：{summary['aggregate_audio_s_per_wall_s']:.2f}× 实时"
        if summary["aggregate_audio_s_per_wall_s"] is not None
        else "- 聚合处理速度：不可用",
        f"- 提示词：{'显式发送官方提示词' if not args.omit_prompt else '使用服务端官方默认提示词'}",
        "",
        "## 转写结果",
        "",
    ]
    first_success = next((result for result in results if result["ok"]), None)
    payload = first_success.get("response", {}) if first_success else {}
    segments = payload.get("segments") if isinstance(payload, dict) else None
    if isinstance(segments, list) and segments:
        for segment in segments:
            if not isinstance(segment, dict):
                continue
            start = segment.get("start")
            end = segment.get("end")
            text = str(segment.get("text", "")).strip()
            lines.append(f"[{start}–{end}] {text}")
            lines.append("")
    elif isinstance(payload, dict) and isinstance(payload.get("text"), str):
        lines.extend([payload["text"].strip(), ""])
    else:
        lines.extend(["转写失败；请查看同名 JSON 文件。", ""])
    args.output.with_suffix(".md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1")
    if args.max_new_tokens < 1:
        raise SystemExit("--max-new-tokens must be at least 1")
    if not args.audio.is_file():
        raise SystemExit(f"Audio file not found: {args.audio}")

    duration = wav_duration(args.audio)
    with tempfile.TemporaryDirectory(prefix="moss-sglang-bench-") as temp_dir:
        audio_paths = build_audio_variants(args, Path(temp_dir))
        wall_started = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(args.concurrency) as executor:
            results = list(
                executor.map(
                    lambda item: transcribe(args, *item),
                    enumerate(audio_paths),
                )
            )
        wall_s = time.perf_counter() - wall_started
    successes = [result for result in results if result["ok"]]
    latencies = [float(result["latency_s"]) for result in successes]
    summary = {
        "engine": "sglang-omni-xpu",
        "audio": str(args.audio.resolve()),
        "audio_duration_s": duration,
        "concurrency": args.concurrency,
        "max_new_tokens": args.max_new_tokens,
        "content_distinct_audio": (
            args.concurrency > 1 and not args.reuse_identical_audio
        ),
        "official_prompt_sent_explicitly": not args.omit_prompt,
        "wall_clock_s": wall_s,
        "successful_requests": len(successes),
        "failed_requests": len(results) - len(successes),
        "latency_mean_s": statistics.mean(latencies) if latencies else None,
        "latency_p50_s": statistics.median(latencies) if latencies else None,
        "latency_max_s": max(latencies) if latencies else None,
        "aggregate_audio_s_per_wall_s": (
            duration * len(successes) / wall_s if wall_s else None
        ),
        "realtime_factor_wall": (
            wall_s / (duration * len(successes)) if successes and duration else None
        ),
    }
    output = {"summary": summary, "requests": results}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_markdown(args, summary, results)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if len(successes) == len(results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
