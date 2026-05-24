# SPDX-License-Identifier: Apache-2.0
"""（wenyao）Launch Ming V1 speech server and run TP4 smoke tests from one terminal.

The script starts ``examples/run_ming_omni_speech_server.py`` in a subprocess,
waits for ``/health``, then exercises the OpenAI-compatible endpoints.

Example::

    CUDA_VISIBLE_DEVICES=3,4,5,6,7 python -u run_ming_tp4.py

Use ``--skip-server`` if a server is already running on the target port.
Use ``--run-mmmu`` to run the MMMU CI subset after the smoke tests.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--model-path", default="inclusionAI/Ming-flash-omni-2.0")
    parser.add_argument(
        "--launcher",
        default=str(
            Path(__file__).resolve().parents[2]
            / "examples"
            / "run_ming_omni_speech_server.py"
        ),
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--cuda-visible-devices",
        default=os.environ.get("CUDA_VISIBLE_DEVICES", "3,4,5,6,7"),
    )
    parser.add_argument("--gpu-thinker", type=int, default=0)
    parser.add_argument("--gpu-talker", type=int, default=4)
    parser.add_argument("--tp-size", type=int, default=4)
    parser.add_argument("--mem-fraction-static", type=float, default=0.80)
    parser.add_argument("--startup-timeout", type=float, default=900)
    parser.add_argument("--request-timeout", type=float, default=300)
    parser.add_argument("--output-dir", default="/data/repo/logs")
    parser.add_argument("--run-log", default=None)
    parser.add_argument("--no-run-log", action="store_true")
    parser.add_argument("--keep-server", action="store_true")
    parser.add_argument("--skip-server", action="store_true")
    parser.add_argument(
        "--run-mmmu",
        action="store_true",
        help="Run the MMMU image-text benchmark after smoke tests.",
    )
    parser.add_argument(
        "--mmmu-repo-id",
        default="zhaochenyang20/mmmu-ci-50",
        help="HuggingFace dataset repo for MMMU. Defaults to the CI-50 subset.",
    )
    parser.add_argument("--mmmu-max-samples", type=int, default=50)
    parser.add_argument("--mmmu-max-tokens", type=int, default=512)
    parser.add_argument("--mmmu-temperature", type=float, default=0.0)
    parser.add_argument("--mmmu-warmup", type=int, default=2)
    parser.add_argument("--mmmu-max-concurrency", type=int, default=1)
    parser.add_argument(
        "--mmmu-output-dir",
        default=None,
        help="Output directory for MMMU results. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--run-mmsu",
        action="store_true",
        help="Run the MMSU audio-text benchmark after smoke tests.",
    )
    parser.add_argument("--mmsu-max-samples", type=int, default=50)
    parser.add_argument("--mmsu-max-tokens", type=int, default=32)
    parser.add_argument("--mmsu-temperature", type=float, default=0.0)
    parser.add_argument("--mmsu-warmup", type=int, default=1)
    parser.add_argument("--mmsu-max-concurrency", type=int, default=1)
    parser.add_argument(
        "--mmsu-modalities",
        choices=["text", "text+audio"],
        default="text+audio",
    )
    parser.add_argument(
        "--mmsu-output-dir",
        default=None,
        help="Output directory for MMSU results. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--run-tts",
        action="store_true",
        help="Run the SeedTTS benchmark after smoke tests.",
    )
    parser.add_argument("--tts-max-samples", type=int, default=20)
    parser.add_argument("--tts-max-new-tokens", type=int, default=256)
    parser.add_argument("--tts-temperature", type=float, default=0.7)
    parser.add_argument("--tts-warmup", type=int, default=1)
    parser.add_argument("--tts-max-concurrency", type=int, default=1)
    parser.add_argument(
        "--tts-meta",
        default="zhaochenyang20/seed-tts-eval-arrow",
        help="HuggingFace Arrow/Parquet dataset repo id or local meta.lst path.",
    )
    parser.add_argument("--tts-lang", choices=["en", "zh"], default="en")
    parser.add_argument(
        "--tts-speaker",
        default="Ethan",
        choices=["Ethan", "Chelsie", "Aiden"],
    )
    parser.add_argument(
        "--tts-generate-only",
        action="store_true",
        help="Skip WER transcription phase (generate audio + measure speed only).",
    )
    parser.add_argument(
        "--tts-output-dir",
        default=None,
        help="Output directory for TTS results. Defaults under --output-dir.",
    )
    parser.add_argument(
        "--quiet-server-log",
        action="store_true",
        help="Do not mirror the server log to stdout while tests are running.",
    )
    return parser.parse_args()


class _TeeStream:
    def __init__(self, *streams):
        self._streams = streams

    def write(self, data: str) -> int:
        for stream in self._streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self._streams:
            stream.flush()


def _url(args: argparse.Namespace, path: str) -> str:
    return f"http://{args.host}:{args.port}{path}"


def _get_json(args: argparse.Namespace, path: str, timeout: float = 10) -> dict:
    request = urllib.request.Request(_url(args, path), method="GET")
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def _post_json(
    args: argparse.Namespace,
    path: str,
    payload: dict,
    timeout: float,
) -> dict:
    request = urllib.request.Request(
        _url(args, path),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail[:1000]}") from exc


def _stream_sse(
    args: argparse.Namespace,
    payload: dict,
    timeout: float,
) -> list[dict]:
    request = urllib.request.Request(
        _url(args, "/v1/chat/completions"),
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    events: list[dict] = []
    with urllib.request.urlopen(request, timeout=timeout) as response:
        while True:
            line = response.readline()
            if not line:
                break
            line_text = line.decode("utf-8", "replace").strip()
            if not line_text.startswith("data: "):
                continue
            data = line_text[len("data: ") :]
            if data == "[DONE]":
                break
            events.append(json.loads(data))
    return events


def _start_server(args: argparse.Namespace, log_path: Path) -> subprocess.Popen:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.cuda_visible_devices
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    command = [
        sys.executable,
        "-u",
        args.launcher,
        "--version",
        "v1",
        "--model-path",
        args.model_path,
        "--tp-size",
        str(args.tp_size),
        "--gpu-thinker",
        str(args.gpu_thinker),
        "--gpu-talker",
        str(args.gpu_talker),
        "--mem-fraction-static",
        str(args.mem_fraction_static),
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    print("[server] " + " ".join(command), flush=True)
    print(f"[server] CUDA_VISIBLE_DEVICES={args.cuda_visible_devices}", flush=True)
    print(f"[server] log: {log_path}", flush=True)
    log_file = open(log_path, "w", buffering=1)
    process = subprocess.Popen(
        command,
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=os.getcwd(),
        start_new_session=True,
        text=True,
    )
    process._log_file = log_file  # type: ignore[attr-defined]
    if not args.quiet_server_log:
        thread = threading.Thread(
            target=_mirror_server_log,
            args=(log_path, process),
            name="server-log-mirror",
            daemon=True,
        )
        thread.start()
        process._log_thread = thread  # type: ignore[attr-defined]
    return process


def _mirror_server_log(log_path: Path, process: subprocess.Popen) -> None:
    """Mirror the server log file to stdout for one-terminal smoke runs."""
    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as reader:
            while process.poll() is None:
                line = reader.readline()
                if line:
                    print(f"[server-log] {line}", end="", flush=True)
                else:
                    time.sleep(0.2)
            for line in reader:
                print(f"[server-log] {line}", end="", flush=True)
    except Exception as exc:
        print(f"[server-log] mirror stopped: {exc}", flush=True)


def _wait_ready(args: argparse.Namespace, process: subprocess.Popen | None) -> None:
    deadline = time.time() + args.startup_timeout
    last_error = ""
    while time.time() < deadline:
        if process is not None and process.poll() is not None:
            raise RuntimeError(f"server exited early with code {process.returncode}")
        try:
            health = _get_json(args, "/health", timeout=5)
            if health.get("status") == "healthy" or health.get("running") is True:
                print("[ready] /health OK", flush=True)
                return
            last_error = str(health)
        except Exception as exc:
            last_error = str(exc)
        time.sleep(2)
    raise TimeoutError(f"server did not become healthy: {last_error}")


def _show_text_response(name: str, body: dict) -> str:
    choices = body.get("choices") or []
    if not choices:
        raise AssertionError(f"{name}: missing choices: {body}")
    message = choices[0].get("message") or {}
    text = message.get("content") or ""
    if not isinstance(text, str):
        raise AssertionError(f"{name}: missing text content: {body}")
    print(f"\n=== {name} OUTPUT ===\n{text}\n=== END {name} OUTPUT ===", flush=True)
    return text


def _show_stream_response(name: str, events: list[dict]) -> None:
    text_parts: list[str] = []
    for event in events:
        choices = event.get("choices") or []
        for choice in choices:
            delta = choice.get("delta") or {}
            content = delta.get("content")
            if isinstance(content, str):
                text_parts.append(content)
    text = "".join(text_parts)
    print(
        f"\n=== {name} STREAM OUTPUT ===\n{text}\n=== END {name} STREAM OUTPUT ===",
        flush=True,
    )


def _run_smoke_tests(args: argparse.Namespace) -> None:
    results: list[tuple[str, bool, float, str]] = []

    def run(name: str, fn) -> None:
        print(f"[test] {name}", flush=True)
        started = time.time()
        try:
            fn()
            results.append((name, True, time.time() - started, ""))
        except Exception as exc:
            results.append((name, False, time.time() - started, str(exc)))
            print(f"[fail] {name}: {exc}", flush=True)
            raise

    run("health", lambda: _get_json(args, "/health", timeout=10))
    run("models", lambda: _get_json(args, "/v1/models", timeout=10))

    def text_chat() -> None:
        body = _post_json(
            args,
            "/v1/chat/completions",
            {
                "model": "ming-omni",
                "messages": [{"role": "user", "content": "What is capital of japan?"}],
                "max_tokens": 256,
                "temperature": 0,
                "top_p": 1,
            },
            args.request_timeout,
        )
        _show_text_response("text_chat", body)

    run("text_chat", text_chat)

    def image_text_chat() -> None:
        body = _post_json(
            args,
            "/v1/chat/completions",
            {
                "model": "ming-omni",
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": "https://picsum.photos/id/237/300/200"
                                },
                            },
                            {
                                "type": "text",
                                "text": "Describe what you see in this image.",
                            },
                        ],
                    }
                ],
                "max_tokens": 256,
            },
            args.request_timeout,
        )
        _show_text_response("image_text_chat", body)

    run("image_text_chat", image_text_chat)

    def stream_text_chat() -> None:
        events = _stream_sse(
            args,
            {
                "model": "ming-omni",
                "messages": [{"role": "user", "content": "Say: stream ok"}],
                "modalities": ["text"],
                "stream": True,
                "max_tokens": 256,
                "temperature": 0.2,
            },
            args.request_timeout,
        )
        if not events:
            raise AssertionError("no SSE events")
        print(f"[ok] stream_text_chat: events={len(events)}", flush=True)
        _show_stream_response("stream_text_chat", events)

    run("stream_text_chat", stream_text_chat)

    failed = [item for item in results if not item[1]]
    print("\n=== SUMMARY ===", flush=True)
    for name, ok, seconds, error in results:
        status = "PASS" if ok else "FAIL"
        print(f"{status} {name:18s} {seconds:8.2f}s {error[:180]}", flush=True)
    if failed:
        raise SystemExit(1)


def _run_mmmu_benchmark(args: argparse.Namespace) -> None:
    output_dir = (
        Path(args.mmmu_output_dir)
        if args.mmmu_output_dir
        else Path(args.output_dir) / "mmmu_ming_ci50"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_mmmu",
        "--model",
        "ming-omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--repo-id",
        args.mmmu_repo_id,
        "--max-samples",
        str(args.mmmu_max_samples),
        "--max-concurrency",
        str(args.mmmu_max_concurrency),
        "--warmup",
        str(args.mmmu_warmup),
        "--max-tokens",
        str(args.mmmu_max_tokens),
        "--temperature",
        str(args.mmmu_temperature),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    print("\n[test] mmmu", flush=True)
    print("[mmmu] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=os.getcwd(), env=env, check=True)


def _run_mmsu_benchmark(args: argparse.Namespace) -> None:
    output_dir = (
        Path(args.mmsu_output_dir)
        if args.mmsu_output_dir
        else Path(args.output_dir) / "mmsu_ming"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_mmsu",
        "--model",
        "ming-omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--modalities",
        args.mmsu_modalities,
        "--max-samples",
        str(args.mmsu_max_samples),
        "--max-concurrency",
        str(args.mmsu_max_concurrency),
        "--warmup",
        str(args.mmsu_warmup),
        "--max-tokens",
        str(args.mmsu_max_tokens),
        "--temperature",
        str(args.mmsu_temperature),
        "--output-dir",
        str(output_dir),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    print("\n[test] mmsu", flush=True)
    print("[mmsu] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=os.getcwd(), env=env, check=True)


_MING_TTS_SYSTEM_PROMPT_EN = (
    "You are a text-to-speech engine. Read aloud only the exact text the user "
    "asks you to speak. Do not add greetings, preambles, suffixes, "
    'explanations, apologies, or refusals. Do not say phrases like "Sure", '
    '"Here is", "In English", or "I am an AI". Output the spoken text '
    "verbatim and nothing else."
)
_MING_TTS_SYSTEM_PROMPT_ZH = (
    "你是一个文本转语音引擎。只朗读用户给出的原文，逐字朗读。"
    "不要添加任何开场白、前缀、后缀、解释、道歉或拒绝。"
    '不要说"好的"、"以下是"、"用中文"或"我是 AI"之类的话。'
    "只输出原文对应的语音，不要任何额外内容。"
)


def _run_tts_benchmark(args: argparse.Namespace) -> None:
    output_dir = (
        Path(args.tts_output_dir)
        if args.tts_output_dir
        else Path(args.output_dir) / "tts_ming"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    system_prompt = (
        _MING_TTS_SYSTEM_PROMPT_ZH
        if args.tts_lang == "zh"
        else _MING_TTS_SYSTEM_PROMPT_EN
    )
    command = [
        sys.executable,
        "-m",
        "benchmarks.eval.benchmark_omni_seedtts",
        "--model",
        "ming-omni",
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--meta",
        args.tts_meta,
        "--lang",
        args.tts_lang,
        "--speaker",
        args.tts_speaker,
        "--max-samples",
        str(args.tts_max_samples),
        "--max-new-tokens",
        str(args.tts_max_new_tokens),
        "--temperature",
        str(args.tts_temperature),
        "--warmup",
        str(args.tts_warmup),
        "--max-concurrency",
        str(args.tts_max_concurrency),
        "--output-dir",
        str(output_dir),
        "--system-prompt",
        system_prompt,
    ]
    if args.tts_generate_only:
        command.append("--generate-only")
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{os.getcwd()}:{env.get('PYTHONPATH', '')}"
    print("\n[test] tts", flush=True)
    print("[tts] " + " ".join(command), flush=True)
    subprocess.run(command, cwd=os.getcwd(), env=env, check=True)


def _stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    print("[server] stopping...", flush=True)
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except Exception:
        process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except Exception:
            process.kill()
    log_thread = getattr(process, "_log_thread", None)
    if log_thread is not None:
        log_thread.join(timeout=2)
    log_file = getattr(process, "_log_file", None)
    if log_file is not None:
        log_file.close()


def _tail(path: Path, n: int = 160) -> None:
    if not path.exists():
        return
    lines = path.read_text(errors="replace").splitlines()
    print(f"\n=== SERVER LOG TAIL: {path} ===", flush=True)
    for line in lines[-n:]:
        print(line, flush=True)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "server.log"
    run_log_file = None
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    if not args.no_run_log:
        run_log_path = Path(args.run_log) if args.run_log else output_dir / "run.log"
        run_log_path.parent.mkdir(parents=True, exist_ok=True)
        run_log_file = open(run_log_path, "w", buffering=1)
        sys.stdout = _TeeStream(original_stdout, run_log_file)  # type: ignore[assignment]
        sys.stderr = _TeeStream(original_stderr, run_log_file)  # type: ignore[assignment]
        print(f"[runner] log: {run_log_path}", flush=True)

    process: subprocess.Popen | None = None
    try:
        if not args.skip_server:
            process = _start_server(args, log_path)
        _wait_ready(args, process)
        _run_smoke_tests(args)
        if args.run_mmmu:
            _run_mmmu_benchmark(args)
        if args.run_mmsu:
            _run_mmsu_benchmark(args)
        if args.run_tts:
            _run_tts_benchmark(args)
        if args.keep_server:
            print(f"[server] keeping server alive; log={log_path}", flush=True)
            process = None
    except Exception:
        _tail(log_path)
        raise
    finally:
        if not args.keep_server:
            _stop_server(process)
        if run_log_file is not None:
            sys.stdout = original_stdout
            sys.stderr = original_stderr
            run_log_file.close()


if __name__ == "__main__":
    main()
