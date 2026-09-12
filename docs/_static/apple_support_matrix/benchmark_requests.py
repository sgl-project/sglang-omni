# SPDX-License-Identifier: Apache-2.0
"""Record serial ASR HTTP requests using only the Python standard library.

The primary audio is loaded once and identical bytes are used for every request.
Latency covers urlopen() through reading the complete response, excluding disk
reads and multipart construction. The server must already be ready. This is a
small end-to-end functional/timing check, not a dataset accuracy evaluation.
"""

import argparse
import hashlib
import json
import mimetypes
import statistics
import sys
import time
import uuid
from datetime import datetime, timezone
from http.client import HTTPException
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def load_audio(path):
    path = Path(path).expanduser().resolve(strict=True)
    data = path.read_bytes()
    if not data:
        raise ValueError(f"Audio file is empty: {path}")
    return {
        "path": str(path),
        "filename": path.name,
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "content_type": mimetypes.guess_type(path.name)[0]
        or "application/octet-stream",
    }, data


def multipart(model, audio, data):
    boundary = "asr-benchmark-" + uuid.uuid4().hex
    parts = []
    for name, value in (
        ("model", model),
        ("response_format", "json"),
        ("temperature", "0"),
        ("stream", "false"),
    ):
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"'
            f"\r\n\r\n{value}\r\n".encode("utf-8")
        )
    # Keep the multipart filename safe while preserving its extension.
    filename = audio["filename"].replace("\\", "_").replace('"', "_")
    filename = filename.replace("\r", "_").replace("\n", "_")
    parts.extend(
        [
            (
                f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
                f'filename="{filename}"\r\n'
                f'Content-Type: {audio["content_type"]}\r\n\r\n'
            ).encode("utf-8"),
            data,
            f"\r\n--{boundary}--\r\n".encode("ascii"),
        ]
    )
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def run_request(endpoint, model, audio, data, phase, index, timeout):
    body, content_type = multipart(model, audio, data)
    request = Request(
        endpoint,
        data=body,
        headers={"Content-Type": content_type, "Accept": "application/json"},
        method="POST",
    )
    record = {
        "phase": phase,
        "index": index,
        "started_at": utc_now(),
        "audio_path": audio["path"],
        "audio_sha256": audio["sha256"],
        "status": None,
        "ok": False,
    }
    started = time.perf_counter()
    try:
        try:
            response = urlopen(request, timeout=timeout)
        except HTTPError as error:
            response = error
        with response:
            record["status"] = response.status
            raw = response.read()
            record["latency_seconds"] = time.perf_counter() - started
            record["response_content_type"] = response.headers.get("Content-Type")
        record["response_text"] = raw.decode("utf-8", errors="replace")
        try:
            record["response_json"] = json.loads(record["response_text"])
        except json.JSONDecodeError:
            record["response_json"] = None
        record["ok"] = 200 <= record["status"] < 300
        if not record["ok"]:
            record["error"] = f'HTTP {record["status"]}'
    except (URLError, OSError, ValueError, HTTPException) as error:
        record["latency_seconds"] = time.perf_counter() - started
        record["error"] = f"{type(error).__name__}: {error}"
    return record


def summarize(records):
    summary = {}
    for phase in ("warmup", "measured", "smoke"):
        phase_records = [row for row in records if row["phase"] == phase]
        successful = [row for row in phase_records if row["ok"]]
        result = {
            "completed": len(phase_records),
            "successful": len(successful),
            "failed": len(phase_records) - len(successful),
        }
        # Smoke inputs are intentionally not pooled into timing statistics.
        if phase == "measured" and successful:
            times = [row["latency_seconds"] for row in successful]
            result["latency_seconds"] = {
                "median": statistics.median(times),
                "mean": statistics.mean(times),
                "min": min(times),
                "max": max(times),
            }
        summary[phase] = result
    return summary


def save_result(path, result):
    result["summary"] = summarize(result["requests"])
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--audio", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--requests", type=int, default=20)
    parser.add_argument("--smoke-audio", action="append", default=[])
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    if args.warmups < 0 or args.requests < 1 or args.timeout <= 0:
        parser.error("warmups must be >= 0; requests and timeout must be > 0")
    output = Path(args.output).expanduser().resolve()
    if output.exists():
        parser.error(f"Refusing to overwrite existing evidence: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        primary = load_audio(args.audio)
        smoke = [load_audio(path) for path in args.smoke_audio]
    except (OSError, ValueError) as error:
        parser.error(str(error))
    base_url = args.base_url.rstrip("/")
    endpoint = base_url + (
        "/audio/transcriptions"
        if base_url.endswith("/v1")
        else "/v1/audio/transcriptions"
    )
    result = {
        "schema_version": 1,
        "started_at": utc_now(),
        "finished_at": None,
        "complete": False,
        "endpoint": endpoint,
        "model": args.model,
        "settings": {
            "concurrency": 1,
            "warmups": args.warmups,
            "measured_requests": args.requests,
            "timeout_seconds": args.timeout,
            "response_format": "json",
            "temperature": 0,
            "stream": False,
            "other_generation_settings": "server defaults",
            "latency_scope": "client HTTP open through complete response read",
            "audio_bytes": "loaded once; identical bytes reused per input",
            "execution_order": "primary warmup, primary measured, smoke inputs",
            "failure_policy": "save error and abort on first failed request",
            "success_criterion": "HTTP 2xx; transcripts require separate inspection",
        },
        "primary_audio": primary[0],
        "smoke_audio": [item[0] for item in smoke],
        "requests": [],
    }
    save_result(output, result)
    batches = [("warmup", args.warmups, primary), ("measured", args.requests, primary)]
    batches.extend(("smoke", 1, item) for item in smoke)
    failed = False
    smoke_index = 0
    try:
        for phase, count, (audio, data) in batches:
            for index in range(1, count + 1):
                if phase == "smoke":
                    smoke_index += 1
                    index = smoke_index
                record = run_request(
                    endpoint, args.model, audio, data, phase, index, args.timeout
                )
                result["requests"].append(record)
                save_result(output, result)
                print(
                    f'{phase} {index}: status={record["status"]} '
                    f'{record["latency_seconds"]:.4f}s',
                    file=sys.stderr,
                    flush=True,
                )
                if not record["ok"]:
                    failed = True
                    result["abort_reason"] = record["error"]
                    break
            if failed:
                break
    except KeyboardInterrupt:
        failed = True
        result["abort_reason"] = "Interrupted by user; in-flight request not completed"
    result["finished_at"] = utc_now()
    result["complete"] = not failed
    save_result(output, result)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
