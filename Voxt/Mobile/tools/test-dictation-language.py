#!/usr/bin/env python3
"""Check that the synthetic Chinese shopping list stays Chinese on both audio routes."""
import argparse
import base64
import datetime
import json
import os
import time
import urllib.error
import urllib.request
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--base-url", required=True)
parser.add_argument("--audio", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
token = os.environ["VOXT_TEST_TOKEN"]


def request(path, body=None, content_type=None, authorization=token):
    headers = {"Authorization": "Bearer " + authorization}
    if content_type:
        headers["Content-Type"] = content_type
    start = time.monotonic()
    req = urllib.request.Request(
        args.base_url.rstrip("/") + path, data=body, headers=headers
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as response:
            return (
                response.status,
                json.load(response),
                round((time.monotonic() - start) * 1000),
            )
    except urllib.error.HTTPError as error:
        return error.code, {}, round((time.monotonic() - start) * 1000)


def is_chinese_list(text):
    return all(word in text for word in ["苹果", "牛奶", "面包"])


assert is_chinese_list("苹果、牛奶、面包。")
assert not is_chinese_list("Apples, milk, and bread.")
assert request("/v1/health", authorization="invalid-language-regression")[0] == 401
assert request("/v1/health")[0] == 200
audio = args.audio.read_bytes()
compose = json.dumps(
    {
        "audio": base64.b64encode(audio).decode(),
        "contentType": "audio/wav",
        "options": {"mode": "dictate", "style": "automatic"},
    }
).encode()
results = []
for path, body, content_type in [
    ("/v1/compose", compose, "application/json"),
    ("/v1/dictations?enhance=0", audio, "audio/wav"),
]:
    status, response, elapsed = request(path, body, content_type)
    row = {
        "path": path,
        "status": status,
        "elapsedMs": elapsed,
        **{key: response.get(key) for key in ["rawText", "text", "asrMs", "enhanceMs"]},
    }
    row["passed"] = status == 200 and all(
        is_chinese_list(response.get(key, "")) for key in ["rawText", "text"]
    )
    results.append(row)
    print(json.dumps(row, ensure_ascii=False), flush=True)
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(
    json.dumps(
        {
            "date": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "results": results,
        },
        ensure_ascii=False,
        indent=2,
    )
    + "\n"
)
assert all(
    row["passed"] for row in results
), "Chinese dictation must retain Chinese raw and final text"
