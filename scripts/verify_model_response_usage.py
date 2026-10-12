#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Verify live AR Chat response/usage contracts; SenseNova Images is not covered."""

import argparse
import json
import os
import urllib.error
import urllib.request
from pathlib import Path


def check_usage(usage, expect_cached_details):
    assert isinstance(usage, dict), "Missing real text usage"
    for key in ("prompt_tokens", "completion_tokens", "total_tokens"):
        assert type(usage.get(key)) is int and usage[key] >= 0, f"Invalid {key}"
    assert usage["total_tokens"] == usage["prompt_tokens"] + usage["completion_tokens"]
    details = usage.get("prompt_tokens_details")
    if expect_cached_details:
        assert (
            isinstance(details, dict) and "cached_tokens" in details
        ), "Missing cache details"
    if details is not None:
        assert isinstance(details, dict), "Invalid prompt_tokens_details"
        for key, value in details.items():
            assert type(value) is int and value >= 0, f"Invalid detail {key}"
        if "cached_tokens" in details:
            assert details["cached_tokens"] <= usage["prompt_tokens"]
    reasoning = usage.get("reasoning_tokens")
    if reasoning is not None:
        assert type(reasoning) is int and 0 <= reasoning <= usage["completion_tokens"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--expect-cached-details", action="store_true")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.timeout <= 0:
        parser.error("max-tokens and timeout must be positive")
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for case, streaming, include_usage in (
        ("nonstream", False, False),
        ("stream-default", True, False),
        ("stream-usage", True, True),
    ):
        record = {"case": case, "passed": False}
        try:
            payload = {
                "model": args.model,
                "messages": [
                    {
                        "role": "user",
                        "content": "Write a sentence about the sky containing at least ten words.",
                    }
                ],
                "max_tokens": args.max_tokens,
                "stream": streaming,
            }
            if include_usage:
                payload["stream_options"] = {"include_usage": True}
            headers = {"Content-Type": "application/json"}
            if os.environ.get("OPENAI_API_KEY"):
                headers["Authorization"] = "Bearer " + os.environ["OPENAI_API_KEY"]
            request = urllib.request.Request(
                args.base_url.rstrip("/") + "/v1/chat/completions",
                data=json.dumps(payload).encode(),
                headers=headers,
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=args.timeout) as response:
                raw = response.read().decode("utf-8")
                assert response.status == 200
            if not streaming:
                data = json.loads(raw)
                assert data.get("object") == "chat.completion"
                assert isinstance(data.get("id"), str) and data["id"]
                assert data.get("choices"), "Missing choices"
                assert data["choices"][0].get("finish_reason"), "Missing finish_reason"
                check_usage(data.get("usage"), args.expect_cached_details)
                record["usage"] = data["usage"]
                (out / f"{case}.json").write_text(
                    json.dumps(data, indent=2), encoding="utf-8"
                )
            else:
                frames = [
                    line[5:].strip()
                    for line in raw.splitlines()
                    if line.startswith("data:")
                ]
                assert frames and frames[-1] == "[DONE]", "Missing terminal DONE"
                events = [json.loads(frame) for frame in frames[:-1]]
                assert events and all(
                    "error" not in event for event in events
                ), "SSE error"
                assert all(
                    event.get("object") == "chat.completion.chunk" for event in events
                )
                finished = [
                    index
                    for index, event in enumerate(events)
                    if any(
                        choice.get("finish_reason")
                        for choice in event.get("choices", [])
                    )
                ]
                assert finished, "Missing finish chunk"
                usages = [
                    (index, event["usage"])
                    for index, event in enumerate(events)
                    if event.get("usage") is not None
                ]
                if include_usage:
                    assert len(usages) == 1, "Expected exactly one usage chunk"
                    index, usage = usages[0]
                    assert index == len(events) - 1 and index >= max(
                        finished
                    ), "Usage must be terminal"
                    assert (
                        events[index].get("choices") == []
                    ), "Usage chunk must have empty choices"
                    check_usage(usage, args.expect_cached_details)
                    record["usage"] = usage
                else:
                    assert not usages, "Default stream unexpectedly includes usage"
                (out / f"{case}.json").write_text(
                    json.dumps(events, indent=2), encoding="utf-8"
                )
            record["passed"] = True
        except urllib.error.HTTPError as error:
            record["error"] = (
                f"HTTP {error.code}: "
                + error.read().decode("utf-8", errors="replace")[:4000]
            )
        except Exception as error:
            record["error"] = str(error)
        results.append(record)
        print(json.dumps(record, ensure_ascii=False), flush=True)
    summary = {
        "scope": "Live AR Chat response and usage contracts; not tokenizer ground truth",
        "results": results,
        "passed": all(result["passed"] for result in results),
        "not_covered": [
            "Images",
            "audio usage",
            "logprobs",
            "device timing",
            "reasoning details without producer",
        ],
        "cache_details_required": args.expect_cached_details,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0 if summary["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
