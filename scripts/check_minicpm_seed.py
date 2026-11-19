# SPDX-License-Identifier: Apache-2.0
"""Probe an existing MiniCPM-o server for request-seeded DSpark sampling."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Literal

import httpx
from pydantic import BaseModel, Field, JsonValue

Mode = Literal["plain", "dspark"]
SeedExpectation = Literal["supported", "rejected"]
CASE_NAMES = ("greedy", "sampled", "seeded", "intervening", "seeded_repeat")
DEFAULT_PROMPT = "Write a two-sentence story about a robot discovering a garden."
INTERVENING_PROMPT = "Name three planets in our solar system."


class RequestRecord(BaseModel):
    name: str
    request: dict[str, JsonValue]
    status: int
    seconds: float
    body: JsonValue


class ChatMessage(BaseModel):
    content: str = Field(min_length=1)


class ChatChoice(BaseModel):
    message: ChatMessage


class SpeculativeDetails(BaseModel):
    spec_verify_ct: int = 0
    spec_num_proposed_drafts: int = 0


class ChatExtensions(BaseModel):
    spec_tokens_details: SpeculativeDetails | None = None


class ChatResponse(BaseModel):
    choices: list[ChatChoice] = Field(min_length=1)
    sglext: ChatExtensions = Field(default_factory=ChatExtensions)


def validate_response(
    record: RequestRecord, *, mode: Mode, expect_seed: SeedExpectation
) -> str | None:
    if (
        record.name in ("seeded", "seeded_repeat")
        and record.request.get("seed") is None
    ):
        raise ValueError("Seeded case is missing its request seed")
    else:
        pass
    if record.request.get("seed") is not None and expect_seed == "rejected":
        if (
            mode == "dspark"
            and record.status == 400
            and isinstance(record.body, dict)
            and record.body.get("detail")
            == "Speculative decoding does not support seed. Unset them for this request."
        ):
            return None
        else:
            raise ValueError(f"Expected seed rejection, got HTTP {record.status}")
    elif record.status != 200:
        raise ValueError(f"Expected HTTP 200, got HTTP {record.status}")
    else:
        response = ChatResponse.model_validate(record.body)
    details = response.sglext.spec_tokens_details
    if mode == "dspark" and (
        details is None
        or details.spec_verify_ct <= 0
        or details.spec_num_proposed_drafts <= 0
    ):
        raise ValueError("Response contains no speculative verification activity")
    else:
        return response.choices[0].message.content


def validate_run(
    records: list[RequestRecord], *, mode: Mode, expect_seed: SeedExpectation
) -> list[str]:
    errors: list[str] = []
    if tuple(record.name for record in records) != CASE_NAMES:
        errors.append(f"Unexpected or missing cases; expected {CASE_NAMES}")
    else:
        pass
    contents: dict[str, str] = {}
    for record in records:
        try:
            content = validate_response(record, mode=mode, expect_seed=expect_seed)
        except ValueError as error:
            errors.append(f"{record.name}: {error}")
        else:
            if content is not None:
                contents[record.name] = content
            else:
                pass
    if expect_seed == "supported" and not errors:
        seeded, repeated = records[2], records[4]
        if seeded.request != repeated.request:
            errors.append("Seeded repeat requests differ")
        elif contents["seeded"] != contents["seeded_repeat"]:
            errors.append("Seeded repeat output differs after the intervening request")
        else:
            pass
    else:
        pass
    return errors


def collect_requests(
    client: httpx.Client, *, model: str, prompt: str, max_tokens: int, seed: int
) -> list[RequestRecord]:
    records: list[RequestRecord] = []
    for name in CASE_NAMES:
        request: dict[str, JsonValue] = {
            "model": model,
            "messages": [
                {
                    "role": "user",
                    "content": INTERVENING_PROMPT if name == "intervening" else prompt,
                }
            ],
            "temperature": 0.0 if name == "greedy" else 0.7,
            "max_tokens": max_tokens,
            "return_spec_tokens_details": True,
        }
        if name in ("seeded", "seeded_repeat"):
            request["seed"] = seed
        else:
            pass
        started = time.perf_counter()
        try:
            response = client.post("v1/chat/completions", json=request)
        except httpx.RequestError as error:
            status, body = 0, str(error)
        else:
            status = response.status_code
            try:
                body = json.loads(response.text)
            except json.JSONDecodeError:
                body = response.text
        records.append(
            RequestRecord(
                name=name,
                request=request,
                status=status,
                seconds=time.perf_counter() - started,
                body=body,
            )
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--base-url", required=True, help="Server root URL, without /v1"
    )
    parser.add_argument("--model", required=True)
    parser.add_argument("--mode", choices=("plain", "dspark"), required=True)
    parser.add_argument(
        "--expect-seed", choices=("supported", "rejected"), default="supported"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout-seconds", type=float, default=180)
    arguments = parser.parse_args()
    if arguments.mode == "plain" and arguments.expect_seed == "rejected":
        parser.error("--expect-seed rejected requires --mode dspark")
    elif arguments.max_tokens < 2 or arguments.timeout_seconds <= 0:
        parser.error("Use at least two output tokens and a positive timeout")
    else:
        pass
    mode: Mode = arguments.mode
    expectation: SeedExpectation = arguments.expect_seed
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client(
        base_url=arguments.base_url.rstrip("/") + "/",
        timeout=arguments.timeout_seconds,
        trust_env=False,
    ) as client:
        records = collect_requests(
            client,
            model=arguments.model,
            prompt=arguments.prompt,
            max_tokens=arguments.max_tokens,
            seed=arguments.seed,
        )
    errors = validate_run(records, mode=mode, expect_seed=expectation)
    arguments.output.write_text(
        json.dumps(
            {
                "base_url": arguments.base_url,
                "mode": mode,
                "expect_seed": expectation,
                "records": [record.model_dump(mode="json") for record in records],
                "errors": errors,
            },
            indent=2,
        )
        + "\n"
    )
    for record in records:
        print(f"{record.name}: HTTP {record.status}")
    for error in errors:
        print(error)
    print(f"Saved {arguments.output}")
    raise SystemExit(1 if errors else 0)


if __name__ == "__main__":
    main()
else:
    pass
