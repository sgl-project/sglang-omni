# SPDX-License-Identifier: Apache-2.0
"""Check that the seed probe distinguishes rejection, fallback and repeatability."""

from __future__ import annotations

import json
import sys
from functools import partial
from pathlib import Path

import httpx
import pytest

from scripts.check_minicpm_seed import (
    RequestRecord,
    collect_requests,
    main,
    validate_response,
    validate_run,
)


def response_record(
    name: str, *, seed: int | None = None, text: str = "Hello", verify_count: int = 2
) -> RequestRecord:
    return RequestRecord(
        name=name,
        request={"seed": seed} if seed is not None else {},
        status=200,
        seconds=0.1,
        body={
            "choices": [{"message": {"content": text}}],
            "usage": {"completion_tokens": 10},
            "sglext": {
                "spec_tokens_details": {
                    "spec_verify_ct": verify_count,
                    "spec_num_proposed_drafts": 14 if verify_count else 0,
                }
            },
        },
    )


def test_seed_rejection_is_a_failure_under_the_supported_contract() -> None:
    """The recorded pre-fix HTTP 400 must fail the feature acceptance check."""
    record = RequestRecord(
        name="seeded",
        request={"seed": 42},
        status=400,
        seconds=0.01,
        body={
            "detail": "Speculative decoding does not support seed. Unset them for this request."
        },
    )
    validate_response(record, mode="dspark", expect_seed="rejected")
    with pytest.raises(ValueError, match="HTTP 400"):
        validate_response(record, mode="dspark", expect_seed="supported")
    record.body = {"detail": "The prompt is too long"}
    with pytest.raises(ValueError, match="seed rejection"):
        validate_response(record, mode="dspark", expect_seed="rejected")


def test_plain_fallback_cannot_pass_a_dspark_probe() -> None:
    record = response_record("sampled", verify_count=0)
    validate_response(record, mode="plain", expect_seed="supported")
    with pytest.raises(ValueError, match="speculative verification"):
        validate_response(record, mode="dspark", expect_seed="supported")


def test_seeded_repeats_are_checked_after_an_intervening_request() -> None:
    records = [
        response_record("greedy"),
        response_record("sampled"),
        response_record("seeded", seed=42),
        response_record("intervening"),
        response_record("seeded_repeat", seed=42),
    ]
    assert validate_run(records, mode="dspark", expect_seed="supported") == []
    records[-1] = response_record("seeded_repeat", seed=42, text="Different")
    assert any(
        "repeat" in error
        for error in validate_run(records, mode="dspark", expect_seed="supported")
    )
    assert any(
        "missing" in error
        for error in validate_run(records[:-1], mode="dspark", expect_seed="supported")
    )


def test_seeded_case_requires_a_seed_in_the_saved_request() -> None:
    with pytest.raises(ValueError, match="request seed"):
        validate_response(
            response_record("seeded"), mode="dspark", expect_seed="supported"
        )


def test_probe_records_requests_and_preserves_server_errors(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert request.url.path == "/v1/chat/completions"
        if payload.get("seed") is not None:
            return httpx.Response(
                400,
                json={
                    "detail": "Speculative decoding does not support seed. Unset them for this request."
                },
            )
        else:
            return httpx.Response(200, json=response_record("sampled").body)

    with httpx.Client(
        base_url="http://model.test/", transport=httpx.MockTransport(respond)
    ) as client:
        records = collect_requests(
            client, model="minicpmo", prompt="Describe a garden", max_tokens=64, seed=42
        )
    assert records[2].request["seed"] == records[4].request["seed"] == 42
    assert records[2].request == records[4].request
    assert records[3].request["messages"] != records[2].request["messages"]
    assert validate_run(records, mode="dspark", expect_seed="rejected") == []
    assert any(
        "HTTP 400" in error
        for error in validate_run(records, mode="dspark", expect_seed="supported")
    )
    output_path = tmp_path / "probe.json"
    monkeypatch.setattr(
        httpx, "Client", partial(httpx.Client, transport=httpx.MockTransport(respond))
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "check_minicpm_seed",
            "--base-url",
            "http://model.test",
            "--model",
            "minicpmo",
            "--mode",
            "dspark",
            "--output",
            str(output_path),
        ],
    )
    with pytest.raises(SystemExit) as result:
        main()
    assert result.value.code == 1
    report = json.loads(output_path.read_text())
    assert report["records"][2]["status"] == 400
    assert "does not support seed" in report["records"][2]["body"]["detail"]
    assert len(report["errors"]) == 2


def test_transport_failures_and_non_json_responses_remain_in_the_record() -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        if payload["temperature"] == 0:
            raise httpx.ReadTimeout("server timed out", request=request)
        else:
            return httpx.Response(502, text="upstream unavailable")

    with httpx.Client(
        base_url="http://model.test/", transport=httpx.MockTransport(respond)
    ) as client:
        records = collect_requests(
            client, model="minicpmo", prompt="Describe a garden", max_tokens=64, seed=42
        )
    assert records[0].status == 0
    assert records[0].body == "server timed out"
    assert records[1].status == 502
    assert records[1].body == "upstream unavailable"
    assert len(validate_run(records, mode="dspark", expect_seed="supported")) == 5
