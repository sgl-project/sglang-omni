# SPDX-License-Identifier: Apache-2.0
"""Protect the checked-in Super request contract across transports."""

from argparse import Namespace

from examples.cosmos3_super_t2i import http_payload


def test_raw_http_payload_flattens_openai_extension_fields():
    args = Namespace(
        prompt="prompt",
        negative_prompt="negative",
        width=640,
        height=640,
        steps=35,
        guidance_scale=6.0,
        flow_shift=3.0,
        seed=0,
    )
    payload = http_payload(args)
    assert "extra_body" not in payload
    assert payload["negative_prompt"] == "negative"
    assert payload["use_resolution_template"] is False
    assert payload["use_system_prompt"] is False
    assert payload["use_guardrails"] is False
