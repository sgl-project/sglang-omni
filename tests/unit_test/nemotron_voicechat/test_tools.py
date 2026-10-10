# SPDX-License-Identifier: Apache-2.0
"""VoiceChat tool-calling text protocol."""

import pytest

from sglang_omni.models.nemotron_voicechat.tools import (
    RANDOM_NUMBER_TOOL,
    format_tool_response,
    generate_random_number,
    parse_tool_calls,
    tool_system_prompt,
)


@pytest.mark.parametrize(
    ("call_text", "expected"),
    [
        (
            '<SPECIAL_20><TOOLCALL>[{"name": "a", "arguments": {"n": 1}}]</TOOLCALL><SPECIAL_21>',
            [{"name": "a", "arguments": {"n": 1}}],
        ),
        (
            '<SPECIAL_20>{"name": "a", "arguments": "{\\"n\\": 1}"}<SPECIAL_21>',
            [{"name": "a", "arguments": {"n": 1}}],
        ),
        ('<SPECIAL_20><TOOLCALL>[{"name": "a", "argu<SPECIAL_21>', []),
        ('<SPECIAL_20><TOOLCALL>[{"arguments": {}}]</TOOLCALL><SPECIAL_21>', []),
    ],
)
def test_parse_tool_calls(call_text: str, expected: list[dict[str, object]]) -> None:
    assert parse_tool_calls(call_text) == expected


def test_tool_prompt_and_response_text() -> None:
    prompt = tool_system_prompt([RANDOM_NUMBER_TOOL])
    assert '<AVAILABLE_TOOLS>[{"description": "Generate a random integer' in prompt
    assert prompt.endswith("or just respond to the user.")
    assert (
        format_tool_response(
            [{"name": "a", "response": {"x": 1}}, {"name": "b", "response": {"y": 2}}]
        )
        == '<TOOL_RESPONSE>[{"x": 1}, {"y": 2}]</TOOL_RESPONSE>'
    )
    assert 1 <= generate_random_number({"min": 1, "max": 3})["result"] <= 3
    with pytest.raises(ValueError, match="min <= max"):
        generate_random_number({"min": 3, "max": 1})
