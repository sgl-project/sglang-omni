# SPDX-License-Identifier: Apache-2.0
"""VoiceChat tool-calling protocol: system prompt, call parsing, and response tokens.

The thinker writes a call on its function channel as
<SPECIAL_20><TOOLCALL>[{"name": ..., "arguments": {...}}]</TOOLCALL><SPECIAL_21>.
The tool result is fed back on the same channel as
<TOOL_RESPONSE>[...]</TOOL_RESPONSE>, after which the model predicts <SPECIAL_22>.
"""

from __future__ import annotations

import json
import random
from collections.abc import Mapping
from typing import TypedDict

from jinja2 import Environment

START_OF_TOOL_CALL_TOKEN = "<SPECIAL_20>"
END_OF_TOOL_CALL_TOKEN = "<SPECIAL_21>"

# The checkpoint was trained with this system message whenever tools are offered.
TOOL_SYSTEM_MESSAGE = (
    "You are an AI voice assistant developed by NVIDIA. "
    "Your name is NVIDIA Voice Chat. "
    "Your job is to be helpful and harmless and have engaging conversations in English. "
    "Maintain a warm and friendly tone. "
    "Keep the dialogue open and ongoing. "
    "Be clear and direct, especially when answering yes or no questions and multiple-choice questions. "
    "Avoid long answers unless the user asks you to provide details or context. "
    "You must provide diverse responses and rephrase answers if the user asks the same question. "
    "DO NOT interrupt the user when they are speaking, let them finish their turn before answering."
    "\n\nWhen you receive a request, follow this decision process:\n"
    "1. Does the request match one of your available tools below? If yes, you MUST call that tool - "
    "never answer it directly from your own knowledge, even if you think you know the answer.\n"
    "2. Is it a general knowledge question (history, science, geography, math, facts, etc.)? "
    "If yes, answer directly from your own knowledge - do not call any tool.\n"
    "3. Does it require an external action or live data that none of your tools cover "
    "(e.g. ordering food, sending email)? If yes, politely say you don't have that capability."
    '\n\nNEVER say "I don\'t have a tool for that" for general knowledge questions you can answer yourself.'
    "\n\nDO NOT use any tools when not needed to answer the user's requests, under no circumstance."
    "\n\nYou are an expert across history, geography, science, math, literature, biographies, languages, "
    "recipes, programming, current affairs, and general knowledge. When the user asks about any of these, "
    "answer directly and conversationally from your own knowledge - no <TOOLCALL>."
    "\n\nCall a tool ONLY when the user's request matches one of the tools listed in <AVAILABLE_TOOLS> below. "
    "For every other request, do not call any tool - just answer from your knowledge. "
    "Never invent or call a tool name that is not literally in <AVAILABLE_TOOLS>."
    "\n\nTool-call arguments must be values the user spoke. "
    "If a required argument is missing, ask the user; never guess."
    "\n\nIf a tool call fails or returns an error, do not retry the tool call for the same request. "
    "Tell the user that the API has an issue."
)

# Rendered by jinja2 itself: its tojson filter sorts keys and HTML-escapes, as at training time.
TOOL_PROMPT_TEMPLATE = (
    "{{- system_message -}}"
    "{%- if tools -%}"
    "{%- if system_message != '' -%}{{- '\\n\\n' -}}{%- endif -%}"
    "{{- 'You can use the following tools to assist the user if required:' -}}"
    "{{- '\\n<AVAILABLE_TOOLS>[' -}}"
    "{%- for tool in tools -%}"
    "{%- set _t = (tool.function if tool.function is defined else tool) -%}"
    "{%- set _d = {} -%}"
    "{%- for k, v in _t.items() if k != 'type' -%}{%- set _ = _d.update({k: v}) -%}{%- endfor -%}"
    "{{- _d | tojson -}}"
    "{{- ', ' if not loop.last else '' -}}"
    "{%- endfor -%}"
    "{{- ']</AVAILABLE_TOOLS>\\n\\n' -}}"
    "{{- 'If you decide to call any tool(s), use the following format:\\n' -}}"
    '{{- \'<TOOLCALL>[{"name": "tool_name1", "arguments": "tool_args1"}, \' -}}'
    '{{- \'{"name": "tool_name2", "arguments": "tool_args2"}]\' -}}'
    "{{- '</TOOLCALL>\\n\\n' -}}"
    "{{- 'The user will execute tool-calls and return responses from tool(s) in this format:\\n' -}}"
    '{{- \'<TOOL_RESPONSE>[{"tool_response1"}, {"tool_response2"}]</TOOL_RESPONSE>\\n\\n\' -}}'
    "{{- 'Based on the tool responses, you can call additional tools if needed, "
    "correct tool calls if any errors are found, or just respond to the user.' -}}"
    "{%- endif -%}"
)


class ToolFunctionSchema(TypedDict):
    name: str
    description: str
    parameters: dict[str, object]


class ToolDefinition(TypedDict):
    function: ToolFunctionSchema


class ToolCall(TypedDict):
    name: str
    arguments: dict[str, object]


RANDOM_NUMBER_TOOL: ToolDefinition = {
    "function": {
        "name": "generate_random_number",
        "description": "Generate a random integer between min and max (inclusive).",
        "parameters": {
            "type": "object",
            "properties": {
                "min": {"type": "integer", "description": "Minimum value (inclusive)"},
                "max": {"type": "integer", "description": "Maximum value (inclusive)"},
            },
            "required": ["min", "max"],
        },
    }
}


def tool_system_prompt(tool_definitions: list[ToolDefinition]) -> str:
    return (
        Environment()
        .from_string(TOOL_PROMPT_TEMPLATE)
        .render(system_message=TOOL_SYSTEM_MESSAGE, tools=tool_definitions)
    )


def parse_tool_calls(call_text: str) -> list[ToolCall]:
    """Parse one decoded function-channel span; a malformed span yields no calls."""
    body = call_text.replace(START_OF_TOOL_CALL_TOKEN, "").replace(
        END_OF_TOOL_CALL_TOKEN, ""
    )
    if "<TOOLCALL>" in body:
        body = body.split("<TOOLCALL>")[1].split("</TOOLCALL>")[0]
    else:
        pass
    try:
        decoded = json.loads(body.strip())
    except json.JSONDecodeError:
        return []
    calls: list[ToolCall] = []
    for call in decoded if isinstance(decoded, list) else [decoded]:
        if not isinstance(call, dict) or not isinstance(call.get("name"), str):
            continue
        else:
            pass
        arguments = call.get("arguments", {})
        if isinstance(arguments, str):
            # Note (Dayuxiaoshui): The prompt's own example shows arguments as a string.
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {}
        else:
            pass
        calls.append(
            {
                "name": call["name"],
                "arguments": arguments if isinstance(arguments, dict) else {},
            }
        )
    return calls


def format_tool_response(responses: list[Mapping[str, object]]) -> str:
    """Function-channel text for the tool node's responses, in call order."""
    body = ", ".join(json.dumps(response["response"]) for response in responses)
    return f"<TOOL_RESPONSE>[{body}]</TOOL_RESPONSE>"


def generate_random_number(arguments: Mapping[str, object]) -> dict[str, object]:
    low, high = arguments["min"], arguments["max"]
    if not isinstance(low, int) or not isinstance(high, int) or low > high:
        raise ValueError("min and max must be integers with min <= max")
    else:
        return {"result": random.randint(low, high)}
