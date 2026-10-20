# SPDX-License-Identifier: Apache-2.0
"""VoiceChat tool-calling text protocol and the tool variant's declaration."""

from pathlib import Path

import pytest

from sglang_omni.config.manager import ConfigManager
from sglang_omni.config.schema import PipelineConfig, StageConfig
from sglang_omni.models.nemotron_voicechat.duplex_config import (
    TOOL_STAGE,
    VOICECHAT_NODE,
    NemotronVoiceChatDuplexPipelineConfig,
    NemotronVoiceChatToolPipelineConfig,
)
from sglang_omni.models.nemotron_voicechat.tools import (
    format_tool_response,
    parse_tool_calls,
    tool_system_prompt,
)
from sglang_omni.utils.imports import import_string


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
    prompt = tool_system_prompt(
        [
            {
                "function": {
                    "name": "lookup",
                    "description": "Look up a value.",
                    "parameters": {"type": "object"},
                }
            }
        ]
    )
    assert '<AVAILABLE_TOOLS>[{"description": "Look up a value."' in prompt
    assert prompt.endswith("or just respond to the user.")
    assert (
        format_tool_response(
            [
                {"id": "call_1_0", "name": "a", "response": {"x": 1}},
                {"id": "call_1_1", "name": "b", "response": {"y": 2}},
            ]
        )
        == '<TOOL_RESPONSE>[{"x": 1}, {"y": 2}]</TOOL_RESPONSE>'
    )


def test_tool_variant_stage_factories_and_tools_resolve() -> None:
    config = NemotronVoiceChatToolPipelineConfig(model_path="voicechat")
    for stage in config.stages:
        assert callable(import_string(stage.factory_path)), stage.name
    assert config.tools == {}
    assert config.stage_factory_kwargs(TOOL_STAGE)["tools"] == {}
    assert "tool_definitions" not in config.stage_named("thinker").factory.model_dump()
    assert config.session_graph().order() == (VOICECHAT_NODE, TOOL_STAGE)


def test_example_injects_tool_schema_and_function() -> None:
    example_path = (
        Path(__file__).resolve().parents[3] / "examples/nemotron_voicechat/tool.yaml"
    )
    config = ConfigManager.from_file(str(example_path)).config
    definitions = config.stage_factory_kwargs("thinker")["tool_definitions"]
    functions = config.stage_factory_kwargs(TOOL_STAGE)["tools"]
    assert [definition["function"]["name"] for definition in definitions] == list(
        functions
    )
    assert "generate_random_number" in tool_system_prompt(definitions)
    generate_random_number = import_string(functions["generate_random_number"])
    assert generate_random_number({"min": 40, "max": 40}) == {"result": 40}
    with pytest.raises(ValueError, match="min <= max"):
        generate_random_number({"min": 3, "max": 1})
    assert config.stage_named("thinker").factory.dtype == "bfloat16"
    assert config.session_graph().order() == (VOICECHAT_NODE, TOOL_STAGE)


def test_tool_registry_rejects_name_drift_and_unknown_selection():
    tool = {
        "function": {"name": "lookup", "parameters": {"type": "object"}},
        "implementation": "unused.module.lookup",
    }
    with pytest.raises(ValueError, match="function.name"):
        NemotronVoiceChatToolPipelineConfig(
            model_path="voicechat", tools={"other": tool}
        )
    with pytest.raises(ValueError, match="unknown tool"):
        NemotronVoiceChatToolPipelineConfig(
            model_path="voicechat", tools={"lookup": tool}, tool_names=["missing"]
        ).check_conflicts()
    config = NemotronVoiceChatToolPipelineConfig(
        model_path="voicechat", tools={"lookup": tool}, tool_names=[]
    )
    assert config.stage_factory_kwargs("thinker")["tool_definitions"] == []
    assert config.stage_factory_kwargs(TOOL_STAGE)["tools"] == {}


@pytest.mark.parametrize(
    "config_cls", [PipelineConfig, NemotronVoiceChatDuplexPipelineConfig]
)
def test_pipeline_without_tool_support_rejects_registry(config_cls):
    stages = [
        StageConfig(name="entry", process="entry", factory_path="unused", terminal=True)
    ]
    config = config_cls(model_path="unused", stages=stages)
    assert "tools" not in config.model_dump()
    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        config_cls(model_path="unused", stages=stages, tools={})
