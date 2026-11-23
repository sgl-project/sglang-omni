# SPDX-License-Identifier: Apache-2.0
"""Deployment tool declarations shared by model prompts and tool executors."""

from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field


class ToolRegistration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    function: dict[str, object]
    implementation: str = Field(min_length=1)


def validate_tool_names(
    tools: dict[str, ToolRegistration]
) -> dict[str, ToolRegistration]:
    for name, tool in tools.items():
        if not name or tool.function.get("name") != name:
            raise ValueError(f"tool {name!r} must match function.name")
        else:
            pass
    return tools


ToolRegistry = Annotated[
    dict[str, ToolRegistration], AfterValidator(validate_tool_names)
]


def tool_definitions(
    tools: ToolRegistry, names: list[str] | None = None
) -> list[dict[str, object]]:
    return [
        {"type": "function", "function": tools[name].function}
        for name in (tools if names is None else names)
    ]


def tool_implementations(
    tools: ToolRegistry, names: list[str] | None = None
) -> dict[str, str]:
    return {
        name: tools[name].implementation for name in (tools if names is None else names)
    }
