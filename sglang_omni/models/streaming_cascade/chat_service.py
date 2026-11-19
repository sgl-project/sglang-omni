# SPDX-License-Identifier: Apache-2.0
"""Launch SGLang's chat server with a qwen3_coder tool call parser that streams string arguments.

Run it in place of python -m sglang.launch_server, with the same arguments.
"""

import json
import runpy

from sglang.srt.entrypoints.openai.protocol import Tool
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector

STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char"})


def json_string_body(text: str) -> str:
    """Text as it stands inside a JSON string, without the quotes."""
    return json.dumps(text, ensure_ascii=False)[1:-1]


class StreamingQwen3CoderDetector(Qwen3CoderDetector):
    """Qwen3CoderDetector whose streamed calls release a string argument as the model writes it.

    note (Jingwen): SGLang 0.5.23's detector holds a parameter until its closing tag, so the text of a tts_start call reached the cascade in one piece 1.05 s into the completion; released as it is written, its first sentence was complete at 0.59 s. The calls are the same as the detector's for any way the text is cut into pieces, and only streaming changes.
    """

    def __init__(self) -> None:
        super().__init__()
        # True while a string value is being released: its opening "name": " was sent and its closing quote was not.
        self.is_string_value_open = False
        # The text that arrived and is not parsed yet.
        self.pending = ""
        # The parameters the schema of the function being called declares.
        self.parameters: dict[str, object] = {}

    def parse_streaming_increment(
        self, new_text: str, tools: list[Tool]
    ) -> StreamingParseResult:
        self.pending += new_text
        calls: list[ToolCallItem] = []
        texts: list[str] = []
        while self.pending and self.consume(tools, calls, texts):
            pass
        return StreamingParseResult(calls=calls, normal_text="".join(texts))

    def consume(
        self, tools: list[Tool], calls: list[ToolCallItem], texts: list[str]
    ) -> bool:
        """Take the next element off the buffer; False when the rest must wait for more text."""
        text = self.pending
        if self.is_string_value_open:
            return self.consume_string_value(text, calls)
        elif text.startswith(self.tool_call_start_token):
            self.is_inside_tool_call = True
            return self.advance(len(self.tool_call_start_token))
        elif text.startswith(self.tool_call_prefix):
            return self.consume_function(text, tools, calls)
        elif text.startswith(self.parameter_prefix):
            return self.consume_parameter(text, calls)
        elif text.startswith(self.function_end_token):
            if not self.json_started:
                calls.append(self.arguments("{"))
                self.json_started = True
            else:
                pass
            calls.append(self.arguments("}"))
            self.current_func_name = None
            return self.advance(len(self.function_end_token))
        elif text.startswith(self.tool_call_end_token):
            self.is_inside_tool_call = False
            return self.advance(len(self.tool_call_end_token))
        else:
            return self.consume_text(text, texts)

    def advance(self, length: int) -> bool:
        self.pending = self.pending[length:]
        return True

    def arguments(self, fragment: str) -> ToolCallItem:
        return ToolCallItem(tool_index=self.current_tool_id, parameters=fragment)

    def consume_function(
        self, text: str, tools: list[Tool], calls: list[ToolCallItem]
    ) -> bool:
        name_end = text.find(">")
        if name_end == -1:
            return False
        else:
            self.current_tool_id += 1
            self.current_tool_name_sent = True
            self.current_tool_param_count = 0
            self.json_started = False
            self.current_func_name = text[len(self.tool_call_prefix) : name_end]
            # The three methods with a leading underscore below are the SGLang detector's own.
            self.parameters = self._get_arguments_config(  # noqa: leading-underscore
                self.current_func_name, tools
            )
            calls.append(
                ToolCallItem(
                    tool_index=self.current_tool_id,
                    name=self.current_func_name,
                    parameters="",
                )
            )
            return self.advance(name_end + 1)

    def key_prefix(self, name: str) -> str:
        """What precedes a parameter's value in the arguments: the opening brace before the first one, a comma before any other, and its key."""
        opening = "" if self.json_started else "{"
        separator = ", " if self.current_tool_param_count > 0 else ""
        self.json_started = True
        return f"{opening}{separator}{json.dumps(name)}: "

    def consume_parameter(self, text: str, calls: list[ToolCallItem]) -> bool:
        name_end = text.find(">")
        if name_end == -1:
            return False
        else:
            pass
        name = text[len(self.parameter_prefix) : name_end]
        rest = text[name_end + 1 :]
        config = self.parameters
        value_end = self.value_end(rest)
        if value_end is not None:
            position, tag_length = value_end
            raw = rest[:position].removeprefix("\n").removesuffix("\n")
            value = self._convert_param_value(  # noqa: leading-underscore
                raw, name, config, self.current_func_name
            )
            calls.append(
                self.arguments(
                    self.key_prefix(name) + json.dumps(value, ensure_ascii=False)
                )
            )
            self.current_tool_param_count += 1
            return self.advance(name_end + 1 + position + tag_length)
        elif name in config and self.is_string(config[name]):
            skipped = 1 if rest.startswith("\n") else 0
            value = rest[skipped:]
            length = self.streamable_length(value)
            # A value of "null" is the JSON null, not a string, so nothing is sent while the value may still be it.
            if length > 0 and not "null".startswith(value[:length].lower()):
                calls.append(
                    self.arguments(
                        f'{self.key_prefix(name)}"{json_string_body(value[:length])}'
                    )
                )
                self.is_string_value_open = True
                return self.advance(name_end + 1 + skipped + length)
            else:
                return False
        else:
            # Any other type needs its whole value to be converted.
            return False

    def is_string(self, schema: object) -> bool:
        return self._get_param_type(schema) in STRING_TYPES  # noqa: leading-underscore

    def consume_string_value(self, text: str, calls: list[ToolCallItem]) -> bool:
        value_end = self.value_end(text)
        if value_end is None:
            length = self.streamable_length(text)
            if length > 0:
                calls.append(self.arguments(json_string_body(text[:length])))
                self.advance(length)
            else:
                pass
            return False
        else:
            position, tag_length = value_end
            calls.append(
                self.arguments(
                    json_string_body(text[:position].removesuffix("\n")) + '"'
                )
            )
            self.current_tool_param_count += 1
            self.is_string_value_open = False
            return self.advance(position + tag_length)

    def value_end(self, text: str) -> tuple[int, int] | None:
        """Where a parameter's value ends in text, as (position, length of the end tag to take with it): at its closing tag, or at the next parameter or the end of the function when the model left the tag out. None while no end is in sight."""
        ends = [
            (text.find(tag), length)
            for tag, length in (
                (self.parameter_end_token, len(self.parameter_end_token)),
                (self.parameter_prefix, 0),
                (self.function_end_token, 0),
            )
        ]
        found = [end for end in ends if end[0] != -1]
        return min(found, key=lambda end: end[0]) if found else None

    def streamable_length(self, text: str) -> int:
        """How much of a string value can be sent now: all of it but a tail that may be the start of the tag that ends the value, and the newline before that tag."""
        held = 0
        for tag in (
            self.parameter_end_token,
            self.parameter_prefix,
            self.function_end_token,
        ):
            for size in range(min(len(tag) - 1, len(text)), 0, -1):
                if text.endswith(tag[:size]):
                    held = max(held, size)
                    break
                else:
                    pass
        end = len(text) - held
        if end > 0 and text[end - 1] == "\n":
            return end - 1
        else:
            return end

    def consume_text(self, text: str, texts: list[str]) -> bool:
        """Text outside a call is the model's reply; inside one it is the whitespace between tags, and dropped."""
        tags = (
            self.tool_call_start_token,
            self.tool_call_end_token,
            self.tool_call_prefix,
            self.function_end_token,
            self.parameter_prefix,
            self.parameter_end_token,
        )
        tag_start = text.find("<")
        if tag_start == 0 and any(tag.startswith(text) for tag in tags):
            # The start of a tag whose rest has not arrived.
            return False
        else:
            pass
        length = 1 if tag_start == 0 else len(text) if tag_start == -1 else tag_start
        if not self.is_inside_tool_call:
            texts.append(text[:length])
        else:
            pass
        return self.advance(length)


def main() -> None:
    FunctionCallParser.ToolCallParserEnum["qwen3_coder"] = StreamingQwen3CoderDetector
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
else:
    pass
