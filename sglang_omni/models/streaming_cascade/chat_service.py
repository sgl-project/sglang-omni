# SPDX-License-Identifier: Apache-2.0
"""Launch SGLang's chat server with tool call parsing that streams the arguments of a call while the model writes them.

For qwen3_coder it releases string arguments and reads a call a continued assistant message opened; for Inkling it releases the arguments object of a call. Run it in place of python -m sglang.launch_server, with the same arguments.
"""

import codecs
import json
import re
import runpy
from collections.abc import AsyncIterator, Sequence

from sglang.srt.entrypoints.openai import serving_inkling
from sglang.srt.entrypoints.openai.protocol import (
    ChatCompletionRequest,
    FunctionResponse,
    Tool,
    ToolCall,
)
from sglang.srt.entrypoints.openai.serving_chat import OpenAIServingChat
from sglang.srt.entrypoints.openai.serving_inkling import InklingChatEncodingAdapter
from sglang.srt.function_call.core_types import StreamingParseResult, ToolCallItem
from sglang.srt.function_call.function_call_parser import FunctionCallParser
from sglang.srt.function_call.json_array_parser import JsonArrayParser
from sglang.srt.function_call.qwen3_coder_detector import Qwen3CoderDetector
from sglang.srt.parser.inkling_output import (  # noqa: leading-underscore  # the delta SGLang's parser builds
    InklingOutputDelta,
    InklingOutputParser,
    InklingToolCall,
    _DeltaBuilder,
)
from sglang.srt.parser.inkling_renderer import load_tml_renderers

STRING_TYPES = frozenset({"string", "str", "text", "varchar", "char"})
TOOL_CALL_START = "<tool_call>"
TOOL_CALL_END = "</tool_call>"
# What an Inkling call's payload writes before its arguments object: {"name":"tts_start","args": and the brace that opens the arguments, which the match leaves in place.
INKLING_CALL_HEAD = re.compile(
    r'\s*\{\s*"name"\s*:\s*"((?:[^"\\]|\\.)*)"\s*,\s*"args"\s*:\s*(?=\{)'
)
# What may still close an Inkling call at the end of the payload that arrived: the brace that closes the payload, and whitespace around it.
INKLING_CALL_TAIL = re.compile(r"\s*\}?\s*\Z")
# An Inkling control token as a continued assistant message writes it, with its name captured.
INKLING_CONTROL_TOKEN = re.compile(r"<\|([a-z_]+)\|>")
INKLING_MESSAGE_MODEL = "<|message_model|>"
# SGLang's own streamed tool call parsing, before main replaces it with stream_continued_call.
SGLANG_TOOL_CALL_STREAM = (
    OpenAIServingChat._process_tool_call_stream  # noqa: leading-underscore
)


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


class StreamingInklingOutputParser(InklingOutputParser):
    """InklingOutputParser whose streamed calls release their arguments while the model writes them.

    note (wenyao): SGLang's parser sends an Inkling call only at its end_message token, so the text of a tts_start call reached the cascade once the whole reply was written. While feed reads a call, this one sends the call's name as soon as the payload reaches its arguments, then each piece of the arguments, holding back only the brace that may close the payload. A call whose name does not come before its arguments, or that arrives in the last step, is sent whole at its end, as SGLang sends it.
    """

    def __init__(self, *, opening_ids: Sequence[int] = (), **kwargs: object) -> None:
        super().__init__(**kwargs)
        self.invoke_tool_json_id = (
            self._tml.tokenizer.encode_special(  # noqa: leading-underscore
                "content_invoke_tool_json"
            )
        )
        # True inside feed, the only place a call streams: SGLang reads a response that is not streamed with finish alone.
        self.is_feeding = False
        # Where the streaming call's arguments start in its payload; None while no call streams.
        self.arguments_start: int | None = None
        # How much of the streaming call's arguments was sent.
        self.sent_length = 0
        # The prompt ended inside a call the request opened, so the sampled tokens carry no header; replay the opening as SGLang replays a continued text block's opener, uncounted and unsent.
        for token_id in opening_ids:
            self._parse(token_id, _DeltaBuilder())  # noqa: leading-underscore

    def feed(self, token_ids: Sequence[int]) -> InklingOutputDelta:
        self.is_feeding = True
        delta = super().feed(token_ids)
        self.is_feeding = False
        return delta

    def _parse(
        self, token_id: int, delta: _DeltaBuilder
    ) -> None:  # noqa: leading-underscore
        super()._parse(token_id, delta)  # noqa: leading-underscore
        if (
            self.is_feeding
            and self._parse_tool_calls  # noqa: leading-underscore
            and self._model_authored  # noqa: leading-underscore
            and self._content_kind_id()  # noqa: leading-underscore
            == self.invoke_tool_json_id
        ):
            self.release_arguments(delta)
        else:
            pass

    def release_arguments(self, delta: _DeltaBuilder) -> None:
        """Send the call's name once its payload reaches the arguments, then the arguments that arrived since."""
        payload = self.payload_text(final=False)
        if self.arguments_start is None:
            head = INKLING_CALL_HEAD.match(payload)
            if head is None:
                return
            else:
                pass
            try:
                name = json.loads(f'"{head.group(1)}"')
            except json.JSONDecodeError:
                return
            self.arguments_start = head.end()
            self.sent_length = 0
            delta.tool_calls.append(self.call_piece(name=name, arguments=""))
        else:
            pass
        arguments = payload[self.arguments_start :]
        end = INKLING_CALL_TAIL.search(arguments).start()
        if end > self.sent_length:
            delta.tool_calls.append(
                self.call_piece(arguments=arguments[self.sent_length : end])
            )
            self.sent_length = end
        else:
            pass

    def _tool_call(
        self, message: object
    ) -> InklingToolCall:  # noqa: leading-underscore
        if self.arguments_start is None:
            return super()._tool_call(message)  # noqa: leading-underscore
        else:
            pass
        arguments = (
            self.payload_text(final=True)
            .rstrip()
            .removesuffix("}")
            .rstrip()[self.arguments_start :]
        )
        rest = arguments[self.sent_length :]
        self.arguments_start = None
        return self.call_piece(arguments=rest)

    def _recover_message(  # noqa: leading-underscore
        self, delta: _DeltaBuilder, exc: Exception, *, block_open: bool = False
    ) -> None:
        # A payload that TML rejects is surfaced as text, and the next call starts afresh.
        self.arguments_start = None
        super()._recover_message(  # noqa: leading-underscore
            delta, exc, block_open=block_open
        )

    def call_piece(self, *, arguments: str, name: str | None = None) -> InklingToolCall:
        """A piece of the call being written: its name with no arguments first, then pieces of its arguments with no name."""
        return InklingToolCall(
            index=self._num_tool_calls,  # noqa: leading-underscore
            name=name,
            arguments=arguments,
        )

    def payload_text(self, *, final: bool) -> str:
        """The call's payload that arrived, without its end_message token and, unless final, without a character whose bytes have not all arrived."""
        token_ids = self._payload_ids()  # noqa: leading-underscore
        if (
            token_ids and token_ids[-1] == self._end_message_id
        ):  # noqa: leading-underscore
            token_ids = token_ids[:-1]
        else:
            pass
        payload = self._tml.tokenizer.decode_bytes(  # noqa: leading-underscore
            token_ids
        )
        return codecs.getincrementaldecoder("utf-8")(errors="replace").decode(
            payload, final=final
        )


class StreamingInklingChatEncodingAdapter(InklingChatEncodingAdapter):
    """InklingChatEncodingAdapter that sends the pieces of a streamed call as an OpenAI stream does: the id and name with the first piece, the arguments alone after it.

    It also reads a continued assistant message written with Inkling's control tokens as <|name|> that ends inside a call, such as the cascade's required_call_opening: the model writes the call's arguments without reasoning first.
    """

    def encode_messages(
        self,
        *,
        messages: list[dict[str, object]],
        request: ChatCompletionRequest,
        **kwargs: object,
    ) -> list[int]:
        opening_ids = self.call_opening_ids(request)
        if opening_ids is None:
            return super().encode_messages(messages=messages, request=request, **kwargs)
        else:
            # note (wenyao): the opened call is the whole reply; without this stop the model went on after it to reason and answer again in 21 of 120 probe replies.
            end_message = load_tml_renderers(
                self._spec  # noqa: leading-underscore
            ).tokenizer.encode_special("end_message")
            request.stop_token_ids = [*(request.stop_token_ids or ()), end_message]
            # note (wenyao): SGLang continues only a text block, whose prefix it encodes as ordinary text; the opened call follows the other turns as tokens instead.
            return [
                *super().encode_messages(
                    messages=messages[:-1], request=request, **kwargs
                ),
                *opening_ids,
            ]

    def _new_output_parser(  # noqa: leading-underscore
        self, request: ChatCompletionRequest
    ) -> InklingOutputParser:
        opening_ids = self.call_opening_ids(request)
        if opening_ids is None:
            return super()._new_output_parser(request)  # noqa: leading-underscore
        else:
            return StreamingInklingOutputParser(
                separate_reasoning=self._separate_reasoning  # noqa: leading-underscore
                and request.separate_reasoning,
                parse_tool_calls=self._hooks.tool_call_parsing_active(  # noqa: leading-underscore
                    request
                ),
                stream_reasoning=request.stream_reasoning,
                spec=self._spec,  # noqa: leading-underscore
                opening_ids=opening_ids,
            )

    def call_opening_ids(self, request: ChatCompletionRequest) -> list[int] | None:
        """The tokens of a continued assistant message written with Inkling's control tokens; None when the request continues no such message."""
        final_message = request.messages[-1].model_dump() if request.messages else {}
        content = final_message.get("content")
        if not (
            request.continue_final_message
            and final_message.get("role") == "assistant"
            and isinstance(content, str)
            and content.startswith(INKLING_MESSAGE_MODEL)
        ):
            return None
        else:
            pass
        tokenizer = load_tml_renderers(self._spec).tokenizer  # noqa: leading-underscore
        opening_ids = []
        # split keeps each control token's name at the odd positions.
        for position, piece in enumerate(INKLING_CONTROL_TOKEN.split(content)):
            if position % 2:
                opening_ids.append(tokenizer.encode_special(piece))
            elif piece:
                opening_ids.extend(tokenizer.encode_ordinary(piece))
            else:
                pass
        return opening_ids

    def _tool_call(  # noqa: leading-underscore
        self, call: InklingToolCall, history_tool_calls_cnt: int
    ) -> ToolCall:
        if call.name is None:
            return ToolCall(
                index=call.index, function=FunctionResponse(arguments=call.arguments)
            )
        else:
            return super()._tool_call(  # noqa: leading-underscore
                call, history_tool_calls_cnt
            )


async def stream_continued_call(
    serving_chat: OpenAIServingChat,
    index: int,
    delta: str,
    parser_dict: dict[int, FunctionCallParser | JsonArrayParser],
    content: dict[str, object],
    request: ChatCompletionRequest,
    has_tool_calls: dict[int, bool],
    created: int,
    continuous_usage_stats: bool = False,
    flush: bool = False,
) -> AsyncIterator[str]:
    """SGLang's streamed tool call parsing, which first reads the call that a continued assistant message opened and left open, so the completion streams as that call's arguments rather than as text; the cascade's openai_chat node sends such a message with required_call_opening."""
    final_message = request.messages[-1]
    opening = ""
    # Note (wenyao): SGLang 0.5.23 shows the prompt only to response-template parsers, so the qwen3_coder detector gets the opening in front of the first delta instead.
    if (
        index not in parser_dict
        and request.continue_final_message
        and final_message.role == "assistant"
        and isinstance(final_message.content, str)
    ):
        call_start = final_message.content.rfind(TOOL_CALL_START)
        if call_start != -1 and TOOL_CALL_END not in final_message.content[call_start:]:
            opening = final_message.content[call_start:]
        else:
            pass
    else:
        pass
    async for chunk in SGLANG_TOOL_CALL_STREAM(
        serving_chat,
        index,
        opening + delta,
        parser_dict,
        content,
        request,
        has_tool_calls,
        created,
        continuous_usage_stats,
        flush,
    ):
        yield chunk


def main() -> None:
    FunctionCallParser.ToolCallParserEnum["qwen3_coder"] = StreamingQwen3CoderDetector
    OpenAIServingChat._process_tool_call_stream = (  # noqa: leading-underscore
        stream_continued_call
    )
    # note (wenyao): With --reasoning-parser inkling, SGLang streams Inkling's output through these two classes, which read control tokens by ID, and not through a tool call parser.
    serving_inkling.InklingOutputParser = StreamingInklingOutputParser
    serving_inkling.InklingChatEncodingAdapter = StreamingInklingChatEncodingAdapter
    runpy.run_module("sglang.launch_server", run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
else:
    pass
