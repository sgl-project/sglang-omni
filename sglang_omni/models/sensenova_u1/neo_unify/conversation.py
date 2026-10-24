"""Conversation prompt templates adapted from FastChat."""

import dataclasses
from enum import IntEnum
from typing import TypedDict

from PIL import Image

# Modified for SGLang; see this directory's README.md for upstream source.


ConversationMessage = str | tuple[str, list[Image.Image], str] | None


class ConversationRecord(TypedDict):
    template_name: str
    system_message: str | None
    roles: tuple[str, str]
    messages: list[list[ConversationMessage]] | tuple[()]
    offset: int


class SeparatorStyle(IntEnum):
    """Separator styles."""

    MPT = 18


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    name: str
    system_template: str = "{system_message}"
    system_message: str | None = ""
    roles: tuple[str, str] = ("USER", "ASSISTANT")
    messages: list[list[ConversationMessage]] | tuple[()] = ()
    offset: int = 0
    sep_style: SeparatorStyle = SeparatorStyle.MPT
    sep: str = "\n"
    sep2: str | None = None
    stop_str: str | list[str] | None = None
    stop_token_ids: list[int] | None = None

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.system_message is not None and self.system_message != "":
            system_prompt = self.system_template.format(
                system_message=self.system_message
            )
        else:
            system_prompt = ""

        if self.sep_style == SeparatorStyle.MPT:
            ret = "" if system_prompt == "" else system_prompt + self.sep
            for i, (role, message) in enumerate(self.messages):
                if message:
                    if type(message) is tuple:
                        message, _, _ = message
                    else:
                        pass
                    ret += role + message + self.sep
                else:
                    if i != len(self.messages) and message is not None:
                        ret += role + self.sep
                    else:
                        ret += role
            return ret
        else:
            raise ValueError(f"Invalid style: {self.sep_style}")

    def set_system_message(self, system_message: str | None) -> None:
        """Set the system message."""
        self.system_message = system_message

    def append_message(self, role: str, message: ConversationMessage) -> None:
        """Append a new message."""
        self.messages.append([role, message])

    def update_last_message(self, message: ConversationMessage) -> None:
        """Update the last output."""
        self.messages[-1][1] = message

    def to_gradio_chatbot(self) -> list[list[ConversationMessage]]:
        """Convert the conversation to gradio chatbot format."""
        ret = []
        for i, (role, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append([msg, None])
            else:
                ret[-1][-1] = msg
        return ret

    def to_openai_api_messages(self) -> list[dict[str, ConversationMessage]]:
        """Convert the conversation to OpenAI chat completion format."""
        ret = [{"role": "system", "content": self.system_message}]

        for i, (_, msg) in enumerate(self.messages[self.offset :]):
            if i % 2 == 0:
                ret.append({"role": "user", "content": msg})
            else:
                if msg is not None:
                    ret.append({"role": "assistant", "content": msg})
                else:
                    pass
        return ret

    def copy(self) -> "Conversation":
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            stop_str=self.stop_str,
            stop_token_ids=self.stop_token_ids,
        )

    def dict(self) -> ConversationRecord:
        return {
            "template_name": self.name,
            "system_message": self.system_message,
            "roles": self.roles,
            "messages": self.messages,
            "offset": self.offset,
        }


def register_conv_template(template: Conversation, override: bool = False) -> None:
    """Register a new conversation template."""
    if not override:
        assert (
            template.name not in conv_templates
        ), f"{template.name} has been registered."
    else:
        pass

    conv_templates[template.name] = template


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()


INTERNVL_SYSTEM_MESSAGE = (
    "\u4f60\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c\u5ba4"
    "\u8054\u5408\u5546\u6c64\u79d1\u6280\u5f00\u53d1\u7684\u4e66\u751f\u591a"
    "\u6a21\u6001\u5927\u6a21\u578b\uff0c\u82f1\u6587\u540d\u53ebInternVL, "
    "\u662f\u4e00\u4e2a\u6709\u7528\u65e0\u5bb3\u7684\u4eba\u5de5\u667a\u80fd"
    "\u52a9\u624b\u3002"
)
INTERNVL2_5_SYSTEM_MESSAGE = (
    "\u4f60\u662f\u4e66\u751f\xb7\u4e07\u8c61\uff0c\u82f1\u6587\u540d\u662f"
    "InternVL\uff0c\u662f\u7531\u4e0a\u6d77\u4eba\u5de5\u667a\u80fd\u5b9e\u9a8c"
    "\u5ba4\u3001\u6e05\u534e\u5927\u5b66\u53ca\u591a\u5bb6\u5408\u4f5c\u5355"
    "\u4f4d\u8054\u5408\u5f00\u53d1\u7684\u591a\u6a21\u6001\u5927\u8bed\u8a00"
    "\u6a21\u578b\u3002"
)


conv_templates: dict[str, Conversation] = {
    "Hermes-2": Conversation(
        name="Hermes-2",
        system_template="<|im_start|>system\n{system_message}",
        system_message=INTERNVL_SYSTEM_MESSAGE,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
        stop_str="<|endoftext|>",
    ),
    "internlm2-chat": Conversation(
        name="internlm2-chat",
        system_template="<|im_start|>system\n{system_message}",
        system_message=INTERNVL_SYSTEM_MESSAGE,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>",
    ),
    "phi3-chat": Conversation(
        name="phi3-chat",
        system_template="<|system|>\n{system_message}",
        system_message=INTERNVL_SYSTEM_MESSAGE,
        roles=("<|user|>\n", "<|assistant|>\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|end|>",
    ),
    "internvl2_5": Conversation(
        name="internvl2_5",
        system_template="<|im_start|>system\n{system_message}",
        system_message=INTERNVL2_5_SYSTEM_MESSAGE,
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    ),
    "neo1_0": Conversation(
        name="neo1_0",
        system_template="<|im_start|>system\n{system_message}",
        system_message="",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep_style=SeparatorStyle.MPT,
        sep="<|im_end|>\n",
    ),
}
