"""
Conversation prompt templates.

We kindly request that you import fastchat instead of copying this file if you wish to use it.
If you have changes in mind, please contribute back so the community can benefit collectively and continue to maintain these valuable templates.

Modified from https://github.com/lm-sys/FastChat/blob/main/fastchat/conversation.py
"""

# Modified for SGLang; see this directory's README.md for upstream source.

import dataclasses
from typing import Dict, List, Tuple


@dataclasses.dataclass
class Conversation:
    """A class that manages prompt templates and keeps all conversation history."""

    # The name of this template
    name: str
    # The template of the system prompt
    system_template: str = "{system_message}"
    # The system message
    system_message: str = ""
    # The names of two roles
    roles: Tuple[str] = ("USER", "ASSISTANT")
    # All messages. Each item is (role, message).
    messages: List[List[str]] = ()
    sep: str = "\n"

    def get_prompt(self) -> str:
        """Get the prompt for generation."""
        if self.system_message is not None and self.system_message != "":
            system_prompt = self.system_template.format(
                system_message=self.system_message
            )
        else:
            system_prompt = ""

        ret = "" if system_prompt == "" else system_prompt + self.sep
        for i, (role, message) in enumerate(self.messages):
            if message:
                if type(message) is tuple:
                    message, _, _ = message
                ret += role + message + self.sep
            else:
                if i != len(self.messages) and message is not None:
                    ret += role + self.sep
                else:
                    ret += role
        return ret

    def append_message(self, role: str, message: str):
        """Append a new message."""
        self.messages.append([role, message])

    def copy(self):
        return Conversation(
            name=self.name,
            system_template=self.system_template,
            system_message=self.system_message,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            sep=self.sep,
        )


conv_templates: Dict[str, Conversation] = {
    "neo1_0": Conversation(
        name="neo1_0",
        system_template="<|im_start|>system\n{system_message}",
        system_message="",
        roles=("<|im_start|>user\n", "<|im_start|>assistant\n"),
        sep="<|im_end|>\n",
    )
}


def get_conv_template(name: str) -> Conversation:
    """Get a conversation template."""
    return conv_templates[name].copy()
