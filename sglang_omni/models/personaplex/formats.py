# SPDX-License-Identifier: Apache-2.0
"""What a PersonaPlex Session's speech node publishes, for the nodes of a Session graph that read it."""

# {"text": str, "pcm": bytes}, the words and the 24 kHz PCM16 audio of one unit of the agent's audio, stamped on the agent's output clock, which runs with the input's from the Session's first unit; silence is the agent not speaking.
PERSONAPLEX_REPLY = "personaplex_reply"

__all__ = ["PERSONAPLEX_REPLY"]
