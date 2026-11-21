# SPDX-License-Identifier: Apache-2.0
"""Who holds a session's floor: the reply speaking, or a reply paused by speech over it until a verdict says whether that speech was addressed to the assistant."""

from __future__ import annotations

import math
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Generic, Literal, TypeVar

OverlapPolicy = Literal["interrupt", "pause"]

Content = TypeVar("Content")


def logsumexp(values: list[float]) -> float:
    peak = max(values)
    if math.isinf(peak):
        return peak
    else:
        return peak + math.log(sum(math.exp(value - peak) for value in values))


@dataclass(frozen=True)
class AddresseeQuestion:
    """The question and answer labels used to score speech heard over a paused reply."""

    text: str
    addressed: tuple[str, ...]
    other: tuple[str, ...]

    @property
    def labels(self) -> frozenset[str]:
        return frozenset(self.addressed + self.other)

    def margin(self, label_logprobs: Mapping[str, float]) -> float | None:
        """log P(addressed) - log P(other) of an answer's first token, each side summed over its labels; a side with no label present is -inf, and None when neither side is."""
        addressed = [
            label_logprobs[label] for label in self.addressed if label in label_logprobs
        ]
        other = [
            label_logprobs[label] for label in self.other if label in label_logprobs
        ]
        if not addressed and not other:
            return None
        else:
            return (logsumexp(addressed) if addressed else -math.inf) - (
                logsumexp(other) if other else -math.inf
            )


@dataclass(frozen=True)
class Pause:
    """Stop the reply of turn where the listener is; its audio waits."""

    turn: int


@dataclass(frozen=True)
class Resume:
    """Play the paused reply of turn on from where it stopped."""

    turn: int


@dataclass(frozen=True)
class Cancel:
    """End the reply of turn through the existing interrupt path."""

    turn: int


@dataclass(frozen=True)
class Ask(Generic[Content]):
    """Request evidence on whether everything heard over the paused reply of turn was said to the assistant; answered with Floor.decided(id, margin)."""

    id: int
    turn: int
    heard: tuple[Content, ...]


@dataclass(frozen=True)
class Answer(Generic[Content]):
    """Queue one user turn made of these utterances, in order."""

    heard: tuple[Content, ...]


FloorAct = Pause | Resume | Cancel | Ask | Answer


@dataclass
class Episode(Generic[Content]):
    """One pause of one reply: from the first speech over it to its resume or cancel."""

    id: int
    turn: int
    pending: int = 0
    heard: list[Content] = field(default_factory=list)
    has_words: bool = False
    ask: int | None = None
    is_cleared: bool = False


class Floor(Generic[Content]):
    """Who holds a session's floor: nobody, a reply speaking, or a reply paused by speech over it.

    An utterance is answered unless a verdict said it was not addressed. Every
    method is idempotent on repeated or stale input. Utterances carry no turn,
    so a FIFO of the started ones tags which belong to the Episode; content
    arrives in the same order.
    """

    def __init__(self, policy: OverlapPolicy, respond_margin: float) -> None:
        self.policy: OverlapPolicy = policy
        self.respond_margin = respond_margin
        self.reply: int | None = None
        self.is_user_speaking = False
        self.episode: Episode[Content] | None = None
        self.utterances: deque[int | None] = deque()
        self.next_ask = 0
        self.next_episode = 0

    @property
    def is_paused(self) -> bool:
        return self.episode is not None

    def held(self) -> tuple[Content, ...]:
        return tuple(self.episode.heard) if self.episode is not None else ()

    def reply_started(self, turn: int) -> list[FloorAct]:
        """A reply begins to hold the floor; one that opens over speech is cancelled or, under the pause policy, paused with that speech over it."""
        if self.reply == turn:
            return []
        else:
            self.reply = turn
        if not self.is_user_speaking:
            return []
        elif self.policy == "interrupt":
            self.reply = None
            return [Cancel(turn)]
        else:
            return self.join_episode()

    def reply_ended(self, turn: int) -> list[FloorAct]:
        """The reply ended outside the floor's own Cancel: its turn.done or a client cancel; what was heard over it is answered unjudged."""
        if self.reply != turn:
            return []
        else:
            episode = self.episode
        self.reply = None
        self.episode = None
        if episode is None:
            return []
        else:
            self.release_episode(episode)
            return [Answer(tuple(episode.heard))] if episode.heard else []

    def speech_started(self) -> list[FloorAct]:
        if self.is_user_speaking:
            return []
        else:
            self.is_user_speaking = True
            self.utterances.append(None)
        if self.reply is None:
            return []
        elif self.policy == "interrupt":
            turn, self.reply = self.reply, None
            return [Cancel(turn)]
        else:
            return self.join_episode()

    def empty_utterance(self) -> None:
        """A commit with no speech still yields one (empty) content: a plain entry keeps the FIFO aligned."""
        self.utterances.append(None)

    def speech_stopped(self) -> list[FloorAct]:
        self.is_user_speaking = False
        return self.settle()

    def heard(self, content: Content, *, is_empty: bool = False) -> list[FloorAct]:
        """The content of the oldest started utterance arrived; one spoken over the paused reply asks for a verdict on all heard so far, except that content with nothing in it needs no model."""
        owner = self.utterances.popleft() if self.utterances else None
        episode = self.episode
        if episode is None or owner != episode.id:
            return [Answer((content,))]
        else:
            episode.pending -= 1
            episode.heard.append(content)
            episode.has_words = episode.has_words or not is_empty
            episode.is_cleared = False
            episode.ask = self.next_ask
            self.next_ask += 1
        if episode.has_words:
            return [Ask(episode.ask, episode.turn, tuple(episode.heard))]
        else:
            return self.decided(episode.ask, -math.inf)

    def decided(self, ask: int, verdict: float | None) -> list[FloorAct]:
        """Evidence on an Ask: log P(addressed) - log P(other), or None for no evidence, which cancels like an addressed verdict."""
        episode = self.episode
        if episode is None or ask != episode.ask:
            return []
        elif verdict is None or verdict >= self.respond_margin:
            self.reply = None
            self.episode = None
            self.release_episode(episode)
            return [Cancel(episode.turn), Answer(tuple(episode.heard))]
        else:
            episode.is_cleared = True
            return self.settle()

    def settle(self) -> list[FloorAct]:
        episode = self.episode
        if (
            episode is not None
            and episode.is_cleared
            and episode.pending == 0
            and not self.is_user_speaking
        ):
            self.episode = None
            return [Resume(episode.turn)]
        else:
            return []

    def join_episode(self) -> list[FloorAct]:
        """The utterance in progress, the newest started, is spoken over the reply holding the floor."""
        assert self.reply is not None
        acts: list[FloorAct] = []
        episode = self.episode
        if episode is None:
            episode = Episode(id=self.next_episode, turn=self.reply, is_cleared=True)
            self.next_episode += 1
            self.episode = episode
            acts.append(Pause(self.reply))
        else:
            pass
        if self.utterances:
            episode.pending += 1
            episode.is_cleared = False
            self.utterances[-1] = episode.id
        else:
            pass
        return acts

    def release_episode(self, episode: Episode[Content]) -> None:
        self.utterances = deque(
            None if owner == episode.id else owner for owner in self.utterances
        )
