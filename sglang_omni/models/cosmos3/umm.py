# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 decisions and native media re-ingestion for the shared UMM loop."""

from __future__ import annotations

import base64
import binascii
import hashlib
import itertools
import json
import math
import re
import time
from collections import deque
from copy import deepcopy
from dataclasses import dataclass, field

from sglang_omni.admission import QueueFullError
from sglang_omni.models.cosmos3.stages import (
    COMMON_REQUEST_FIELDS,
    DEADLINE_METADATA_KEY,
    INLINE_MEDIA_LIMIT_BYTES,
    check_media_source,
    data_url,
    media_policy_configured,
    resolve_generation_options,
)
from sglang_omni.pipeline.umm import (
    CloseReason,
    MonotonicClock,
    UMMContext,
    UMMOutput,
    UMMRequestError,
    UMMStep,
)
from sglang_omni.preprocessing.resource_connector import get_global_resource_connector
from sglang_omni.proto import OmniRequest, StagePayload
from sglang_omni.scheduling.native_chat_scheduler import (
    build_chat_fields,
    set_sampling_fields,
)

# Adapter phase names, mapped to stages by the orchestrator's factory.phases.
REASON = "reason"
GENERATE = "generate"
# The stage name clients use for Reasoner overrides in stage_sampling and
# stage_params.
REASONER_STAGE = "reasoner"

# Server limits on the decision text and the generation prompt. The retained
# byte reservation plans for them, so longer model output fails the request
# as a server error. The schema leaves free text strings without pattern or
# length keywords, because native grammar decoding then admits no JSON escape
# in them, which would cut answers at their first newline or quote.
MAX_TEXT_CHARS = 8192
MAX_PROMPT_CHARS = 4096

# Native generation refuses a prompt without a character outside ASCII
# whitespace, so such a prompt returns to the Reasoner as a receipt.
NONBLANK = re.compile(r"\S", re.ASCII)

FINAL_DECISION = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["final"]},
        "text": {"type": "string"},
        "generation": {"type": "null"},
    },
    "required": ["kind", "text", "generation"],
    "additionalProperties": False,
}

GENERATE_DECISION = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["generate"]},
        "text": {"type": "string"},
        "generation": {
            "type": "object",
            "properties": {
                "modality": {"type": "string", "enum": ["image", "video"]},
                "prompt": {"type": "string"},
            },
            "required": ["modality", "prompt"],
            "additionalProperties": False,
        },
    },
    "required": ["kind", "text", "generation"],
    "additionalProperties": False,
}

SYSTEM_INSTRUCTION = (
    "You can understand images and videos and request image or video generation. "
    "Respond only with a JSON object matching the supplied schema. "
    "Use kind generate when media is needed, with a precise generation prompt and "
    "modality image or video. The text field is an optional message for the user. "
    "After a generation request, the generated media will be returned to you. "
    "Inspect that media and continue the user's task. You may request another "
    "generation when needed. Use kind final, generation null and a nonempty text "
    "answer only when the task is complete. Do not claim to have generated media "
    "before receiving the generation result."
)

# Decisions sample greedily unless the client sets a sampling field, and only
# the decision schema bounds their length: a client length or stop setting
# would truncate the JSON and fail the request.
DECISION_SAMPLING = {"temperature": 0.0}
SAMPLING_FIELDS = ("temperature", "top_p", "top_k", "min_p", "repetition_penalty")
DECISION_OWNED_FIELDS = frozenset(
    {
        "chat_template_kwargs",
        "ignore_eos",
        "max_completion_tokens",
        "max_new_tokens",
        "max_tokens",
        "n",
        "response_format",
        "stop",
        "stop_token_ids",
        "stream",
    }
)
# Interleaved results carry no logprobs or rollout artifacts.
ROLLOUT_FIELDS = (
    "return_logprob",
    "return_omni_rollout",
    "return_routed_experts",
    "return_indexer_topk",
)

MEDIA_TYPES = {
    "image": ("image/png", "image/jpeg", "image/webp"),
    "video": ("video/mp4",),
}

# The fields native Cosmos3 declares for its image and video routes, which
# the generation stage adds to COMMON_REQUEST_FIELDS through native SGLang.
# The orchestrator does not import native SGLang, so they are listed here,
# and a unit test pins them to the native declaration.
COSMOS3_REQUEST_FIELDS = frozenset(
    {
        "action",
        "action_fps",
        "action_mode",
        "action_normalization",
        "action_view_point",
        "condition_frame_indexes",
        "condition_video_keep",
        "control_guidance",
        "control_guidance_interval",
        "control_hint",
        "control_path",
        "domain_id",
        "domain_name",
        "guidance_interval",
        "max_frames",
        "num_conditional_frames",
        "num_first_chunk_conditional_frames",
        "num_video_frames_per_chunk",
        "raw_action_dim",
        "share_vision_temporal_positions",
        "show_control_condition",
        "show_input",
        "sound_duration",
        "use_duration_template",
        "use_guardrails",
        "use_resolution_template",
        "use_system_prompt",
    }
)


@dataclass(frozen=True)
class Decision:
    kind: str
    text: str
    generation: dict[str, str] | None


@dataclass
class Cosmos3History:
    """Chat history one request resubmits to the Reasoner, and its bytes.

    A receipt refers to its generated media instead of copying it, since the
    controller already retains that media as an output. media_urls holds the
    message index of each receipt and the data URL that output carries.
    """

    reasons_left: int
    generations_left: int
    sampling: dict[str, object]
    options: dict[str, object]
    context: UMMContext
    messages: list[dict[str, object]] = field(default_factory=list)
    media_urls: list[tuple[int, str]] = field(default_factory=list)
    retained_bytes: int = 0
    decision: Decision | None = None
    dispatched_at: float = 0.0
    # The deadline and the admission index, which order generation turns.
    order: tuple[float, int] = (math.inf, 0)
    # The most generations the request may still run, a pending one included.
    # A plan at raised measured times lowers it, and a grant adds one when the
    # plan leaves room.
    planned: int = 0
    # Whether the Reasoner turn in flight may answer with a generation.
    offered: bool = False


class GenerationBacklogError(QueueFullError):
    """A request that admitted generation work cannot finish in time (HTTP 503)."""

    def __init__(self, detail: str) -> None:
        super().__init__(f"{QueueFullError.MESSAGE} {detail}")


class RecentMaximum:
    """The largest value recorded within a trailing time window.

    Keeps only values that no later and larger value dominates, so each
    record and read takes amortized constant time.
    """

    sign = 1.0

    def __init__(self) -> None:
        self.entries: deque[tuple[float, float]] = deque()

    def record(self, now: float, value: object) -> None:
        if type(value) not in (int, float) or not 0 <= value < math.inf:
            return
        else:
            pass
        while self.entries and self.entries[-1][1] <= self.sign * value:
            self.entries.pop()
        self.entries.append((now, self.sign * value))

    def recent(self, now: float, window_s: float) -> float | None:
        while self.entries and self.entries[0][0] <= now - window_s:
            self.entries.popleft()
        return self.sign * self.entries[0][1] if self.entries else None

    def raise_by(self, now: float, window_s: float, value: object) -> bool:
        """Record value and report whether it raised the window maximum."""
        before = self.recent(now, window_s)
        self.record(now, value)
        after = self.recent(now, window_s)
        return after is not None and (before is None or after > before)


class RecentMinimum(RecentMaximum):
    """The smallest value recorded within a trailing time window."""

    sign = -1.0


@dataclass(frozen=True)
class MeasuredTimes:
    """The longest times measured within the last timeout_s: any generation,
    a generation of each modality, and a Reasoner turn, and the shortest
    generation of each modality."""

    generation: float
    by_modality: dict[str, float]
    reasoning: float
    fastest: dict[str, float]

    def pending(
        self, history: Cosmos3History, *, fastest: bool = False
    ) -> float | None:
        """Seconds of the request's pending generation, None without one, at
        the longest time of its modality, or with fastest at the shortest and
        at 0 before its modality was measured."""
        if history.decision is None:
            return None
        else:
            modality = history.decision.generation["modality"]
            if fastest:
                return self.fastest.get(modality, 0.0)
            else:
                return self.by_modality.get(modality, self.generation)

    def most_generations(
        self, spare: float, ahead: float, pending: float | None, limit: int
    ) -> int:
        """The most generations, a pending one included, that one request can
        still run and end within spare seconds, at most limit.

        The generation stage serves one request at a time, earliest deadline
        first, so each generation of the request waits at most for the
        generation seconds planned for requests it serves earlier (ahead), one
        younger request's running generation, and its own service. A request
        that is not generating runs a Reasoner turn before its next
        generation, and every generation is followed by one. A pending
        generation takes the longest time measured for its modality, any
        other the longest of all.
        """
        service, reasoning = self.generation, self.reasoning
        if pending is None:
            first, room = 0, spare - ahead - reasoning
        else:
            first, room = 1, spare - ahead - pending - service - reasoning
        per_generation = 2 * service + reasoning
        if room < 0:
            return 0
        elif per_generation <= 0:
            return limit
        else:
            return min(limit, first + math.floor(room / per_generation))

    def planned_seconds(self, history: Cosmos3History) -> float:
        """Generation seconds the plan reserves for one request."""
        pending = self.pending(history)
        if pending is None:
            return history.planned * self.generation
        elif history.planned <= 0:
            return 0.0
        else:
            return pending + (history.planned - 1) * self.generation


def unique_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Cosmos3 decision contains duplicate JSON keys")
        else:
            pass
        result[key] = value
    return result


def parse_decision(value: object) -> Decision:
    if not isinstance(value, dict) or set(value) != {"kind", "text", "generation"}:
        raise ValueError("Cosmos3 requires a structured final or generate decision")
    else:
        pass
    kind, text, generation = value["kind"], value["text"], value["generation"]
    if not isinstance(text, str):
        raise ValueError("Cosmos3 decision text must be a string")
    elif len(text) > MAX_TEXT_CHARS:
        raise ValueError(
            f"Cosmos3 decision text exceeds the server limit of {MAX_TEXT_CHARS} "
            "characters"
        )
    elif kind == "final":
        if generation is not None:
            raise ValueError("Final Cosmos3 decisions carry no generation")
        else:
            return Decision(kind, text, None)
    elif kind != "generate":
        raise ValueError("Unknown Cosmos3 decision kind")
    elif not isinstance(generation, dict) or set(generation) != {"modality", "prompt"}:
        raise ValueError("Cosmos3 generation requires only modality and prompt")
    elif generation["modality"] not in MEDIA_TYPES:
        raise ValueError("Cosmos3 interleaving currently supports images and videos")
    elif not isinstance(generation["prompt"], str):
        raise ValueError("Cosmos3 generation prompt must be a string")
    elif len(generation["prompt"]) > MAX_PROMPT_CHARS:
        raise ValueError(
            "Cosmos3 generation prompt exceeds the server limit of "
            f"{MAX_PROMPT_CHARS} characters"
        )
    else:
        return Decision(
            kind,
            text,
            {"modality": generation["modality"], "prompt": generation["prompt"]},
        )


def interpret_reasoner(result: object) -> Decision:
    if not isinstance(result, dict) or not isinstance(result.get("text"), str):
        raise ValueError("Cosmos3 Reasoner did not return a structured text decision")
    elif result.get("finish_reason") not in (None, "stop"):
        raise ValueError("Cosmos3 Reasoner decision was truncated or interrupted")
    else:
        pass
    try:
        parsed = json.loads(result["text"], object_pairs_hook=unique_object)
    except (ValueError, RecursionError) as exc:
        raise ValueError("Cosmos3 Reasoner returned an invalid JSON decision") from exc
    return parse_decision(parsed)


def generated_media(result: object) -> dict[str, object]:
    """The one owned inline item of a generation turn, with verified content."""
    if not isinstance(result, dict) or not isinstance(result.get("media"), list):
        raise ValueError("Cosmos3 generation did not return media")
    elif len(result["media"]) != 1:
        raise ValueError(
            "Cosmos3 interleaving requires one generated media item per turn"
        )
    else:
        pass
    item = result["media"][0]
    if not isinstance(item, dict) or "path" in item:
        raise ValueError("Cosmos3 interleaving requires owned inline media")
    else:
        pass
    kind, mime, size = item.get("kind"), item.get("mime_type"), item.get("size_bytes")
    if not isinstance(kind, str) or mime not in MEDIA_TYPES.get(kind, ()):
        raise ValueError("Unsupported inline Cosmos3 media type")
    elif type(size) is not int or not 0 < size <= INLINE_MEDIA_LIMIT_BYTES:
        raise ValueError("Cosmos3 inline media exceeds its byte limit")
    else:
        pass
    url = item.get("url")
    prefix = data_url(mime, b"")
    if not isinstance(url, str) or not url.startswith(prefix):
        raise ValueError("Cosmos3 media must be a typed data URL")
    elif len(url) - len(prefix) > 4 * ((INLINE_MEDIA_LIMIT_BYTES + 2) // 3):
        raise ValueError("Cosmos3 inline media exceeds its encoded byte limit")
    else:
        pass
    try:
        content = base64.b64decode(url[len(prefix) :], validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("Malformed Cosmos3 inline media") from exc
    if len(content) != size or hashlib.sha256(content).hexdigest() != item.get(
        "sha256"
    ):
        raise ValueError("Cosmos3 inline media identity does not match its content")
    else:
        return {
            "kind": kind,
            "mime_type": mime,
            "size_bytes": size,
            "sha256": item["sha256"],
            "url": url,
        }


def stage_mapping(params: dict[str, object], key: str) -> dict[str, object]:
    """The Reasoner entry of a per-stage mapping, which must be a mapping."""
    value = params.get(key)
    if value is None:
        return {}
    elif not isinstance(value, dict):
        raise UMMRequestError(f"{key} must be a mapping")
    else:
        reasoner = value.get("reasoner", {})
        if not isinstance(reasoner, dict):
            raise UMMRequestError(f"{key}.reasoner must be a mapping")
        else:
            return reasoner


def decision_sampling(request: OmniRequest) -> dict[str, object]:
    """Native sampling fields for every decision of one request.

    The adapter's defaults apply to fields the client did not set. The client
    sets fields at the top level, through stage_sampling.reasoner, or as
    native fields in stage_params.reasoner. A field counts as set when the
    request carries it, since Omni leaves every unset sampling field out.
    """
    params = request.params
    requested = [name for name in ROLLOUT_FIELDS if params.get(name)]
    if requested:
        raise UMMRequestError(
            "Cosmos3 interleaving returns no logprobs or rollout artifacts. "
            f"Leave {', '.join(requested)} unset or false"
        )
    else:
        pass
    sampling: dict[str, object] = dict(DECISION_SAMPLING)
    sampling.update(
        set_sampling_fields(
            {name: params[name] for name in SAMPLING_FIELDS if name in params}
        )
    )
    if params.get("seed") is not None:
        sampling["seed"] = params["seed"]
    else:
        pass
    stage_sampling = set_sampling_fields(stage_mapping(params, "stage_sampling"))
    for name, value in stage_sampling.items():
        if name in DECISION_OWNED_FIELDS:
            continue
        else:
            sampling[name] = value
    reasoner = stage_mapping(params, "stage_params")
    for name, value in reasoner.items():
        if name not in DECISION_OWNED_FIELDS:
            sampling[name] = value
        else:
            pass
    template = reasoner.get("chat_template_kwargs", params.get("chat_template_kwargs"))
    if template is not None and not isinstance(template, dict):
        raise UMMRequestError("chat_template_kwargs must be a mapping")
    else:
        sampling["chat_template_kwargs"] = {
            **(template or {}),
            "enable_thinking": False,
        }
    return sampling


def generation_options(request: OmniRequest) -> dict[str, object]:
    """The client's native generation options, checked before any turn runs."""
    try:
        options = resolve_generation_options(deepcopy(request.params))
    except ValueError as exc:
        raise UMMRequestError(str(exc)) from exc
    refused = sorted(set(options) - COMMON_REQUEST_FIELDS - COSMOS3_REQUEST_FIELDS)
    action_mode = options.get("action_mode")
    frames = options.get("num_frames")
    if refused:
        # The generation stage would refuse them at the first generation.
        raise UMMRequestError(
            f"Cosmos3 generation cannot set {', '.join(refused)}. Clients may "
            "set the fields that the native image and video routes accept, and "
            "the server owns paths it writes or reads on its own"
        )
    elif action_mode is not None and (
        not isinstance(action_mode, str)
        or action_mode.strip().lower() != "forward_dynamics"
    ):
        raise UMMRequestError(
            "Cosmos3 visual interleaving cannot return action-only results"
        )
    elif frames is not None and (type(frames) is not int or frames <= 1):
        # Image decisions always generate one frame, so a client frame count
        # applies to videos only.
        raise UMMRequestError("Video interleaving requires more than one frame")
    else:
        return options


def check_media_policy(options: dict[str, object]) -> None:
    """Refuse generation media paths the server media policy refuses, before
    any turn.

    The paths are checked without being read, and the generation stage
    applies the same policy when it reads them. The Reasoner stage applies
    the policy to chat media before native SGLang loads any.
    """
    connector = get_global_resource_connector()
    if not media_policy_configured(connector):
        return
    else:
        pass
    sources: list[tuple[str, object]] = []
    for name, value in options.items():
        if name.endswith("_path"):
            items = value if isinstance(value, list) else [value]
            sources.extend((name, item) for item in items)
        else:
            pass
    for name, source in sources:
        try:
            check_media_source(connector, name, source)
        except ValueError as exc:
            raise UMMRequestError(
                f"{name} refused by the server media policy: {exc}"
            ) from exc


def encoded_size(message: dict[str, object]) -> int:
    return len(json.dumps(message, ensure_ascii=False).encode("utf-8"))


def decision_message(decision: Decision) -> dict[str, object]:
    content = json.dumps(
        {
            "kind": decision.kind,
            "text": decision.text,
            "generation": decision.generation,
        },
        ensure_ascii=False,
    )
    return {"role": "assistant", "content": content}


# The Reasoner's next turn learns how its generation request ended. Only a
# completed generation attaches media.
RECEIPTS = {
    "completed": (
        "Generation execution completed",
        "The attached media is the result of your preceding generation "
        "request. Inspect the received results against the original user's "
        "task. If the task is complete, choose kind final with generation "
        "null and your answer in text. Otherwise request the next needed "
        "generation. Repeat a completed generation only when the user's "
        "task or your assessment of its result calls for a revision.",
    ),
    "not_executed": (
        "Generation was not executed because its prompt was blank",
        "No media is attached. Request the generation again with a precise "
        "nonblank prompt, or choose kind final with generation null and your "
        "answer in text.",
    ),
    "no_time": (
        "Generation was not executed because the request deadline leaves no "
        "time for it",
        "No media is attached. Choose kind final with generation null and "
        "answer the user with what you have.",
    ),
}


def receipt(decision: Decision, status: str) -> dict[str, object]:
    """The user turn that returns a generation outcome. A completed one
    attaches its media with the URL left empty."""
    summary, guidance = RECEIPTS[status]
    kind = decision.generation["modality"]
    record = json.dumps(
        {
            "status": status,
            "modality": kind,
            "requested_prompt": decision.generation["prompt"],
        },
        ensure_ascii=False,
    )
    content: list[dict[str, object]] = [
        {"type": "text", "text": f"{summary}. Result receipt: {record}\n{guidance}"}
    ]
    if status == "completed":
        part = f"{kind}_url"
        content.append({"type": part, part: {"url": ""}})
    else:
        pass
    return {"role": "user", "content": content}


def worst_case_bytes() -> tuple[int, int]:
    """Most bytes one Reasoner turn and one generation turn can add, for
    answers and prompts within MAX_TEXT_CHARS and MAX_PROMPT_CHARS.

    A Reasoner turn adds its text output, counted in UTF-8 at up to four bytes
    a character, and its decision in the history, where a control character
    costs the most: seven bytes once the decision and then the message are
    JSON encoded. A generation turn adds its media output, encoded the way the
    controller counts outputs, and its receipt.
    """
    worst = Decision(
        "generate",
        "\x01" * MAX_TEXT_CHARS,
        {"modality": "video", "prompt": "\x01" * MAX_PROMPT_CHARS},
    )
    reason = 4 * MAX_TEXT_CHARS + encoded_size(decision_message(worst))
    mime = max((m for mimes in MEDIA_TYPES.values() for m in mimes), key=len)
    item = {
        "kind": "video",
        "mime_type": mime,
        "size_bytes": INLINE_MEDIA_LIMIT_BYTES,
        "sha256": "0" * 64,
        "url": data_url(mime, b""),
    }
    # Base64 characters need no JSON escaping.
    media = encoded_size(item) + 4 * ((INLINE_MEDIA_LIMIT_BYTES + 2) // 3)
    return reason, media + max(encoded_size(receipt(worst, s)) for s in RECEIPTS)


WORST_CASE_BYTES = worst_case_bytes()


def max_model_bytes(max_dispatches: int) -> int:
    """Most bytes a request's model turns can retain within max_dispatches.

    The first dispatch is a Reasoner turn, and each generation is followed by
    a Reasoner turn that inspects it.
    """
    generations = (max_dispatches - 1) // 2
    reason, generation = WORST_CASE_BYTES
    return (generations + 1) * reason + generations * generation


class Cosmos3UMMAdapter:
    """Translate Cosmos3 decisions while native runtimes keep model execution.

    Each request alternates between the Reasoner (phase reason), which answers
    with a structured decision, and native generation (phase generate), whose
    media returns to the Reasoner in the next turn.
    """

    def __init__(
        self, model_path: str, *, clock: MonotonicClock = time.monotonic
    ) -> None:
        """clock must be the controller's clock, which sets each deadline."""
        del model_path
        self.histories: dict[str, Cosmos3History] = {}
        self.clock = clock
        # Measured within the last timeout_s, the deadline every request has.
        self.window_s = math.inf
        self.generation_times = {kind: RecentMaximum() for kind in MEDIA_TYPES}
        self.fastest_times = {kind: RecentMinimum() for kind in MEDIA_TYPES}
        self.reason_times = RecentMaximum()
        self.admissions = itertools.count()

    def measured(self, now: float) -> MeasuredTimes | None:
        """The times plans use, once a generation was measured."""
        by_modality = {
            kind: value
            for kind, times in self.generation_times.items()
            if (value := times.recent(now, self.window_s)) is not None
        }
        if not by_modality:
            return None
        else:
            return MeasuredTimes(
                max(by_modality.values()),
                by_modality,
                self.reason_times.recent(now, self.window_s) or 0.0,
                {
                    kind: value
                    for kind, times in self.fastest_times.items()
                    if (value := times.recent(now, self.window_s)) is not None
                },
            )

    def active(self, now: float) -> list[Cosmos3History]:
        """Requests not being ended, in the generation stage's service order."""
        histories = tuple(self.histories.values())
        return sorted(
            (
                h
                for h in histories
                if h.context.deadline > now and not h.context.shedding
            ),
            key=lambda h: h.order,
        )

    def admit(self, context: UMMContext, now: float, generations: int) -> int:
        """The generations a new request plans, as many as end by its deadline.

        The request is served after every active one, behind the generations
        planned for them, at the measured times (most_generations), and is
        refused when they leave it fewer generations than an idle server
        would. Before any generation was measured it plans them all, and the
        plan at the first measurement settles it.
        """
        times = self.measured(now)
        if times is None or generations <= 0:
            return generations
        else:
            pass
        ahead = sum(
            times.planned_seconds(h)
            for h in tuple(self.histories.values())
            if h.context.deadline > now and not h.context.shedding
        )
        spare = context.deadline - now
        fits = times.most_generations(spare, ahead, None, generations)
        idle = times.most_generations(spare, 0.0, None, generations)
        if fits < idle:
            raise GenerationBacklogError(
                f"Admitted Cosmos3 work leaves time for {fits} of the {idle} "
                "generations this request could run before its deadline at "
                f"{times.generation:.1f} s per generation. Retry later"
            )
        else:
            return fits

    def replan(self, now: float, current: Cosmos3History) -> None:
        """Plan every active request again at newly raised measured times.

        Earlier plans assumed the previous maxima, so a raised one can break
        them. In the generation stage's service order, each request gets the
        most generations that end by its deadline behind the generations just
        planned for the requests ahead of it, never more than before. A
        request that cannot finish even its pending generation and a final
        answer, with that generation at the shortest time measured for its
        modality, or its Reasoner turn in flight, is shed: the controller's
        deadline sweep fails it with the capacity error and aborts its queued
        or running generation. A pending generation that fits only at that
        shortest time stays planned, since the longest time of its modality
        may come from a longer generation of another request. reason ends
        current, the request in transition, if not even its next turn fits.
        """
        times = self.measured(now)
        if times is None:
            return
        else:
            pass
        ahead = 0.0
        for history in self.active(now):
            context = history.context
            pending = times.pending(history)
            history.planned = times.most_generations(
                context.deadline - now, ahead, pending, history.planned
            )
            if history.planned > 0 or history is current:
                pass
            elif (
                pending is not None
                and times.most_generations(
                    context.deadline - now,
                    ahead,
                    times.pending(history, fastest=True),
                    1,
                )
                == 1
            ):
                history.planned = 1
            elif (
                pending is not None
                or history.dispatched_at + times.reasoning > context.deadline
            ):
                context.shed()
            else:
                pass
            ahead += times.planned_seconds(history)

    def grant(self, now: float, current: Cosmos3History) -> bool:
        """Whether current, in transition with no generation planned, may
        still run one: it ends by its deadline behind the requests ahead of
        it, and every request served after it keeps its planned generations.

        Plans charge every generation the longest measured time, so requests
        that ran faster leave room that a lowered budget can use again.
        """
        times = self.measured(now)
        if times is None:
            return False
        else:
            pass
        ahead = 0.0
        behind = False
        for history in self.active(now):
            spare = history.context.deadline - now
            if history is current:
                fits = times.most_generations(spare, ahead, None, 1) == 1
                ahead += times.generation
                behind = True
            else:
                pending = times.pending(history)
                kept = times.most_generations(spare, ahead, pending, history.planned)
                fits = not behind or kept == history.planned
                ahead += times.planned_seconds(history)
            if not fits:
                return False
            else:
                pass
        return behind

    def measure(
        self, times: RecentMaximum, now: float, value: object, history: Cosmos3History
    ) -> None:
        if times.raise_by(now, self.window_s, value):
            self.replan(now, history)
        else:
            pass

    def start(self, context: UMMContext) -> UMMStep:
        try:
            # The chat history in native form, as the Reasoner stage builds it.
            request = OmniRequest(context.request.inputs, {}, context.request.metadata)
            payload = StagePayload(context.execution_id, request, None)
            messages = build_chat_fields(payload, "", set(), REASONER_STAGE)["messages"]
        except ValueError as exc:
            raise UMMRequestError(str(exc)) from exc
        if any(
            message.get("role") not in ("system", "user", "assistant")
            for message in messages
        ):
            raise UMMRequestError(
                "Cosmos3 interleaving accepts native user and assistant chat history"
            )
        elif messages[0].get("role") == "system" and isinstance(
            messages[0].get("content"), str
        ):
            messages[0]["content"] = (
                SYSTEM_INSTRUCTION + "\n\n" + messages[0]["content"]
            )
        else:
            messages.insert(0, {"role": "system", "content": SYSTEM_INSTRUCTION})
        options = generation_options(context.request)
        check_media_policy(options)
        generations = (context.remaining_dispatches - 1) // 2
        now = self.clock()
        self.window_s = context.deadline - now
        planned = self.admit(context, now, generations)
        history = Cosmos3History(
            reasons_left=generations + 1,
            generations_left=generations,
            sampling=decision_sampling(context.request),
            options=options,
            context=context,
            order=(context.deadline, next(self.admissions)),
            planned=planned,
        )
        self.histories[context.execution_id] = history
        for message in messages:
            self.remember(history, message)
        return self.reason(context, history, ())

    @staticmethod
    def remember(history: Cosmos3History, message: dict[str, object]) -> None:
        history.messages.append(message)
        history.retained_bytes += encoded_size(message)

    def reason(
        self,
        context: UMMContext,
        history: Cosmos3History,
        outputs: tuple[UMMOutput, ...],
    ) -> UMMStep:
        now = self.clock()
        reasoning = self.reason_times.recent(now, self.window_s)
        if reasoning is not None and now + reasoning > context.deadline:
            # Not even this turn can end in time, so the request ends now with
            # the capacity error instead of dispatching a turn to be aborted.
            raise QueueFullError(
                f"{QueueFullError.MESSAGE} Reasoner turns measured at up to "
                f"{reasoning:.1f} s leave this Cosmos3 request no time for its "
                "next turn before its deadline. Retry later"
            )
        else:
            pass
        # A generation and the Reasoner turn that inspects it need two more
        # dispatches after this one, so with fewer left only final remains.
        # Every earlier generate decision used one of the allowed generations,
        # and the plan at the measured times must leave one.
        allowed = (
            context.deadline > now
            and context.remaining_dispatches >= 3
            and history.generations_left > 0
        )
        if allowed and history.planned <= 0 and self.grant(now, history):
            history.planned = 1
        else:
            pass
        history.offered = allowed and history.planned > 0
        if not history.offered:
            history.planned = 0
            schema = deepcopy(FINAL_DECISION)
        else:
            generate = deepcopy(GENERATE_DECISION)
            if history.options.get("action_mode") is not None:
                # Known actions condition a video, so only videos remain.
                modality = generate["properties"]["generation"]["properties"]
                modality["modality"]["enum"] = ["video"]
            else:
                pass
            schema = {"anyOf": [deepcopy(FINAL_DECISION), generate]}
        params = {
            **deepcopy(history.sampling),
            "stream": False,
            "n": 1,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "cosmos3_umm_decision",
                    "strict": True,
                    "schema": schema,
                },
            },
        }
        messages = list(history.messages)
        for index, url in history.media_urls:
            message = messages[index]
            text, media = message["content"]
            part = media["type"]
            messages[index] = {
                **message,
                "content": [text, {"type": part, part: {"url": url}}],
            }
        request = OmniRequest({"messages": messages}, params, {})
        history.dispatched_at = now
        return UMMStep(REASON, request, outputs)

    def advance(self, context: UMMContext, phase: str, result: object) -> UMMStep:
        history = self.histories[context.execution_id]
        now = self.clock()
        if phase == REASON:
            offered, history.offered = history.offered, False
            self.measure(self.reason_times, now, now - history.dispatched_at, history)
            decision = interpret_reasoner(result)
            history.reasons_left -= 1
            outputs = (UMMOutput("text", decision.text),) if decision.text else ()
            if decision.kind == "final":
                return UMMStep(outputs=outputs)
            elif (
                history.options.get("action_mode") is not None
                and decision.generation["modality"] != "video"
            ):
                raise ValueError("Cosmos3 forward_dynamics requires a video decision")
            elif NONBLANK.search(decision.generation["prompt"]) is None:
                # Native generation refuses a blank prompt, so the Reasoner
                # learns that nothing ran and decides again.
                history.generations_left -= 1
                history.planned = max(history.planned - 1, 0)
                self.remember(history, decision_message(decision))
                self.remember(history, receipt(decision, "not_executed"))
                return self.reason(context, history, outputs)
            elif not offered:
                raise ValueError(
                    "Cosmos3 Reasoner requested a generation its turn did not offer"
                )
            elif history.planned <= 0 and not self.grant(now, history):
                # A plan at newly measured times left no time for it, so the
                # Reasoner learns that nothing ran and answers final.
                history.generations_left -= 1
                self.remember(history, decision_message(decision))
                self.remember(history, receipt(decision, "no_time"))
                return self.reason(context, history, outputs)
            else:
                history.planned = max(history.planned, 1)
                history.decision = decision
                self.remember(history, decision_message(decision))
                request = self.generation_request(history, decision)
                return UMMStep(GENERATE, request, outputs)
        elif phase == GENERATE and history.decision is not None:
            decision, history.decision = history.decision, None
            history.generations_left -= 1
            history.planned = max(history.planned - 1, 0)
            if isinstance(result, dict):
                modality = decision.generation["modality"]
                self.fastest_times[modality].record(now, result.get("service_s"))
                self.measure(
                    self.generation_times[modality],
                    now,
                    result.get("service_s"),
                    history,
                )
            else:
                pass
            item = generated_media(result)
            if item["kind"] != decision.generation["modality"]:
                raise ValueError(
                    "Generated Cosmos3 modality differs from the model decision"
                )
            else:
                pass
            history.media_urls.append((len(history.messages), item["url"]))
            self.remember(history, receipt(decision, "completed"))
            return self.reason(context, history, (UMMOutput(item["kind"], item),))
        else:
            raise ValueError(f"Cosmos3 adapter received an unexpected phase {phase!r}")

    def usage(self, context: UMMContext) -> int:
        """Retained history bytes plus the most the remaining turns can add.

        The controller checks this against max_retained_bytes at start, where
        a breach is the client's input, so a request it admits can never
        exceed the bound through model output.
        """
        history = self.histories.get(context.execution_id)
        if history is None:
            return 0
        else:
            reason, generation = WORST_CASE_BYTES
            return (
                history.retained_bytes
                + history.reasons_left * reason
                + history.generations_left * generation
            )

    def close(self, context: UMMContext, reason: CloseReason) -> None:
        self.histories.pop(context.execution_id, None)

    @staticmethod
    def generation_request(history: Cosmos3History, decision: Decision) -> OmniRequest:
        options = deepcopy(history.options)
        options["prompt"] = decision.generation["prompt"]
        options["num_outputs_per_prompt"] = 1
        if decision.generation["modality"] == "image":
            options["num_frames"] = 1
        else:
            options.setdefault("num_frames", 33)
        return OmniRequest(
            {"prompt": decision.generation["prompt"]},
            {"diffusion": options, "stream": False},
            # The generation stage serves the oldest request first.
            {DEADLINE_METADATA_KEY: list(history.order)},
        )
