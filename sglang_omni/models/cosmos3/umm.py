# SPDX-License-Identifier: Apache-2.0
"""Cosmos3 decisions and native media re-ingestion for the shared UMM loop."""

from __future__ import annotations

import base64
import binascii
import hashlib
import json
from copy import deepcopy
from typing import Any

from sglang_omni.models.cosmos3.reasoner import build_chat_messages
from sglang_omni.models.cosmos3.stages import (
    INLINE_MEDIA_LIMIT_BYTES,
    resolve_generation_options,
)
from sglang_omni.pipeline.umm import UMMController, UMMDecision, UMMLimits
from sglang_omni.proto import OmniRequest

MAX_TEXT_CHARS = 8192
MAX_PROMPT_CHARS = 4096

_DECISION_SCHEMA = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["final", "generate"]},
        "text": {"type": "string", "maxLength": MAX_TEXT_CHARS},
        "generation": {
            "anyOf": [
                {
                    "type": "object",
                    "properties": {
                        "modality": {"type": "string", "enum": ["image", "video"]},
                        "prompt": {
                            "type": "string",
                            "minLength": 1,
                            "maxLength": MAX_PROMPT_CHARS,
                        },
                    },
                    "required": ["modality", "prompt"],
                    "additionalProperties": False,
                },
                {"type": "null"},
            ]
        },
    },
    "required": ["kind", "text", "generation"],
    "additionalProperties": False,
}

_SYSTEM_INSTRUCTION = (
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


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Cosmos3 decision contains duplicate JSON keys")
        result[key] = value
    return result


def _decision(value: Any) -> UMMDecision:
    if not isinstance(value, dict) or set(value) != {"kind", "text", "generation"}:
        raise ValueError("Cosmos3 requires a structured final or generate decision")
    kind, text, generation = value["kind"], value["text"], value["generation"]
    if not isinstance(text, str) or len(text) > MAX_TEXT_CHARS:
        raise ValueError("Cosmos3 decision text exceeds its string limit")
    if kind == "final":
        if generation is not None or not text.strip():
            raise ValueError("Final Cosmos3 decisions require text and no generation")
    elif kind == "generate":
        if not isinstance(generation, dict) or set(generation) != {
            "modality",
            "prompt",
        }:
            raise ValueError("Cosmos3 generation requires only modality and prompt")
        if generation["modality"] not in ("image", "video"):
            raise ValueError(
                "Cosmos3 interleaving currently supports images and videos"
            )
        prompt = generation["prompt"]
        if (
            not isinstance(prompt, str)
            or not prompt.strip()
            or len(prompt) > MAX_PROMPT_CHARS
        ):
            raise ValueError("Cosmos3 generation requires a bounded nonempty prompt")
    else:
        raise ValueError("Unknown Cosmos3 decision kind")
    return UMMDecision(kind=kind, text=text, generation=deepcopy(generation))


def _media_items(result: Any) -> list[dict]:
    if not isinstance(result, dict) or not isinstance(result.get("media"), list):
        raise ValueError("Cosmos3 generation did not return media")
    media = result["media"]
    if len(media) != 1:
        raise ValueError(
            "Cosmos3 interleaving requires one generated media item per turn"
        )
    for item in media:
        if not isinstance(item, dict) or "path" in item:
            raise ValueError("Cosmos3 interleaving requires owned inline media")
        mime = item.get("mime_type")
        kind = item.get("kind")
        allowed = {
            "image": ("image/png", "image/jpeg", "image/webp"),
            "video": ("video/mp4",),
        }
        if (
            not isinstance(kind, str)
            or kind not in allowed
            or mime not in allowed[kind]
        ):
            raise ValueError("Unsupported inline Cosmos3 media type")
        size = item.get("size_bytes")
        if type(size) is not int or not 0 < size <= INLINE_MEDIA_LIMIT_BYTES:
            raise ValueError("Cosmos3 inline media exceeds its byte limit")
        url = item.get("url")
        prefix = f"data:{mime};base64,"
        if not isinstance(url, str) or not url.startswith(prefix):
            raise ValueError("Cosmos3 media must be a typed data URL")
        encoded = url[len(prefix) :]
        if len(encoded) > 4 * ((INLINE_MEDIA_LIMIT_BYTES + 2) // 3):
            raise ValueError("Cosmos3 inline media exceeds its encoded byte limit")
        try:
            content = base64.b64decode(encoded, validate=True)
        except (ValueError, binascii.Error) as exc:
            raise ValueError("Malformed Cosmos3 inline media") from exc
        if len(content) != size or hashlib.sha256(content).hexdigest() != item.get(
            "sha256"
        ):
            raise ValueError("Cosmos3 inline media identity does not match its content")
    return media


class Cosmos3UMMAdapter:
    """Translate model decisions while native runtimes retain model execution."""

    def start(self, request: OmniRequest) -> list[dict]:
        messages = build_chat_messages(request.inputs)
        if any(
            message.get("role") not in ("system", "user", "assistant")
            for message in messages
        ):
            raise ValueError(
                "Cosmos3 interleaving accepts native user and assistant chat history"
            )
        if messages[0].get("role") == "system" and isinstance(
            messages[0].get("content"), str
        ):
            messages[0]["content"] = (
                _SYSTEM_INSTRUCTION + "\n\n" + messages[0]["content"]
            )
        else:
            messages.insert(0, {"role": "system", "content": _SYSTEM_INSTRUCTION})
        return messages

    def reasoner_request(
        self, history: list[dict], request: OmniRequest
    ) -> OmniRequest:
        params = deepcopy(request.params)
        # Flatten native reasoner overrides first so the internal decision
        # contract cannot be replaced by a stage-specific stream or schema.
        for key in ("stage_sampling", "stage_params"):
            value = params.pop(key, {})
            if not isinstance(value, dict):
                raise ValueError(f"{key} must be a mapping")
            reasoner = value.get("reasoner", {})
            if not isinstance(reasoner, dict):
                raise ValueError("Reasoner stage parameters must be a mapping")
            params.update(reasoner)
        template = params.get("chat_template_kwargs") or {}
        if not isinstance(template, dict):
            raise ValueError("chat_template_kwargs must be a mapping")
        params.update(
            stream=False,
            n=1,
            response_format={
                "type": "json_schema",
                "json_schema": {
                    "name": "cosmos3_umm_decision",
                    "strict": True,
                    "schema": deepcopy(_DECISION_SCHEMA),
                },
            },
            chat_template_kwargs={**template, "enable_thinking": False},
        )
        params.setdefault("temperature", 0.0)
        params.setdefault("max_tokens", 2048)
        return OmniRequest(
            {"messages": deepcopy(history)}, params, deepcopy(request.metadata)
        )

    def interpret_reasoner(self, result: Any) -> UMMDecision:
        if not isinstance(result, dict) or not isinstance(result.get("text"), str):
            raise ValueError(
                "Cosmos3 Reasoner did not return a structured text decision"
            )
        if result.get("finish_reason") not in (None, "stop"):
            raise ValueError("Cosmos3 Reasoner decision was truncated or interrupted")
        text = result["text"]
        if len(text) > 2 * (MAX_TEXT_CHARS + MAX_PROMPT_CHARS):
            raise ValueError("Cosmos3 decision exceeds its serialized size limit")
        try:
            parsed = json.loads(text, object_pairs_hook=_unique_object)
        except (ValueError, RecursionError) as exc:
            raise ValueError(
                "Cosmos3 Reasoner returned an invalid JSON decision"
            ) from exc
        return _decision(parsed)

    def generation_request(
        self, decision: UMMDecision, request: OmniRequest
    ) -> OmniRequest:
        validated = _decision(
            {
                "kind": decision.kind,
                "text": decision.text,
                "generation": decision.generation,
            }
        )
        if validated.kind != "generate":
            raise ValueError("Only generation decisions can enter the generation stage")
        options = resolve_generation_options(deepcopy(request.params))
        action_mode = options.get("action_mode")
        if action_mode is not None and (
            not isinstance(action_mode, str)
            or action_mode.strip().lower() != "forward_dynamics"
        ):
            raise ValueError(
                "Cosmos3 visual interleaving cannot return action-only results"
            )
        generation = validated.generation
        options["prompt"] = generation["prompt"]
        options["num_outputs_per_prompt"] = 1
        if generation["modality"] == "image":
            if action_mode is not None:
                raise ValueError("Cosmos3 forward_dynamics requires a video decision")
            options["num_frames"] = 1
        else:
            options.setdefault("num_frames", 33)
            if type(options["num_frames"]) is not int or options["num_frames"] <= 1:
                raise ValueError("Video interleaving requires more than one frame")
        return OmniRequest(
            {"prompt": generation["prompt"]},
            {"diffusion": options, "stream": False},
            deepcopy(request.metadata),
        )

    def incorporate_media(
        self, history: list[dict], decision: UMMDecision, result: Any
    ) -> list[dict]:
        media = _media_items(result)
        if media[0]["kind"] != decision.generation["modality"]:
            raise ValueError(
                "Generated Cosmos3 modality differs from the model decision"
            )
        messages = deepcopy(history)
        messages.append(
            {
                "role": "assistant",
                "content": json.dumps(
                    {
                        "kind": decision.kind,
                        "text": decision.text,
                        "generation": decision.generation,
                    },
                    ensure_ascii=False,
                ),
            }
        )
        content = [
            {
                "type": "text",
                "text": (
                    "Generation execution completed. Result receipt: "
                    + json.dumps(
                        {
                            "status": "completed",
                            "modality": media[0]["kind"],
                            "requested_prompt": decision.generation["prompt"],
                        },
                        ensure_ascii=False,
                    )
                    + "\nThe attached media is the result of your preceding generation "
                    "request. Inspect the received results against the original user's "
                    "task. If the task is complete, choose kind final with generation "
                    "null and your answer in text. Otherwise request the next needed "
                    "generation. Repeat a completed generation only when the user's "
                    "task or your assessment of its result calls for a revision."
                ),
            }
        ]
        for item in media:
            kind = f"{item['kind']}_url"
            content.append({"type": kind, kind: {"url": item["url"]}})
        messages.append({"role": "user", "content": content})
        return messages

    def media_segments(self, result: Any) -> list[dict]:
        return [
            {"kind": item["kind"], "data": deepcopy(item)}
            for item in _media_items(result)
        ]


def create_umm_scheduler(
    model_path: str,
    *,
    max_turns: int = 8,
    max_segments: int = 16,
    max_sessions: int = 32,
    max_context_bytes: int = 64 * 1024 * 1024,
    timeout_s: float = 3600.0,
) -> UMMController:
    """Construct the CPU controller without loading another model runtime."""
    return UMMController(
        Cosmos3UMMAdapter(),
        limits=UMMLimits(
            max_turns=max_turns,
            max_segments=max_segments,
            max_sessions=max_sessions,
            max_context_bytes=max_context_bytes,
            timeout_s=timeout_s,
        ),
    )
