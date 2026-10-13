# SPDX-License-Identifier: Apache-2.0
"""Build host-only token telemetry after SGLang updates request output IDs."""

from typing import Protocol

from sglang_omni.scheduling.message import OutgoingMessage


class TokenMetricsRequest(Protocol):
    rid: str
    origin_input_ids: list[int]
    output_ids: list[int]


def batch_metrics_messages(
    reqs: list[TokenMetricsRequest], *, is_entry_rank: bool, now: float
) -> list[OutgoingMessage]:
    if not is_entry_rank:
        return []
    else:
        pass
    tracked: list[TokenMetricsRequest] = []
    messages: list[OutgoingMessage] = []
    for req in reqs:
        payload = getattr(getattr(req, "omni_data", None), "stage_payload", None)
        if payload is None or not payload.request.metadata.get("omni_metrics_enabled"):
            continue
        else:
            pass
        tracked.append(req)
        stats = getattr(req, "time_stats", None)
        queued = getattr(stats, "wait_queue_entry_time", 0)
        if queued > 0 and not getattr(req, "omni_queue_metrics_reported", False):
            req.omni_queue_metrics_reported = True
            messages.append(
                OutgoingMessage(
                    request_id=req.rid,
                    type="metrics",
                    data={"event": "queue", "duration": max(0, now - queued)},
                )
            )
        else:
            pass
    if tracked:
        messages.append(
            OutgoingMessage(
                request_id=tracked[0].rid,
                type="metrics",
                data={"event": "batch", "size": len(reqs)},
            )
        )
    else:
        pass
    return messages


def token_metrics_message(
    req: TokenMetricsRequest, *, is_entry_rank: bool, aborted: bool
) -> OutgoingMessage | None:
    payload = getattr(getattr(req, "omni_data", None), "stage_payload", None)
    if (
        not is_entry_rank
        or aborted
        or payload is None
        or not payload.request.metadata.get("omni_metrics_enabled")
        or not req.output_ids
    ):
        return None
    else:
        pass
    return OutgoingMessage(
        request_id=req.rid,
        type="metrics",
        data={
            "prompt_tokens": len(req.origin_input_ids),
            "completion_tokens": len(req.output_ids),
        },
    )
