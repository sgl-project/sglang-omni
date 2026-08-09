# SPDX-License-Identifier: Apache-2.0
"""Exact-owner routing state for uploaded TTS voices."""

from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Set
from dataclasses import dataclass
from typing import Any, Literal

import httpx

from sglang_omni_router.config import can_own_uploaded_voices
from sglang_omni_router.worker import Worker

logger = logging.getLogger(__name__)

DEFAULT_VOICE_NAME = "default"
MAX_VOICE_REGISTRY_RESPONSE_BYTES = 1024 * 1024


class VoiceRegistryResponseTooLargeError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceMutation:
    operation: Literal["upload", "delete"]
    name: str

    @classmethod
    def create(
        cls,
        operation: Literal["upload", "delete"],
        name: str,
    ) -> VoiceMutation | None:
        normalized = _normalize_voice_name(name)
        if normalized is None:
            return None
        return cls(operation=operation, name=normalized)


class VoiceRoutingState:
    """Track which voice names require the configured state owner."""

    def __init__(
        self,
        *,
        workers: list[Worker],
        owner_url: str | None,
        client: httpx.AsyncClient,
        timeout_secs: int,
        retry_interval_secs: int,
    ) -> None:
        self._workers = workers
        self._owner_url = owner_url
        self._client = client
        self._timeout_secs = timeout_secs
        self._retry_interval_secs = retry_interval_secs
        self._uploaded_names: set[str] = set()
        self._is_active = False
        self._is_hydrated = False
        self._is_registry_dirty = False
        self._is_reconciling = False
        self._mutations_inflight = 0
        self._last_refresh_error: str | None = None
        self._registry_generation = 0
        self._refresh_requested = asyncio.Event()
        self._task: asyncio.Task[None] | None = None

    def owner(self) -> Worker | None:
        """Return the assigned owner without changing routing state."""
        if self._owner_url is None:
            return None
        return next(
            (worker for worker in self._workers if worker.url == self._owner_url),
            None,
        )

    def ensure_owner(self) -> Worker | None:
        """Assign the first eligible automatic owner at a routing boundary."""
        owner = self.owner()
        if owner is not None or self._owner_url is not None:
            return owner
        owner = next(
            (
                worker
                for worker in self._workers
                if worker.is_routable and can_own_uploaded_voices(worker.capabilities)
            ),
            None,
        )
        if owner is not None:
            self._owner_url = owner.url
        return owner

    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run_reconciliation())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def request_refresh(self) -> None:
        if self._is_active and not self._is_reconciling:
            self._refresh_requested.set()

    def activate(self) -> None:
        if self._is_active:
            return
        self._is_active = True
        self.request_refresh()

    def mark_uncertain(self) -> None:
        self._is_active = True
        self._is_registry_dirty = True
        self._last_refresh_error = "mutation outcome is uncertain"
        self._refresh_requested.set()

    def begin_mutation(self) -> None:
        self._is_active = True
        self._mutations_inflight += 1
        self._registry_generation += 1

    def mark_mutation_dispatched(self) -> None:
        assert self._mutations_inflight > 0
        self._is_registry_dirty = True

    def end_mutation(self) -> None:
        assert self._mutations_inflight > 0
        self._mutations_inflight -= 1
        if self._mutations_inflight == 0:
            self._refresh_requested.set()

    def to_dict(self) -> dict[str, Any]:
        owner = self.owner()
        if owner is None:
            registry_state = "unassigned"
        elif not self._is_active:
            registry_state = "inactive"
        elif self._mutations_inflight:
            registry_state = "mutating"
        elif self._is_reconciling:
            registry_state = "refreshing" if self._is_hydrated else "hydrating"
        elif not self._is_hydrated:
            registry_state = (
                "degraded" if self._last_refresh_error is not None else "pending"
            )
        elif self._is_registry_dirty:
            registry_state = "degraded"
        elif self._last_refresh_error is not None:
            registry_state = "stale"
        else:
            registry_state = "ready"
        return {
            "owner_worker_id": owner.worker_id if owner is not None else None,
            "owner_url": owner.url if owner is not None else None,
            "owner_routable": owner.is_routable if owner is not None else False,
            "registry_state": registry_state,
            "uploaded_voice_count": len(self._uploaded_names),
            "last_refresh_error": self._last_refresh_error,
        }

    def requires_owner(
        self,
        voice_names: Set[str],
        *,
        is_body_over_metadata_limit: bool = False,
    ) -> bool:
        if self.ensure_owner() is None:
            return False
        if is_body_over_metadata_limit:
            return True
        names = {
            normalized
            for name in voice_names
            if (normalized := _normalize_voice_name(name)) is not None
        }
        names.discard(DEFAULT_VOICE_NAME)
        if not names:
            return False
        self.activate()
        if (
            not self._is_hydrated
            or self._is_registry_dirty
            or self._mutations_inflight > 0
        ):
            self.request_refresh()
            # The owner can serve both presets and uploaded voices. Falling back
            # to it preserves correctness when registry discovery is unavailable.
            return True
        return bool(names & self._uploaded_names)

    async def _run_reconciliation(self) -> None:
        loop = asyncio.get_running_loop()
        while True:
            await self._refresh_requested.wait()
            self._refresh_requested.clear()
            if not await self._reconcile_once():
                continue
            retry = loop.call_later(
                self._retry_interval_secs, self._refresh_requested.set
            )
            try:
                await self._refresh_requested.wait()
            finally:
                retry.cancel()

    async def _reconcile_once(self) -> bool:
        if self._mutations_inflight > 0:
            return False
        owner = self.owner()
        if owner is None or not owner.is_routable:
            return not self._is_hydrated or self._is_registry_dirty

        registry_generation = self._registry_generation
        self._is_reconciling = True
        try:
            uploaded_names = await self._fetch_uploaded_voice_names(owner)
        except (httpx.HTTPError, ValueError) as exc:
            refresh_error = _registry_refresh_error(exc)
            if self._last_refresh_error != refresh_error:
                logger.warning(
                    f"voice_registry_hydration_failed worker={owner.display_id} "
                    f"error={refresh_error}"
                )
            self._last_refresh_error = refresh_error
            return True
        else:
            if (
                self._registry_generation != registry_generation
                or self._mutations_inflight > 0
            ):
                return True
            should_log = (
                not self._is_hydrated
                or self._is_registry_dirty
                or self._last_refresh_error is not None
                or self._uploaded_names != uploaded_names
            )
            self._uploaded_names = uploaded_names
            self._is_hydrated = True
            self._is_registry_dirty = False
            self._last_refresh_error = None
            if should_log:
                logger.info(
                    f"voice_registry_hydrated worker={owner.display_id} "
                    f"uploaded_voices={len(uploaded_names)}"
                )
            return False
        finally:
            self._is_reconciling = False

    async def _fetch_uploaded_voice_names(self, owner: Worker) -> set[str]:
        async with self._client.stream(
            "GET",
            f"{owner.url}/v1/audio/voices",
            params={"names_only": "true"},
            timeout=self._timeout_secs,
        ) as response:
            response.raise_for_status()
            body = bytearray()
            async for chunk in response.aiter_bytes():
                if len(body) + len(chunk) > MAX_VOICE_REGISTRY_RESPONSE_BYTES:
                    raise VoiceRegistryResponseTooLargeError
                body.extend(chunk)
        try:
            payload = json.loads(body)
        except (json.JSONDecodeError, UnicodeDecodeError):
            raise ValueError("voice registry response must be valid JSON") from None
        return _uploaded_voice_names(payload)

    def apply(self, mutation: VoiceMutation) -> None:
        assert self._mutations_inflight > 0
        if mutation.operation == "upload":
            self._uploaded_names.add(mutation.name)
        else:
            self._uploaded_names.discard(mutation.name)


def _uploaded_voice_names(payload: Any) -> set[str]:
    if not isinstance(payload, dict):
        raise ValueError("voice list response must be an object")
    uploaded_names = payload.get("uploaded_voice_names")
    if uploaded_names is not None:
        if not isinstance(uploaded_names, list):
            raise ValueError("uploaded_voice_names must be a list")
        names: set[str] = set()
        for item in uploaded_names:
            normalized = _normalize_voice_name(item)
            if normalized is None:
                raise ValueError("uploaded_voice_names must contain names")
            names.add(normalized)
        return names

    uploaded = payload.get("uploaded_voices")
    if not isinstance(uploaded, list):
        raise ValueError("voice list response must include uploaded_voices")
    names: set[str] = set()
    for item in uploaded:
        if not isinstance(item, dict):
            raise ValueError("uploaded voice metadata must be an object")
        normalized = _normalize_voice_name(item.get("name"))
        if normalized is None:
            raise ValueError("uploaded voice metadata must include a name")
        names.add(normalized)
    return names


def _registry_refresh_error(exc: Exception) -> str:
    if isinstance(exc, httpx.HTTPStatusError):
        return f"HTTPStatusError:status={exc.response.status_code}"
    if isinstance(exc, VoiceRegistryResponseTooLargeError):
        return "VoiceRegistryResponseTooLargeError"
    return type(exc).__name__


def _normalize_voice_name(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    normalized = value.strip().lower()
    return normalized or None
