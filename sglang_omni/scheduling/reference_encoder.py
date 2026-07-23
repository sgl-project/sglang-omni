# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
import time
from dataclasses import asdict, dataclass
from typing import Any, Generic, TypeVar, cast

import torch

from sglang_omni.scheduling.stage_cache import StageOutputCache

logger = logging.getLogger(__name__)

InputT = TypeVar("InputT")
ArtifactT = TypeVar("ArtifactT")
StoredT = TypeVar("StoredT")


@dataclass(frozen=True)
class ReferenceEncodeKey:
    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str
    input_key: str
    options_key: str = ""

    def to_string(self) -> str:
        return json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))


class ReferenceEncodeHook(Generic[InputT, ArtifactT, StoredT]):
    def normalize_input(self, raw_input: Any) -> InputT:
        raise NotImplementedError

    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None:
        raise NotImplementedError

    def encode_one(self, item: InputT) -> ArtifactT:
        raise NotImplementedError

    def store_artifact(self, artifact: ArtifactT) -> StoredT:
        raise NotImplementedError

    def load_artifact(self, stored: StoredT) -> ArtifactT:
        raise NotImplementedError

    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool:
        return True

    def encode_batch(self, items: list[InputT]) -> list[ArtifactT]:
        """Reserved; ReferenceEncodeService does not call this until batching."""
        return [self.encode_one(item) for item in items]


class KeyedReferenceEncodeHook(ReferenceEncodeHook[InputT, ArtifactT, StoredT]):
    """Defaults for hooks with structured identity and option keys."""

    model_id: str
    model_revision: str
    encoder_id: str
    encoder_config_hash: str
    artifact_kind: str

    def normalize_input(self, raw_input: Any) -> InputT:
        return cast(InputT, raw_input)

    def input_key(self, item: InputT) -> str | None:
        raise NotImplementedError

    def options_key(self, item: InputT) -> str:
        return ""

    def cache_key(self, item: InputT) -> ReferenceEncodeKey | None:
        input_key = self.input_key(item)
        if input_key is None:
            return None
        return ReferenceEncodeKey(
            model_id=self.model_id,
            model_revision=self.model_revision,
            encoder_id=self.encoder_id,
            encoder_config_hash=self.encoder_config_hash,
            artifact_kind=self.artifact_kind,
            input_key=input_key,
            options_key=self.options_key(item),
        )

    def revalidate(self, item: InputT, key: ReferenceEncodeKey) -> bool:
        return (
            self.input_key(item) == key.input_key
            and self.options_key(item) == key.options_key
        )


class TensorReferenceEncodeHook(
    KeyedReferenceEncodeHook[InputT, torch.Tensor, torch.Tensor]
):
    """Defaults for reference encoders that cache CPU tensor artifacts."""

    storage_dtype: torch.dtype | None = None
    output_dtype: torch.dtype | None = None

    def store_artifact(self, artifact: torch.Tensor) -> torch.Tensor:
        return artifact.detach().to(device="cpu", dtype=self.storage_dtype, copy=True)

    def load_artifact(self, stored: torch.Tensor) -> torch.Tensor:
        return stored.detach().to(dtype=self.output_dtype, copy=True)


def _fresh_exception(exc: BaseException) -> BaseException:
    try:
        fresh = type(exc)(*getattr(exc, "args", ()))
    except Exception:
        fresh = RuntimeError(str(exc))
    for note in getattr(exc, "__notes__", ()):
        add_note = getattr(fresh, "add_note", None)
        if callable(add_note):
            add_note(note)
    return fresh


class ReferenceEncodeService(Generic[InputT, ArtifactT, StoredT]):
    _LOG_INTERVAL_S = 60.0

    def __init__(
        self,
        hook: ReferenceEncodeHook[InputT, ArtifactT, StoredT],
        *,
        max_items: int | None = 256,
        max_bytes: int | None = 64 * 1024 * 1024,
        timeout_s: float = 130.0,
        log_prefix: str | None = None,
    ) -> None:
        if max_items is not None and max_items < 1:
            raise ValueError(f"max_items must be >= 1, got {max_items}")
        if max_bytes is not None and max_bytes < 1:
            raise ValueError(f"max_bytes must be >= 1, got {max_bytes}")
        self._hook = hook
        self._cache = StageOutputCache(max_size=max_items, max_bytes=max_bytes)
        self._timeout_s = float(timeout_s)
        self._log_prefix = log_prefix
        self._lock = threading.Lock()
        self._inflight: dict[str, concurrent.futures.Future[StoredT]] = {}
        self._hits = 0
        self._misses = 0
        self._merged = 0
        self._failed = 0
        self._uncacheable = 0
        self._last_log_time = 0.0

    def get_or_encode(self, raw_input: Any, *, desc: str | None = None) -> ArtifactT:
        item = self._hook.normalize_input(raw_input)
        key = self._hook.cache_key(item)
        if key is None:
            with self._lock:
                self._uncacheable += 1
            try:
                return self._hook.encode_one(item)
            except BaseException as exc:
                self._add_exception_note(exc, desc)
                with self._lock:
                    self._failed += 1
                raise

        cache_key = key.to_string()
        leader_fut: concurrent.futures.Future[StoredT] | None = None
        follower_fut: concurrent.futures.Future[StoredT] | None = None
        stored: StoredT | None = None
        with self._lock:
            stored = self._cache.get(cache_key)
            if stored is not None:
                self._hits += 1
            elif cache_key in self._inflight:
                self._merged += 1
                follower_fut = self._inflight[cache_key]
            else:
                self._misses += 1
                leader_fut = concurrent.futures.Future()
                self._inflight[cache_key] = leader_fut

        if stored is not None:
            self._maybe_log()
            return self._hook.load_artifact(stored)

        if follower_fut is not None:
            try:
                stored = follower_fut.result(timeout=self._timeout_s)
            except concurrent.futures.TimeoutError as exc:
                self._add_exception_note(exc, desc)
                raise
            except BaseException as exc:
                self._add_exception_note(exc, desc)
                raise _fresh_exception(exc) from exc
            return self._hook.load_artifact(stored)

        assert leader_fut is not None
        # revalidate() and cache.put() run inside the same guard as encode: any
        # exception here must still drop the in-flight entry and fail the future,
        # otherwise same-key followers block on a future that never resolves and
        # every later request for this key re-follows a dead leader (permanent
        # poison + timeout-length hangs). A hook's revalidate() may legitimately
        # raise (e.g. a reference file mutated during the encode window).
        try:
            artifact = self._hook.encode_one(item)
            stored = self._hook.store_artifact(artifact)
            should_cache = self._hook.revalidate(item, key)
            with self._lock:
                if should_cache:
                    self._cache.put(cache_key, stored)
                self._inflight.pop(cache_key, None)
        except BaseException as exc:
            self._add_exception_note(exc, desc)
            with self._lock:
                self._inflight.pop(cache_key, None)
                self._failed += 1
            leader_fut.set_exception(exc)
            raise
        leader_fut.set_result(stored)
        self._maybe_log()
        return self._hook.load_artifact(stored)

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
                "evictions": self._cache.eviction_count,
                "failed": self._failed,
                "uncacheable": self._uncacheable,
            }

    @staticmethod
    def _add_exception_note(exc: BaseException, desc: str | None) -> None:
        if not desc:
            return
        add_note = getattr(exc, "add_note", None)
        if callable(add_note):
            add_note(f"Reference encode context: {desc}")

    def _maybe_log(self) -> None:
        if self._log_prefix is None:
            return
        now = time.monotonic()
        if now - self._last_log_time < self._LOG_INTERVAL_S:
            return
        with self._lock:
            if now - self._last_log_time < self._LOG_INTERVAL_S:
                return
            self._last_log_time = now
            stats = {
                "hits": self._hits,
                "misses": self._misses,
                "merged": self._merged,
                "entries": len(self._cache),
                "bytes": self._cache.current_bytes,
                "evictions": self._cache.eviction_count,
                "failed": self._failed,
                "uncacheable": self._uncacheable,
            }
        logger.info("%s reference encode stats: %s", self._log_prefix, stats)
