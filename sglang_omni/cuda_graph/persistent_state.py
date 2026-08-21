# SPDX-License-Identifier: Apache-2.0
"""Persistent cross-step state registry for CUDA-graph capture.

A CUDA graph binds to the *addresses* of the buffers it touches. State that
crosses decode steps (e.g. a streaming codec attention cache) must therefore live
at a fixed address and be written in place; if the eager path reassigns it each
step, a captured graph replays against stale memory and silently corrupts output
(MOSS PR #798). Stateless stages (Higgs PR #729) carry no such state.

This registry makes the contract explicit and testable: a stage declares each
cross-step buffer via an *accessor* (a callable returning the live buffer), the
framework resets the state before capture, and ``assert_addresses_stable`` re-runs
the accessors to verify nothing was reassigned. Stateless stages declare nothing
and every method is a no-op.
"""
from __future__ import annotations

from collections.abc import Callable

import torch

BufferAccessor = Callable[[], torch.Tensor]


class PersistentStateError(RuntimeError):
    """Raised on a misuse or a violated address-stability contract."""


class PersistentStateRegistry:
    def __init__(self) -> None:
        self._accessors: dict[str, BufferAccessor] = {}
        self._spec: dict[str, tuple[tuple[int, ...], torch.dtype]] = {}
        self._reset_hooks: list[Callable[[], None]] = []
        self._addr: dict[str, int] = {}

    def declare(
        self,
        name: str,
        accessor: BufferAccessor,
        *,
        expected_shape: tuple[int, ...] | None = None,
        expected_dtype: torch.dtype | None = None,
    ) -> None:
        """Declare one cross-step buffer by an accessor returning the live tensor.

        Raises on a duplicate name (declare-twice) or a shape/dtype mismatch
        against ``expected_shape``/``expected_dtype``.
        """
        if name in self._accessors:
            raise PersistentStateError(f"persistent state {name!r} already declared")
        if not callable(accessor):
            raise PersistentStateError(
                f"persistent state {name!r} accessor must be callable"
            )
        tensor = accessor()
        if not isinstance(tensor, torch.Tensor):
            raise PersistentStateError(
                f"persistent state {name!r} accessor must return a tensor, "
                f"got {type(tensor).__name__}"
            )
        if expected_shape is not None and tuple(tensor.shape) != tuple(expected_shape):
            raise PersistentStateError(
                f"persistent state {name!r} shape mismatch: declared "
                f"{tuple(expected_shape)} but buffer is {tuple(tensor.shape)}"
            )
        if expected_dtype is not None and tensor.dtype != expected_dtype:
            raise PersistentStateError(
                f"persistent state {name!r} dtype mismatch: declared "
                f"{expected_dtype} but buffer is {tensor.dtype}"
            )
        self._accessors[name] = accessor
        self._spec[name] = (tuple(tensor.shape), tensor.dtype)

    def add_reset_hook(self, fn: Callable[[], None]) -> None:
        """Register a callable that resets cross-step state to a clean start.
        Run by :meth:`reset` after warmup and before capture."""
        self._reset_hooks.append(fn)

    def is_empty(self) -> bool:
        """True for a stateless stage: nothing declared, no reset hooks."""
        return not self._accessors and not self._reset_hooks

    def reset(self) -> None:
        """Run every reset hook (no-op when empty)."""
        for fn in self._reset_hooks:
            fn()

    def snapshot_addresses(self) -> None:
        """Record each declared buffer's ``data_ptr`` (call right after capture)."""
        self._addr = {name: acc().data_ptr() for name, acc in self._accessors.items()}

    def assert_addresses_stable(self) -> None:
        """Re-run accessors and verify no declared buffer moved (was reassigned)
        or changed shape/dtype since :meth:`snapshot_addresses`. This is #798's
        guarantee as a first-class, testable contract."""
        for name, acc in self._accessors.items():
            if name not in self._addr:
                raise PersistentStateError(
                    f"no address snapshot for {name!r}; call snapshot_addresses() "
                    "after capture before asserting"
                )
            tensor = acc()
            spec_shape, spec_dtype = self._spec[name]
            if tuple(tensor.shape) != spec_shape or tensor.dtype != spec_dtype:
                raise PersistentStateError(
                    f"persistent state {name!r} shape/dtype changed since declare"
                )
            if tensor.data_ptr() != self._addr[name]:
                raise PersistentStateError(
                    f"persistent state {name!r} moved "
                    f"(0x{self._addr[name]:x} -> 0x{tensor.data_ptr():x}); it was "
                    "reassigned instead of written in place, so the captured "
                    "graph reads/writes stale memory"
                )

    def declared_names(self) -> list[str]:
        return list(self._accessors)
