# SPDX-License-Identifier: Apache-2.0
"""Resolve weight-IPC configuration from env / explicit overrides."""

from __future__ import annotations

import os

from sglang_omni.distributed.weight_ipc.types import WeightIpcConfig, WeightIpcRole

_ROLE_ENV_KEYS = (
    "SGLANG_OMNI_WEIGHT_IPC_ROLE",
    "WEIGHT_IPC_ROLE",
)
_STORE_ENV_KEYS = (
    "SGLANG_OMNI_WEIGHT_IPC_STORE",
    "WEIGHT_IPC_STORE",
)
_TIMEOUT_ENV_KEYS = (
    "SGLANG_OMNI_WEIGHT_IPC_TIMEOUT_S",
    "WEIGHT_IPC_TIMEOUT_S",
)


def _first_env(*keys: str) -> str | None:
    for key in keys:
        value = os.environ.get(key)
        if value is not None and value.strip() != "":
            return value.strip()
    return None


def resolve_weight_ipc_config(
    explicit: WeightIpcConfig | None = None,
) -> WeightIpcConfig:
    """Return the active weight-IPC config (explicit > env > ``role=off``)."""
    if explicit is not None:
        _validate(explicit)
        return explicit

    role_raw = (_first_env(*_ROLE_ENV_KEYS) or "off").lower()
    if role_raw not in ("off", "leader", "follower"):
        raise ValueError(
            f"invalid weight IPC role {role_raw!r}; expected off|leader|follower"
        )
    role: WeightIpcRole = role_raw  # type: ignore[assignment]
    store = _first_env(*_STORE_ENV_KEYS)
    timeout_raw = _first_env(*_TIMEOUT_ENV_KEYS)
    timeout_s = float(timeout_raw) if timeout_raw is not None else 120.0
    config = WeightIpcConfig(
        role=role,
        store_dir=store,
        timeout_s=timeout_s,
    )
    _validate(config)
    return config


def _validate(config: WeightIpcConfig) -> None:
    if config.role == "off":
        return
    if not config.store_dir:
        raise ValueError(
            "weight IPC role is "
            f"{config.role!r} but store_dir / WEIGHT_IPC_STORE is unset"
        )
    if config.timeout_s <= 0:
        raise ValueError(
            f"weight IPC timeout_s must be positive, got {config.timeout_s}"
        )


def is_weight_ipc_follower() -> bool:
    return resolve_weight_ipc_config().role == "follower"


def apply_weight_ipc_cli_env(
    *,
    role: str | None,
    store: str | None,
    timeout_s: float | None = None,
) -> None:
    """Set process env from CLI flags (no-op when both role and store are None)."""
    if role is None and store is None and timeout_s is None:
        return
    if role is not None:
        os.environ["SGLANG_OMNI_WEIGHT_IPC_ROLE"] = role
        os.environ["WEIGHT_IPC_ROLE"] = role
    if store is not None:
        os.environ["SGLANG_OMNI_WEIGHT_IPC_STORE"] = store
        os.environ["WEIGHT_IPC_STORE"] = store
    if timeout_s is not None:
        os.environ["SGLANG_OMNI_WEIGHT_IPC_TIMEOUT_S"] = str(timeout_s)
        os.environ["WEIGHT_IPC_TIMEOUT_S"] = str(timeout_s)
    # Note (guozhihao): validate early so serve fails before spawning stages.
    resolve_weight_ipc_config()
