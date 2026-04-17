"""Optional precision debug dumps for Qwen3-Omni parity work."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import torch

_REQUEST_ID_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _dump_dir() -> Path | None:
    raw = os.environ.get("QWEN_PRECISION_DUMP_DIR")
    if not raw:
        return None
    path = Path(raw)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _safe_request_id(request_id: str) -> str:
    return _REQUEST_ID_SAFE.sub("_", request_id)


def dump_precision_pt(
    *,
    prefix: str,
    request_id: str,
    payload: Any,
    step: int | None = None,
    suffix: str = ".pt",
) -> str | None:
    dump_dir = _dump_dir()
    if dump_dir is None:
        return None

    safe_request_id = _safe_request_id(request_id)
    filename = f"{prefix}_{safe_request_id}"
    if step is not None:
        filename += f"_step{int(step):03d}"
    filename += suffix

    path = dump_dir / filename
    torch.save(payload, path)
    return str(path)


def precision_dump_path(
    *,
    prefix: str,
    request_id: str,
    step: int | None = None,
    suffix: str = ".pt",
) -> Path | None:
    dump_dir = _dump_dir()
    if dump_dir is None:
        return None

    safe_request_id = _safe_request_id(request_id)
    filename = f"{prefix}_{safe_request_id}"
    if step is not None:
        filename += f"_step{int(step):03d}"
    filename += suffix
    return dump_dir / filename
