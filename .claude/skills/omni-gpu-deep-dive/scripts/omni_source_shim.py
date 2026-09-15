# SPDX-License-Identifier: Apache-2.0
"""Teach the vendored llm-torch-profiler-analysis backend about omni source paths.

The backend ranks every python frame in a trace against a fixed allowlist of
repo prefixes (``python/sglang/``, ``vllm/``, ``tensorrt_llm/``, ...) and drops
frames that miss it, so an omni run's kernel table shows torch frames instead of
``sglang_omni/`` code. Three places encode that allowlist:

- ``profile_common._normalize_repo_relative_path_cached`` -- turns an absolute
  path into a repo-relative one. Its bare ``sglang/`` fallback also matches any
  *parent* directory called ``sglang``, so a checkout at
  ``/sgl-workspace/sglang/test3/sglang-omni`` gets rewritten to the fabricated
  ``python/sglang/test3/sglang-omni/...``.
- ``triage_kernel_helpers.source_location_priority`` / ``.is_preferred_source_location``
  -- rank and gate a resolved location.
- ``triage_kernel_helpers.frame_priority`` -- ranks a raw stack frame.
- ``triage_overlap_helpers.choose_best_scope`` / ``.source_scope_priority`` -- a
  second, independent ladder behind the overlap table's "Python scope" column.
  It scores ``sglang_omni/`` the same as any ``.py(`` frame and, unlike the
  kernel-side ladder, has no ``torch/`` demotion at all, so the deepest frame in
  the stack wins and the column reads ``torch/nn/modules/linear.py``.

This module patches those entry points and leaves every other backend behaviour
untouched: omni prefixes are *added* to the ladders by wrapping the originals,
never by reimplementing their tiers, so a backend update does not silently drop
the non-omni rules.

Import order matters. ``triage_kernel_helpers`` and ``triage_overlap_helpers``
``from``-import ``normalize_repo_relative_path``, so the normalizer must be
patched before those modules are imported; ``apply()`` handles that.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Ranked above python/sglang/ (300): in an omni run the engine internals are the
# floor and the model/pipeline code is what a reader needs to act on, so when a
# stack contains both, the omni frame should win.
OMNI_TIER = 305

# Longest first: sglang_omni_router/ must not be matched by sglang_omni/.
OMNI_MARKERS = ("sglang_omni_router/", "sglang_omni/")

# Repo-relative roots of omni code that is not inside the package (harnesses a
# profile legitimately points at).
OMNI_EXTRA_PREFIXES = ("benchmarks/", "tests/")

OMNI_PREFIXES = OMNI_MARKERS + OMNI_EXTRA_PREFIXES

# Frames inside the torch runtime answer "which op ran", never "which code asked
# for it". The kernel-side ladder already demotes them to 20, below third-party
# model code at 120; the overlap ladder has no such rule, so the shim mirrors it.
_TORCH_RUNTIME_MARKERS = ("torch/",)

# Upstream's marker table (profile_common.py). Kept verbatim so the patched
# normalizer stays a superset, with the bare ``sglang/`` fallback tightened.
_UPSTREAM_MARKERS = (
    ("python/sglang/", "python/sglang/"),
    ("sgl_kernel/", "sgl_kernel/"),
    ("vllm/", "vllm/"),
    ("python/tokenspeed/", "python/tokenspeed/"),
    ("tokenspeed/", "tokenspeed/"),
    ("tensorrt_llm/", "tensorrt_llm/"),
    ("tensorrt-llm/", "tensorrt_llm/"),
)

# The bare ``sglang/`` fallback only means "the sglang package" when a real
# top-level subpackage follows it. ``sglang/test3/`` is a sibling checkout.
_SGLANG_SUBPACKAGES = (
    "srt/",
    "kernels/",
    "lang/",
    "test/",
    "eval/",
    "profiler/",
    "compilation/",
    "utils/",
    "__init__.py",
)

DEFAULT_BACKEND = (
    "/sgl-workspace/sglang/.claude/skills/llm-torch-profiler-analysis/scripts"
)

_applied = False


def backend_scripts_dir() -> Path:
    """Locate the vendored backend's scripts/ directory.

    Order: ``OMNI_PROFILER_BACKEND`` env override, then a walk up from this file
    looking for a sibling ``.claude/skills/llm-torch-profiler-analysis``, then
    the known vendored location.
    """
    override = os.environ.get("OMNI_PROFILER_BACKEND")
    if override:
        path = Path(override).expanduser()
        if path.name != "scripts":
            path = path / "scripts"
        if not path.is_dir():
            raise FileNotFoundError(
                f"OMNI_PROFILER_BACKEND={override} has no scripts/ directory"
            )
        return path
    for parent in Path(__file__).resolve().parents:
        candidate = (
            parent / ".claude" / "skills" / "llm-torch-profiler-analysis" / "scripts"
        )
        if candidate.is_dir():
            return candidate
    fallback = Path(DEFAULT_BACKEND)
    if fallback.is_dir():
        return fallback
    raise FileNotFoundError(
        "cannot locate the llm-torch-profiler-analysis backend; set "
        "OMNI_PROFILER_BACKEND to its skill directory"
    )


def is_omni_location(text: str) -> bool:
    return str(text).lstrip("/").startswith(OMNI_PREFIXES)


def is_torch_runtime_location(text: str) -> bool:
    stripped = str(text).lstrip("/")
    return stripped.startswith(_TORCH_RUNTIME_MARKERS) or "/torch/" in stripped


def _normalize_repo_relative_path(path: object) -> str:
    """Replacement for ``profile_common.normalize_repo_relative_path``."""
    text = str("" if path is None else path).strip().replace("\\", "/")
    lowered = text.lower()
    for marker in OMNI_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            return text[index:].lstrip("/")
    for marker, normalized_marker in _UPSTREAM_MARKERS:
        index = lowered.find(marker)
        if index != -1:
            suffix = text[index + len(marker) :].lstrip("/")
            return f"{normalized_marker}{suffix}".lstrip("/")
    index = lowered.find("sglang/")
    if index != -1:
        suffix = text[index + len("sglang/") :].lstrip("/")
        if suffix.lower().startswith(_SGLANG_SUBPACKAGES):
            return f"python/sglang/{suffix}".lstrip("/")
    return text.lstrip("/")


def apply() -> tuple[object, object]:
    """Patch the backend in-place and return ``(kernel_helpers, analyzer)``."""
    global _applied
    scripts = backend_scripts_dir()
    if str(scripts) not in sys.path:
        sys.path.insert(0, str(scripts))

    import profile_common

    if not _applied:
        profile_common.normalize_repo_relative_path = _normalize_repo_relative_path

    # Imported only after the normalizer is patched: both helper modules bind it
    # by value at import time.
    import triage_kernel_helpers as kernel_helpers

    if not _applied:
        _patch_priority_ladders(kernel_helpers)

    import triage_overlap_helpers as overlap_helpers

    if not _applied:
        _patch_overlap_ladders(overlap_helpers)

    import analyze_llm_torch_profile as analyzer

    _applied = True
    return kernel_helpers, analyzer


def _patch_priority_ladders(kernel_helpers) -> None:
    original_priority = kernel_helpers.source_location_priority
    original_preferred = kernel_helpers.is_preferred_source_location
    original_frame = kernel_helpers.frame_priority
    is_low_signal = kernel_helpers.is_low_signal_source_location
    normalize_location = kernel_helpers.normalize_source_location

    def source_location_priority(location: str) -> int:
        text = str(location).strip()
        if is_omni_location(text):
            return OMNI_TIER - (80 if is_low_signal(text) else 0)
        return original_priority(location)

    def is_preferred_source_location(location: str) -> bool:
        if is_omni_location(str(location).strip()):
            return True
        return original_preferred(location)

    def frame_priority(frame_name: str) -> int:
        normalized = normalize_location(str(frame_name).strip())
        if is_omni_location(normalized):
            return OMNI_TIER - (80 if is_low_signal(normalized) else 0)
        return original_frame(frame_name)

    kernel_helpers.source_location_priority = source_location_priority
    kernel_helpers.is_preferred_source_location = is_preferred_source_location
    kernel_helpers.frame_priority = frame_priority


def _preferred_scope_subset(scope_chain, *, canonicalize) -> tuple[str, ...]:
    """Narrow a stack to the frames worth ranking, or () to rank all of them."""
    omni = tuple(
        scope for scope in scope_chain if is_omni_location(canonicalize(scope))
    )
    if omni:
        return omni
    # No omni frame: a third-party model file (transformers/, torchaudio/) is
    # still a real answer, so keep those and drop only the torch runtime.
    model_code = tuple(
        scope
        for scope in scope_chain
        if not is_torch_runtime_location(canonicalize(scope))
    )
    if model_code and len(model_code) != len(scope_chain):
        return model_code
    return ()


def _patch_overlap_ladders(overlap_helpers) -> None:
    original_choose = overlap_helpers.choose_best_scope
    original_priority = overlap_helpers.source_scope_priority
    canonicalize = overlap_helpers.canonicalize_python_scope_name
    is_low_signal = overlap_helpers.is_low_signal_scope

    def choose_best_scope(scope_chain):
        # Delegate on the narrowed chain so upstream's own scoring and penalties
        # still decide which omni frame wins.
        subset = _preferred_scope_subset(scope_chain, canonicalize=canonicalize)
        return original_choose(subset or scope_chain)

    def source_scope_priority(scope) -> int:
        normalized = canonicalize(scope or "")
        if is_omni_location(normalized):
            return OMNI_TIER - (80 if is_low_signal(normalized) else 0)
        if is_torch_runtime_location(normalized):
            return 20
        return original_priority(scope)

    overlap_helpers.choose_best_scope = choose_best_scope
    overlap_helpers.source_scope_priority = source_scope_priority
