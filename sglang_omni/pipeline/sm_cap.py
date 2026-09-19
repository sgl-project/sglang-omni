"""Per-stage SM caps backed by CUDA Green Contexts.

A stage that issues short bursts of wide kernels can hold most of the device's
SMs for milliseconds at a time, stalling a latency-sensitive stage sharing the
GPU through MPS. Setting ``sm_cap`` on the bursty stage bounds its SM footprint.

Whether a cap helps, and which stage to cap, is a property of the pipeline and
has to be measured: see ``docs/basic_usage/stage_sm_cap.md``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sglang_omni.config.schema import StageConfig

# Note (Jiaxin Deng): the driver splits a device into groups of at least this
# many SMs; H100 and H200 both quantize to 8. Other granularities are not
# supported, because the group count is derived here rather than negotiated
# with the driver.
SM_GROUP_SIZE = 8

BOOTSTRAP_ENV = "SGLANG_OMNI_SM_CAP_BOOTSTRAP"

# Derived from `sm_cap`, so a value arriving from anywhere else would silently
# describe a different partition than the one that was asked for.
RESERVED_ENV = ("GREEN_CTX_SM", "GREEN_CTX_SPLIT", "GREEN_CTX_GROUP_COUNT")


class SmCapError(ValueError):
    """Raised when a requested SM cap cannot be applied."""


def resolve_bootstrap_path(explicit: str | None = None) -> str:
    """Return the bootstrap library path, or raise if it is not configured."""
    path = explicit or os.environ.get(BOOTSTRAP_ENV, "")
    if not path:
        raise SmCapError(
            "sm_cap needs the green-context bootstrap library: build it with "
            f"`make -C tools/green_ctx` and point ${BOOTSTRAP_ENV} at the .so"
        )
    if not os.path.isfile(path):
        raise SmCapError(f"sm_cap bootstrap library not found: {path}")
    return path


def merged_ld_preload(bootstrap: str, inherited: str | None) -> str:
    """Return an ``LD_PRELOAD`` that keeps *inherited* entries and adds *bootstrap*."""
    entries = [entry for entry in (inherited or "").replace(",", " ").split() if entry]
    if bootstrap in entries:
        return " ".join(entries)
    return " ".join([bootstrap, *entries])


def sm_cap_env(
    sm_cap: int, bootstrap: str, inherited_preload: str | None = None
) -> dict[str, str]:
    """Return the spawn env that caps a stage process to *sm_cap* SMs.

    The cap bounds the provisioned SM set; it is not a reservation. Two capped
    stages may cover overlapping SMs, and an uncapped stage still sees the
    whole device.
    """
    if sm_cap <= 0 or sm_cap % SM_GROUP_SIZE:
        raise SmCapError(
            f"sm_cap={sm_cap} must be a positive multiple of {SM_GROUP_SIZE}"
        )
    return {
        "GREEN_CTX_SM": str(sm_cap),
        "GREEN_CTX_SPLIT": str(SM_GROUP_SIZE),
        "GREEN_CTX_GROUP_COUNT": str(sm_cap // SM_GROUP_SIZE),
        "LD_PRELOAD": merged_ld_preload(bootstrap, inherited_preload),
        # Note (Jiaxin Deng): kept separately because verification must name the
        # library it expects, not whatever LD_PRELOAD ended up as.
        BOOTSTRAP_ENV: bootstrap,
    }


def validate_capped_process(stages: list[StageConfig]) -> None:
    """Reject placements where a cap would apply to more than it names.

    A green context is process-wide, so every stage sharing an OS process with
    a capped stage runs capped too. Requiring them to agree keeps the config
    honest about what is capped.
    """
    caps = {stage.sm_cap for stage in stages}
    if len(caps) > 1:
        listed = ", ".join(f"{stage.name}={stage.sm_cap}" for stage in stages)
        raise SmCapError(
            f"stages sharing one process disagree about sm_cap ({listed}); a "
            "green context is process-wide, so every stage in the process must "
            "declare the same cap, or the capped stage must own its process"
        )


def validate_stage_sm_cap(stage_cfg: StageConfig) -> str | None:
    """Check a stage's cap at config time and return the bootstrap path.

    Returns ``None`` when the stage declares no cap. The env itself is built at
    spawn time by :func:`sm_cap_env`, because it has to override the parent
    environment rather than default under it.
    """
    if stage_cfg.sm_cap is None:
        return None
    declared = [name for name in RESERVED_ENV if name in stage_cfg.env]
    if declared:
        raise SmCapError(
            f"stage {stage_cfg.name!r} sets both sm_cap and {declared}; these "
            "are derived from sm_cap and may not be set directly"
        )
    inherited = [name for name in RESERVED_ENV if name in os.environ]
    if inherited:
        raise SmCapError(
            f"stage {stage_cfg.name!r} sets sm_cap but {inherited} is already "
            "set in the parent environment, where it would shadow the derived "
            "value; unset it"
        )
    # Note (Jiaxin Deng): MPS can scale a client's SM set beyond the green
    # context's provisioned groups, which would make the cap advisory.
    if "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE" in os.environ:
        raise SmCapError(
            f"stage {stage_cfg.name!r} sets sm_cap but "
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE is set; MPS may then run "
            "kernels on more SMs than the cap provisions"
        )
    bootstrap = resolve_bootstrap_path()
    # Surface a bad cap size now rather than at stage startup.
    sm_cap_env(stage_cfg.sm_cap, bootstrap)
    return bootstrap


def verify_sm_cap(bootstrap: str, expected_sm: int) -> int:
    """Check that this process really runs capped, and return its SM count.

    Raises ``SmCapError`` when the cap did not take effect. Both the calling
    thread and a freshly created one are checked, against the capped context's
    identity rather than its SM count: an extension that rebound the main
    thread to the primary context would otherwise stay invisible, and a library
    merely ``dlopen``-ed instead of preloaded does not interpose
    ``pthread_create`` and so cannot bind a new thread.
    """
    import ctypes
    import threading

    library = ctypes.CDLL(bootstrap)
    library.green_ctx_actual_sm.restype = ctypes.c_uint
    library.green_ctx_current_sm.restype = ctypes.c_uint
    library.green_ctx_current_is_capped.restype = ctypes.c_int

    actual = int(library.green_ctx_actual_sm())
    if actual != expected_sm:
        raise SmCapError(
            f"sm_cap={expected_sm} requested but the green context has {actual} SMs"
        )
    if not int(library.green_ctx_current_is_capped()):
        raise SmCapError(
            f"sm_cap={expected_sm} is not current on the stage's main thread; "
            "something rebound it to another context after startup"
        )

    observed: list[tuple[int, int]] = []

    def probe() -> None:
        observed.append(
            (
                int(library.green_ctx_current_is_capped()),
                int(library.green_ctx_current_sm()),
            )
        )

    thread = threading.Thread(target=probe, name="sm-cap-verify")
    thread.start()
    thread.join()
    if observed != [(1, actual)]:
        raise SmCapError(
            f"sm_cap={expected_sm} did not reach a new thread (observed "
            f"{observed}); check that LD_PRELOAD names {bootstrap}"
        )
    return actual
