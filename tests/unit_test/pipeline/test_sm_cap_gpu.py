"""GPU checks for the SM cap bootstrap itself.

The pure-Python tests cover the config surface. These cover the part that can
only be observed on a device: that a preloaded bootstrap really confines the
process, and that verification rejects the ways it can fail to.

Requires a CUDA device and the built bootstrap
(``make -C tools/green_ctx``, then ``SGLANG_OMNI_SM_CAP_BOOTSTRAP``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from sglang_omni.pipeline.sm_cap import BOOTSTRAP_ENV, SM_GROUP_SIZE, sm_cap_env

CAP_SM = 3 * SM_GROUP_SIZE

bootstrap_path = os.environ.get(BOOTSTRAP_ENV, "")
requires_bootstrap = pytest.mark.skipif(
    not bootstrap_path or not os.path.isfile(bootstrap_path),
    reason=f"requires a built bootstrap at ${BOOTSTRAP_ENV}",
)


def _run(script: str, env: dict[str, str]) -> subprocess.CompletedProcess:
    child_env = {**os.environ, **env}
    return subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        env=child_env,
        capture_output=True,
        text=True,
        timeout=300,
    )


@requires_bootstrap
def test_preloaded_bootstrap_caps_the_process():
    """A preloaded bootstrap confines the process, and verification accepts it."""
    result = _run(
        f"""
        from sglang_omni.pipeline.sm_cap import verify_sm_cap
        print("SM", verify_sm_cap({bootstrap_path!r}, {CAP_SM}))
        """,
        sm_cap_env(CAP_SM, bootstrap_path),
    )
    assert result.returncode == 0, result.stderr
    assert f"SM {CAP_SM}" in result.stdout
    assert f"actual_sm={CAP_SM}" in result.stderr


@requires_bootstrap
def test_capped_process_sees_fewer_sms_than_the_device():
    """The cap is observable through the CUDA runtime, not just self-reported."""
    env = sm_cap_env(CAP_SM, bootstrap_path)
    result = _run(
        """
        import ctypes, os
        lib = ctypes.CDLL(os.environ["LD_PRELOAD"].split()[0])
        lib.green_ctx_current_sm.restype = ctypes.c_uint
        print("CURRENT", lib.green_ctx_current_sm())
        """,
        env,
    )
    assert result.returncode == 0, result.stderr
    capped = int(result.stdout.split("CURRENT")[1])

    uncapped = _run(
        """
        import ctypes, os
        lib = ctypes.CDLL(os.environ["BOOTSTRAP"])
        lib.green_ctx_current_sm.restype = ctypes.c_uint
        import ctypes.util
        cuda = ctypes.CDLL("libcuda.so.1")
        cuda.cuInit(0)
        dev = ctypes.c_int()
        cuda.cuDeviceGet(ctypes.byref(dev), 0)
        count = ctypes.c_int()
        cuda.cuDeviceGetAttribute(ctypes.byref(count), 16, dev)
        print("TOTAL", count.value)
        """,
        {"BOOTSTRAP": bootstrap_path},
    )
    assert uncapped.returncode == 0, uncapped.stderr
    total = int(uncapped.stdout.split("TOTAL")[1])
    assert capped == CAP_SM < total


@requires_bootstrap
def test_verification_rejects_a_late_load_instead_of_a_preload():
    """Loading the library without LD_PRELOAD must not pass as a capped process.

    Without the preload the ``pthread_create`` interposer is never installed,
    so the probe thread cannot be bound and the check has to fail.
    """
    env = sm_cap_env(CAP_SM, bootstrap_path)
    env.pop("LD_PRELOAD")
    result = _run(
        f"""
        from sglang_omni.pipeline.sm_cap import SmCapError, verify_sm_cap
        try:
            verify_sm_cap({bootstrap_path!r}, {CAP_SM})
        except SmCapError as error:
            print("REJECTED", error)
        else:
            print("ACCEPTED")
        """,
        env,
    )
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout, result.stdout


@requires_bootstrap
def test_verification_rejects_a_mismatched_cap():
    """Verification compares against what was asked for, not what it found."""
    result = _run(
        f"""
        from sglang_omni.pipeline.sm_cap import SmCapError, verify_sm_cap
        try:
            verify_sm_cap({bootstrap_path!r}, {CAP_SM + SM_GROUP_SIZE})
        except SmCapError as error:
            print("REJECTED", error)
        else:
            print("ACCEPTED")
        """,
        sm_cap_env(CAP_SM, bootstrap_path),
    )
    assert result.returncode == 0, result.stderr
    assert "REJECTED" in result.stdout, result.stdout
