# SPDX-License-Identifier: Apache-2.0
"""Check the shared lock used by CosyVoice3's MPS execution contexts."""

from concurrent.futures import ThreadPoolExecutor

import pytest

from sglang_omni.models.fun_cosyvoice3.mps_lock import MPS_LOCK, MpsSerializedStep


def acquire_from_other_thread() -> bool:
    acquired = MPS_LOCK.acquire(blocking=False)
    if acquired:
        MPS_LOCK.release()
    else:
        pass
    return acquired


@pytest.mark.parametrize("raise_in_step", [False, True])
def test_mps_steps_share_reentrant_lock_and_release_on_exit(
    raise_in_step: bool,
) -> None:
    """Nested AR/vocoder contexts exclude another thread and release after errors."""
    step = MpsSerializedStep()
    with ThreadPoolExecutor(max_workers=1) as executor:
        try:
            with step:
                with MpsSerializedStep():
                    assert not executor.submit(acquire_from_other_thread).result()
                assert not executor.submit(acquire_from_other_thread).result()
                if raise_in_step:
                    raise ValueError("step failed")
                else:
                    pass
        except ValueError as error:
            assert raise_in_step
            assert str(error) == "step failed"
        else:
            assert not raise_in_step
        assert executor.submit(acquire_from_other_thread).result()
        with step:
            assert not executor.submit(acquire_from_other_thread).result()
        assert executor.submit(acquire_from_other_thread).result()
