# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import numpy as np

from sglang_omni.realtime.media import mono_float32


def test_mono_float32_scales_integer_stereo_before_mixdown():
    stereo = np.array(
        [[16384, -16384, 8192, -8192], [16384, -16384, 8192, -8192]],
        dtype=np.int16,
    )

    mono = mono_float32(stereo)

    np.testing.assert_allclose(
        mono,
        np.array([0.5, -0.5, 0.25, -0.25], dtype=np.float32),
        atol=1e-5,
    )
