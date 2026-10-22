# SPDX-License-Identifier: Apache-2.0
"""Voxtral MLX acoustic-head token contracts."""

import pytest

pytest.importorskip("mlx.core")

from sglang_omni_mlx.voxtral_tts.acoustic_head import (  # noqa: E402 - MLX is optional in test environments
    EMPTY_AUDIO_TOKEN_ID,
    END_AUDIO_TOKEN_ID,
)


def test_acoustic_special_token_ids_match_checkpoint_order() -> None:
    assert EMPTY_AUDIO_TOKEN_ID == 0
    assert END_AUDIO_TOKEN_ID == 1
