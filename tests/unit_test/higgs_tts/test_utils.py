# SPDX-License-Identifier: Apache-2.0

import pytest

from sglang_omni.models.higgs_tts.utils import load_audio_to_24k


@pytest.mark.parametrize("reference_audio", [123, ["audio.wav"], object()])
def test_load_audio_to_24k_rejects_invalid_reference_audio_type(
    reference_audio,
) -> None:
    with pytest.raises(TypeError, match="reference_audio must be"):
        load_audio_to_24k(reference_audio)


@pytest.mark.parametrize(
    "reference_audio",
    [
        {},
        {"media_type": "audio/wav"},
        {"audio_path": None},
        {"path": None},
        {"bytes": None},
    ],
)
def test_load_audio_to_24k_rejects_dict_without_audio_payload(
    reference_audio,
) -> None:
    with pytest.raises(ValueError, match="reference_audio dict must provide"):
        load_audio_to_24k(reference_audio)
