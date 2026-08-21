# SPDX-License-Identifier: Apache-2.0

import importlib.util
from array import array
from pathlib import Path

import pytest

CLIENT_PATH = Path(__file__).parents[3] / "examples" / "nemotron_voicechat_client.py"
CLIENT_SPEC = importlib.util.spec_from_file_location(
    "nemotron_voicechat_client", CLIENT_PATH
)
assert CLIENT_SPEC is not None and CLIENT_SPEC.loader is not None
client = importlib.util.module_from_spec(CLIENT_SPEC)
CLIENT_SPEC.loader.exec_module(client)


def test_resample_pcm16_uses_exact_80ms_device_frame_sizes():
    microphone = array("h", range(1_920)).tobytes()
    model_input = client.resample_pcm16(microphone, 24_000, client.INPUT_RATE)
    assert len(model_input) // 2 == client.FRAME_SAMPLES

    model_output = array("h", range(1_764)).tobytes()
    playback = client.resample_pcm16(model_output, client.OUTPUT_RATE, 48_000)
    assert len(playback) // 2 == 3_840


def test_validate_session_created_accepts_voicechat_audio_contract():
    client.validate_session_created(
        {
            "type": "session.created",
            "session": {
                "input_audio_format": "pcm16",
                "input_sample_rate": client.INPUT_RATE,
                "output_audio_format": "pcm16",
                "output_sample_rate": client.OUTPUT_RATE,
                "frame_samples": client.FRAME_SAMPLES,
            },
        }
    )


def test_validate_session_created_rejects_incompatible_audio_contract():
    with pytest.raises(RuntimeError, match="output_sample_rate=24000"):
        client.validate_session_created(
            {
                "type": "session.created",
                "session": {
                    "input_audio_format": "pcm16",
                    "input_sample_rate": client.INPUT_RATE,
                    "output_audio_format": "pcm16",
                    "output_sample_rate": 24_000,
                    "frame_samples": client.FRAME_SAMPLES,
                },
            }
        )


def test_client_defaults_to_microphone_with_playback():
    args = client.build_parser().parse_args([])

    assert args.url == "ws://127.0.0.1:18080/v1/realtime"
    assert not args.no_playback
    assert args.trailing_silence == 2.0
