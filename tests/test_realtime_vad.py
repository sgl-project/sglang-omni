# SPDX-License-Identifier: Apache-2.0

import numpy as np

from sglang_omni.realtime.vad import EnergyVad, VadConfig


def test_energy_vad_detects_start_and_stop():
    vad = EnergyVad(
        VadConfig(
            start_threshold=0.02,
            stop_threshold=0.01,
            min_speech_s=0.2,
            min_silence_s=0.4,
        )
    )

    chunk_silence = np.zeros(1600, dtype=np.float32)  # 100 ms @ 16 kHz
    chunk_speech = np.full(1600, 0.2, dtype=np.float32)

    events = [vad.process(chunk_silence) for _ in range(4)]
    assert not any(event.speech_started or event.speech_stopped for event in events)

    events = [vad.process(chunk_speech) for _ in range(3)]
    assert any(event.speech_started for event in events)
    assert not any(event.speech_stopped for event in events)

    events = [vad.process(chunk_silence) for _ in range(5)]
    assert any(event.speech_stopped for event in events)
