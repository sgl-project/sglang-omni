"""Recorder-side model options for the duplex gates (not part of the client).

Imported by sitecustomize.py in this directory, so it runs in every recorder before the client
modules are used. It leaves the recorder tree (BENCH_TREE) unchanged:

- adds the profile personaplex-native (80 ms frames, 24 kHz output) to benchmarks.duplex.profiles;
- DUPLEX_INPUT_SAMPLE_RATE=R (default 16000): the dataset audio is resampled to R and sent in 80 ms
  packets of R samples per second (PersonaPlex takes 24 kHz input; the dataset is 16 kHz);
- DUPLEX_SESSION_UPDATE=<JSON object>: merged into the session of every session.update (for example
  a PersonaPlex voice: {"sglang": {"voice": "NATM1"}}).
"""

import json
import os

INPUT_SAMPLE_RATE = int(os.environ.get("DUPLEX_INPUT_SAMPLE_RATE") or 16000)
SESSION_UPDATE = json.loads(os.environ.get("DUPLEX_SESSION_UPDATE") or "{}")


def merge(base, extra):
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def add_profiles():
    from benchmarks.duplex import profiles

    profiles.PROFILES.setdefault(
        "personaplex-native",
        profiles.DuplexProfile(
            native_unit_ms=80,
            output_sample_rate=24000,
            output_modalities=("audio",),
            stop_requires_eos=True,
            continuous_output=True,
        ),
    )


def set_input_sample_rate(rate):
    from benchmarks.duplex import client, v15_audio, v15_runner

    # the modules read these globals at call time
    client.SAMPLE_RATE = rate
    client.PACKET_BYTES = rate * client.PACKET_MS // 1000 * 2
    v15_audio.SAMPLE_RATE = rate
    v15_runner.SAMPLE_RATE = rate


def merge_session_update(extra):
    import websockets

    connection_cls = websockets.ClientConnection
    original_send = connection_cls.send

    async def send(self, message, *args, **kwargs):
        if isinstance(message, str) and '"session.update"' in message:
            event = json.loads(message)
            merge(event.setdefault("session", {}), extra)
            message = json.dumps(event)
        else:
            pass
        return await original_send(self, message, *args, **kwargs)

    connection_cls.send = send


add_profiles()
if INPUT_SAMPLE_RATE != 16000:
    set_input_sample_rate(INPUT_SAMPLE_RATE)
else:
    pass
if SESSION_UPDATE:
    merge_session_update(SESSION_UPDATE)
else:
    pass
