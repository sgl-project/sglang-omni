"""Recorder-side video probe and model options for the duplex gates (not part of the client).

With DUPLEX_FRAMES_PER_UNIT=N > 0 and DUPLEX_FRAME_DIR=<dir of .jpg files>, every recorder
websocket sends N sglang.input_image.append events for each 1 s unit, just before the first
input audio packet that starts in that unit. The frames rotate over the sorted JPEG files
(unit u, frame j uses file (u * N + j) mod count) and carry t_ms = unit start + 1 + j, so the
server attaches them to that unit. The dataset audio is unchanged. The image events are not
written to the recorder trace; the server's sglang.input_image.accepted replies are.
The model options live in model_client.py, imported at the end.
"""

import base64
import json
import os
import uuid

FRAMES_PER_UNIT = int(os.environ.get("DUPLEX_FRAMES_PER_UNIT", "0") or 0)
FRAME_DIR = os.environ.get("DUPLEX_FRAME_DIR", "")
UNIT_MS = 1000


def load_frames(frame_dir):
    names = sorted(name for name in os.listdir(frame_dir) if name.endswith(".jpg"))
    frames = []
    for name in names:
        with open(os.path.join(frame_dir, name), "rb") as f:
            frames.append(base64.b64encode(f.read()).decode("ascii"))
    return frames


def install(frames):
    import websockets

    connection_cls = websockets.ClientConnection
    original_send = connection_cls.send

    async def send(self, message, *args, **kwargs):
        if isinstance(message, str) and '"input_audio_buffer.append"' in message:
            event = json.loads(message)
            unit = int(event.get("sglang", {}).get("t_start_ms", 0) // UNIT_MS)
            if unit > getattr(self, "frame_probe_last_unit", -1):
                self.frame_probe_last_unit = unit
                for j in range(FRAMES_PER_UNIT):
                    image_event = {
                        "type": "sglang.input_image.append",
                        "event_id": uuid.uuid4().hex,
                        "image": frames[(unit * FRAMES_PER_UNIT + j) % len(frames)],
                        "sglang": {"t_ms": float(unit * UNIT_MS + 1 + j)},
                    }
                    await original_send(self, json.dumps(image_event))
            else:
                pass
        else:
            pass
        return await original_send(self, message, *args, **kwargs)

    connection_cls.send = send


if FRAMES_PER_UNIT > 0:
    install(load_frames(FRAME_DIR))
else:
    pass
# model options of the recorder (PersonaPlex profile, input sample rate, session.update fields)
import model_client  # noqa: E402,F401
