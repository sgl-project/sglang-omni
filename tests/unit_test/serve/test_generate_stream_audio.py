# SPDX-License-Identifier: Apache-2.0
import base64
import io
import json
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

from sglang_omni.client import Client
from sglang_omni.proto import CompleteMessage, StreamMessage
from sglang_omni.serve import create_app


@pytest.mark.parametrize("terminal_audio", [False, True])
def test_generate_stream_serializes_audio_and_reaps_iterator(terminal_audio):
    samples = np.array([0.0, 0.25, -0.25, 0.5], dtype=np.float32)

    class Coordinator:
        closed = False

        def health(self):
            return {"running": True}

        async def stream(self, request_id, request):
            try:
                data = {"audio": samples, "sample_rate": 16000, "modality": "audio"}
                if not terminal_audio:
                    yield StreamMessage(request_id, "audio", data, modality="audio")
                    data = {"text": "", "finish_reason": "stop"}
                yield CompleteMessage(request_id, "audio", True, result=data)
            finally:
                self.closed = True

    coordinator = Coordinator()
    with TestClient(
        create_app(Client(coordinator), model_name="audio-model")
    ) as client:
        response = client.post(
            "/generate",
            json={
                "prompt": "hello",
                "stream": True,
                "return_logprob": False,
            },
        )
    assert response.status_code == 200
    assert "event: error" not in response.text
    assert response.text.endswith("data: [DONE]\n\n")
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ") and line != "data: [DONE]"
    ]
    audio = next(event["audio"] for event in events if event.get("audio"))
    with wave.open(io.BytesIO(base64.b64decode(audio["data"])), "rb") as stream:
        assert stream.getframerate() == 16000
        assert stream.getnframes() == len(samples)
    assert coordinator.closed
