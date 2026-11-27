# MOSS-Speech

MOSS-Speech answers English and Chinese text or spoken questions through `/v1/chat/completions`. Select one output modality per request. Speech replies require a voice reference and return a complete 24 kHz waveform.

## Launch

Follow the [MOSS-Speech codec installation steps](../get_started/installation.md#moss-speech-codec-dependencies), then run these commands from the SGLang-Omni checkout in the same environment.

```bash
hf download OpenMOSS-Team/MOSS-Speech --revision cff025bb41d8459d59abac0b5e44aba7f659ec9e --local-dir ./weights/MOSS-Speech
hf download OpenMOSS-Team/MOSS-Speech-Codec --revision eeec733e4e1dea7da444d332d8e1621ef257414c --local-dir ./weights/MOSS-Speech-Codec
sgl-omni serve --config examples/configs/moss_speech.yaml --model-name moss-speech --port 8008
```

The example uses one CUDA GPU with BF16 autoregressive weights, FlashInfer attention, and decode CUDA Graphs. It sets `ar_engine.engine.max_running_requests` to 16 (C16). The model config defaults to four concurrent AR requests. The YAML supplies the codec path to both the speech input and the audio output stages.

## Send text and spoken questions

Save a spoken question as `question.wav` and a voice reference as `voice.wav` in your client directory. This example requests text and speech replies for each input. It prints text replies and saves speech replies as WAV files.

```python
import base64
from pathlib import Path

import requests


def encode_wav(path: str) -> str:
    return base64.b64encode(Path(path).read_bytes()).decode("ascii")


spoken_question = [
    {
        "type": "input_audio",
        "input_audio": {"format": "wav", "data": encode_wav("question.wav")},
    }
]
voice_reference = "data:audio/wav;base64," + encode_wav("voice.wav")
for source, content in [("text", "What is the capital of France?"), ("speech", spoken_question)]:
    for modality in ("text", "audio"):
        body = {
            "model": "moss-speech",
            "messages": [{"role": "user", "content": content}],
            "modalities": [modality],
            "stream": False,
            "max_tokens": 512,
        }
        if modality == "audio":
            body["audio"] = {"format": "wav", "ref_audio": voice_reference}
        response = requests.post("http://localhost:8008/v1/chat/completions", json=body, timeout=300)
        response.raise_for_status()
        message = response.json()["choices"][0]["message"]
        if modality == "audio":
            Path(f"{source}-reply.wav").write_bytes(base64.b64decode(message["audio"]["data"]))
        else:
            print(message["content"])
```
