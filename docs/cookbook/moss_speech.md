# MOSS-Speech

MOSS-Speech answers English and Chinese questions through `/v1/chat/completions`. Send a text or spoken question and select one output modality. Speech replies require a voice reference and return a complete 24 kHz waveform. The deployment uses one CUDA GPU, TP1, BF16 autoregressive weights, FlashInfer attention and eager decoding. The default AR concurrency is four.

## Launch

Install the [codec dependencies and pinned Python sources](../get_started/installation.md#moss-speech-codec-dependencies). Run these commands from the SGLang-Omni checkout. In a fresh shell, activate the same environment and restore the source paths first.

```bash
source .venv/bin/activate
moss_speech_sources="$VIRTUAL_ENV/src/moss-speech-codec"
export PYTHONPATH="$moss_speech_sources:$moss_speech_sources/MOSS-Speech:$moss_speech_sources/Matcha-TTS${PYTHONPATH:+:$PYTHONPATH}"
hf download OpenMOSS-Team/MOSS-Speech --revision cff025bb41d8459d59abac0b5e44aba7f659ec9e --local-dir ./weights/MOSS-Speech
hf download OpenMOSS-Team/MOSS-Speech-Codec --revision eeec733e4e1dea7da444d332d8e1621ef257414c --local-dir ./weights/MOSS-Speech-Codec
sgl-omni serve --config examples/configs/moss_speech.yaml --model-name moss-speech --port 8008
```

The YAML sets `preprocessing.factory.codec_path` for speech input and `audio_vocoder.factory.codec_path` for speech output. To enable optional decode CUDA Graphs, append `--ar_engine.engine.disable_cuda_graph false` to the launch command.

## Send text and spoken questions

Save a recording of a question as `question.wav` and a voice reference as `voice.wav` in the client directory. The example sends each combination once. References accept 80 ms to 30 seconds of finite audio. Each media item is limited to 10 MiB. The combined prompt and generation budget is 40,960 token rows, with speech input encoded at 12.5 tokens per second.

```python
import base64
from pathlib import Path

import requests

def wav_data(filename):
    return base64.b64encode(Path(filename).read_bytes()).decode("ascii")

question = [{"type": "input_audio", "input_audio": {"format": "wav", "data": wav_data("question.wav")}}]
voice = "data:audio/wav;base64," + wav_data("voice.wav")
for source, content in [("text", "What is the capital of France?"), ("speech", question)]:
    for output in ["text", "audio"]:
        body = {
            "messages": [{"role": "user", "content": content}],
            "modalities": [output],
            "stream": False,
            "max_tokens": 512,
        }
        if output == "audio":
            body["audio"] = {"format": "wav", "ref_audio": voice}
        response = requests.post("http://localhost:8008/v1/chat/completions", json=body, timeout=300)
        response.raise_for_status()
        result = response.json()
        choice = result["choices"][0]
        if output == "audio":
            Path(f"{source}-reply.wav").write_bytes(base64.b64decode(choice["message"]["audio"]["data"]))
        else:
            print(choice["message"]["content"])
        print(choice["finish_reason"], result["usage"])
```

Use an `audio_url` part with `{"type": "audio_url", "audio_url": {"url": "..."}}` for server-resolved recordings. URLs and references follow the server's media policy. Local files require `--allowed-local-media-path`. An embedded server should pass `architectures=["MossSpeechForCausalLM"]` to `create_app` along with its media policy. The configured server returns HTTP 400 for streaming chat and for requests to the verbatim synthesis endpoint `/v1/audio/speech`.
