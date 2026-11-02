# Ming-3.0-audio-flash

[Ming-3.0-audio-flash](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash) transcribes speech, answers questions about a recording, and synthesizes speech. Voice cloning uses one reference recording together with the transcript of that recording. The pipeline runs a Whisper audio encoder, a tensor-parallel Ling V3 thinker, and a continuous-latent speech decoder.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

The checkpoint contains the thinker shards and a complete `talker/` directory. Serving needs two GPUs. `--thinker.tp_size 2` places the thinker on both visible devices. The audio encoder uses GPU 0, and speech synthesis uses GPU 1.

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/Ming-3.0-audio-flash \
  --thinker.tp_size 2
```

## Transcription

```bash
curl http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-audio-flash \
  -F file=@input.wav \
  -F temperature=0
```

The model answers with a language label, a tab, and the transcript. This endpoint returns only the transcript. `temperature` 0 selects greedy decoding.

## Speech synthesis

```python
import base64
from pathlib import Path

import requests

reference = base64.b64encode(Path("reference.wav").read_bytes()).decode()
response = requests.post(
    "http://127.0.0.1:8000/v1/audio/speech",
    json={
        "model": "inclusionAI/Ming-3.0-audio-flash",
        "input": "Get the trust fund to the bank early.",
        "ref_audio": f"data:audio/wav;base64,{reference}",
        "ref_text": "The exact transcript of the reference recording.",
        "response_format": "wav",
        "seed": 123,
    },
    timeout=180,
)
response.raise_for_status()
Path("speech.wav").write_bytes(response.content)
```

The response body is one complete 44.1 kHz waveform. The speech stage handles one request at a time. `ref_text` should be the words spoken in `reference.wav`. `seed` fixes the flow-matching noise.

## Audio chat

`audios` entries are file paths or URLs that the server reads.

```python
import requests

response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/Ming-3.0-audio-flash",
        "messages": [
            {"role": "user", "content": "What does the speaker say?"},
        ],
        "audios": ["input.wav"],
        "modalities": ["text"],
    },
    timeout=180,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

The chat response keeps the generated answer. Set `modalities` to `["text", "audio"]` to synthesize that answer as well. The same recording can be sent inside a message as `input_audio` or `audio_url` instead of `audios`.
