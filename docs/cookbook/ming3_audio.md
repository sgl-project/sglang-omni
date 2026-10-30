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

## Apple Silicon: native MLX

The standalone `ming3_audio_server` serves `/v1/chat/completions` on an Apple Silicon Mac using [Ming-3.0-audio-tiny](https://huggingface.co/inclusionAI/Ming-3.0-audio-tiny). Inference runs in C++ with the native runtime's MLX 0.32.3 toolchain, without Python, PyTorch or CUDA.

```bash
python sglang_omni_mlx/native/ci/provision_models.py "$MING3_AUDIO_DATA" \
  --model inclusionAI/Ming-3.0-audio-tiny
sglang_omni_mlx/native/scripts/build_runtime.sh "$MING3_AUDIO_RUNTIME" "$MING3_AUDIO_BUILD"
"$MING3_AUDIO_RUNTIME/bin/ming3_audio_server" \
  --model-path "$MING3_AUDIO_DATA/models/inclusionAI_Ming-3.0-audio-tiny" \
  --model-name inclusionAI/Ming-3.0-audio-tiny --port 8000
```

Set `MING3_AUDIO_DATA`, `MING3_AUDIO_RUNTIME` and `MING3_AUDIO_BUILD` to separate local directories. Model provisioning uses the existing CI Python requirements and verifies the revision and hashes in `ci/model_pins.json`. The original BF16 weights occupy about 17.7 GB; allow additional unified memory for loading and inference. Quantized checkpoints are not supported.

Send the recording as base64 in an `input_audio` content part:

```python
import base64
from pathlib import Path

import requests

audio_base64 = base64.b64encode(Path("input.wav").read_bytes()).decode()
response = requests.post(
    "http://127.0.0.1:8000/v1/chat/completions",
    json={
        "model": "inclusionAI/Ming-3.0-audio-tiny",
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_audio",
                        "input_audio": {"format": "wav", "data": audio_base64},
                    },
                    {"type": "text", "text": "What does the speaker say?"},
                ],
            }
        ],
        "temperature": 0,
        "max_completion_tokens": 64,
        "stream": False,
    },
    timeout=180,
)
response.raise_for_status()
print(response.json()["choices"][0]["message"]["content"])
```

The native server supports audio questions with text responses, with these limits:

- One user message containing exactly one audio clip, with an optional leading system message. Text may appear before or after the audio part.
- Input is 16 kHz mono PCM16 or float32 WAV, from 25 ms to 30 seconds. Top-level `audios`, file paths and `audio_url` are not supported.
- Generation is greedy (`temperature=0`), with 1–1,024 output tokens and an 8,192-token total context limit. `max_completion_tokens` takes precedence over `max_tokens`.
- Streaming, speech output and multi-turn conversations are not supported. The transcription and speech endpoints above are not available on this server.
- Request bodies are limited to 4 MiB. One generation runs at a time; concurrent requests receive 429. `/health` and `/v1/models` are supported.

To convert another recording to the accepted WAV format:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav
```

For offline inference, `ming3_audio_chat` accepts the same checkpoint and WAV input and prints JSON containing the answer, finish reason and token counts:

```bash
"$MING3_AUDIO_RUNTIME/bin/ming3_audio_chat" \
  --model-path "$MING3_AUDIO_DATA/models/inclusionAI_Ming-3.0-audio-tiny" \
  --audio input.wav --prompt 'What does the speaker say?' --max-new-tokens 64
```

Validated on an M4 Pro with 48 GiB memory, macOS 26.6.1 and MLX 0.32.3, using checkpoint revision `3c9578f641b6745abf2d1239dc8419726f37c0e2`. A 2.256-second English clip and a 5.904-second Chinese clip produced identical prompt and generated token IDs to a reference using the official model classes with PyTorch CPU substitutes for FLA kernels. This checks two short greedy cases, not official CUDA parity or identical intermediate activations. Of three additional high-error corpus cases, two produced identical text and one differed by one Chinese character; that difference remains unresolved. The 28 non-corpus API tests in `native/ci/test_ming3_audio_api.py` passed against the real model, covering English and Chinese transcription, a question about the recording's content, busy-request rejection, disconnect cancellation and shutdown during generation.

The same suite exercises all 392 frozen corpus clips: 359 inputs within 30 seconds are decoded, and 33 longer inputs must return HTTP 400. With greedy decoding and a 256-token output limit, the M4 Pro run measured 1.821% English WER (131 clips), 6.469% Chinese CER (156 clips) and 9.340% mixed-language MER (64 clips), using the native CI normalization. No speech clip was truncated. The remaining eight silence/noise clips all reached the token limit; their outputs are retained but excluded from these error rates under the existing corpus scoring rules. These are native measurements, not an official CUDA comparison or a validated accuracy threshold. CI uploads every response and the aggregate metrics.

On the same machine and checkpoint, serial HTTP transcription of the first 3, 15 and 30 seconds of frozen corpus clip `0344_en_long` measured the following latencies. Each case used two warmup requests followed by five measured requests, greedy decoding and a 256-token output limit. All requests finished with stop rather than reaching the token limit.

| Audio duration | Median HTTP latency | Completion tokens | RTF | RTFx |
| --- | --- | --- | --- | --- |
| 3 s | 1.191 s | 15 | 0.397 | 2.52 |
| 15 s | 3.511 s | 57 | 0.234 | 4.27 |
| 30 s | 6.237 s | 102 | 0.208 | 4.81 |

Launch to the supervised ready event took 4.17 seconds. The server's peak memory footprint across all three cases was 18.84 GB (17.55 GiB), measured with macOS `/usr/bin/time -l`; maximum RSS was 2.30 GB and does not represent the full unified-memory footprint. These measurements cover one recording's prefixes, not a representative performance corpus, and do not establish a speedup over a reference implementation.
