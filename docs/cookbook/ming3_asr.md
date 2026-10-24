# Ming-3.0-asr-tiny

[Ming-3.0-asr-tiny](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny) transcribes Chinese and English speech. A Whisper audio encoder and a Ling V3 mixture-of-experts language model share the work. The encoder's embeddings fill the audio placeholder tokens of the transcription prompt.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

Startup uses one GPU. Preprocessing runs on the CPU. The audio encoder and the language model share that GPU.

```bash
sgl-omni serve --model-path inclusionAI/Ming-3.0-asr-tiny
```

Send one audio file to `POST /v1/audio/transcriptions`. The examples below use `tests/data/query_to_cars.wav` and return `how many cars are there in the picture`.

## Transcription

The prompt is the checkpoint's transcription instruction followed by the audio. The chat template is not used. The model answers with a language label, a tab, and the transcript, and this endpoint returns only the transcript. Decoding is greedy unless `temperature` is set. Generation stops at the end-of-text token or after 256 new tokens.

1. Use curl

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F response_format=json
```

2. Use Python

```python
import requests

with open("tests/data/query_to_cars.wav", "rb") as audio_file:
    response = requests.post(
        "http://127.0.0.1:8000/v1/audio/transcriptions",
        data={
            "model": "inclusionAI/Ming-3.0-asr-tiny",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", audio_file, "audio/wav")},
        timeout=180,
    )

response.raise_for_status()
print(response.json()["text"])
```

## Language

`language` accepts `en` or `zh` and prefills that language label. Without `language`, the model identifies the language itself. On this English recording, omitting `language` and setting `language=en` both return the transcript above. A Chinese recording of `明天早上八点在会议室见。` returns `明天早上八点在会议室见` with `language=zh` and also with the hint omitted. Setting `language=en` on that Chinese recording returns `<other>`. Any other language is HTTP 400.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json
```

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F language=fr
```

The second command returns HTTP 400 and `language must be one of ['en', 'zh']`.

## Prompt

`prompt` adds one line of topic or hotword context after the instruction. A prompt that contains a newline is HTTP 400.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F prompt=cars \
  -F response_format=json
```

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  --form-string $'prompt=first line\nsecond line'
```

## Response formats

`response_format` accepts `json`, `text`, and `verbose_json`. `json` is the default. `text` returns the transcript alone. `verbose_json` adds the duration and one segment covering the whole file. `srt` and `vtt` are HTTP 400 because this model does not emit word timestamps.

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=text
```

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=verbose_json
```

An upload longer than 30 seconds is split into 30-second clips, and the clip transcripts are joined.

## Stream transcription

Set `stream=true` to receive the transcript as server-sent events. Use `curl -N` so curl does not buffer the response. This recording returns the finished transcript and then the `[DONE]` sentinel.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```
