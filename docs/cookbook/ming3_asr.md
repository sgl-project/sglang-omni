# Ming-3.0-asr-tiny

[Ming-3.0-asr-tiny](https://huggingface.co/inclusionAI/Ming-3.0-asr-tiny) transcribes Chinese and English speech. A Whisper audio encoder and a Ling V3 mixture-of-experts language model share the work. The encoder's embeddings fill the audio placeholder tokens of the transcription prompt.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).
For Intel GPUs, follow the [Ming-3.0-ASR-Tiny XPU recipe](../other_platforms/installation_xpu.md#ming-30-asr-tiny-speech-to-text-single-xpu).

Startup uses one GPU. Preprocessing runs on the CPU. The audio encoder and the language model share that GPU. The pipeline is preprocessing, then thinker.

```bash
sgl-omni serve --model-path inclusionAI/Ming-3.0-asr-tiny
```

Send one audio file to `POST /v1/audio/transcriptions`. The examples below use `tests/data/query_to_cars.wav` and return `how many cars are there in the picture`.

### Async Decode

Async decode is enabled by default on CUDA. It launches the next decode step before collecting the previous step's CPU results, for both streaming and non-streaming transcription. It remains disabled on other devices. To use synchronous decoding:

```bash
sgl-omni serve --model-path inclusionAI/Ming-3.0-asr-tiny \
  --thinker.factory.enable_async_decode false
```

`--thinker.factory.async_decode_min_batch_size 2` limits lookahead to decode batches with at least two requests; the default threshold is one. Prefill drains pending decode results before admission when predicted by the scheduler. Sampling that depends on output history, including `repetition_penalty` other than `1.0`, uses the synchronous path.

This option does not make audio encoding or prefill asynchronous.

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

Set `stream=true` to receive incremental transcript text as server-sent events. Use `curl -N` so curl does not buffer the response. Append the text from each `transcript.text.delta` event. The final `transcript.text.done` event repeats the complete transcript, followed by the `[DONE]` sentinel.

```bash
curl -N -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=inclusionAI/Ming-3.0-asr-tiny \
  -F file=@tests/data/query_to_cars.wav \
  -F language=en \
  -F response_format=json \
  -F stream=true
```

An illustrative response is:

```text
data: {"type":"transcript.text.delta","delta":"..."}

data: {"type":"transcript.text.done","text":"...","usage":{"type":"duration","seconds":5}}

data: [DONE]
```

The first stable text is emitted immediately; later updates use a 50 ms interval (`--thinker.factory.stream_emit_interval_seconds`). Completion flushes pending text, the language header is removed, and concatenated deltas equal the done event's `text`. When `max_new_tokens` is reached, `transcript.text.done` includes `"finish_reason":"length"` and the partial transcript.

## KDA diagonal anchor

The CUDA thinker applies the KDA diagonal anchor fix from [sglang#48745](https://github.com/sgl-project/sglang/pull/48745) while it is absent from the pinned dependency. The patch aligns the unfused gate normalization anchor with the fused path and is installed before model construction and CUDA Graph capture.
