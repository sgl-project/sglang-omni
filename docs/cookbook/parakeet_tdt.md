# Parakeet TDT and Bark 3.0

[Underdog Bark 3.0 Pro](https://huggingface.co/ConwayResearch/Underdog-Bark-3.0-Pro) and [NVIDIA Parakeet TDT 0.6B v3](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3) are FastConformer encoders with a Token-and-Duration Transducer (TDT) decoder. Decoding is greedy and not autoregressive over a language model, so one pass over the encoder output produces the transcript. Bark 3.0 Pro is an English fine-tune of Parakeet TDT 0.6B v3 and writes lowercase text without punctuation. Parakeet TDT 0.6B v3 transcribes 25 European languages with punctuation and casing.

Any Transformers checkpoint whose architecture is `ParakeetForTDT` uses this pipeline. The compact Bark 3.0, Plus and Clean checkpoints ship only a custom `model.bark` file and are not supported yet.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

Startup uses one GPU. One stage decodes the audio, computes features and runs the model in bf16.

```bash
sgl-omni serve --model-path ConwayResearch/Underdog-Bark-3.0-Pro
```

```bash
sgl-omni serve --model-path nvidia/parakeet-tdt-0.6b-v3
```

The examples below use `tests/data/query_to_cars.wav`. Bark 3.0 Pro returns `how many cars are there in the picture`. Parakeet TDT 0.6B v3 returns `How many cars are there in the picture?`.

## Transcription

1. Use curl

```bash
curl -X POST http://127.0.0.1:8000/v1/audio/transcriptions \
  -F model=ConwayResearch/Underdog-Bark-3.0-Pro \
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
            "model": "ConwayResearch/Underdog-Bark-3.0-Pro",
            "response_format": "json",
        },
        files={"file": ("query_to_cars.wav", audio_file, "audio/wav")},
        timeout=180,
    )

response.raise_for_status()
print(response.json()["text"])
```

`response_format` accepts `json`, `text` and `verbose_json`. `verbose_json` returns one segment covering the whole file. `srt` and `vtt` are HTTP 400. `stream=true` returns the finished transcript as one event followed by `[DONE]`.

The stage prepends 0.25 seconds of silence to every clip, as in the Bark 3.0 evaluation.

## Request Parameters

`language` is accepted and echoed in `verbose_json`, but it does not steer decoding. `prompt`, `max_new_tokens`, `repetition_penalty` and a non-zero `temperature` are HTTP 400, because greedy transducer decoding cannot honor them. `/v1/audio/translations` is HTTP 400.

## Batching and Long Audio

Requests that arrive within 5 ms of each other run as one padded batch of up to 16 clips. An undecodable upload fails only its own request. Set `--asr.factory.max_batch_size` and `--asr.factory.max_batch_wait_ms` to change this.

An upload longer than 30 seconds is split into 30-second clips, and the clip transcripts are joined. The encoder attends over the whole clip, so memory grows with the square of its length. `audio_chunking.max_audio_clip_s` may be raised up to 120 seconds, and the stage rejects a single clip longer than that.

## Benchmark

SeedTTS EN (1088 clips) on one RTX 5090, measured with `benchmarks.eval.benchmark_asr_seedtts`:

| Model | Concurrency | Corpus WER | Requests/s | RTFx | Mean latency |
|---|---|---|---|---|---|
| Bark 3.0 Pro | 32 | 1.35% | 81.6 | 387 | 0.386 s |
| Bark 3.0 Pro | 1 | 1.34% | 8.3 | 39 | 0.120 s |
| Parakeet TDT 0.6B v3 | 32 | 1.45% | 82.7 | 392 | 0.382 s |
