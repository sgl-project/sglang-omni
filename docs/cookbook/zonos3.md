# Zonos3

[Zonos3-Mini](https://huggingface.co/Zyphra-staging/Zonos3-Mini-Staging-Release) is a text-to-speech model from Zyphra. An 11-layer hybrid backbone (Gated DeltaNet, global attention and a latent MoE) generates **10 codec codebooks** at 15 frames per second in a **delay pattern**, and the codec decodes them to **44.1 kHz** speech. It clones a voice from a reference clip. In SGLang-Omni it runs as a `preprocessing → tts_engine → vocoder` pipeline behind the OpenAI-compatible `/v1/audio/speech` endpoint.

## Installation

Install `sglang-omni` by following the [Installation](../get_started/installation.md) guide. No additional model-specific dependencies are required.

## Launch the Server

\```bash
sgl-omni serve --model-path Zyphra/ZONOS3-Mini
\```

Every generation opens the audio stream with three codec-silence frames (0.2 s), which the output does not contain; `--preprocessing.factory.silence_prefill_frames 0` disables it server-wide, and `"prepend_silence": false` disables it per request.

## Synthesizing Speech

Pass a reference clip in `references[0].audio_path` (or `ref_audio`) as a local path, HTTP URL, or base64 data URI. Zonos3 conditions on the audio only, so a transcript is not needed; a clip must be at least 0.5 s long, and one longer than 60 s keeps its first 60 s. Omit the reference to use the default voice.

```bash
curl -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "SGLang-Omni is a great project!",
    "references": [{
      "audio_path": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav"
    }]
  }' \
  --output output.wav
```

**Python**

```python
import requests

resp = requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={
        "input": "Get the trust fund to the bank early.",
        "ref_audio": "https://huggingface.co/datasets/zhaochenyang20/seed-tts-eval-mini/resolve/main/en/prompt-wavs/common_voice_en_10119832.wav",
        "seed": 7,
    },
)
resp.raise_for_status()
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

**Streaming**

With `"stream": true` and `"response_format": "pcm"` the response is `audio/pcm` (16-bit mono, 44.1 kHz) and its chunks arrive while the model is still generating. By default the first chunk targets 12 frames (0.8 s) and later chunks 16 frames; a short utterance may flush fewer. On Hopper GPUs (FlashAttention-3), streams with an available decoder slot retain the codec's key/value windows and convolution histories; streams without a slot decode growing prefixes. Streaming PCM is not guaranteed to be bit-exact with non-streaming WAV.

```bash
curl -N -X POST http://localhost:8000/v1/audio/speech \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Get the trust fund to the bank early.",
    "stream": true,
    "response_format": "pcm"
  }' \
  --output output.pcm
```

```python
import requests

with requests.post(
    "http://localhost:8000/v1/audio/speech",
    json={"input": "Get the trust fund to the bank early.", "stream": True, "response_format": "pcm"},
    stream=True,
) as resp:
    resp.raise_for_status()
    with open("output.pcm", "wb") as output:
        for pcm16_chunk in resp.iter_content(chunk_size=None):
            output.write(pcm16_chunk)
```

`output.pcm` is raw signed 16-bit little-endian mono audio at 44.1 kHz, not a WAV file.

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `max_new_tokens` | `2048` | Maximum generated delay-pattern rows; with 10 codebooks, `N` rows yield at most `N - 9` complete audio frames (15 per second) |
| `temperature` | `0.75` | Applied to every codebook; `0` decodes greedily |
| `top_k` | `80` | |
| `top_p` | `0.69` | Applied after top-k |
| `repetition_penalty` | `2.4` | Over the last 96 generated rows on the first 6 codebooks; `1.0` disables it |
| `seed` | unset | Per-request sampling seed; a request without one draws a random seed |
| `prepend_silence` | server default (`true`) | Codec-silence prefill of the audio stream |
| `condition_buckets` | all unknown | `{"loudness": 0-9, "fmax": 0-9, "speaking_rate": 0-29}`; a table's bucket count means unknown |
| `condition_values` | unset | Raw loudness (LUFS), fmax (Hz) and speaking rate (UTF-8 bytes per second) mapped through the checkpoint's bucket edges |

The decoder's output is scaled by the codec registry's declared 3 dB headroom before PCM16 conversion.

## Benchmark Results

SeedTTS EN voice cloning (1088 samples), 1x H100, concurrency 16, the default `sgl-omni serve` launch above, no request seed, scored with Qwen3-ASR-1.7B. SIM is the WavLM-large speaker similarity between each output and its reference, in percentage points. Each value is the mean of 30 passes (six passes per warm server on five GPUs). Corpus WER is dominated by about 20 whole-sentence failures that the model samples in each pass, so a single pass moves by about 0.6 points; WER below 50 is the stable figure.

| Mode | WER (corpus) | WER below 50 | Samples above 50% WER | SIM | RTF mean | Throughput (qps) | First audio (s) |
|---|---|---|---|---|---|---|---|
| Non-streaming | 3.03% | 1.26% | 21.1 | 50.8 | 0.060 | 55.67 | - |
| Streaming | 3.13% | 1.25% | 22.2 | 50.8 | 0.070 | 48.03 | 0.14 |

Throughput by concurrency on the same H100 (one pass per level; concurrency 16 repeats the table above):

| Concurrency | 1 | 4 | 16 | 32 | 64 |
|---|---|---|---|---|---|
| Non-streaming (qps) | 9.05 | 25.61 | 55.67 | 70.32 | 74.35 |
| Streaming (qps) | 8.54 | 23.22 | 48.03 | 57.97 | 61.66 |
| Streaming first audio (s) | 0.057 | 0.078 | 0.140 | 0.221 | 0.264 |
