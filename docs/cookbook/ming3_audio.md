# Ming-3.0-audio-flash

[Ming-3.0-audio-flash](https://huggingface.co/inclusionAI/Ming-3.0-audio-flash) transcribes speech, answers questions about a recording, and synthesizes speech. Voice cloning uses one reference recording together with the transcript of that recording. The pipeline runs a Whisper audio encoder, a tensor-parallel Ling V3 thinker, and a continuous-latent speech decoder.

## Installation and Startup

Install SGLang-Omni by following [Installation](../get_started/installation.md).

The checkpoint contains the thinker shards and a complete `talker/` directory. Serving needs two GPUs. `--thinker.tp_size 2` places the thinker on both visible devices. The audio encoder uses GPU 0, and speech synthesis uses GPU 1.

Download the checkpoint once into a local directory. Run the commands below from the same working directory so every stage uses the same weights, tokenizer and audio configuration.

```bash
hf download inclusionAI/Ming-3.0-audio-flash \
  --revision db94e0de19f6d05adc0413c255a28eb480cd1fae \
  --local-dir inclusionAI/Ming-3.0-audio-flash
```

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/Ming-3.0-audio-flash \
  --thinker.tp_size 2
```

## MTP decoding

Multi-token prediction (MTP) can accelerate transcription and audio-chat text generation using the prediction layer included in the checkpoint. MTP is off by default. Enable it with:

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/Ming-3.0-audio-flash \
  --thinker.tp_size 2 \
  --thinker.engine.revision db94e0de19f6d05adc0413c255a28eb480cd1fae \
  --thinker.engine.speculative_algorithm NEXTN
```

MTP uses additional GPU memory. Speedup depends on the workload and output length. Standalone speech synthesis is unaffected.

Usage notes:

- To disable MTP in a config that enables it, add `--thinker.engine.disable_speculative_decoding true`.
- MTP uses synchronous decoding (`--thinker.factory.enable_async_decode false`, the default).
- Keep `speculative_eagle_topk=1` and `repetition_penalty=1`. Prefix caching and chunked prefill are unavailable for this model.

For MTP acceptance statistics, set `return_spec_tokens_details=true` on chat or transcription requests and read `sglext.spec_tokens_details` in the JSON response.

## DFlash decoding

The thinker can also verify tokens proposed by the DFlash draft [`wilsonzheng0327/Ming-3.0-audio-flash-dflash`](https://huggingface.co/wilsonzheng0327/Ming-3.0-audio-flash-dflash), which was trained on English and Chinese transcription and proposes 15 tokens per step. Use it for a server that mainly transcribes; for audio chat, use MTP. The draft is off by default. Enable it with:

```bash
CUDA_VISIBLE_DEVICES=0,1 sgl-omni serve \
  --model-path inclusionAI/Ming-3.0-audio-flash \
  --thinker.tp_size 2 \
  --thinker.engine.speculative_algorithm DFLASH \
  --thinker.engine.speculative_draft_model_path wilsonzheng0327/Ming-3.0-audio-flash-dflash \
  --thinker.engine.speculative_num_draft_tokens 16
```

Speed against the same server without a draft, on two H200s with SGLang 0.5.23, greedy decoding, the thinker's CUDA graphs on and the default 8 running requests. Transcription uses held-out English and Chinese FLEURS, LibriSpeech, VoxPopuli and YODAS2 clips that the draft never saw. Audio questions are MMSU questions with a 512-token budget, which the model answers with an explanation. Audio chat asks what other held-out clips are about:

| Workload, 1 at a time / 8 at a time | No draft | MTP | DFlash |
|---|---:|---:|---:|
| Transcription | 2.62 / 8.46 clips/s | 1.33x / 1.21x | 1.64x / 1.46x |
| Audio questions, about 130-token answers | 136 / 529 tokens/s | 1.58x / 1.25x | 0.67x / 0.63x |
| Audio chat in English, about 34-token answers | 98 / 284 tokens/s | 1.30x / 1.20x | 0.98x / 0.98x |
| Audio chat in Chinese, about 29-token answers | 89 / 253 tokens/s | 1.26x / 1.16x | 0.96x / 0.90x |

The draft accepts about 6.8 tokens per step on transcripts but only 1.3-2.5 on chat answers, so it slows chat down; MTP accepts 2.4-2.8 on every workload. Each server takes one algorithm, so pick the one that matches most of its traffic.

More running requests on two GPUs, transcribing held-out clips with the load equal to `max_running_requests`:

| Running requests | No draft | MTP | DFlash |
|---|---:|---:|---:|
| 8 | 8.46 clips/s | 10.24 (1.21x) | 12.36 (1.46x) |
| 16 | 10.59 clips/s | 13.36 (1.26x) | Does not start |
| 32 | 13.46 clips/s | 16.63 (1.24x) | Does not start |
| 64 | 16.40 clips/s | Does not start | Does not start |

Usage notes:

- Verifying on the Kimi delta attention layers keeps one state per proposed token per running request, about 4.9 GB per GPU for the draft at 8 requests. Where a table says the server does not start, it stops at startup with `Loaded weights leave no GPU memory for the KV cache`, and raising `mem_fraction_static` does not help.
- On four GPUs (`--thinker.tp_size 4`) the draft starts with 32 running requests and served 18.33 held-out clips/s, against 16.13 with MTP and 13.96 without a draft at the same cap.
- SeedTTS English WER is 1.227% and Chinese CER 0.747% with the draft, against 1.219% and 0.741% without it. One at a time, MTP output is identical to the server without a draft; the draft's is identical on 81 of 84 held-out transcripts and differs more on long answers. At 8 at a time, both agree with the server without a draft on 263-264 of 280 transcripts, because greedy decoding on this thinker depends on the batch.
- The draft decodes synchronously. Keep `repetition_penalty=1`. Standalone speech synthesis is unaffected.

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

Set `MING3_AUDIO_DATA`, `MING3_AUDIO_RUNTIME` and `MING3_AUDIO_BUILD` to separate local directories. Model provisioning uses the existing CI Python requirements and verifies the revision and hashes in `ci/model_pins.json`. The BF16 weights occupy about 17.7 GB; on an M4 Pro with 48 GiB of memory, the server's peak memory footprint was 18.8 GB. Quantized checkpoints are not supported.

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
- Input is nonempty 16 kHz mono PCM16 or float32 WAV. Audio is encoded in 30-second chunks. Top-level `audios`, file paths and `audio_url` are not supported.
- Generation is greedy (`temperature=0`), with a positive output token budget. Both `ming3_audio_server` and `ming3_audio_chat` accept `--context-length N` (default: 8,192); the effective limit is the smaller of this value and the checkpoint's `max_position_embeddings`. The prompt, including audio tokens, and the requested output budget must fit within that limit. `max_completion_tokens` takes precedence over `max_tokens`.
- Streaming, speech output and multi-turn conversations are not supported. The transcription and speech endpoints above are not available on this server.
- Request bodies are limited to 4 MiB. One generation runs at a time; concurrent requests receive 429. `/health` and `/v1/models` are supported.

To convert another recording to the accepted WAV format:

```bash
ffmpeg -i recording.mp3 -ar 16000 -ac 1 -c:a pcm_s16le input.wav
```

For offline inference, `ming3_audio_chat` accepts the same checkpoint and WAV input and prints JSON containing the answer, finish reason and token counts. `--prompt` is the text after the audio, and `--system` adds a system message:

```bash
"$MING3_AUDIO_RUNTIME/bin/ming3_audio_chat" \
  --model-path "$MING3_AUDIO_DATA/models/inclusionAI_Ming-3.0-audio-tiny" \
  --audio input.wav --prompt 'What does the speaker say?' --max-new-tokens 64
```
