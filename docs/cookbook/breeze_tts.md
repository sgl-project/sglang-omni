# Breeze-TTS-2 (experimental)

[Breeze-TTS-2](https://huggingface.co/BreezeBlue/Breeze-TTS-2) supports English
and Chinese voice cloning, reference-free voice design, and instruction-guided
voice direction. This integration implements a native three-stage pipeline:

```text
preprocessing (T5Gemma2 + reference codec)
  -> tts_engine (SGLang Qwen3 backbone + depth decoder)
  -> vocoder (bundled Qwen3-TTS tokenizer, 24 kHz mono)
```

The depth decoder belongs inside the AR stage: its 16-codebook frame is summed
into the next backbone input. The legacy `codec_config` in the outer checkpoint
mentions Mimi, but the released inference runtime actually encodes and decodes
with the Qwen tokenizer in `audio_tokenizer/`. No Mimi weights or runtime are
needed here.

**Validation status:** Full-checkpoint CUDA serving has been exercised on H20
for English/Chinese cloning, design and direction, including streaming,
uploaded voice references, cancellation and queued-request isolation. CPU tests
cover request contracts, component loading, CFG/depth decoding and state ownership.
This remains an experimental eager implementation: successful synthesis does not
imply production latency or perceptual quality. The AR backbone and depth decoder
are continuously batched, but CUDA graphs and compilation remain disabled. Do not
interpret the upstream H100 latency numbers as SGLang-Omni results.
Tracking issue: [#1973](https://github.com/sgl-project/sglang-omni/issues/1973).

## Installation and launch

Use the project's [CUDA installation](../get_started/installation.md). Keep
its Torch / SGLang / Transformers versions, rather than installing the upstream
Breeze requirements into that environment. Install the same optional codec
dependency used by [Qwen3-TTS](qwen3_tts.md):

```bash
apt-get update && apt-get install -y sox
uv pip install --no-deps sox einops
uv pip install --no-deps qwen-tts==0.1.1
```

Both `--no-deps` flags are important: do not replace Omni's Transformers with
`qwen-tts`'s older pin. The existing Qwen compatibility shim is applied before
loading the audio tokenizer. The Breeze text encoder and depth decoder use
Omni's pinned Transformers directly; the official Breeze Python repository is
not an extra runtime dependency.

For reproducible validation, download the checkpoint revision used for this port:

```bash
hf download BreezeBlue/Breeze-TTS-2 \
  --revision 799624c0b4a1daa8db6d28bbd9850043c0270734 \
  --local-dir models/Breeze-TTS-2

sgl-omni serve \
  --model-path models/Breeze-TTS-2 \
  --config examples/configs/breeze_tts.yaml \
  --port 8000
```

All stages default to the same GPU/process. The CUDA/BF16 eager AR engine runs up
to **16 logical requests concurrently** by default. Each logical request expands
to adjacent conditional/unconditional CFG rows internally; the public
`tts_engine.engine.max_running_requests` setting counts logical requests, not
those internal rows. Preprocessing and vocoder execution overlap AR generation.

SGLang deterministic inference with FA3 attention is required so a seeded request
keeps the same output as neighboring requests join or leave the live batch. TP,
quantization, CUDA graphs, chunked prefill, radix prefix reuse and AR retraction
are not implemented. Disabling prefix reuse is deliberate: placeholder IDs do not
identify the continuous prompt/reference embeddings. The scheduler admits only
complete CFG pairs, reserves KV for every configured row's bounded lifetime, and
requires `max_prefill_tokens > 2048` so SGLang cannot split a maximal pair at its
strict page-rounded prefill boundary.

## Voice design

```bash
curl --fail http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{
    "input": "Welcome aboard. Your journey begins now.",
    "instructions": "A warm, thoughtful voice with calm delivery.",
    "cfg_scale": 4,
    "seed": 42,
    "response_format": "wav"
  }' --output design.wav
```

Match the instruction language to the target text. For example, use Chinese
text with `instructions: "声音温暖、清晰，语气轻快。"`.

## Voice cloning and direction

Use one reference clip and its exact transcript. A data URL avoids granting
server-side filesystem or remote-media access:

```bash
REF_AUDIO="data:audio/wav;base64,$(base64 -w0 reference.wav)"
jq -n --arg ref "$REF_AUDIO" '{
  input: "We need to discuss what happened last night.",
  ref_audio: $ref,
  ref_text: "This is the exact transcript of the reference audio.",
  instructions: "Speak slowly with a restrained, serious tone.",
  cfg_scale: 4,
  seed: 42,
  stream: true,
  response_format: "pcm"
}' | curl --fail http://localhost:8000/v1/audio/speech \
  -H 'Content-Type: application/json' --data-binary @- \
  --output direction.pcm
```

The streaming response is signed 16-bit little-endian, mono 24 kHz PCM.
For voice cloning without direction, omit `instructions` and `cfg_scale`.
`references: [{"audio_path": "...", "text": "..."}]` is also accepted under
Omni's existing media access policy. More than one reference or a missing
reference transcript is rejected. Reference codes condition AR only; they are
not prepended to the generated codec stream.

## Request controls

| Field | Default / meaning |
|---|---|
| `instructions` | Voice description or direction; required without a reference |
| `cfg_scale` | `1`; nonnegative guidance applied to backbone and every depth head |
| `seed` | Random if absent; owns a separate RNG per request, not the global device RNG |
| `temperature` | `0.9`; `0` uses greedy sampling |
| `top_k` | `50`; `-1` or `0` disables top-k |
| `top_p` | `1.0` |
| `repetition_penalty` | `1.1`, applied once to generated backbone-code history |
| `max_new_tokens` | Up to `750` audio frames, further bounded by remaining context |
| `voice` | `default`, or an uploaded voice reference; no built-in named voices |
| `speed` | Only `1`; express pace through `instructions` |

The prompt plus generated frames is bounded to 1024 backbone positions. Long
text/reference prompts are rejected rather than silently truncated. Request RNG
is isolated, but byte-identical outputs across hardware/library versions or
agreement with the upstream implementation's RNG sequence are not guaranteed.

## Validation before promoting this integration

Run the CPU regression suite:

```bash
CUDA_VISIBLE_DEVICES='' .venv/bin/python -m pytest tests/unit_test/breeze_tts -q
```

The opt-in real-weight suite runs against a dedicated server; it does not launch
or stop a server. Prepare `en/samples.json` and `zh/samples.json` under a fixture
directory, each containing a list of entries with `ref_audio` (an absolute path
on the server), `ref_text` and `target_text`. Use matching transcripts and permit
that directory with the server's `--allowed-local-media-path` option. Then run:

```bash
BREEZE_TEST_BASE_URL=http://127.0.0.1:8000 \
BREEZE_TEST_FIXTURE_DIR=/path/to/seedtts-fixtures \
  .venv/bin/python -m pytest tests/test_model/test_breeze_tts.py -v
```

The suite covers clone/design/direction in both languages, supported audio formats,
uploaded references, streaming/offline sample preservation, single-frame terminal
flushes, long-form output, invalid inputs, seeded batch invariance, and cancellation
while another request remains in the same AR batch. Without
`BREEZE_TEST_BASE_URL`, the real-weight tests are skipped.

For performance and quality evaluation, use a controlled CUDA device and record
checkpoint/code revisions, environment, raw WER/CER and sample/failure/outlier
counts, TTFA, RTF, end-to-end latency, observed AR batch size and GPU memory.
The shared SeedTTS harness should cover all 1088 English and 2020 Chinese rows at
concurrency 16; any smaller subset must be identified explicitly.

## Benchmark results

SeedTTS-Eval full set on one H20, BF16/FA3 deterministic eager execution,
concurrency 16, seed 42 and `max_new_tokens=750`. The checkpoint revision is
`799624c0b4a1daa8db6d28bbd9850043c0270734` and the dataset revision is
`27f4c1adee83b5b29b7c4b375f6b976324bda308`. These are single-GPU reference
numbers for this configuration, not a tuned upper bound; per-request RTF is
still above realtime and is being optimized.

| Lang | Samples / failures | QPS | Audio s/s | Latency mean / p95 (s) | TTFA p95 (s) | RTF mean |
|---|---:|---:|---:|---:|---:|---:|
| EN | 1088 / 0 | 0.673 | 2.852 | 23.668 / 35.830 | 1.585 | 5.609 |
| ZH | 2020 / 0 | 0.555 | 3.008 | 28.762 / 40.088 | 1.475 | 5.311 |

`/model_info` polling observed 32 live SGLang rows (16 logical requests); the
median active sample also had 32 rows. Peak sampled GPU memory was 60.77 GiB.
RTF is per request and includes time sharing the batch; aggregate `Audio s/s`
shows that the c16 server produced more than two seconds of audio per wall-clock
second even though each request's RTF was above one.

Quality uses Qwen3-ASR-1.7B revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`, WavLM cosine similarity ×100 and
predicted UTMOS. All generated rows were scored; no sample exceeded 50% WER/CER.

| Lang | Corpus WER/CER | >50% outliers | WavLM SIM | UTMOS |
|---|---:|---:|---:|---:|
| EN | 1.507% WER | 0 / 1088 | 69.461 | 3.927 |
| ZH | 1.039% CER | 0 / 2020 | 74.488 | 3.193 |

These automated scores are not human listening tests. Voice similarity,
instruction following and naturalness still require perceptual review. CUDA
graphs also require separate validation.

## Performance characteristics

Per-request RTF is above realtime, and the cost is host dispatch rather than
compute. The codec emits 12.5 frames per second, so a step must finish within
80 ms to reach RTF 1. Three measured effects explain the gap.

First, `num_codebooks` is 16, so every audio frame runs 15 sequential 12-layer
depth forwards (180 layer passes) against 28 backbone layer passes. Each depth
forward costs a fixed ~6.3 ms of host dispatch versus ~4.3 ms of device time,
unchanged from 2 to 32 rows, so `decode_frames` needs about 101 ms at one
request and 137 ms at sixteen.

Second, the vocoder stage decodes one stream at a time (`max_batch_size=1`) with
CUDA graphs disabled, at about 10.8 ms per 2-frame chunk regardless of chunk
size. At concurrency 16 that is roughly 86 ms of serial vocoder work per
autoregressive step.

Third, all three stages run as threads in one process, so the autoregressive and
vocoder threads contend for the interpreter instead of overlapping: with a
concurrent vocoder thread a depth frame grows from 99 to 227 ms at one request
and from 134 to 307 ms at sixteen, while chunk decode grows from 10.8 to 23.9 ms.

Because these costs do not grow with batch size, concurrency raises aggregate
throughput but cannot bring per-request RTF near 1 on its own. Known
optimization candidates, none of them applied or measured end to end yet, are
CUDA-graphing the static depth loop, running the vocoder in its own process,
re-enabling cross-stream vocoder batching and graph capture, vectorizing
per-request sampling, removing per-step host synchronization, and re-enabling
backbone CUDA graphs.

Reference implementation: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts/tree/43e2ea1595297c4059477e2e4a300653761c759b).
