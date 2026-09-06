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
This remains an experimental implementation: successful synthesis does not
imply production latency or perceptual quality. Do not interpret the upstream
H100 latency numbers as SGLang-Omni results.
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

The AR engine and preprocessing share one GPU process; codec decoding runs in a
second process on the same GPU, because it is host-dispatch bound like the AR
stage and would otherwise contend with it for the interpreter. The CUDA/BF16 AR
engine runs up to **16 logical requests concurrently** by default. Each logical
request expands to adjacent conditional/unconditional CFG rows internally; the
public `tts_engine.engine.max_running_requests` setting counts logical requests,
not those internal rows. Preprocessing and codec decoding overlap AR generation.

SGLang deterministic inference with FA3 attention is required so a seeded request
keeps the same output as neighboring requests join or leave the live batch. The
backbone and depth decode steps are CUDA-graph captured. TP, quantization,
torch.compile, chunked prefill, radix prefix reuse and AR retraction
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

SeedTTS-Eval full set on one H20, BF16/FA3 deterministic execution, concurrency
16, seed 42 and `max_new_tokens=750`. The checkpoint revision is
`799624c0b4a1daa8db6d28bbd9850043c0270734` and the dataset revision is
`27f4c1adee83b5b29b7c4b375f6b976324bda308`. These are single-GPU reference
numbers for this configuration, not a tuned upper bound.

| Lang | Samples / failures | QPS | Audio s/s | Latency mean / p95 (s) | TTFA p95 (s) | RTF mean |
|---|---:|---:|---:|---:|---:|---:|
| EN | 1088 / 0 | 2.868 | 12.25 | 5.553 / 8.065 | 2.483 | 1.336 |
| ZH | 2020 / 0 | 2.634 | 14.46 | 6.061 / 8.106 | 1.974 | 1.113 |

`/model_info` polling observed 32 live SGLang rows (16 logical requests); the
median active sample also had 32 rows. RTF is per request and includes time
spent sharing the batch; aggregate `Audio s/s` shows the c16 server producing
over twelve seconds of audio per wall-clock second.

Quality uses Qwen3-ASR-1.7B revision
`7278e1e70fe206f11671096ffdd38061171dd6e5`, WavLM cosine similarity ×100 and
predicted UTMOS. All generated rows were scored; no sample exceeded 50% WER/CER.

| Lang | Corpus WER/CER | >50% outliers | WavLM SIM | UTMOS |
|---|---:|---:|---:|---:|
| EN | 1.767% WER | 0 / 1088 | 69.159 | 3.910 |
| ZH | 0.968% CER | 0 / 2020 | 74.567 | 3.184 |

These automated scores are not human listening tests. Voice similarity,
instruction following and naturalness still require perceptual review.

## Performance characteristics

The codec emits 12.5 frames per second, so an autoregressive step must finish
within 80 ms to reach RTF 1. The cost is host dispatch rather than compute:
`num_codebooks` is 16, so every audio frame runs 15 sequential 12-layer depth
forwards (180 layer passes) against 28 backbone layer passes, and each forward
costs a fixed ~6.3 ms of host dispatch versus ~4.3 ms of device time, unchanged
from 2 to 32 rows.

Sampling is vectorized across the batch through per-row seeds instead of one call
per request, which makes the depth loop's cost independent of batch size (137 ms
at sixteen requests before, ~104 ms after). The whole fifteen-step loop then runs
under a CUDA graph with a static KV cache, captured per batch bucket at startup,
which cuts it to 23.6 ms at one request and 28.9 ms at sixteen. Replays reproduce
the eager tokens exactly.

The backbone step is graph captured too. A decode step feeds it a continuous
frame embedding rather than a token, and SGLang's decode graph only refreshes
registered ForwardBatch slots on replay, so the step's embeddings are staged in a
fixed table with `input_ids` holding their row indices -- a stable pointer the
graph can replay.

Codec decode is launch bound: a chunk costs ~10.6 ms at one frame and ~11.3 ms at
sixteen, so the per-frame cost falls from 5.40 ms to 0.70 ms as the chunk grows.
The stage therefore ramps its chunk size (2, 4, 8, then 16 frames) instead of
decoding every 2 frames, which keeps the first chunk small for time to first
audio. Batching several streams into one decode was measured as the worse trade:
it costs more per stream-frame than one large single-stream chunk (0.776 ms
against 0.703 ms) and is not bit-exact against decoding a stream alone.

The codec stage also runs in its own process. It is host-dispatch bound like the
autoregressive stage, so sharing one process made the two contend for the
interpreter rather than overlap; a concurrent codec thread inflated an eager
depth frame from 99 to 227 ms at one request. Streaming frames cross the process
boundary host-side, so the transport never needs CUDA IPC. Preprocessing overlaps
four requests for the same reason.

Measured end to end on the fixed 32-sample streaming sweep, one H20, against the
same sweep on the pre-optimization implementation. Zero failures either side:

| Group | QPS | RTF mean | TTFA p95 (s) |
|---|---|---|---|
| EN c1 | 0.097 -> 0.507 | 2.337 -> 0.475 | 0.24 |
| EN c8 | 0.446 -> 2.022 | 3.648 -> 0.916 | 1.89 |
| EN c16 | 0.613 -> 2.540 | 4.961 -> 1.336 | 4.35 |
| EN c32 | 0.638 -> 2.496 | 8.505 -> 2.331 | 9.64 |
| ZH c1 | 0.080 -> 0.377 | 2.339 -> 0.464 | 0.24 |
| ZH c8 | 0.389 -> 1.599 | 3.607 -> 0.832 | 2.04 |
| ZH c16 | 0.516 -> 2.222 | 5.117 -> 1.148 | 4.18 |
| ZH c32 | 0.510 -> 2.196 | 8.597 -> 1.952 | 10.33 |

A request runs below realtime up to concurrency 8. Throughput saturates at the
16-logical-request running limit, so c32 mainly adds queue delay. Seeded output
stayed byte-identical across c1/c8/c16/c32 throughout.

Still not enabled: torch.compile, quantization, chunked prefill, retraction and
radix prefix reuse. Per-step host work in the runner still grows with batch size
and is the next thing to attack.

Reference implementation: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts/tree/43e2ea1595297c4059477e2e4a300653761c759b).
