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
This remains an experimental eager baseline: successful synthesis does not imply
production latency, perceptual quality or support for the optimizations below.
Do not interpret the upstream H100 latency numbers as SGLang-Omni results.
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

All stages default to the same GPU/process. This is a CUDA/BF16 eager baseline;
**one logical request (two CFG branch rows) runs in the AR engine at a time**.
Additional HTTP requests queue. Preprocessing and vocoder work can overlap AR,
but this is not a claim of continuous-batched Breeze generation. Do not increase
`tts_engine.engine.max_running_requests` above its required value of `2`: it
counts the conditional and unconditional rows, not two independent requests.

TP, quantization, CUDA graphs, chunked prefill, radix prefix reuse and AR
retraction are not implemented. Disabling prefix reuse is deliberate: placeholder
IDs do not identify the continuous prompt/reference embeddings. The complete
CFG pair is admitted together, with enough KV capacity for its bounded lifetime.

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
flushes, long-form output, cancellation, queued-request isolation and input errors.
Without `BREEZE_TEST_BASE_URL`, the real-weight tests are skipped.

For performance and quality evaluation, use a controlled CUDA device and record
checkpoint/code revisions, environment, WER/CER and sample counts, TTFA, RTF,
end-to-end latency and GPU memory at concurrency 1/8/32. Those concurrency values
measure queued-load behavior for this baseline, not AR batch sizes. Listening
checks remain necessary for voice similarity and instruction-following quality;
dynamic batching and CUDA graphs require separate validation.

Reference implementation: [breezeblue-ai/breeze-tts](https://github.com/breezeblue-ai/breeze-tts/tree/43e2ea1595297c4059477e2e4a300653761c759b).
